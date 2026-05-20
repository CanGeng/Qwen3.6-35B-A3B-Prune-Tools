"""Tests for layer-by-layer streaming teacher inference.

These tests build a tiny Qwen3.5 MoE text-only model with random weights,
save its state dict as a fake sharded teacher directory, and verify that:

1. ``load_layer_to_gpu`` reproduces the layer's forward bit-for-bit (against
   the layer's instance from the full model).
2. The router forward hook on ``mlp.gate`` captures ``(seq, num_experts)``
   logits matching the expected ``F.linear`` of the gate weight.
3. End-to-end ``LayerStreamer`` (router stats + cache) matches a direct
   ``model(output_hidden_states=True, output_router_logits=True)`` pass on
   per-sample hidden states and on the recommended-experts list.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

torch.manual_seed(0)

try:
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (  # noqa: F401
        Qwen3_5MoeTextConfig,
        Qwen3_5MoeTextModel,
    )
except ImportError:  # pragma: no cover
    pytest.skip(
        "transformers without qwen3_5_moe (need >=4.6 / 5.x) — streamer tests skipped",
        allow_module_level=True,
    )


def _build_toy_text_config() -> "Qwen3_5MoeTextConfig":
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextConfig

    return Qwen3_5MoeTextConfig(
        vocab_size=64,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=2,
        layer_types=["linear_attention", "full_attention"],
        rms_norm_eps=1e-6,
        max_position_embeddings=64,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        attention_bias=False,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000,
            "partial_rotary_factor": 0.25,
            "mrope_interleaved": True,
            "mrope_section": [3, 3, 2],
        },
    )


def _save_fake_teacher(model: "Qwen3_5MoeTextModel", teacher_dir: Path) -> None:
    """Persist the toy model's state_dict as a sharded HF checkpoint.

    Keys are rewritten to the multimodal trunk path
    ``model.language_model.<key>`` so the streamer's prefix logic finds them.
    """
    from safetensors.torch import save_file

    teacher_dir.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    remapped: dict[str, torch.Tensor] = {
        f"model.language_model.{k}": v.detach().clone() for k, v in state.items()
    }
    weight_map = {k: "model.safetensors" for k in remapped}
    save_file(remapped, str(teacher_dir / "model.safetensors"))
    (teacher_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8"
    )


def _layer_forward(layer, hidden, attention_mask, text_config, layer_idx, device):
    """Run a Qwen3.5 MoE decoder layer with a freshly-built rotary + mask trio."""
    from moe_prune_distill.distill.layer_streamer import build_position_inputs

    pos = build_position_inputs(
        seq_len=hidden.shape[1],
        batch_size=hidden.shape[0],
        device=device,
        dtype=hidden.dtype,
        text_config=text_config,
        attention_mask_2d=attention_mask,
    )
    layer_mask = (
        pos.linear_attn_mask
        if text_config.layer_types[layer_idx] == "linear_attention"
        else pos.causal_mask
    )
    return layer(
        hidden,
        position_embeddings=(pos.cos, pos.sin),
        attention_mask=layer_mask,
        position_ids=pos.text_pos,
        past_key_values=None,
    )


def _build_model_and_save(tmp_path: Path):
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextModel

    text_config = _build_toy_text_config()
    model = Qwen3_5MoeTextModel(text_config)
    model.eval()
    teacher_dir = tmp_path / "teacher"
    _save_fake_teacher(model, teacher_dir)
    return text_config, model, teacher_dir


def test_layer_loader_roundtrip(tmp_path: Path) -> None:
    from moe_prune_distill.distill.layer_streamer import (
        load_layer_to_gpu,
        read_shard_index,
    )

    text_config, model, teacher_dir = _build_model_and_save(tmp_path)
    weight_map = read_shard_index(teacher_dir)
    device = torch.device("cpu")

    for layer_idx in range(text_config.num_hidden_layers):
        loaded = load_layer_to_gpu(
            text_config, layer_idx, weight_map, teacher_dir, device, torch.float32
        )
        ref = model.layers[layer_idx]
        ref.eval()

        attn_mask = torch.ones(1, 6, dtype=torch.long)
        h = torch.randn(1, 6, text_config.hidden_size, dtype=torch.float32)

        with torch.inference_mode():
            out_loaded = _layer_forward(loaded, h, attn_mask, text_config, layer_idx, device)
            out_ref = _layer_forward(ref, h, attn_mask, text_config, layer_idx, device)

        torch.testing.assert_close(out_loaded, out_ref, atol=1e-4, rtol=1e-4)


def test_router_hook_captures_logits(tmp_path: Path) -> None:
    import torch.nn.functional as F

    from moe_prune_distill.distill.layer_streamer import (
        load_layer_to_gpu,
        read_shard_index,
    )

    text_config, _, teacher_dir = _build_model_and_save(tmp_path)
    weight_map = read_shard_index(teacher_dir)

    layer = load_layer_to_gpu(
        text_config, 1, weight_map, teacher_dir, torch.device("cpu"), torch.float32
    )

    captured: list[torch.Tensor] = []

    def hook(_m, _inp, out):
        captured.append(out[0].detach())

    layer.mlp.gate.register_forward_hook(hook)

    attn_mask = torch.ones(1, 5, dtype=torch.long)
    h = torch.randn(1, 5, text_config.hidden_size, dtype=torch.float32)
    with torch.inference_mode():
        _layer_forward(layer, h, attn_mask, text_config, 1, torch.device("cpu"))

    assert captured, "router hook did not fire"
    rl = captured[0]
    assert rl.shape == (5, text_config.num_experts), rl.shape

    # router_logits = F.linear(post_attn_layernormed_hidden, gate.weight)
    # Reproduce by running the layer up to the gate and comparing.
    gate_w = layer.mlp.gate.weight.detach()
    # Re-run forward but stash the gate input via a separate hook.
    gate_inputs: list[torch.Tensor] = []

    def in_hook(_m, inp):
        gate_inputs.append(inp[0].detach())

    layer.mlp.gate.register_forward_pre_hook(in_hook)
    captured.clear()
    with torch.inference_mode():
        _layer_forward(layer, h, attn_mask, text_config, 1, torch.device("cpu"))

    expected = F.linear(gate_inputs[0], gate_w)
    torch.testing.assert_close(captured[0], expected, atol=1e-5, rtol=1e-5)


def test_stream_vs_full_model_equivalence(tmp_path: Path) -> None:
    from moe_prune_distill.distill.layer_streamer import (
        LayerStreamer,
        StreamSample,
        read_shard_index,
    )
    from moe_prune_distill.distill.router_stats import write_router_stats
    from moe_prune_distill.distill.teacher_cache import load_sample_cache

    text_config, model, teacher_dir = _build_model_and_save(tmp_path)
    weight_map = read_shard_index(teacher_dir)

    cache_layers = [0, 1]

    # Capture per-layer raw outputs via plain forward hooks. We deliberately
    # do not use ``output_hidden_states=True``: in transformers 5.x the
    # OutputRecorder ends up holding the post-final-norm tensor for the last
    # entry, which doesn't match what the streamer (correctly) caches.
    captured_layer_out: dict[int, list[torch.Tensor]] = {li: [] for li in cache_layers}
    captured_router: dict[int, list[torch.Tensor]] = {li: [] for li in cache_layers}

    def make_layer_hook(li: int):
        def hook(_m, _args, output):
            captured_layer_out[li].append(output.detach().clone())
        return hook

    def make_router_hook(li: int):
        def hook(_m, _args, output):
            captured_router[li].append(output[0].detach().clone())
        return hook

    handles = []
    for li in cache_layers:
        handles.append(model.layers[li].register_forward_hook(make_layer_hook(li)))
        handles.append(model.layers[li].mlp.gate.register_forward_hook(make_router_hook(li)))

    samples: list[StreamSample] = []
    for i in range(3):
        ids = torch.randint(0, text_config.vocab_size, (6,))
        attn = torch.ones(6, dtype=torch.long)
        samples.append(StreamSample(sid=f"s{i}", input_ids=ids, attention_mask=attn))
        with torch.inference_mode():
            model(input_ids=ids.unsqueeze(0), attention_mask=attn.unsqueeze(0), use_cache=False)

    for h in handles:
        h.remove()

    cache_dir = tmp_path / "cache"
    streamer = LayerStreamer(
        text_config=text_config,
        teacher_dir=teacher_dir,
        weight_map=weight_map,
        samples=samples,
        scratch_dir=tmp_path / "scratch",
        device=torch.device("cpu"),
        dtype=torch.float32,
        cache_dir=cache_dir,
        cache_layers=cache_layers,
        cache_dtype=torch.float32,
        cache_router_logits=True,
    )
    counts = streamer.run()

    # --- per-sample hidden / router equivalence ---
    for i, s in enumerate(samples):
        loaded = load_sample_cache(cache_dir, s.sid)
        for li in cache_layers:
            torch.testing.assert_close(
                loaded["hidden"][li].to(torch.float32),
                captured_layer_out[li][i].squeeze(0),
                atol=5e-3,
                rtol=5e-3,
            )
            torch.testing.assert_close(
                loaded["router"][li].to(torch.float32),
                captured_router[li][i],
                atol=5e-3,
                rtol=5e-3,
            )

    # --- router_stats recommended list equivalence ---
    direct_counts = torch.zeros(
        text_config.num_hidden_layers, text_config.num_experts, dtype=torch.long
    )
    top_k = text_config.num_experts_per_tok
    for li in cache_layers:
        for r in captured_router[li]:
            _, topi = r.float().topk(top_k, dim=-1)
            direct_counts[li].index_add_(
                0, topi.reshape(-1).cpu(), torch.ones(topi.numel(), dtype=torch.long)
            )

    torch.testing.assert_close(counts.float(), direct_counts.float())

    out_path = tmp_path / "router_stats.json"
    out_direct = tmp_path / "router_stats_direct.json"
    write_router_stats(
        out_path,
        model_id="toy",
        num_samples=len(samples),
        num_layers=text_config.num_hidden_layers,
        num_experts=text_config.num_experts,
        top_k=top_k,
        counts=counts,
        target_num_experts=2,
    )
    write_router_stats(
        out_direct,
        model_id="toy",
        num_samples=len(samples),
        num_layers=text_config.num_hidden_layers,
        num_experts=text_config.num_experts,
        top_k=top_k,
        counts=direct_counts,
        target_num_experts=2,
    )
    a = json.loads(out_path.read_text(encoding="utf-8"))
    b = json.loads(out_direct.read_text(encoding="utf-8"))
    for li in cache_layers:
        assert a["layers"][str(li)]["recommended"] == b["layers"][str(li)]["recommended"]


def test_batched_cache_layout_emitted(tmp_path: Path) -> None:
    """After ``run()``, the v2 batched layout is on disk and round-trips.

    Verifies:
      * ``cache_index.json`` exists and lists every sample with a chunk id.
      * ``cache_meta.safetensors`` exists.
      * ``cache_layer_{i}_chunk_{j}.safetensors`` files exist for every
        cached layer × chunk that contains samples.
      * No legacy ``{sid}.safetensors`` files were written.
      * ``load_sample_cache`` reads back the same hidden tensors as the
        streamer wrote (auto-dispatch via the index).
    """
    from moe_prune_distill.distill.layer_streamer import (
        LayerStreamer,
        StreamSample,
        read_shard_index,
    )
    from moe_prune_distill.distill.teacher_cache import (
        BATCHED_INDEX_NAME,
        BATCHED_META_NAME,
        _invalidate_batched_index,
        load_sample_cache,
    )

    text_config, _, teacher_dir = _build_model_and_save(tmp_path)
    weight_map = read_shard_index(teacher_dir)

    samples = [
        StreamSample(
            sid=f"s{i}",
            input_ids=torch.randint(0, text_config.vocab_size, (5,)),
            attention_mask=torch.ones(5, dtype=torch.long),
        )
        for i in range(5)
    ]

    cache_dir = tmp_path / "cache"
    streamer = LayerStreamer(
        text_config=text_config,
        teacher_dir=teacher_dir,
        weight_map=weight_map,
        samples=samples,
        scratch_dir=tmp_path / "scratch",
        device=torch.device("cpu"),
        dtype=torch.float32,
        cache_dir=cache_dir,
        cache_layers=[0, 1],
        cache_dtype=torch.float32,
        cache_router_logits=True,
        chunk_size=2,  # forces 3 chunks for 5 samples (0,1 / 2,3 / 4)
    )
    streamer.run()

    # Index + meta
    assert (cache_dir / BATCHED_INDEX_NAME).is_file()
    assert (cache_dir / BATCHED_META_NAME).is_file()
    index = json.loads((cache_dir / BATCHED_INDEX_NAME).read_text(encoding="utf-8"))
    assert index["version"] == 2
    assert index["chunk_size"] == 2
    assert index["num_chunks"] == 3
    assert sorted(index["samples"].keys()) == [f"s{i}" for i in range(5)]
    # Sample-stable chunking: s0,s1 -> 0; s2,s3 -> 1; s4 -> 2
    assert index["samples"]["s0"]["chunk"] == 0
    assert index["samples"]["s2"]["chunk"] == 1
    assert index["samples"]["s4"]["chunk"] == 2

    # Per-layer chunk files: 2 layers × 3 chunks = 6 files
    chunk_files = sorted(cache_dir.glob("cache_layer_*_chunk_*.safetensors"))
    assert len(chunk_files) == 6, [p.name for p in chunk_files]

    # No legacy per-sample files snuck in
    legacy_files = sorted(cache_dir.glob("s*.safetensors"))
    assert legacy_files == []

    # Round-trip via load_sample_cache (auto-dispatch should pick v2)
    _invalidate_batched_index(cache_dir)
    for s in samples:
        loaded = load_sample_cache(cache_dir, s.sid)
        torch.testing.assert_close(loaded["input_ids"].to(torch.int64), s.input_ids)
        torch.testing.assert_close(
            loaded["attention_mask"].to(torch.int64), s.attention_mask
        )
        assert set(loaded["hidden"].keys()) == {0, 1}
        assert set(loaded["router"].keys()) == {0, 1}
        for li in (0, 1):
            assert loaded["hidden"][li].shape == (5, text_config.hidden_size)
            assert loaded["router"][li].shape == (5, text_config.num_experts)


def test_skip_existing_via_batched_index(tmp_path: Path) -> None:
    """A sample listed in ``cache_index.json`` is reported as cached.

    Mirrors what ``stream_teacher.py``'s ``--skip-existing`` filter relies
    on: ``cache_exists`` must dispatch to the v2 reader and return True for
    sids already present in the index, regardless of whether per-sample
    legacy files exist on disk.
    """
    from moe_prune_distill.distill.layer_streamer import (
        LayerStreamer,
        StreamSample,
        read_shard_index,
    )
    from moe_prune_distill.distill.teacher_cache import (
        _invalidate_batched_index,
        cache_exists,
    )

    text_config, _, teacher_dir = _build_model_and_save(tmp_path)
    weight_map = read_shard_index(teacher_dir)

    samples = [
        StreamSample(
            sid="cached_one",
            input_ids=torch.randint(0, text_config.vocab_size, (4,)),
            attention_mask=torch.ones(4, dtype=torch.long),
        )
    ]

    cache_dir = tmp_path / "cache"
    streamer = LayerStreamer(
        text_config=text_config,
        teacher_dir=teacher_dir,
        weight_map=weight_map,
        samples=samples,
        scratch_dir=tmp_path / "scratch",
        device=torch.device("cpu"),
        dtype=torch.float32,
        cache_dir=cache_dir,
        cache_layers=[0],
        cache_dtype=torch.float32,
        cache_router_logits=True,
        chunk_size=4,
    )
    streamer.run()
    _invalidate_batched_index(cache_dir)

    assert cache_exists(cache_dir, "cached_one")
    assert not cache_exists(cache_dir, "never_seen")


def test_scratch_chunked_layout(tmp_path: Path) -> None:
    """Scratch directory uses chunked files, not 2 per sample.

    With 5 samples + chunk_size=2 the chunked layout is exactly 3 ``.cur``
    files (chunks 0,1,2) during the run; per-sample ``{sid}.cur/.next``
    files must never appear. After ``run()``, ``_cleanup_scratch`` empties
    the directory.

    A forward hook on the last decoder layer captures the directory state
    at the moment we know all but the final layer pass has run, so we get
    a deterministic mid-run probe without depending on parallel processes.
    """
    from moe_prune_distill.distill.layer_streamer import (
        LayerStreamer,
        StreamSample,
        read_shard_index,
    )

    text_config, _, teacher_dir = _build_model_and_save(tmp_path)
    weight_map = read_shard_index(teacher_dir)

    samples = [
        StreamSample(
            sid=f"s{i}",
            input_ids=torch.randint(0, text_config.vocab_size, (4,)),
            attention_mask=torch.ones(4, dtype=torch.long),
        )
        for i in range(5)
    ]

    scratch_dir = tmp_path / "scratch"
    cache_dir = tmp_path / "cache"

    snapshots: dict[str, list[Path]] = {}

    streamer = LayerStreamer(
        text_config=text_config,
        teacher_dir=teacher_dir,
        weight_map=weight_map,
        samples=samples,
        scratch_dir=scratch_dir,
        device=torch.device("cpu"),
        dtype=torch.float32,
        cache_dir=cache_dir,
        cache_layers=[0, 1],
        cache_dtype=torch.float32,
        cache_router_logits=True,
        chunk_size=2,
    )

    # Wrap _layer_pass so we can snapshot scratch dir contents right before
    # cleanup happens.
    original_layer_pass = streamer._layer_pass

    def _instrumented(layer_idx: int) -> None:
        original_layer_pass(layer_idx)
        snapshots[f"after_layer_{layer_idx}"] = sorted(scratch_dir.iterdir())

    streamer._layer_pass = _instrumented  # type: ignore[method-assign]
    streamer.run()

    for name, files in snapshots.items():
        names = [p.name for p in files]
        cur_files = sorted(p.name for p in files if p.name.endswith(".cur.safetensors"))
        # 5 samples / chunk_size=2 → ceil = 3 chunks
        assert cur_files == [
            "scratch_chunk_0.cur.safetensors",
            "scratch_chunk_1.cur.safetensors",
            "scratch_chunk_2.cur.safetensors",
        ], (name, names)
        # No legacy per-sample scratch files snuck in
        assert not any(n.startswith("s") and ".cur" in n for n in names), (name, names)
        assert not any(n.startswith("s") and ".next" in n for n in names), (name, names)
        # No leftover atomic-rename tmp files between layers
        assert not any(n.endswith(".tmp") for n in names), (name, names)

    # _cleanup_scratch swept everything after run().
    assert sorted(scratch_dir.iterdir()) == []


def test_writer_flush_chunk_frees_buffer(tmp_path: Path) -> None:
    """``BatchedCacheWriter.flush_chunk`` writes one file and clears its buffer.

    After flushing a single (layer, chunk), the chunk file is on disk and the
    buffer entry is gone, so a subsequent ``flush_layer`` for the same layer
    is a no-op for that pair.
    """
    from moe_prune_distill.distill.teacher_cache import (
        BatchedCacheWriter,
        _chunk_filename,
    )

    cache_dir = tmp_path / "cache"
    sids = ["s0", "s1", "s2", "s3", "s4"]
    writer = BatchedCacheWriter(
        cache_dir,
        sample_ids=sids,
        cache_layers=[3],
        num_experts=4,
        cache_dtype=torch.float32,
        cache_router_logits=False,
        chunk_size=2,  # → chunks {s0,s1}=0, {s2,s3}=1, {s4}=2
    )

    # Fill chunk 0 (s0, s1) and chunk 1 (s2, s3); leave chunk 2 (s4) empty.
    for sid in ["s0", "s1", "s2", "s3"]:
        writer.add_layer_sample(
            sid, 3, torch.randn(4, 8), router=None
        )

    assert (3, 0) in writer._buffers
    assert (3, 1) in writer._buffers

    writer.flush_chunk(3, 0)

    # Chunk 0 file is on disk, its buffer is gone, chunk 1 buffer is untouched.
    assert (cache_dir / _chunk_filename(3, 0)).is_file()
    assert (3, 0) not in writer._buffers
    assert (3, 1) in writer._buffers

    # Idempotent: calling again is a no-op.
    writer.flush_chunk(3, 0)
    assert (3, 0) not in writer._buffers

    # flush_layer drains the remaining chunk only.
    writer.flush_layer(3)
    assert (cache_dir / _chunk_filename(3, 1)).is_file()
    assert not (cache_dir / _chunk_filename(3, 2)).is_file()  # never had data
    assert writer._buffers == {}


def test_batched_forward_matches_unbatched(tmp_path: Path) -> None:
    """Per-layer ``batch_size > 1`` produces the same hidden / router /
    counts as the unbatched (B=1) path.

    Uses three samples with **mixed** sequence lengths so the padding code
    in ``_load_and_pad_batch`` is exercised. With ``batch_size=4`` they all
    go in one padded batch; with ``batch_size=1`` they go one-by-one.

    Linear ops (attention QKV/O, MoE expert matmul) are row-independent,
    so in fp32 the two paths should agree to machine precision (we use a
    loose tolerance to absorb cuBLAS algorithm picking and any incidental
    fp32→fp32 rounding inside transformers helpers).
    """
    from moe_prune_distill.distill.layer_streamer import (
        LayerStreamer,
        StreamSample,
        read_shard_index,
    )
    from moe_prune_distill.distill.teacher_cache import (
        _invalidate_batched_index,
        load_sample_cache,
    )

    text_config, _, teacher_dir = _build_model_and_save(tmp_path)
    weight_map = read_shard_index(teacher_dir)

    # Mixed seq lengths — chosen so at least one sample needs padding in
    # the batched run (max_in_batch = 7, samples have 7 / 5 / 4 tokens).
    rng = torch.Generator().manual_seed(123)
    seq_lens = [7, 5, 4]
    samples = []
    for i, sl in enumerate(seq_lens):
        ids = torch.randint(0, text_config.vocab_size, (sl,), generator=rng)
        attn = torch.ones(sl, dtype=torch.long)
        samples.append(StreamSample(sid=f"s{i}", input_ids=ids, attention_mask=attn))

    cache_layers = [0, 1]

    def _run(batch_size: int, sub_path: Path):
        streamer = LayerStreamer(
            text_config=text_config,
            teacher_dir=teacher_dir,
            weight_map=weight_map,
            samples=samples,
            scratch_dir=sub_path / "scratch",
            device=torch.device("cpu"),
            dtype=torch.float32,
            cache_dir=sub_path / "cache",
            cache_layers=cache_layers,
            cache_dtype=torch.float32,
            cache_router_logits=True,
            batch_size=batch_size,
            chunk_size=8,
        )
        counts = streamer.run().clone()
        _invalidate_batched_index(sub_path / "cache")
        return counts, sub_path / "cache"

    counts_b1, cache_b1 = _run(1, tmp_path / "b1")
    counts_b4, cache_b4 = _run(4, tmp_path / "b4")

    # Top-k counts are summed over all samples, order-independent — bit-exact.
    torch.testing.assert_close(counts_b1.float(), counts_b4.float())

    # Per-sample tensors equivalence.
    for s in samples:
        ld1 = load_sample_cache(cache_b1, s.sid)
        ld4 = load_sample_cache(cache_b4, s.sid)
        torch.testing.assert_close(
            ld1["input_ids"].to(torch.int64), ld4["input_ids"].to(torch.int64)
        )
        torch.testing.assert_close(
            ld1["attention_mask"].to(torch.int64),
            ld4["attention_mask"].to(torch.int64),
        )
        for li in cache_layers:
            torch.testing.assert_close(
                ld1["hidden"][li], ld4["hidden"][li], atol=1e-5, rtol=1e-5
            )
            torch.testing.assert_close(
                ld1["router"][li], ld4["router"][li], atol=1e-5, rtol=1e-5
            )

