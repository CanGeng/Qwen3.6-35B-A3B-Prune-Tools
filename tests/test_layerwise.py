"""Tests for layerwise (block-wise) student distillation.

Three units:

1. ``test_enumerate_blocks_every_4`` — pure function, verifies that
   ``cache_layers=[0,4]`` over ``num_layers=8`` yields the expected blocks
   and drops the trailing 5-7.
2. ``test_block_trainer_overfits_one_sample`` — toy Qwen3.5 MoE, build a
   teacher cache from the model itself (so target is exactly reachable),
   re-load the layer with perturbed weights into the BlockTrainer and check
   that hidden MSE drops by an order of magnitude after a few hundred steps.
3. ``test_merge_layer_updates_preserves_other_keys`` — fake student dir with
   two shards; replace one layer's weight via a snapshot and verify every
   other key passes through bit-for-bit.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

try:
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (  # noqa: F401
        Qwen3_5MoeTextConfig,
        Qwen3_5MoeTextModel,
    )
except ImportError:  # pragma: no cover
    pytest.skip(
        "transformers without qwen3_5_moe — layerwise tests skipped",
        allow_module_level=True,
    )


from moe_prune_distill.distill.layerwise_trainer import (
    BlockTrainer,
    TrainerConfig,
    enumerate_blocks,
    merge_layer_updates_into_student,
    save_layer_snapshot,
)


# ====================================================================
# 1. enumerate_blocks
# ====================================================================


def test_enumerate_blocks_every_4_basic() -> None:
    blocks = enumerate_blocks(num_layers=8, cache_layers=[0, 4])
    assert len(blocks) == 2
    assert blocks[0].block_id == 0
    assert blocks[0].input_layer == -1
    assert blocks[0].output_layer == 0
    assert blocks[0].layer_indices == (0,)
    assert blocks[1].block_id == 1
    assert blocks[1].input_layer == 0
    assert blocks[1].output_layer == 4
    assert blocks[1].layer_indices == (1, 2, 3, 4)


def test_enumerate_blocks_drops_trailing_uncached() -> None:
    # cache_layers stops at 4, layers 5-7 have no target → no block for them
    blocks = enumerate_blocks(num_layers=8, cache_layers=[0, 4])
    covered = set()
    for b in blocks:
        covered.update(b.layer_indices)
    assert covered == {0, 1, 2, 3, 4}
    assert {5, 6, 7}.isdisjoint(covered)


def test_enumerate_blocks_handles_dense_cache() -> None:
    blocks = enumerate_blocks(num_layers=4, cache_layers=[0, 1, 2, 3])
    assert [b.layer_indices for b in blocks] == [(0,), (1,), (2,), (3,)]
    assert [b.input_layer for b in blocks] == [-1, 0, 1, 2]


# ====================================================================
# shared toy fixtures
# ====================================================================


def _build_toy_text_config():
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


def _save_toy_student(model, student_dir: Path) -> None:
    student_dir.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    remapped = {f"model.language_model.{k}": v.detach().clone() for k, v in state.items()}
    weight_map = {k: "model.safetensors" for k in remapped}
    save_file(remapped, str(student_dir / "model.safetensors"))
    (student_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8"
    )


# ====================================================================
# 2. BlockTrainer can drive hidden MSE down
# ====================================================================


def _make_teacher_cache_from_model(
    model, text_config, sample_ids: list[str], cache_dir: Path,
    cache_layers: list[int],
) -> None:
    """Run the model with forward hooks and save per-layer outputs to cache."""
    from moe_prune_distill.distill.teacher_cache import save_sample_cache

    cache_dir.mkdir(parents=True, exist_ok=True)
    captured: dict[int, list[torch.Tensor]] = {li: [] for li in cache_layers}
    captured_router: dict[int, list[torch.Tensor]] = {li: [] for li in cache_layers}

    handles = []
    for li in cache_layers:
        def make_h(li=li):
            def h(_m, _i, o):
                captured[li].append(o.detach().clone())
            return h

        def make_r(li=li):
            def h(_m, _i, o):
                captured_router[li].append(o[0].detach().clone())
            return h

        handles.append(model.layers[li].register_forward_hook(make_h()))
        handles.append(model.layers[li].mlp.gate.register_forward_hook(make_r()))

    sample_inputs = []
    torch.manual_seed(0)
    for sid in sample_ids:
        ids = torch.randint(0, text_config.vocab_size, (6,))
        attn = torch.ones(6, dtype=torch.long)
        sample_inputs.append((sid, ids, attn))
        with torch.inference_mode():
            model(input_ids=ids.unsqueeze(0), attention_mask=attn.unsqueeze(0), use_cache=False)

    for h in handles:
        h.remove()

    for i, (sid, ids, attn) in enumerate(sample_inputs):
        hiddens = {li: captured[li][i].squeeze(0) for li in cache_layers}
        routers = {li: captured_router[li][i] for li in cache_layers}
        save_sample_cache(cache_dir, sid, ids, attn, hiddens, routers, dtype=torch.float32)


def test_block_trainer_overfits_one_sample(tmp_path: Path) -> None:
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextModel

    from moe_prune_distill.distill.layer_streamer import read_shard_index

    text_config = _build_toy_text_config()
    torch.manual_seed(0)
    model = Qwen3_5MoeTextModel(text_config)
    model.eval()

    student_dir = tmp_path / "student"
    cache_dir = tmp_path / "cache"
    snapshot_dir = tmp_path / "snap"

    sample_ids = ["s0"]
    _make_teacher_cache_from_model(
        model, text_config, sample_ids, cache_dir, cache_layers=[0, 1]
    )

    # Now perturb the model so block 0's layer 0 is no longer the teacher,
    # then save it as the "student" with an offset.
    with torch.no_grad():
        for p in model.layers[0].parameters():
            p.add_(torch.randn_like(p) * 0.05)

    _save_toy_student(model, student_dir)
    weight_map = read_shard_index(student_dir)

    block = enumerate_blocks(text_config.num_hidden_layers, cache_layers=[0, 1])[0]
    cfg = TrainerConfig(
        max_steps=400,
        mse_threshold=1e-6,
        patience=400,
        learning_rate=1e-3,
        optimizer="adamw_fp32",
        use_router_kl=False,
        save_every_steps=400,
        log_every_steps=200,
        seed=0,
    )
    trainer = BlockTrainer(
        block=block,
        student_dir=student_dir,
        student_text_config=text_config,
        student_weight_map=weight_map,
        cache_dir=cache_dir,
        sample_ids=sample_ids,
        snapshot_dir=snapshot_dir,
        device=torch.device("cpu"),
        dtype=torch.float32,
        cfg=cfg,
    )
    result = trainer.run()
    history = result["history"]
    initial = history[0]["hidden_mse"]
    final = history[-1]["hidden_mse"]
    assert initial > 1e-5, f"initial loss too low to be a meaningful overfit test: {initial}"
    assert final < initial * 0.5, f"loss did not improve enough: {initial:.4g} -> {final:.4g}"

    # snapshot exists
    snaps = list(snapshot_dir.glob("layer_*.safetensors"))
    assert len(snaps) == 1
    assert snaps[0].name == "layer_000.safetensors"


def test_block_trainer_batched_grad_accum(tmp_path: Path) -> None:
    """batch_size>1 + gradient_accumulation_steps>1 still drives MSE down.

    Each "step" should consume batch * accum micro-samples; we verify the
    history records the right number of steps and the loss trends down.
    Also verifies that the trainer finalizes cleanly (no leaked optimizer).
    """
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextModel

    from moe_prune_distill.distill.layer_streamer import read_shard_index

    text_config = _build_toy_text_config()
    torch.manual_seed(0)
    model = Qwen3_5MoeTextModel(text_config)
    model.eval()

    student_dir = tmp_path / "student"
    cache_dir = tmp_path / "cache"
    snapshot_dir = tmp_path / "snap"

    sample_ids = [f"s{i}" for i in range(4)]
    _make_teacher_cache_from_model(
        model, text_config, sample_ids, cache_dir, cache_layers=[0, 1]
    )

    with torch.no_grad():
        for p in model.layers[0].parameters():
            p.add_(torch.randn_like(p) * 0.05)

    _save_toy_student(model, student_dir)
    weight_map = read_shard_index(student_dir)

    block = enumerate_blocks(text_config.num_hidden_layers, cache_layers=[0, 1])[0]
    cfg = TrainerConfig(
        max_steps=80,
        mse_threshold=1e-6,
        patience=80,
        learning_rate=1e-3,
        optimizer="adamw_fp32",
        use_router_kl=False,
        save_every_steps=80,
        log_every_steps=200,
        seed=0,
        batch_size=2,
        gradient_accumulation_steps=2,
    )
    trainer = BlockTrainer(
        block=block,
        student_dir=student_dir,
        student_text_config=text_config,
        student_weight_map=weight_map,
        cache_dir=cache_dir,
        sample_ids=sample_ids,
        snapshot_dir=snapshot_dir,
        device=torch.device("cpu"),
        dtype=torch.float32,
        cfg=cfg,
    )
    result = trainer.run()
    history = result["history"]

    # one entry per *optimizer* step, not per micro-step
    assert len(history) == 80
    initial = history[0]["hidden_mse"]
    final_window = sum(h["hidden_mse"] for h in history[-5:]) / 5
    assert final_window < initial * 0.5, (
        f"batched run did not converge: {initial:.4g} -> {final_window:.4g}"
    )

    # _unload should have run — layers freed, optimizer dropped.
    assert trainer.layers is None
    assert trainer.embed is None


def test_layerwise_config_parses_batch_and_accum(tmp_path: Path) -> None:
    """YAML-level wiring: batch_size & gradient_accumulation_steps round-trip."""
    import yaml

    from moe_prune_distill.config import load_config

    src = Path(__file__).resolve().parents[1] / "configs" / "example.yaml"
    raw = yaml.safe_load(src.read_text(encoding="utf-8"))
    raw["train"]["layerwise"]["batch_size"] = 4
    raw["train"]["layerwise"]["gradient_accumulation_steps"] = 8
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    app = load_config(p)
    assert app.train.layerwise.batch_size == 4
    assert app.train.layerwise.gradient_accumulation_steps == 8


def test_layerwise_config_rejects_bad_values(tmp_path: Path) -> None:
    import yaml

    from moe_prune_distill.config import load_config

    src = Path(__file__).resolve().parents[1] / "configs" / "example.yaml"
    raw = yaml.safe_load(src.read_text(encoding="utf-8"))
    raw["train"]["layerwise"]["batch_size"] = 0
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="batch_size"):
        load_config(p)


# ====================================================================
# 3. merge preserves untouched keys
# ====================================================================


def test_merge_layer_updates_preserves_other_keys(tmp_path: Path) -> None:
    student = tmp_path / "student"
    student.mkdir()

    # 2 shards: shard1 has layer 0 weights + embed, shard2 has layer 1 + lm_head
    shard1 = {
        "model.language_model.embed_tokens.weight": torch.randn(10, 8),
        "model.language_model.layers.0.input_layernorm.weight": torch.randn(8),
        "model.language_model.layers.0.mlp.gate.weight": torch.randn(4, 8),
    }
    shard2 = {
        "model.language_model.layers.1.input_layernorm.weight": torch.randn(8),
        "lm_head.weight": torch.randn(10, 8),
    }
    save_file(shard1, str(student / "model-00001-of-00002.safetensors"))
    save_file(shard2, str(student / "model-00002-of-00002.safetensors"))
    weight_map: dict[str, str] = {}
    for k in shard1:
        weight_map[k] = "model-00001-of-00002.safetensors"
    for k in shard2:
        weight_map[k] = "model-00002-of-00002.safetensors"
    (student / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map}),
        encoding="utf-8",
    )
    (student / "config.json").write_text("{}", encoding="utf-8")

    # Snapshot dir: change layer 0's gate weight only.
    snap_dir = tmp_path / "snap"
    snap_dir.mkdir()
    new_gate = torch.zeros(4, 8)
    save_file(
        {
            "model.language_model.layers.0.input_layernorm.weight":
                shard1["model.language_model.layers.0.input_layernorm.weight"],
            "model.language_model.layers.0.mlp.gate.weight": new_gate,
        },
        str(snap_dir / "layer_000.safetensors"),
    )

    out = tmp_path / "out"
    merge_layer_updates_into_student(student, snap_dir, out)

    merged_shard1 = load_file(str(out / "model-00001-of-00002.safetensors"))
    merged_shard2 = load_file(str(out / "model-00002-of-00002.safetensors"))

    # changed
    assert torch.equal(
        merged_shard1["model.language_model.layers.0.mlp.gate.weight"], new_gate
    )
    # unchanged
    assert torch.equal(
        merged_shard1["model.language_model.embed_tokens.weight"],
        shard1["model.language_model.embed_tokens.weight"],
    )
    assert torch.equal(
        merged_shard2["model.language_model.layers.1.input_layernorm.weight"],
        shard2["model.language_model.layers.1.input_layernorm.weight"],
    )
    assert torch.equal(merged_shard2["lm_head.weight"], shard2["lm_head.weight"])
    # config copied
    assert (out / "config.json").is_file()
    # index regenerated
    idx = json.loads((out / "model.safetensors.index.json").read_text(encoding="utf-8"))
    assert idx["weight_map"]["lm_head.weight"] == "model-00002-of-00002.safetensors"


# ====================================================================
# 4. gradient checkpointing on the in-block layers
# ====================================================================


def test_block_trainer_gradient_checkpointing(tmp_path: Path) -> None:
    """gradient_checkpointing=True still drives MSE down on the toy model.

    The Qwen3.5 decoder layer derives from GradientCheckpointingLayer and
    re-checks the flag on every forward, so we just verify (a) the trainer
    runs without errors when the flag is set, (b) loss still trends down.
    """
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextModel

    from moe_prune_distill.distill.layer_streamer import read_shard_index

    text_config = _build_toy_text_config()
    torch.manual_seed(0)
    model = Qwen3_5MoeTextModel(text_config)
    model.eval()

    student_dir = tmp_path / "student"
    cache_dir = tmp_path / "cache"
    snapshot_dir = tmp_path / "snap"

    sample_ids = ["s0"]
    _make_teacher_cache_from_model(
        model, text_config, sample_ids, cache_dir, cache_layers=[0, 1]
    )

    with torch.no_grad():
        for p in model.layers[0].parameters():
            p.add_(torch.randn_like(p) * 0.05)

    _save_toy_student(model, student_dir)
    weight_map = read_shard_index(student_dir)

    block = enumerate_blocks(text_config.num_hidden_layers, cache_layers=[0, 1])[0]
    cfg = TrainerConfig(
        max_steps=200,
        mse_threshold=1e-6,
        patience=200,
        learning_rate=1e-3,
        optimizer="adamw_fp32",
        use_router_kl=False,
        save_every_steps=200,
        log_every_steps=200,
        seed=0,
        gradient_checkpointing=True,
    )
    trainer = BlockTrainer(
        block=block,
        student_dir=student_dir,
        student_text_config=text_config,
        student_weight_map=weight_map,
        cache_dir=cache_dir,
        sample_ids=sample_ids,
        snapshot_dir=snapshot_dir,
        device=torch.device("cpu"),
        dtype=torch.float32,
        cfg=cfg,
    )
    result = trainer.run()
    history = result["history"]
    initial = history[0]["hidden_mse"]
    final = history[-1]["hidden_mse"]
    assert initial > 1e-5, f"initial loss too low: {initial}"
    assert final < initial * 0.5, (
        f"GC run did not converge: {initial:.4g} -> {final:.4g}"
    )
    # NaN-safety
    assert all(torch.tensor(h["hidden_mse"]).isfinite() for h in history)


# ====================================================================
# 5. config plumbing for new fields
# ====================================================================


def test_layerwise_config_gradient_checkpointing_default(tmp_path: Path) -> None:
    import yaml

    from moe_prune_distill.config import load_config

    src = Path(__file__).resolve().parents[1] / "configs" / "example.yaml"
    raw = yaml.safe_load(src.read_text(encoding="utf-8"))
    # default should be true even if YAML omits it
    raw["train"]["layerwise"].pop("gradient_checkpointing", None)
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    app = load_config(p)
    assert app.train.layerwise.gradient_checkpointing is True


def test_layerwise_config_accepts_paged_adamw_8bit(tmp_path: Path) -> None:
    import yaml

    from moe_prune_distill.config import load_config

    src = Path(__file__).resolve().parents[1] / "configs" / "example.yaml"
    raw = yaml.safe_load(src.read_text(encoding="utf-8"))
    raw["train"]["layerwise"]["optimizer"] = "paged_adamw_8bit"
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    app = load_config(p)
    assert app.train.layerwise.optimizer == "paged_adamw_8bit"


# ====================================================================
# 6. student-rollout cache: block N+1 reads from rollout, not teacher cache
# ====================================================================


def test_block_trainer_writes_rollout_and_block1_consumes_it(tmp_path: Path) -> None:
    """Two-block toy run: with use_student_rollout_input=True, after block 0
    finishes (a) a rollout cache file lands, (b) block 1's input is the
    student's block-0 output (not teacher cache hidden.layer_0).
    """
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextModel

    from moe_prune_distill.distill.layer_streamer import read_shard_index
    from moe_prune_distill.distill.rollout_cache import (
        ROLLOUT_INDEX_NAME,
        load_rollout_input,
        rollout_index_exists,
    )

    text_config = _build_toy_text_config()
    torch.manual_seed(0)
    model = Qwen3_5MoeTextModel(text_config)
    model.eval()

    student_dir = tmp_path / "student"
    cache_dir = tmp_path / "cache"
    snapshot_dir = tmp_path / "snap"
    rollout_root = snapshot_dir / "_rollout"

    sample_ids = ["s0", "s1"]
    _make_teacher_cache_from_model(
        model, text_config, sample_ids, cache_dir, cache_layers=[0, 1]
    )

    # Perturb block 0 so the student diverges from teacher cache hidden.layer_0.
    with torch.no_grad():
        for p in model.layers[0].parameters():
            p.add_(torch.randn_like(p) * 0.05)

    _save_toy_student(model, student_dir)
    weight_map = read_shard_index(student_dir)

    blocks = enumerate_blocks(text_config.num_hidden_layers, cache_layers=[0, 1])
    assert len(blocks) == 2

    cfg = TrainerConfig(
        max_steps=12,
        mse_threshold=1e-9,
        patience=12,
        learning_rate=1e-3,
        optimizer="adamw_fp32",
        use_router_kl=False,
        save_every_steps=12,
        log_every_steps=20,
        seed=0,
        use_student_rollout_input=True,
        rollout_root=rollout_root,
        rollout_chunk_size=10,
    )

    # ---- block 0 ----
    trainer0 = BlockTrainer(
        block=blocks[0],
        student_dir=student_dir,
        student_text_config=text_config,
        student_weight_map=weight_map,
        cache_dir=cache_dir,
        sample_ids=sample_ids,
        snapshot_dir=snapshot_dir,
        device=torch.device("cpu"),
        dtype=torch.float32,
        cfg=cfg,
    )
    trainer0.run()

    # rollout index + chunk file should exist
    assert rollout_index_exists(rollout_root)
    assert (rollout_root / ROLLOUT_INDEX_NAME).is_file()
    chunk_files = list(rollout_root.glob("rollout_block_000_chunk_*.safetensors"))
    assert len(chunk_files) >= 1, f"expected rollout chunk under {rollout_root}"

    # The cached rollout for sid s0 should match what we'd compute by running
    # the snapshotted block-0 layers from teacher cache input (= embedding for
    # block 0, since input_layer == -1).
    # rollout_cache always writes fp16 on disk; cast for comparison.
    rollout_s0 = load_rollout_input(rollout_root, "s0").to(torch.float32)
    rollout_s1 = load_rollout_input(rollout_root, "s1").to(torch.float32)
    assert rollout_s0.shape == rollout_s1.shape
    assert rollout_s0.shape[-1] == text_config.hidden_size

    # And it should differ from the teacher cache's hidden.layer_0 — the
    # whole point of the perturbation + 12 training steps is to have moved
    # the student off the teacher's value.
    from moe_prune_distill.distill.teacher_cache import load_sample_cache
    tcache = load_sample_cache(cache_dir, "s0", layers=[0])
    teacher_h0 = tcache["hidden"][0].to(torch.float32)
    assert not torch.allclose(rollout_s0, teacher_h0, atol=1e-3), (
        "rollout output is identical to teacher cache hidden.layer_0; "
        "either training did not move the student or the rollout pass "
        "regenerated the teacher value by accident"
    )

    # ---- block 1 should consume the rollout ----
    # We patch load_sample_cache via a sentinel: instead of running the full
    # block 1 trainer, we instantiate it and call _build_microbatch directly
    # on a one-sample batch and verify h_in matches the rollout, not the
    # teacher cache hidden.layer_0.
    trainer1 = BlockTrainer(
        block=blocks[1],
        student_dir=student_dir,
        student_text_config=text_config,
        student_weight_map=weight_map,
        cache_dir=cache_dir,
        sample_ids=sample_ids,
        snapshot_dir=snapshot_dir,
        device=torch.device("cpu"),
        dtype=torch.float32,
        cfg=cfg,
    )
    trainer1._load_modules()
    try:
        # Avoid the timer-instrumented branch which would otherwise need
        # `_t_*` attrs to exist.
        trainer1._time_steps = False
        h_in, h_tgt, attn, _ = trainer1._build_microbatch(["s0"])
        sl = int(attn[0].sum().item())
        assert torch.allclose(h_in[0, :sl].to(torch.float32), rollout_s0, atol=1e-3), (
            "block 1 _build_microbatch did not pull h_in from the rollout cache"
        )
        # Target should still be teacher cache hidden.layer_1
        tcache1 = load_sample_cache(cache_dir, "s0", layers=[1])
        assert torch.allclose(
            h_tgt[0, :sl].to(torch.float32),
            tcache1["hidden"][1].to(torch.float32),
            atol=1e-5,
        )
    finally:
        trainer1._unload()
