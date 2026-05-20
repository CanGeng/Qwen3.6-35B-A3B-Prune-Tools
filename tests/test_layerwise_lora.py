"""Tests for layerwise LoRA + 4bit (attention sub-modules)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

try:
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (  # noqa: F401
        Qwen3_5MoeTextConfig,
        Qwen3_5MoeTextModel,
    )
except ImportError:  # pragma: no cover
    pytest.skip(
        "transformers without qwen3_5_moe — layerwise LoRA tests skipped",
        allow_module_level=True,
    )

from moe_prune_distill.distill.layer_lora import (
    LoRAWrapper,
    apply_lora_to_layer,
    freeze_base_train_lora,
    snapshot_layer_with_merged_lora,
)
from moe_prune_distill.distill.layer_streamer import (
    load_layer_to_gpu,
    read_shard_index,
)
from moe_prune_distill.distill.layerwise_trainer import (
    BlockTrainer,
    TrainerConfig,
    enumerate_blocks,
)

from tests.test_layerwise import (
    _build_toy_text_config,
    _save_toy_student,
)


_ATTN_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
)


# ====================================================================
# 1. apply_lora_to_layer / freeze_base_train_lora
# ====================================================================


def test_apply_lora_wraps_attention_linears(tmp_path: Path) -> None:
    """LoRA wrappers replace exactly the attention Linears, leaving MoE alone."""
    text_config = _build_toy_text_config()
    torch.manual_seed(0)
    model = Qwen3_5MoeTextModel(text_config)
    model.eval()
    student_dir = tmp_path / "student"
    _save_toy_student(model, student_dir)
    weight_map = read_shard_index(student_dir)

    layer = load_layer_to_gpu(
        text_config, 1, weight_map, student_dir, torch.device("cpu"), torch.float32
    )

    meta = apply_lora_to_layer(
        layer,
        target_modules=_ATTN_TARGETS,
        r=4,
        alpha=8,
        dropout=0.0,
        load_in_4bit=False,
        compute_dtype=torch.float32,
    )

    # Layer 1 is full_attention -> q/k/v/o get wrapped
    assert "self_attn.q_proj" in meta
    assert "self_attn.k_proj" in meta
    assert "self_attn.v_proj" in meta
    assert "self_attn.o_proj" in meta

    # MoE Linears are not wrapped (shared_expert.gate_proj etc.)
    assert "mlp.shared_expert.gate_proj" not in meta
    assert "mlp.shared_expert.up_proj" not in meta
    assert "mlp.shared_expert.down_proj" not in meta

    # Each wrapped path now resolves to a LoRAWrapper.
    for path in meta:
        assert isinstance(layer.get_submodule(path), LoRAWrapper)


def test_freeze_base_train_lora_policy(tmp_path: Path) -> None:
    """Norms frozen, base frozen, LoRA A/B trainable, rest trainable."""
    text_config = _build_toy_text_config()
    torch.manual_seed(0)
    model = Qwen3_5MoeTextModel(text_config)
    model.eval()
    student_dir = tmp_path / "student"
    _save_toy_student(model, student_dir)
    weight_map = read_shard_index(student_dir)

    layer = load_layer_to_gpu(
        text_config, 1, weight_map, student_dir, torch.device("cpu"), torch.float32
    )
    meta = apply_lora_to_layer(
        layer,
        target_modules=_ATTN_TARGETS,
        r=4,
        alpha=8,
        dropout=0.0,
        load_in_4bit=False,
        compute_dtype=torch.float32,
    )
    freeze_base_train_lora(layer, meta)

    # Norms must be frozen.
    norm_names = {
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
    }
    seen_norms = set()
    for name, p in layer.named_parameters():
        if name in norm_names:
            seen_norms.add(name)
            assert not p.requires_grad, f"{name} should be frozen"
    assert seen_norms == norm_names

    # LoRA wrappers: base frozen, A/B trainable.
    for path in meta:
        wrapper = layer.get_submodule(path)
        assert not wrapper.base.weight.requires_grad
        assert wrapper.lora_A.weight.requires_grad
        assert wrapper.lora_B.weight.requires_grad

    # MoE expert stack params trainable.
    for name, p in layer.named_parameters():
        if "mlp.experts.gate_up_proj" in name or "mlp.experts.down_proj" in name:
            assert p.requires_grad, f"{name} should be trainable"

    # Router gate trainable.
    assert layer.mlp.gate.weight.requires_grad


# ====================================================================
# 2. snapshot_layer_with_merged_lora — keys / shapes / values
# ====================================================================


def test_snapshot_merges_lora_into_base_keys(tmp_path: Path) -> None:
    """Snapshot dict matches original layer's state_dict keys + shapes,
    and folds (B @ A) * scaling into the base weight."""
    text_config = _build_toy_text_config()
    torch.manual_seed(0)
    model = Qwen3_5MoeTextModel(text_config)
    model.eval()
    student_dir = tmp_path / "student"
    _save_toy_student(model, student_dir)
    weight_map = read_shard_index(student_dir)

    # Reference state_dict (no LoRA): keys we must reproduce.
    layer_ref = load_layer_to_gpu(
        text_config, 1, weight_map, student_dir, torch.device("cpu"), torch.float32
    )
    ref_sd = {k: v.detach().clone() for k, v in layer_ref.state_dict().items()}

    # Wrap a fresh copy.
    layer = load_layer_to_gpu(
        text_config, 1, weight_map, student_dir, torch.device("cpu"), torch.float32
    )
    meta = apply_lora_to_layer(
        layer,
        target_modules=_ATTN_TARGETS,
        r=4,
        alpha=8,
        dropout=0.0,
        load_in_4bit=False,
        compute_dtype=torch.float32,
    )
    # Pick one path to make the LoRA delta non-zero and check the math.
    target_path = "self_attn.q_proj"
    wrapper = layer.get_submodule(target_path)
    with torch.no_grad():
        wrapper.lora_A.weight.copy_(torch.randn_like(wrapper.lora_A.weight) * 0.1)
        wrapper.lora_B.weight.copy_(torch.randn_like(wrapper.lora_B.weight) * 0.1)
    base_w = wrapper.base.weight.detach().clone()
    expected_delta = (
        wrapper.lora_B.weight.detach().to(torch.float32)
        @ wrapper.lora_A.weight.detach().to(torch.float32)
    ) * float(wrapper.scaling)

    sd = snapshot_layer_with_merged_lora(layer, meta, out_dtype=torch.float32)

    # All keys from the reference layer must be present, with matching shapes.
    for k, v in ref_sd.items():
        assert k in sd, f"missing key in snapshot: {k}"
        assert sd[k].shape == v.shape, f"shape mismatch for {k}"

    # No stray base./lora_A./lora_B. keys leak through.
    for k in sd:
        assert ".base." not in k
        assert ".lora_A" not in k
        assert ".lora_B" not in k

    # Merged weight matches base + delta on the perturbed path.
    merged_q = sd[f"{target_path}.weight"]
    assert torch.allclose(merged_q, base_w + expected_delta, atol=1e-5, rtol=1e-4)

    # Untouched paths (k_proj LoRA still zero) reproduce the reference base.
    untouched_k = "self_attn.k_proj.weight"
    assert torch.allclose(sd[untouched_k], ref_sd[untouched_k], atol=1e-6)


# ====================================================================
# 3. BlockTrainer end-to-end with LoRA enabled
# ====================================================================


def test_block_trainer_with_lora_drives_loss_down(tmp_path: Path) -> None:
    """Toy block trainer with LoRA enabled (no 4bit) reduces hidden MSE.

    Joins the existing Triton-CPU deselect group (see memory
    ``feedback_triton_cpu_test_skip.md``) — the toy model forward goes through
    a Triton kernel and ``aten::_grouped_mm``, both unavailable on CPU.
    Skipped automatically on hosts without a CUDA GPU.
    """
    if not torch.cuda.is_available():
        pytest.skip("toy MoE forward needs CUDA (Triton + _grouped_mm); skipping on CPU")
    device = torch.device("cuda")
    from moe_prune_distill.distill.layer_streamer import build_position_inputs
    from moe_prune_distill.distill.teacher_cache import save_sample_cache

    text_config = _build_toy_text_config()
    torch.manual_seed(0)
    model = Qwen3_5MoeTextModel(text_config).to(device)
    model.eval()

    student_dir = tmp_path / "student"
    cache_dir = tmp_path / "cache"
    snapshot_dir = tmp_path / "snap"

    sample_ids = ["s0", "s1"]
    rng = torch.Generator().manual_seed(0)
    samples: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for sid in sample_ids:
        ids = torch.randint(0, text_config.vocab_size, (6,), generator=rng).to(device)
        attn = torch.ones(6, dtype=torch.long, device=device)
        with torch.no_grad():
            h_in = model.embed_tokens(ids.unsqueeze(0)).squeeze(0)
        samples.append((sid, ids, attn, h_in))

    captured_router: dict[str, torch.Tensor] = {}

    def router_hook(_m, _i, out):
        captured_router["last"] = out[0].detach().clone()

    handle = model.layers[1].mlp.gate.register_forward_hook(router_hook)
    try:
        with torch.inference_mode():
            for sid, ids, attn, h_in in samples:
                pos = build_position_inputs(
                    seq_len=h_in.shape[0],
                    batch_size=1,
                    device=device,
                    dtype=torch.float32,
                    text_config=text_config,
                    attention_mask_2d=attn.unsqueeze(0),
                )
                h_out = model.layers[1](
                    h_in.unsqueeze(0),
                    position_embeddings=(pos.cos, pos.sin),
                    attention_mask=pos.causal_mask,
                    position_ids=pos.text_pos,
                    past_key_values=None,
                )
                save_sample_cache(
                    cache_dir,
                    sid,
                    ids,
                    attn,
                    hiddens={0: h_in, 1: h_out.squeeze(0)},
                    routers={1: captured_router["last"]},
                    dtype=torch.float32,
                )
    finally:
        handle.remove()

    with torch.no_grad():
        for p in model.layers[1].parameters():
            p.add_(torch.randn_like(p) * 0.05)

    _save_toy_student(model.cpu(), student_dir)
    weight_map = read_shard_index(student_dir)

    block = enumerate_blocks(text_config.num_hidden_layers, cache_layers=[0, 1])[1]
    cfg = TrainerConfig(
        max_steps=80,
        mse_threshold=1e-6,
        patience=80,
        learning_rate=1e-2,
        optimizer="adamw_fp32",
        use_router_kl=False,
        save_every_steps=80,
        log_every_steps=200,
        seed=0,
        gradient_checkpointing=False,
        lora_enabled=True,
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        lora_target_modules=_ATTN_TARGETS,
        lora_load_in_4bit=False,
        lora_compute_dtype="float32",
    )
    trainer = BlockTrainer(
        block=block,
        student_dir=student_dir,
        student_text_config=text_config,
        student_weight_map=weight_map,
        cache_dir=cache_dir,
        sample_ids=sample_ids,
        snapshot_dir=snapshot_dir,
        device=device,
        dtype=torch.float32,
        cfg=cfg,
    )
    result = trainer.run()
    history = result["history"]
    initial = history[0]["hidden_mse"]
    final_window = sum(h["hidden_mse"] for h in history[-5:]) / 5
    assert initial > 1e-5, f"initial loss too low: {initial}"
    assert final_window < initial * 0.7, (
        f"LoRA run did not converge: {initial:.4g} -> {final_window:.4g}"
    )

    # Snapshot exists and its keys match the un-wrapped layer's keys.
    snaps = list(snapshot_dir.glob("layer_*.safetensors"))
    assert len(snaps) == 1
    from safetensors.torch import load_file
    snap_sd = load_file(str(snaps[0]))
    layer_ref = load_layer_to_gpu(
        text_config, 1, weight_map, student_dir, torch.device("cpu"), torch.float32
    )
    expected_keys = {
        f"model.language_model.layers.1.{k}" for k in layer_ref.state_dict().keys()
    }
    assert set(snap_sd.keys()) == expected_keys


def test_layerwise_lora_config_round_trip(tmp_path: Path) -> None:
    """YAML wiring: lora subsection parses with defaults and overrides."""
    import yaml

    from moe_prune_distill.config import load_config

    cfg_path = Path(__file__).resolve().parent.parent / "configs" / "example.yaml"
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["train"]["layerwise"]["lora"] = {
        "enabled": True,
        "r": 8,
        "alpha": 16,
        "dropout": 0.1,
        "target_modules": ["q_proj", "k_proj"],
        "load_in_4bit": False,
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_quant_type": "nf4",
    }
    out_path = tmp_path / "cfg.yaml"
    out_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    app = load_config(out_path)
    lora = app.train.layerwise.lora
    assert lora.enabled is True
    assert lora.r == 8
    assert lora.alpha == 16
    assert lora.dropout == 0.1
    assert lora.target_modules == ("q_proj", "k_proj")
    assert lora.load_in_4bit is False


def test_layerwise_lora_default_disabled(tmp_path: Path) -> None:
    """Missing layerwise.lora subsection → enabled=False, parsing succeeds."""
    import yaml

    from moe_prune_distill.config import load_config

    cfg_path = Path(__file__).resolve().parent.parent / "configs" / "example.yaml"
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["train"]["layerwise"].pop("lora", None)
    out_path = tmp_path / "cfg.yaml"
    out_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    app = load_config(out_path)
    assert app.train.layerwise.lora.enabled is False


# ====================================================================
# 4. partition_for_muon / partition_for_sso route LoRA params to AdamW branch
# ====================================================================


def test_partition_for_muon_sends_lora_to_adamw_branch() -> None:
    """LoRA A/B params (2D matrices) should land in the AdamW branch, not Muon."""
    from moe_prune_distill.distill.muon_triton import partition_for_muon

    # Simulate a layer's named_parameters: a "real" 2D weight, MoE 3D stack,
    # and 2 LoRA matrices. All requires_grad=True.
    real = nn.Parameter(torch.randn(8, 8))
    moe = nn.Parameter(torch.randn(4, 8, 8))
    a = nn.Parameter(torch.randn(4, 8))
    b = nn.Parameter(torch.randn(8, 4))
    named = [
        ("self_attn.q_proj.base.weight", real),
        ("mlp.experts.gate_up_proj", moe),
        ("self_attn.q_proj.lora_A.weight", a),
        ("self_attn.q_proj.lora_B.weight", b),
    ]
    muon_p, adamw_p = partition_for_muon(named)
    muon_ids = {id(p) for p in muon_p}
    adamw_ids = {id(p) for p in adamw_p}
    assert id(real) in muon_ids
    assert id(moe) in muon_ids
    assert id(a) in adamw_ids
    assert id(b) in adamw_ids


def test_partition_for_sso_sends_lora_to_adamw_branch() -> None:
    from moe_prune_distill.distill.optimizer import partition_for_sso

    real = nn.Parameter(torch.randn(8, 8))
    a = nn.Parameter(torch.randn(4, 8))
    b = nn.Parameter(torch.randn(8, 4))
    named = [
        ("self_attn.q_proj.base.weight", real),
        ("self_attn.q_proj.lora_A.weight", a),
        ("self_attn.q_proj.lora_B.weight", b),
    ]
    sso_p, adamw_p = partition_for_sso(named)
    sso_ids = {id(p) for p in sso_p}
    adamw_ids = {id(p) for p in adamw_p}
    assert id(real) in sso_ids
    assert id(a) in adamw_ids
    assert id(b) in adamw_ids
