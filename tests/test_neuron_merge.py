"""Unit tests for SwiGLU-aware neuron-level expert merging.

Covers:
* ``_split_gate_up`` / ``_build_super_vec`` math.
* ``build_neuron_merge_plan`` with both ``neuron_swiglu_local`` and
  ``neuron_swiglu_global`` strategies.
* End-to-end slicer integration: ``gate_up_proj`` is left as plain
  index_select (no mixing — preserves SwiGLU activation boundary), and
  ``down_proj`` columns are accumulated by hand-computed buckets.
* Macro regression: when ``mode=='macro'`` the slicer still runs the
  legacy scaled-add path verbatim.
* ``write_merge_report`` produces a Markdown file with the expected
  section headings.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from safetensors.torch import save_file as st_save

from moe_prune_distill.adapters.qwen_moe import QwenMoeAdapter
from moe_prune_distill.prune.expert_merge import (
    build_neuron_merge_plan,
    serialize_merge_plan,
    write_merge_report,
)
from moe_prune_distill.prune.expert_merge import (  # private helpers worth testing
    _build_super_vec,
    _load_router_stats_scale,
    _normalize_alpha_scale_for_dropped,
    _split_gate_up,
)
from moe_prune_distill.prune.slicer import MergePlan, prune_state_dict_sharded


# === fixture ===========================================================


def _make_real_layout_teacher(
    teacher: Path,
    *,
    num_layers: int = 2,
    num_experts: int = 4,
    hidden: int = 8,
    inter: int = 6,
    seed: int = 0,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Synthesise a tiny Qwen3.5-MoE-style teacher using the *real* shapes.

    Real Qwen3.5 layout (cf. transformers qwen3_5_moe modeling_qwen3_5_moe.py
    around line 734):
      * gate_up_proj: [E, 2 * intermediate, hidden]
      * down_proj   : [E, hidden, intermediate]
      * gate weight : [E, hidden]
    """
    torch.manual_seed(seed)
    teacher.mkdir(parents=True, exist_ok=True)
    sd: dict[str, torch.Tensor] = {
        "model.language_model.embed_tokens.weight": torch.randn(10, hidden),
        "lm_head.weight": torch.randn(10, hidden),
    }
    for layer in range(num_layers):
        prefix = f"model.language_model.layers.{layer}"
        sd[f"{prefix}.mlp.gate.weight"] = torch.randn(num_experts, hidden)
        sd[f"{prefix}.mlp.experts.gate_up_proj"] = torch.randn(
            num_experts, 2 * inter, hidden
        )
        sd[f"{prefix}.mlp.experts.down_proj"] = torch.randn(
            num_experts, hidden, inter
        )
    st_save(
        {k: v.contiguous() for k, v in sd.items()},
        str(teacher / "model.safetensors"),
    )
    hf = {
        "model_type": "qwen3_5_moe",
        "text_config": {
            "num_hidden_layers": num_layers,
            "num_experts": num_experts,
            "num_experts_per_tok": 2,
            "hidden_size": hidden,
        },
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
    }
    (teacher / "config.json").write_text(json.dumps(hf), encoding="utf-8")
    return sd, hf


# === unit: super-vector math ===========================================


def test_split_gate_up_partitions_first_half_to_gate_second_half_to_up():
    e, i, h = 3, 5, 7
    gate_up = torch.arange(e * 2 * i * h, dtype=torch.float32).reshape(e, 2 * i, h)
    gate, up = _split_gate_up(gate_up)
    assert gate.shape == (e, i, h)
    assert up.shape == (e, i, h)
    assert torch.equal(gate, gate_up[:, :i, :])
    assert torch.equal(up, gate_up[:, i:, :])


def test_split_gate_up_rejects_odd_middle_dim():
    bad = torch.zeros(2, 7, 4)  # 7 is odd
    with pytest.raises(ValueError):
        _split_gate_up(bad)


def test_build_super_vec_shape_and_unit_norm():
    gate_up = torch.randn(3, 8, 5)  # E=3, 2*I=8 -> I=4, H=5
    super_n, i, h = _build_super_vec(gate_up)
    assert i == 4 and h == 5
    assert super_n.shape == (3, 4, 10)  # 2*H=10
    norms = super_n.norm(dim=2)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


# === local strategy =====================================================


def test_neuron_swiglu_local_keeps_gate_up_unchanged_and_adds_down_columns(tmp_path: Path):
    teacher = tmp_path / "teacher"
    sd, hf = _make_real_layout_teacher(teacher, num_layers=1, num_experts=4)
    surviving = {0: [0, 2]}
    alpha = 0.5
    sim_threshold = -1.0  # accept everything so we exercise the full math

    a = QwenMoeAdapter()
    plan = build_neuron_merge_plan(
        teacher,
        a,
        hf,
        surviving,
        strategy="neuron_swiglu_local",
        alpha=alpha,
        sim_threshold=sim_threshold,
    )

    assert plan.mode == "neuron_swiglu"
    assert plan.neuron_meta["strategy"] == "neuron_swiglu_local"
    assert 0 in plan.neuron_down_contrib

    student = tmp_path / "student"
    prune_state_dict_sharded(
        teacher,
        student,
        a,
        hf,
        target_num_experts=2,
        keep_shared_experts=True,
        surviving_per_layer=surviving,
        merge_plan=plan,
    )

    out = load_file(str(student / "model.safetensors"))

    # 1. gate_up_proj is *exactly* the kept slice — no mixing.
    gu_orig = sd["model.language_model.layers.0.mlp.experts.gate_up_proj"]
    gu_actual = out["model.language_model.layers.0.mlp.experts.gate_up_proj"]
    assert torch.equal(gu_actual, gu_orig[surviving[0]])

    # 2. down_proj equals kept slice + the per-layer bucket.
    dp_orig = sd["model.language_model.layers.0.mlp.experts.down_proj"]
    dp_actual = out["model.language_model.layers.0.mlp.experts.down_proj"]
    expected = (
        dp_orig[surviving[0]].to(torch.float32)
        + plan.neuron_down_contrib[0].to(torch.float32)
    ).to(dp_orig.dtype)
    assert torch.allclose(dp_actual, expected, atol=1e-5)


def test_neuron_swiglu_local_threshold_filters_dropped_neurons(tmp_path: Path):
    """A very high threshold (>1) drops every neuron -> bucket is all zeros."""
    teacher = tmp_path / "teacher"
    _, hf = _make_real_layout_teacher(teacher, num_layers=1, num_experts=4, seed=42)
    surviving = {0: [0, 2]}

    a = QwenMoeAdapter()
    plan = build_neuron_merge_plan(
        teacher, a, hf, surviving,
        strategy="neuron_swiglu_local", alpha=1.0, sim_threshold=2.0,  # impossible
    )

    bucket = plan.neuron_down_contrib[0]
    assert torch.allclose(bucket, torch.zeros_like(bucket))
    stats = plan.neuron_stats[0]
    assert stats["hosted"] == 0
    assert stats["dropped_below_thr"] == stats["total_dropped_neurons"]


def test_neuron_swiglu_local_alpha_zero_yields_zero_bucket(tmp_path: Path):
    """alpha=0 + any threshold -> bucket is identically zero (vanilla prune)."""
    teacher = tmp_path / "teacher"
    _, hf = _make_real_layout_teacher(teacher, num_layers=1, num_experts=4, seed=7)
    surviving = {0: [0, 2]}

    a = QwenMoeAdapter()
    plan = build_neuron_merge_plan(
        teacher, a, hf, surviving,
        strategy="neuron_swiglu_local", alpha=0.0, sim_threshold=-1.0,
    )
    bucket = plan.neuron_down_contrib[0]
    assert torch.allclose(bucket, torch.zeros_like(bucket))


# === global strategy ====================================================


def test_neuron_swiglu_global_keeps_gate_up_unchanged_and_writes_bucket(tmp_path: Path):
    teacher = tmp_path / "teacher"
    sd, hf = _make_real_layout_teacher(
        teacher, num_layers=1, num_experts=4, hidden=8, inter=4, seed=3
    )
    surviving = {0: [0, 2]}

    a = QwenMoeAdapter()
    plan = build_neuron_merge_plan(
        teacher, a, hf, surviving,
        strategy="neuron_swiglu_global",
        alpha=0.5,
        sim_threshold=-1.0,
        top_k=4,
    )
    assert plan.mode == "neuron_swiglu"
    assert plan.neuron_meta["strategy"] == "neuron_swiglu_global"
    assert plan.neuron_meta["top_k"] == 4

    student = tmp_path / "student"
    prune_state_dict_sharded(
        teacher, student, a, hf,
        target_num_experts=2, keep_shared_experts=True,
        surviving_per_layer=surviving, merge_plan=plan,
    )
    out = load_file(str(student / "model.safetensors"))
    gu_orig = sd["model.language_model.layers.0.mlp.experts.gate_up_proj"]
    assert torch.equal(
        out["model.language_model.layers.0.mlp.experts.gate_up_proj"],
        gu_orig[surviving[0]],
    )
    dp_orig = sd["model.language_model.layers.0.mlp.experts.down_proj"]
    expected = (
        dp_orig[surviving[0]].to(torch.float32)
        + plan.neuron_down_contrib[0].to(torch.float32)
    ).to(dp_orig.dtype)
    assert torch.allclose(
        out["model.language_model.layers.0.mlp.experts.down_proj"],
        expected,
        atol=1e-5,
    )


def test_neuron_swiglu_global_caches_topk_to_scratch_dir(tmp_path: Path):
    teacher = tmp_path / "teacher"
    _, hf = _make_real_layout_teacher(
        teacher, num_layers=1, num_experts=4, hidden=8, inter=4, seed=11
    )
    surviving = {0: [0, 2]}
    scratch = tmp_path / "scratch"
    a = QwenMoeAdapter()

    plan1 = build_neuron_merge_plan(
        teacher, a, hf, surviving,
        strategy="neuron_swiglu_global",
        alpha=0.5, sim_threshold=-1.0, top_k=3,
        scratch_dir=scratch,
    )
    cache_file = scratch / "neuron_match_layer0.pt"
    assert cache_file.is_file()

    plan2 = build_neuron_merge_plan(
        teacher, a, hf, surviving,
        strategy="neuron_swiglu_global",
        alpha=0.5, sim_threshold=-1.0, top_k=3,
        scratch_dir=scratch,
    )
    # Same inputs + same cache -> identical bucket.
    assert torch.allclose(
        plan1.neuron_down_contrib[0], plan2.neuron_down_contrib[0], atol=1e-6
    )


# === macro regression ===================================================


def test_macro_path_unchanged_when_neuron_fields_are_empty(tmp_path: Path):
    """A MergePlan with mode='macro' must run the legacy scaled-add slicer code path,
    bit-for-bit identical to pre-neuron behaviour."""
    teacher = tmp_path / "teacher"
    sd, hf = _make_real_layout_teacher(teacher, num_layers=1, num_experts=4, seed=99)
    surviving = {0: [0, 2]}

    # Hand-build a macro MergePlan: 2 dropped (1, 3), 2 kept (0, 2), uniform mixing.
    plan = MergePlan(alpha=0.25, mode="macro")
    plan.surviving_per_layer = {0: [0, 2]}
    plan.dropped_per_layer = {0: [1, 3]}
    plan.weights[0] = torch.tensor(
        [[0.7, 0.3], [0.4, 0.6]], dtype=torch.float32
    )

    a = QwenMoeAdapter()
    student = tmp_path / "student"
    prune_state_dict_sharded(
        teacher, student, a, hf,
        target_num_experts=2, keep_shared_experts=True,
        surviving_per_layer=surviving, merge_plan=plan,
    )
    out = load_file(str(student / "model.safetensors"))

    for proj in ("gate_up_proj", "down_proj"):
        orig = sd[f"model.language_model.layers.0.mlp.experts.{proj}"].to(torch.float32)
        kept = orig[surviving[0]]
        dropped = orig[plan.dropped_per_layer[0]]
        d_flat = dropped.reshape(dropped.shape[0], -1)
        contrib = (
            plan.weights[0].t() @ d_flat
        ).reshape(kept.shape)
        expected = (kept + 0.25 * contrib).to(orig.dtype)
        actual = out[f"model.language_model.layers.0.mlp.experts.{proj}"]
        assert torch.allclose(actual, expected, atol=1e-5)


# === serialize / report =================================================


def test_serialize_neuron_plan_schema(tmp_path: Path):
    teacher = tmp_path / "teacher"
    _, hf = _make_real_layout_teacher(teacher, num_layers=1, num_experts=4, seed=1)
    a = QwenMoeAdapter()
    plan = build_neuron_merge_plan(
        teacher, a, hf, {0: [0, 2]},
        strategy="neuron_swiglu_local", alpha=0.5, sim_threshold=0.0,
    )
    payload = serialize_merge_plan(plan)
    assert payload["mode"] == "neuron_swiglu"
    assert payload["strategy"] == "neuron_swiglu_local"
    assert payload["sim_threshold"] == 0.0
    assert "0" in payload["layers"]
    layer0 = payload["layers"]["0"]
    assert "host_pairs" in layer0  # local-only field
    assert "neuron_stats" in layer0
    assert {"hosted", "dropped_below_thr", "total_dropped_neurons"} <= set(
        layer0["neuron_stats"].keys()
    )


def test_write_merge_report_neuron_contains_expected_sections(tmp_path: Path):
    teacher = tmp_path / "teacher"
    _, hf = _make_real_layout_teacher(teacher, num_layers=2, num_experts=4, seed=5)
    a = QwenMoeAdapter()
    plan = build_neuron_merge_plan(
        teacher, a, hf, {0: [0, 2], 1: [1, 3]},
        strategy="neuron_swiglu_local", alpha=0.5, sim_threshold=-1.0,
    )
    student = tmp_path / "student"
    student.mkdir()
    out = write_merge_report(
        plan,
        student,
        teacher_arch="Qwen3_5MoeForConditionalGeneration",
        num_layers_total=2,
        num_experts_total=4,
        target_num_experts=2,
        target_num_experts_per_tok=2,
    )
    assert out.is_file()
    text = out.read_text(encoding="utf-8")
    for needle in (
        "# Expert Merge Report",
        "## Aggregate",
        "## Per-layer summary",
        "## Host load distribution",
        "## Warnings",
        "neuron_swiglu_local",
    ):
        assert needle in text, f"missing in merge_report.md: {needle!r}"


def test_write_merge_report_macro_branch(tmp_path: Path):
    plan = MergePlan(alpha=0.5, mode="macro")
    plan.weights[0] = torch.eye(2)
    plan.surviving_per_layer = {0: [0, 1]}
    plan.dropped_per_layer = {0: [2, 3]}
    out = write_merge_report(plan, tmp_path)
    text = out.read_text(encoding="utf-8")
    assert "## Aggregate (macro)" in text


# === dynamic alpha helper ==============================================


def test_load_router_stats_scale_round_trip(tmp_path: Path):
    payload = {
        "model_id": "fake",
        "num_samples": 100,
        "num_layers": 2,
        "num_experts": 4,
        "top_k": 2,
        "layers": {
            "0": {
                "usage_counts": {"0": 10, "1": 20, "2": 5, "3": 65},
                "recommended": [1, 3],
            },
            "1": {
                "usage_counts": {"0": 100, "1": 100, "2": 100, "3": 100},
                "recommended": [0, 1],
            },
        },
    }
    p = tmp_path / "router_stats.json"
    p.write_text(json.dumps(payload), encoding="utf-8")

    full = _load_router_stats_scale(p, num_layers=2)
    assert full[0] == {0: 10.0, 1: 20.0, 2: 5.0, 3: 65.0}
    # Layer 0: dropped = [0, 2] -> mean usage = (10+5)/2 = 7.5
    scale = _normalize_alpha_scale_for_dropped(full, layer=0, dropped=[0, 2])
    assert scale is not None
    assert pytest.approx(scale[0]) == 10.0 / 7.5
    assert pytest.approx(scale[2]) == 5.0 / 7.5
    # Layer 1: uniform usage -> all dropped get scale 1.0.
    scale1 = _normalize_alpha_scale_for_dropped(full, layer=1, dropped=[2, 3])
    assert pytest.approx(scale1[2]) == 1.0
    assert pytest.approx(scale1[3]) == 1.0


def test_neuron_swiglu_local_with_router_stats_scales_alpha(tmp_path: Path):
    """Higher dropped-expert frequency -> higher down-column contribution."""
    teacher = tmp_path / "teacher"
    sd, hf = _make_real_layout_teacher(
        teacher, num_layers=1, num_experts=4, hidden=8, inter=4, seed=21
    )
    surviving = {0: [0, 2]}

    # Build a router_stats.json where dropped=[1, 3] have very different freqs.
    rs = {
        "model_id": "fake",
        "num_samples": 100,
        "num_layers": 1,
        "num_experts": 4,
        "top_k": 2,
        "layers": {
            "0": {
                "usage_counts": {"0": 50, "1": 30, "2": 50, "3": 10},
                "recommended": [0, 2],
            },
        },
    }
    rs_path = tmp_path / "router_stats.json"
    rs_path.write_text(json.dumps(rs), encoding="utf-8")

    a = QwenMoeAdapter()
    plan_no_scale = build_neuron_merge_plan(
        teacher, a, hf, surviving,
        strategy="neuron_swiglu_local", alpha=0.5, sim_threshold=-1.0,
    )
    plan_scaled = build_neuron_merge_plan(
        teacher, a, hf, surviving,
        strategy="neuron_swiglu_local", alpha=0.5, sim_threshold=-1.0,
        router_stats_path=rs_path,
    )

    assert plan_scaled.neuron_meta["router_stats_used"] is True
    assert plan_no_scale.neuron_meta["router_stats_used"] is False
    # With a heavy mass on expert 1 vs light on expert 3, the scaled bucket
    # cannot be identical to the unscaled one.
    assert not torch.allclose(
        plan_no_scale.neuron_down_contrib[0],
        plan_scaled.neuron_down_contrib[0],
        atol=1e-5,
    )


# === streaming bucket_dir ===============================================


def test_bucket_dir_writes_one_file_per_layer_and_drops_in_memory(tmp_path: Path):
    teacher = tmp_path / "teacher"
    _, hf = _make_real_layout_teacher(
        teacher, num_layers=3, num_experts=4, hidden=8, inter=4, seed=12
    )
    surviving = {0: [0, 2], 1: [1, 3], 2: [0, 1]}
    bucket_dir = tmp_path / "buckets"

    a = QwenMoeAdapter()
    plan = build_neuron_merge_plan(
        teacher, a, hf, surviving,
        strategy="neuron_swiglu_local", alpha=0.5, sim_threshold=-1.0,
        bucket_dir=bucket_dir,
    )

    # In-memory dict empty, on-disk paths populated.
    assert plan.neuron_down_contrib == {}
    assert set(plan.neuron_down_contrib_paths.keys()) == {0, 1, 2}
    for layer in (0, 1, 2):
        f = Path(plan.neuron_down_contrib_paths[layer])
        assert f.is_file()
        assert f.parent == bucket_dir
        loaded = torch.load(f, map_location="cpu", weights_only=True)
        # Shape matches kept down_proj: [Nk=2, hidden=8, intermediate=4].
        assert loaded.shape == (2, 8, 4)


def test_streaming_and_in_memory_buckets_produce_identical_student(tmp_path: Path):
    """Slicer output bit-equal whether buckets live in RAM or are loaded from disk."""
    teacher = tmp_path / "teacher"
    sd, hf = _make_real_layout_teacher(
        teacher, num_layers=2, num_experts=4, hidden=8, inter=4, seed=33
    )
    surviving = {0: [0, 2], 1: [1, 3]}
    a = QwenMoeAdapter()

    plan_mem = build_neuron_merge_plan(
        teacher, a, hf, surviving,
        strategy="neuron_swiglu_local", alpha=0.4, sim_threshold=-1.0,
    )
    plan_disk = build_neuron_merge_plan(
        teacher, a, hf, surviving,
        strategy="neuron_swiglu_local", alpha=0.4, sim_threshold=-1.0,
        bucket_dir=tmp_path / "stream",
    )

    student_mem = tmp_path / "student_mem"
    student_disk = tmp_path / "student_disk"
    prune_state_dict_sharded(
        teacher, student_mem, a, hf,
        target_num_experts=2, keep_shared_experts=True,
        surviving_per_layer=surviving, merge_plan=plan_mem,
    )
    prune_state_dict_sharded(
        teacher, student_disk, a, hf,
        target_num_experts=2, keep_shared_experts=True,
        surviving_per_layer=surviving, merge_plan=plan_disk,
    )
    out_mem = load_file(str(student_mem / "model.safetensors"))
    out_disk = load_file(str(student_disk / "model.safetensors"))
    assert set(out_mem.keys()) == set(out_disk.keys())
    for k in out_mem:
        assert torch.equal(out_mem[k], out_disk[k]), f"mismatch on key {k}"


def test_streaming_global_strategy_via_bucket_dir(tmp_path: Path):
    """Global path must also stream buckets when bucket_dir is provided."""
    teacher = tmp_path / "teacher"
    _, hf = _make_real_layout_teacher(
        teacher, num_layers=1, num_experts=4, hidden=8, inter=4, seed=77
    )
    surviving = {0: [0, 2]}
    bucket_dir = tmp_path / "g_buckets"
    a = QwenMoeAdapter()
    plan = build_neuron_merge_plan(
        teacher, a, hf, surviving,
        strategy="neuron_swiglu_global",
        alpha=0.5, sim_threshold=-1.0, top_k=3,
        bucket_dir=bucket_dir,
    )
    assert plan.neuron_down_contrib == {}
    assert plan.neuron_down_contrib_paths == {0: str(bucket_dir / "layer0.pt")}
    assert (bucket_dir / "layer0.pt").is_file()
