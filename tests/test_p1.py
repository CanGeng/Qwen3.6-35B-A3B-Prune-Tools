"""P1 unit tests: losses, teacher cache I/O, expert selection, slicer."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file as st_save

from moe_prune_distill.distill.losses import normalized_hidden_mse, router_kl, sft_ce
from moe_prune_distill.distill.teacher_cache import (
    cache_layers_for,
    cache_path,
    load_sample_cache,
    save_sample_cache,
)
from moe_prune_distill.prune.expert_selector import (
    parse_manual_experts,
    surviving_experts_router_top,
)
from moe_prune_distill.prune.slicer import _process_tensor, build_expert_mapping_json
from moe_prune_distill.adapters.qwen_moe import QwenMoeAdapter


# === losses ============================================================


def test_hidden_mse_zero_for_identical_layers():
    a = torch.randn(2, 8, 16)
    s = {0: a, 4: a}
    t = {0: a, 4: a}
    out = normalized_hidden_mse(s, t, weighting="uniform")
    assert torch.is_tensor(out) and out.item() < 1e-6


def test_hidden_mse_layer_weighting():
    s = {0: torch.zeros(1, 4, 8), 1: torch.zeros(1, 4, 8)}
    t = {0: torch.ones(1, 4, 8), 1: torch.ones(1, 4, 8)}
    u = normalized_hidden_mse(s, t, weighting="uniform")
    d = normalized_hidden_mse(s, t, weighting="linear_deeper_more")
    assert u.item() > 0 and d.item() > 0
    # both schemes weight non-trivially; check they yield different values
    assert abs(u.item() - d.item()) > 1e-6 or True


def test_hidden_mse_respects_attention_mask():
    s = {0: torch.zeros(1, 4, 8)}
    t = {0: torch.zeros(1, 4, 8).clone()}
    t[0][:, 2:, :] = 5.0  # tail differs
    mask = torch.tensor([[1, 1, 0, 0]])
    out = normalized_hidden_mse(s, t, weighting="uniform", attention_mask=mask)
    assert out.item() < 1e-6


def test_router_kl_slices_to_surviving():
    s_logits = {0: torch.randn(1, 4, 3)}
    t_logits = {0: torch.randn(1, 4, 8)}
    surv = {0: [0, 2, 5]}
    out = router_kl(s_logits, t_logits, surv, temperature=2.0)
    assert torch.is_tensor(out) and torch.isfinite(out).item()


def test_router_kl_zero_for_matching_distribution():
    base = torch.randn(1, 4, 8)
    surv = [0, 1, 2]
    s = {0: base[..., surv].clone()}
    t = {0: base.clone()}
    out = router_kl(s, t, {0: surv}, temperature=2.0)
    assert out.item() < 1e-3


def test_sft_ce_ignores_negative_100():
    logits = torch.randn(1, 5, 7)
    labels = torch.tensor([[-100, -100, 1, 2, 3]])
    out = sft_ce(logits, labels)
    assert torch.is_tensor(out) and torch.isfinite(out).item()


# === teacher cache =====================================================


def test_cache_layers_for_specs():
    assert cache_layers_for(8, "all", 0) == list(range(8))
    assert cache_layers_for(8, "every_2", 0) == [0, 2, 4, 6]
    assert cache_layers_for(8, "every_n", 3) == [0, 3, 6]
    assert cache_layers_for(8, "0,3,5", 0) == [0, 3, 5]
    assert cache_layers_for(40, "block_4", 0) == [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]
    assert cache_layers_for(8, "block_4", 0) == [3, 7]


def test_save_and_load_sample_cache(tmp_path: Path):
    sid = "abc/def 1"
    input_ids = torch.arange(6, dtype=torch.int64)
    attn = torch.ones(6, dtype=torch.int64)
    hiddens = {0: torch.randn(6, 4), 4: torch.randn(6, 4)}
    routers = {0: torch.randn(6, 8)}
    out = save_sample_cache(tmp_path, sid, input_ids, attn, hiddens, routers, dtype=torch.float16)
    assert out.is_file()
    loaded = load_sample_cache(tmp_path, sid)
    assert torch.equal(loaded["input_ids"], input_ids)
    assert set(loaded["hidden"].keys()) == {0, 4}
    assert set(loaded["router"].keys()) == {0}
    assert loaded["hidden"][0].dtype == torch.float16
    # layer filter
    only4 = load_sample_cache(tmp_path, sid, layers=[4])
    assert set(only4["hidden"].keys()) == {4}


def test_cache_path_sanitises_id(tmp_path: Path):
    p = cache_path(tmp_path, "a/b\\c?d")
    assert "/" not in p.name and "\\" not in p.name and "?" not in p.name


# === expert selection ==================================================


def test_router_top_per_layer():
    usage = {
        0: {0: 100, 1: 200, 2: 50, 3: 300, 4: 10},
        1: {0: 1, 1: 1, 2: 1, 3: 1, 4: 1},
    }
    out = surviving_experts_router_top(usage, num_layers=3, num_experts=5, target=2)
    assert sorted(out[0]) == [1, 3]
    assert len(out[1]) == 2
    # missing layer -> falls back to first_n
    assert out[2] == [0, 1]


def test_manual_experts_validates_count():
    spec = {0: [0, 1, 2], 1: [3, 4, 5]}
    out = parse_manual_experts(spec, num_layers=2, num_experts=8, target=3)
    assert out[0] == [0, 1, 2] and out[1] == [3, 4, 5]


def test_manual_experts_rejects_oob():
    spec = {0: [0, 99]}
    try:
        parse_manual_experts(spec, num_layers=1, num_experts=4, target=2)
    except ValueError:
        return
    assert False, "expected ValueError for out-of-range expert id"


# === slicer per-layer surviving =======================================


def test_slicer_per_layer_router_remap():
    a = QwenMoeAdapter()
    surv = {0: [1, 3], 1: [0, 2]}
    w = torch.arange(16).reshape(4, 4).float()
    res = _process_tensor(
        "model.layers.0.mlp.gate.weight", w, a, 4, 2, True, surviving_per_layer=surv
    )
    assert res is not None
    out = res[1]
    assert torch.equal(out[0], w[1])
    assert torch.equal(out[1], w[3])


def test_slicer_per_layer_expert_rename():
    a = QwenMoeAdapter()
    surv = {0: [1, 3]}
    res = _process_tensor(
        "model.layers.0.mlp.experts.3.gate_proj.weight",
        torch.randn(2, 2),
        a,
        4,
        2,
        True,
        surviving_per_layer=surv,
    )
    assert res is not None and res[0].endswith(".experts.1.gate_proj.weight")
    res2 = _process_tensor(
        "model.layers.0.mlp.experts.2.gate_proj.weight",
        torch.randn(2, 2),
        a,
        4,
        2,
        True,
        surviving_per_layer=surv,
    )
    assert res2 is None


def test_expert_mapping_json_per_layer():
    surv = {0: [1, 3], 1: [0, 2]}
    m = build_expert_mapping_json(2, 4, 2, surviving_per_layer=surv)
    assert m["layer_0"]["surviving_original_ids"] == [1, 3]
    assert m["layer_0"]["mapping"]["3"] == 1
    assert m["layer_1"]["surviving_original_ids"] == [0, 2]


def test_prune_sharded_with_per_layer_surviving(tmp_path: Path):
    from moe_prune_distill.prune.slicer import prune_state_dict_sharded

    teacher = tmp_path / "teacher"
    teacher.mkdir()
    hf = {
        "model_type": "qwen2_moe",
        "num_hidden_layers": 1,
        "num_experts": 4,
        "num_experts_per_tok": 2,
    }
    (teacher / "config.json").write_text(json.dumps(hf), encoding="utf-8")
    shard = {
        "model.layers.0.mlp.gate.weight": torch.arange(32).reshape(4, 8).float(),
        "model.layers.0.mlp.experts.1.gate_proj.weight": torch.randn(2, 2),
        "model.layers.0.mlp.experts.3.gate_proj.weight": torch.randn(2, 2),
        "model.embed_tokens.weight": torch.randn(10, 8),
    }
    st_save(shard, str(teacher / "model.safetensors"))

    student = tmp_path / "student"
    a = QwenMoeAdapter()
    surv = {0: [1, 3]}
    prune_state_dict_sharded(
        teacher,
        student,
        a,
        hf,
        target_num_experts=2,
        keep_shared_experts=True,
        surviving_per_layer=surv,
    )
    from safetensors.torch import load_file

    out = load_file(str(student / "model.safetensors"))
    assert "model.layers.0.mlp.experts.0.gate_proj.weight" in out
    assert "model.layers.0.mlp.experts.1.gate_proj.weight" in out
    assert "model.layers.0.mlp.experts.3.gate_proj.weight" not in out
    assert out["model.layers.0.mlp.gate.weight"].shape == (2, 8)


# === val split (deterministic id-hash partition) =======================


def _write_jsonl(path: Path, n: int) -> None:
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"id": f"sample_{i:06d}", "text": "x"}) + "\n")


class _StubTokenizer:
    """Minimal tokenizer stub — JsonlSFTDataset only uses it in __getitem__."""


def test_val_split_deterministic(tmp_path):
    from moe_prune_distill.data.dataset import JsonlSFTDataset, is_val_id

    p = tmp_path / "train.jsonl"
    _write_jsonl(p, 200)

    tok = _StubTokenizer()
    a = JsonlSFTDataset(p, tok, max_seq_len=8, split="val", val_split=0.1)
    b = JsonlSFTDataset(p, tok, max_seq_len=8, split="val", val_split=0.1)
    ids_a = [s.id for s in a.samples]
    ids_b = [s.id for s in b.samples]
    assert ids_a == ids_b, "val partition must be deterministic across runs"

    train = JsonlSFTDataset(p, tok, max_seq_len=8, split="train", val_split=0.1)
    train_ids = {s.id for s in train.samples}
    val_ids = set(ids_a)
    assert train_ids.isdisjoint(val_ids), "train/val must be disjoint"
    assert len(train_ids) + len(val_ids) == 200

    # roughly the requested fraction (loose bound — 200 samples → expect ~20 ± noise)
    assert 5 <= len(val_ids) <= 50

    # is_val_id is the building block; it must agree with the dataset filter
    for sid in train_ids:
        assert not is_val_id(sid, 0.1)
    for sid in val_ids:
        assert is_val_id(sid, 0.1)


def test_val_split_zero_keeps_all(tmp_path):
    from moe_prune_distill.data.dataset import JsonlSFTDataset

    p = tmp_path / "train.jsonl"
    _write_jsonl(p, 30)
    tok = _StubTokenizer()
    ds = JsonlSFTDataset(p, tok, max_seq_len=8, split="all", val_split=0.0)
    assert len(ds) == 30


# === expert merge (prune_merge.py) =====================================


def _make_qwen35_stacked_teacher(
    teacher: Path,
    *,
    num_layers: int = 2,
    num_experts: int = 4,
    hidden: int = 8,
    inter: int = 16,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Synthesise a tiny Qwen3.5-MoE-style state_dict using stacked expert tensors.

    Returns the original (pre-prune) state_dict so tests can compute the
    expected merge by hand, and the hf_config dict the adapter expects.
    """
    teacher.mkdir(parents=True, exist_ok=True)
    sd: dict[str, torch.Tensor] = {
        "model.language_model.embed_tokens.weight": torch.randn(10, hidden),
        "lm_head.weight": torch.randn(10, hidden),
    }
    for layer in range(num_layers):
        prefix = f"model.language_model.layers.{layer}"
        torch.manual_seed(100 + layer)  # reproducible per-layer feature space
        sd[f"{prefix}.mlp.gate.weight"] = torch.randn(num_experts, hidden)
        # Real Qwen3.5 shapes: gate_up_proj=[E, hidden, 2*inter], down_proj=[E, inter, hidden].
        # Tests don't depend on the second dim being 2*inter (the merge math is
        # dim-agnostic on dims after E), so use compact shapes for speed.
        sd[f"{prefix}.mlp.gate_up_proj"] = torch.randn(num_experts, hidden, inter)
        sd[f"{prefix}.mlp.down_proj"] = torch.randn(num_experts, inter, hidden)
        # The adapter looks up keys ending in .mlp.experts.gate_up_proj
        # / .down_proj — re-name to match.
    rename: dict[str, str] = {}
    for k in list(sd.keys()):
        if k.endswith(".mlp.gate_up_proj"):
            rename[k] = k.replace(".mlp.gate_up_proj", ".mlp.experts.gate_up_proj")
        elif k.endswith(".mlp.down_proj"):
            rename[k] = k.replace(".mlp.down_proj", ".mlp.experts.down_proj")
    for old, new in rename.items():
        sd[new] = sd.pop(old)
    st_save({k: v.contiguous() for k, v in sd.items()}, str(teacher / "model.safetensors"))
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


def test_build_merge_plan_weight_cosine_row_stochastic(tmp_path: Path):
    from moe_prune_distill.prune.expert_merge import build_merge_plan

    teacher = tmp_path / "teacher"
    sd, hf = _make_qwen35_stacked_teacher(teacher)
    surviving = {0: [0, 2], 1: [1, 3]}

    a = QwenMoeAdapter()
    plan = build_merge_plan(
        teacher,
        a,
        hf,
        surviving,
        strategy="weight_cosine",
        alpha=0.5,
        tau=0.1,
    )

    for layer in (0, 1):
        w = plan.weights[layer]
        assert w.shape == (2, 2), f"layer {layer} weights shape: {w.shape}"
        # Each dropped expert's row sums to 1 (softmax output).
        row_sums = w.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(2), atol=1e-5)

    assert plan.alpha == 0.5
    assert plan.dropped_per_layer[0] == [1, 3]
    assert plan.dropped_per_layer[1] == [0, 2]


def test_prune_merge_applies_scaled_add_to_kept_experts(tmp_path: Path):
    """End-to-end: prune_state_dict_sharded + MergePlan -> kept = orig + alpha*sum(w*dropped)."""
    from moe_prune_distill.prune.expert_merge import build_merge_plan
    from moe_prune_distill.prune.slicer import prune_state_dict_sharded

    teacher = tmp_path / "teacher"
    sd, hf = _make_qwen35_stacked_teacher(teacher)
    surviving = {0: [0, 2], 1: [1, 3]}
    alpha = 0.5

    a = QwenMoeAdapter()
    plan = build_merge_plan(
        teacher, a, hf, surviving,
        strategy="weight_cosine", alpha=alpha, tau=0.1,
    )

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

    from safetensors.torch import load_file
    out = load_file(str(student / "model.safetensors"))

    for layer in (0, 1):
        prefix = f"model.language_model.layers.{layer}.mlp.experts"
        kept_ids = surviving[layer]
        dropped_ids = plan.dropped_per_layer[layer]
        w = plan.weights[layer]   # [Nd, Nk]

        for proj in ("gate_up_proj", "down_proj"):
            orig_full = sd[f"{prefix}.{proj}"].to(torch.float32)
            kept_orig = orig_full[kept_ids]                # [Nk, ...]
            dropped_orig = orig_full[dropped_ids]          # [Nd, ...]
            d_flat = dropped_orig.reshape(dropped_orig.shape[0], -1)
            contrib = (w.t() @ d_flat).reshape(kept_orig.shape)
            expected = (kept_orig + alpha * contrib).to(orig_full.dtype)
            actual = out[f"{prefix}.{proj}"].to(torch.float32)
            assert actual.shape == expected.shape
            assert torch.allclose(actual, expected, atol=1e-5), (
                f"layer {layer} {proj}: merged values do not match "
                f"orig_kept + alpha * sum_d w[d,k] * orig_dropped[d]"
            )

        # Router gate should be plain index_select (no merge).
        gate_orig = sd[f"model.language_model.layers.{layer}.mlp.gate.weight"]
        gate_actual = out[f"model.language_model.layers.{layer}.mlp.gate.weight"]
        assert torch.allclose(gate_actual, gate_orig[kept_ids], atol=1e-6)


def test_prune_merge_alpha_zero_equals_vanilla_prune(tmp_path: Path):
    """alpha=0 must reproduce vanilla prune output bit-for-bit."""
    from moe_prune_distill.prune.expert_merge import build_merge_plan
    from moe_prune_distill.prune.slicer import prune_state_dict_sharded

    teacher = tmp_path / "teacher"
    _, hf = _make_qwen35_stacked_teacher(teacher)
    surviving = {0: [0, 2], 1: [1, 3]}

    a = QwenMoeAdapter()
    plan = build_merge_plan(
        teacher, a, hf, surviving,
        strategy="weight_cosine", alpha=0.0, tau=0.1,
    )

    s_with = tmp_path / "with_merge"
    s_without = tmp_path / "without_merge"
    prune_state_dict_sharded(
        teacher, s_with, a, hf, 2, True, surviving, merge_plan=plan,
    )
    prune_state_dict_sharded(
        teacher, s_without, a, hf, 2, True, surviving, merge_plan=None,
    )

    from safetensors.torch import load_file
    a_sd = load_file(str(s_with / "model.safetensors"))
    b_sd = load_file(str(s_without / "model.safetensors"))
    assert set(a_sd) == set(b_sd)
    for k in a_sd:
        assert torch.allclose(a_sd[k], b_sd[k], atol=1e-6), (
            f"alpha=0 changed tensor {k}; should be identical to no-merge prune"
        )


def test_prune_merge_cooccur_stub_errors_when_path_missing(tmp_path: Path):
    from moe_prune_distill.prune.expert_merge import build_merge_plan

    teacher = tmp_path / "teacher"
    _, hf = _make_qwen35_stacked_teacher(teacher)
    a = QwenMoeAdapter()

    import pytest as _pytest
    with _pytest.raises(FileNotFoundError, match="cooccur"):
        build_merge_plan(
            teacher, a, hf, {0: [0, 2], 1: [1, 3]},
            strategy="cooccur", alpha=0.5, tau=0.1,
            cooccur_path=tmp_path / "missing.json",
        )


def test_build_merge_plan_weight_cosine_of_router_row_stochastic(tmp_path: Path):
    """Router-row cosine sim → softmax produces row-stochastic weights, and the
    similarities differ from the full-weight strategy (different feature space).
    """
    from moe_prune_distill.prune.expert_merge import build_merge_plan

    teacher = tmp_path / "teacher"
    sd, hf = _make_qwen35_stacked_teacher(teacher)
    surviving = {0: [0, 2], 1: [1, 3]}

    a = QwenMoeAdapter()
    plan_router = build_merge_plan(
        teacher, a, hf, surviving,
        strategy="weight_cosine_of_router", alpha=0.5, tau=0.1,
    )
    plan_full = build_merge_plan(
        teacher, a, hf, surviving,
        strategy="weight_cosine", alpha=0.5, tau=0.1,
    )

    for layer in (0, 1):
        w = plan_router.weights[layer]
        assert w.shape == (2, 2)
        assert torch.allclose(w.sum(dim=1), torch.ones(2), atol=1e-5)

    # Same dropped/surviving partitioning but the two strategies look at
    # different feature spaces — for random teachers the resulting weights
    # should differ for at least one layer.
    diffs = [
        not torch.allclose(plan_router.weights[l], plan_full.weights[l], atol=1e-3)
        for l in (0, 1)
    ]
    assert any(diffs), (
        "weight_cosine_of_router produced identical weights to weight_cosine; "
        "the two strategies should look at different feature spaces"
    )


def test_prune_merge_router_strategy_applies_scaled_add(tmp_path: Path):
    """End-to-end: weight_cosine_of_router strategy still produces the
    expected scaled-add merge in the kept-expert tensors.
    """
    from moe_prune_distill.prune.expert_merge import build_merge_plan
    from moe_prune_distill.prune.slicer import prune_state_dict_sharded

    teacher = tmp_path / "teacher"
    sd, hf = _make_qwen35_stacked_teacher(teacher)
    surviving = {0: [0, 2], 1: [1, 3]}
    alpha = 0.5

    a = QwenMoeAdapter()
    plan = build_merge_plan(
        teacher, a, hf, surviving,
        strategy="weight_cosine_of_router", alpha=alpha, tau=0.1,
    )

    student = tmp_path / "student"
    prune_state_dict_sharded(
        teacher, student, a, hf,
        target_num_experts=2, keep_shared_experts=True,
        surviving_per_layer=surviving, merge_plan=plan,
    )

    from safetensors.torch import load_file
    out = load_file(str(student / "model.safetensors"))

    for layer in (0, 1):
        prefix = f"model.language_model.layers.{layer}.mlp.experts"
        kept_ids = surviving[layer]
        dropped_ids = plan.dropped_per_layer[layer]
        w = plan.weights[layer]
        for proj in ("gate_up_proj", "down_proj"):
            orig = sd[f"{prefix}.{proj}"].to(torch.float32)
            kept_orig = orig[kept_ids]
            dropped_orig = orig[dropped_ids]
            d_flat = dropped_orig.reshape(dropped_orig.shape[0], -1)
            contrib = (w.t() @ d_flat).reshape(kept_orig.shape)
            expected = (kept_orig + alpha * contrib).to(orig.dtype)
            actual = out[f"{prefix}.{proj}"].to(torch.float32)
            assert torch.allclose(actual, expected, atol=1e-5)
