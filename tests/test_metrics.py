"""Unit tests for distill/metrics.py — diagnostic-only metric helpers."""

from __future__ import annotations

import math

import torch

from moe_prune_distill.distill.metrics import (
    batch_token_stats,
    hidden_metrics,
    router_diagnostics,
)


def test_hidden_metrics_zero_for_identical():
    a = torch.randn(2, 5, 8)
    out = hidden_metrics({0: a, 4: a}, {0: a, 4: a}, attention_mask=torch.ones(2, 5))
    assert out["nmse"] < 1e-6
    assert out["cos_loss"] < 1e-6
    assert abs(out["teacher_norm"] - out["student_norm"]) < 1e-5


def test_hidden_metrics_unmasked_equals_full_mean():
    a = torch.randn(1, 6, 4)
    b = torch.randn(1, 6, 4)
    no_mask = hidden_metrics({0: a}, {0: b}, attention_mask=None)
    full_mask = hidden_metrics({0: a}, {0: b}, attention_mask=torch.ones(1, 6))
    assert abs(no_mask["hidden_mse"] - full_mask["hidden_mse"]) < 1e-5
    assert abs(no_mask["nmse"] - full_mask["nmse"]) < 1e-5
    assert abs(no_mask["cos_loss"] - full_mask["cos_loss"]) < 1e-5


def test_hidden_metrics_omits_keys_when_no_layers():
    out = hidden_metrics({}, {}, attention_mask=None)
    assert out == {}


def test_cos_loss_orthogonal_one():
    s = torch.tensor([[[1.0, 0.0]]])     # [1, 1, 2]
    t = torch.tensor([[[0.0, 1.0]]])
    out = hidden_metrics({0: s}, {0: t}, attention_mask=torch.ones(1, 1))
    assert abs(out["cos_loss"] - 1.0) < 1e-5


def test_router_entropy_uniform_logits_log_E():
    s = {0: torch.zeros(1, 4, 6)}        # uniform softmax -> H = log(6)
    out = router_diagnostics(s, None, None, attention_mask=torch.ones(1, 4))
    assert abs(out["router_entropy"] - math.log(6)) < 1e-5
    assert "removed_expert_mass" not in out


def test_removed_expert_mass_full_keep_zero():
    t = {0: torch.randn(1, 3, 8)}
    surv = {0: list(range(8))}
    out = router_diagnostics({}, t, surv, attention_mask=torch.ones(1, 3))
    assert abs(out["removed_expert_mass"]) < 1e-5
    assert "router_entropy" not in out


def test_removed_expert_mass_keep_none_close_to_one():
    # surviving=[0] keeps 1/8 of a uniform softmax -> mass = 7/8
    t = {0: torch.zeros(1, 3, 8)}
    surv = {0: [0]}
    out = router_diagnostics({}, t, surv, attention_mask=torch.ones(1, 3))
    assert abs(out["removed_expert_mass"] - 7.0 / 8.0) < 1e-5


def test_batch_token_stats_pad():
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    out = batch_token_stats(mask)
    assert out["valid_tokens"] == 5
    assert abs(out["mean_seq_len"] - 2.5) < 1e-6


def test_router_diagnostics_handles_2d_logits():
    # HF often emits [B*T, E]; the entropy helper should still produce a number
    s = {0: torch.zeros(8, 4)}
    out = router_diagnostics(s, None, None, attention_mask=None)
    assert "router_entropy" in out
    assert abs(out["router_entropy"] - math.log(4)) < 1e-5


def test_jsonl_metrics_writer_appends(tmp_path):
    from moe_prune_distill.utils.metrics_log import JsonlMetricsWriter

    path = tmp_path / "logs" / "train_log.jsonl"     # parent does not exist yet
    w = JsonlMetricsWriter(path)
    w.log({"step": 1, "loss": 0.5})
    w.log({"step": 2, "loss": 0.4, "tensor": torch.tensor(0.25)})
    w.close()

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    import json as _json
    rows = [_json.loads(line) for line in lines]
    assert rows[0]["step"] == 1 and rows[0]["loss"] == 0.5
    assert abs(rows[1]["tensor"] - 0.25) < 1e-6
