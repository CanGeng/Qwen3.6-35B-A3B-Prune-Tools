"""Build a per-layer expert merge plan for ``scripts/prune_merge.py``.

Given the per-layer surviving / dropped expert ids resolved by
``_resolve_surviving`` (the same selector ``scripts/prune.py`` uses), this
module walks the teacher shards once, computes a per-layer
``[len(dropped), len(surviving)]`` row-stochastic weight matrix, and packs
it into a :class:`~moe_prune_distill.prune.slicer.MergePlan` that the
existing streaming slicer can consume in a single pass.

Strategies fall into two families:

* **Macro** — full-tensor mixing of dropped into kept (one row-stochastic
  ``[Nd, Nk]`` per layer). All three of ``weight_cosine``,
  ``weight_cosine_of_router``, and ``cooccur`` live here.
* **Neuron-level (SwiGLU-aware)** — ``neuron_swiglu_local`` and
  ``neuron_swiglu_global`` keep the kept expert's ``gate_up_proj``
  untouched (preserving the SwiGLU activation boundary) and only fold the
  dropped expert's ``down_proj`` columns into matched kept neurons. See
  ``build_neuron_merge_plan``.

Macro strategies:

* **weight_cosine** — flatten each expert's ``gate_up_proj`` +
  ``down_proj`` into a single vector, L2-normalize, compute pairwise cosine
  similarity ``sim[d, k]``, then ``w[d, k] = softmax(sim[d, k] / tau)``
  over k. Captures full functional similarity at the cost of loading every
  expert's full weight stack.
* **weight_cosine_of_router** — use each expert's row of the router gate
  matrix (``mlp.gate.weight``) as its feature vector. Cheap to compute
  (one ``[num_experts, hidden]`` tensor per layer, two orders of magnitude
  smaller than the expert stack) and reflects how the *teacher* itself
  decides expert similarity at the routing layer. Same softmax
  normalization as ``weight_cosine``.
* **cooccur** — read pair frequencies produced by a (future) router
  co-occurrence collector. v1 ships a stub for the collector, so this path
  is selectable but errors out clearly until that script is implemented.

Memory budget per layer (macro):

* ``weight_cosine``: ``2 × num_experts × per_expert_bytes`` in fp32
  (load both proj stacks together to build features). For Qwen3.5 256
  experts × ~7M params/expert × 4 bytes ≈ ~7 GB peak.
* ``weight_cosine_of_router``: ``num_experts × hidden × 4 bytes`` in fp32
  (~4 MB at 256 × 4096). Negligible.
* ``cooccur``: ``num_experts^2 × 4 bytes`` (~250 KB at 256 experts). Negligible.

The resulting ``MergePlan`` is also serialised to ``merge_plan.json``
under the student dir so a downstream reader can audit which dropped
expert went where (per-layer ``[d_id, k_id, weight]`` triples for macro,
aggregate stats for neuron mode).
"""

from __future__ import annotations

import gc
import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from moe_prune_distill.adapters.base import MoEAdapter
from moe_prune_distill.prune.slicer import MergePlan

_GATE_UP_TAIL = ".mlp.experts.gate_up_proj"
_DOWN_TAIL = ".mlp.experts.down_proj"
_ROUTER_TAIL = ".mlp.gate.weight"


def _log_process_mem(tag: str, log=None):
    import os
    import gc
    import ctypes

    gc.collect()

    # 【核心防御1】Windows 强制内存回收核武器！逼迫 PyTorch/OS 释放残影
    try:
        psapi = ctypes.WinDLL("psapi")
        kernel32 = ctypes.WinDLL("kernel32")
        handle = kernel32.OpenProcess(0x1F0FFF, False, os.getpid())
        psapi.EmptyWorkingSet(handle)
        kernel32.CloseHandle(handle)
    except Exception:
        pass

    try:
        import psutil
        p = psutil.Process(os.getpid())
        mi = p.memory_info()

        msg = f"[MEM] {tag}: rss={mi.rss / 1024 ** 3:.2f} GB, vms={mi.vms / 1024 ** 3:.2f} GB"

        full = p.memory_full_info()
        if hasattr(full, "uss"):
            msg += f", uss={full.uss / 1024 ** 3:.2f} GB"
        if hasattr(full, "private"):
            msg += f", private={full.private / 1024 ** 3:.2f} GB"

        if log:
            log.info(msg)
        else:
            print(msg)
    except ImportError:
        pass


def _build_weight_map(teacher_dir: Path) -> dict[str, str]:
    idx = teacher_dir / "model.safetensors.index.json"
    if idx.is_file():
        return dict(json.loads(idx.read_text(encoding="utf-8"))["weight_map"])
    files = sorted(p for p in teacher_dir.glob("*.safetensors") if p.is_file())
    if not files:
        raise FileNotFoundError(f"no safetensors under {teacher_dir}")
    if len(files) == 1:
        with safe_open(str(files[0]), framework="pt", device="cpu") as f:
            keys = list(f.keys())
        return {k: files[0].name for k in keys}
    out: dict[str, str] = {}
    for p in files:
        with safe_open(str(p), framework="pt", device="cpu") as f:
            for k in f.keys():
                out[k] = p.name
    return out


def _layer_expert_keys(weight_map: dict[str, str], num_layers: int) -> dict[int, dict[str, str]]:
    out: dict[int, dict[str, str]] = defaultdict(dict)
    for key in weight_map:
        for tail, name in ((_GATE_UP_TAIL, "gate_up_proj"), (_DOWN_TAIL, "down_proj")):
            if not key.endswith(tail):
                continue
            stem = key[: -len(tail)]
            try:
                layer = int(stem.rsplit(".", 1)[-1])
            except ValueError:
                continue
            out[layer][name] = key
    return {l: out[l] for l in range(num_layers) if "gate_up_proj" in out[l] and "down_proj" in out[l]}


def _layer_router_keys(weight_map: dict[str, str], num_layers: int) -> dict[int, str]:
    out: dict[int, str] = {}
    for key in weight_map:
        if not key.endswith(_ROUTER_TAIL):
            continue
        parts = key.split(".")
        try:
            i = parts.index("layers")
            layer = int(parts[i + 1])
        except (ValueError, IndexError):
            continue
        out[layer] = key
    return {l: out[l] for l in range(num_layers) if l in out}


def _load_layer_expert_stack(
        teacher_dir: Path,
        weight_map: dict[str, str],
        layer_keys: dict[str, str],
) -> tuple[torch.Tensor, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for which, key in layer_keys.items():
        shard = weight_map[key]
        with safe_open(str(teacher_dir / shard), framework="pt", device="cpu") as f:
            # 【核心防御2】坚决不升高精度！保留原有的 float16 或 bfloat16，从源头上切断 50% 内存暴涨
            out[which] = f.get_tensor(key)
    return out["gate_up_proj"], out["down_proj"]


def _load_layer_router(
        teacher_dir: Path,
        weight_map: dict[str, str],
        router_key: str,
) -> torch.Tensor:
    shard = weight_map[router_key]
    with safe_open(str(teacher_dir / shard), framework="pt", device="cpu") as f:
        return f.get_tensor(router_key).to(torch.float32)


def _flatten_features(gate_up: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
    n = gate_up.shape[0]
    assert down.shape[0] == n
    a = gate_up.to(torch.float32).reshape(n, -1)
    b = down.to(torch.float32).reshape(n, -1)
    return torch.cat([a, b], dim=1)


def _weight_cosine_similarity(feat: torch.Tensor, dropped: list[int], surviving: list[int]) -> torch.Tensor:
    norms = feat.norm(dim=1, keepdim=True).clamp_min(1e-12)
    feat_n = feat / norms
    fd = feat_n[dropped]
    fk = feat_n[surviving]
    return fd @ fk.t()


def _softmax_rows(sim: torch.Tensor, tau: float) -> torch.Tensor:
    if sim.numel() == 0:
        return sim
    return torch.softmax(sim / max(tau, 1e-6), dim=1)


def _normalize_cooccur_rows(cooccur: torch.Tensor) -> torch.Tensor:
    row_sum = cooccur.sum(dim=1, keepdim=True)
    safe = row_sum.clamp_min(1e-12)
    out = cooccur / safe
    if (row_sum == 0).any():
        nk = cooccur.shape[1]
        zero_rows = (row_sum.squeeze(-1) == 0)
        out[zero_rows] = 1.0 / max(nk, 1)
    return out


def build_merge_plan(
        teacher_dir: str | Path,
        adapter: MoEAdapter,
        hf_config: dict[str, Any],
        surviving_per_layer: dict[int, list[int]],
        *,
        strategy: str = "weight_cosine",
        alpha: float = 0.5,
        tau: float = 0.1,
        cooccur_path: str | Path | None = None,
        log: logging.Logger | None = None,
) -> MergePlan:
    log = log or logging.getLogger("moe_prune_distill.expert_merge")
    teacher_dir = Path(teacher_dir)
    num_layers = adapter.get_num_layers(hf_config)
    num_experts = adapter.get_num_experts(hf_config)

    weight_map = _build_weight_map(teacher_dir)
    per_layer_expert_keys = _layer_expert_keys(weight_map, num_layers)
    per_layer_router_keys = _layer_router_keys(weight_map, num_layers)

    cooccur_data: dict[str, Any] | None = None
    if strategy == "cooccur":
        if cooccur_path is None or not Path(cooccur_path).is_file():
            raise FileNotFoundError(f"strategy=cooccur needs router_cooccur.json")
        cooccur_data = json.loads(Path(cooccur_path).read_text(encoding="utf-8"))
    elif strategy not in ("weight_cosine", "weight_cosine_of_router"):
        raise ValueError(f"unknown merge strategy: {strategy!r}")

    plan = MergePlan(alpha=float(alpha))
    plan.surviving_per_layer = {l: list(v) for l, v in surviving_per_layer.items()}
    plan.dropped_per_layer = {
        l: [e for e in range(num_experts) if e not in set(surviving_per_layer.get(l, []))]
        for l in range(num_layers)
    }

    skipped: list[tuple[int, str]] = []
    for layer in range(num_layers):
        surviving = list(surviving_per_layer.get(layer, []))
        dropped = plan.dropped_per_layer[layer]
        if not surviving or not dropped:
            continue

        if strategy == "weight_cosine":
            if layer not in per_layer_expert_keys:
                skipped.append((layer, "no stacked expert keys"))
                continue
            gate_up, down = _load_layer_expert_stack(
                teacher_dir, weight_map, per_layer_expert_keys[layer]
            )
            feat = _flatten_features(gate_up, down)
            del gate_up, down
            sim = _weight_cosine_similarity(feat, dropped, surviving)
            del feat
            w = _softmax_rows(sim, tau)
        elif strategy == "weight_cosine_of_router":
            if layer not in per_layer_router_keys:
                skipped.append((layer, "no router gate key"))
                continue
            router_w = _load_layer_router(
                teacher_dir, weight_map, per_layer_router_keys[layer]
            )
            if router_w.ndim != 2 or router_w.shape[0] != num_experts:
                skipped.append((layer, f"router shape {tuple(router_w.shape)} ≠ ({num_experts}, hidden)"))
                continue
            sim = _weight_cosine_similarity(router_w, dropped, surviving)
            del router_w
            w = _softmax_rows(sim, tau)
        else:
            assert cooccur_data is not None
            layer_pair = cooccur_data.get("layers", {}).get(str(layer), {})
            pair = layer_pair.get("pair_counts")
            if pair is None:
                raise ValueError(f"router_cooccur.json missing pair_counts for layer {layer}")
            mat = torch.zeros(num_experts, num_experts, dtype=torch.float32)
            for d_str, kvs in pair.items():
                d = int(d_str)
                for k_str, c in kvs.items():
                    mat[d, int(k_str)] = float(c)
            sub = mat[dropped][:, surviving]
            w = _normalize_cooccur_rows(sub)
        plan.weights[layer] = w.to(torch.float32).contiguous()

    if skipped:
        preview = [f"{l}({why})" for l, why in skipped[:8]]
        if len(skipped) > 8:
            preview.append("...")
        log.warning("merge_plan: %d layer(s) skipped (fall back to index-select-only): %s", len(skipped), preview)

    return plan


def serialize_merge_plan(plan: MergePlan) -> dict[str, Any]:
    if plan.mode == "neuron_swiglu":
        return _serialize_neuron_plan(plan)

    out: dict[str, Any] = {
        "mode": "macro",
        "alpha": float(plan.alpha),
        "layers": {},
    }
    for layer, w in plan.weights.items():
        dropped = plan.dropped_per_layer.get(layer, [])
        surviving = plan.surviving_per_layer.get(layer, [])
        triples: list[list[float]] = []
        wt = w.to(torch.float32)
        for di, d_id in enumerate(dropped):
            for ki, k_id in enumerate(surviving):
                triples.append([int(d_id), int(k_id), float(wt[di, ki])])
        out["layers"][str(layer)] = {
            "dropped": list(map(int, dropped)),
            "surviving": list(map(int, surviving)),
            "weights": triples,
        }
    return out


def _serialize_neuron_plan(plan: MergePlan) -> dict[str, Any]:
    out: dict[str, Any] = {
        "mode": "neuron_swiglu",
        "strategy": plan.neuron_meta.get("strategy"),
        "alpha": float(plan.alpha),
        "sim_threshold": plan.neuron_meta.get("sim_threshold"),
        "top_k": plan.neuron_meta.get("top_k"),
        "router_stats_used": bool(plan.neuron_meta.get("router_stats_used", False)),
        "layers": {},
    }
    for layer in sorted(plan.neuron_stats.keys()):
        s = plan.neuron_stats[layer]
        layer_out = {
            "surviving": list(map(int, plan.surviving_per_layer.get(layer, []))),
            "dropped": list(map(int, plan.dropped_per_layer.get(layer, []))),
            "neuron_stats": {
                k: v for k, v in s.items() if k not in ("host_pairs",)
            },
        }
        if "host_pairs" in s:
            layer_out["host_pairs"] = {str(k): int(v) for k, v in s["host_pairs"].items()}
        out["layers"][str(layer)] = layer_out
    return out


def write_merge_report(
        plan: MergePlan,
        student_dir: str | Path,
        *,
        teacher_arch: str | None = None,
        num_layers_total: int | None = None,
        num_experts_total: int | None = None,
        target_num_experts: int | None = None,
        target_num_experts_per_tok: int | None = None,
) -> Path:
    student_dir = Path(student_dir)
    student_dir.mkdir(parents=True, exist_ok=True)
    out_path = student_dir / "merge_report.md"

    lines: list[str] = ["# Expert Merge Report", ""]
    if plan.mode == "neuron_swiglu":
        lines += [f"- Strategy: `{plan.neuron_meta.get('strategy')}`"]
    else:
        lines += ["- Strategy: `macro` (legacy weighted scaled-add)"]
    if teacher_arch:
        teacher_line = f"- Teacher: `{teacher_arch}`"
        if num_layers_total is not None and num_experts_total is not None:
            teacher_line += f" ({num_layers_total} layers, {num_experts_total} experts)"
        lines.append(teacher_line)
    if target_num_experts is not None:
        st = f"- Student: {target_num_experts} experts"
        if target_num_experts_per_tok is not None:
            st += f" (per-token = {target_num_experts_per_tok})"
        lines.append(st)
    lines += [f"- Alpha: {plan.alpha}"]

    if plan.mode == "neuron_swiglu":
        lines += [
            f"- Sim threshold: {plan.neuron_meta.get('sim_threshold')}",
            f"- Top-K (global only): {plan.neuron_meta.get('top_k')}",
            f"- Router-stats dynamic alpha: "
            f"{'enabled' if plan.neuron_meta.get('router_stats_used') else 'disabled'}",
        ]
    lines.append("")

    if plan.mode == "neuron_swiglu" and plan.neuron_stats:
        total_dropped = sum(s["total_dropped_neurons"] for s in plan.neuron_stats.values())
        total_hosted = sum(s["hosted"] for s in plan.neuron_stats.values())
        total_drop_thr = sum(s["dropped_below_thr"] for s in plan.neuron_stats.values())
        weighted_sim = (
            sum(s["sim_mean_hosted"] * s["hosted"] for s in plan.neuron_stats.values()) / total_hosted
            if total_hosted else 0.0
        )
        lines += [
            "## Aggregate", "",
            "| Metric | Value |", "|---|---|",
            f"| Layers processed | {len(plan.neuron_stats)} |",
            f"| Total dropped neurons across model | {total_dropped:,} |",
            f"| Hosted (sim ≥ threshold) | {total_hosted:,} ({100.0 * total_hosted / max(1, total_dropped):.1f}%) |",
            f"| Dropped below threshold | {total_drop_thr:,} ({100.0 * total_drop_thr / max(1, total_dropped):.1f}%) |",
            f"| Mean cosine sim of hosted matches | {weighted_sim:.3f} |", ""
        ]

        per_layer = sorted(
            plan.neuron_stats.items(),
            key=lambda kv: kv[1]["hosted"] / max(1, kv[1]["total_dropped_neurons"]),
            reverse=True,
        )
        lines += [
            "## Per-layer summary (top 5 highest hosted-rate)", "",
            "| Layer | Hosted | Dropped<thr | Hosted-rate | Mean sim | Host-load max |", "|---|---|---|---|---|---|",
        ]
        for layer, s in per_layer[:5]:
            rate = 100.0 * s["hosted"] / max(1, s["total_dropped_neurons"])
            lines.append(
                f"| {layer} | {s['hosted']:,} | {s['dropped_below_thr']:,} | {rate:.1f}% | {s['sim_mean_hosted']:.3f} | {s['host_load_max']} |")
        lines.append("")

        lines += [
            "## Per-layer summary (top 5 lowest hosted-rate — investigate)", "",
            "| Layer | Hosted | Dropped<thr | Hosted-rate | Mean sim | Host-load max |", "|---|---|---|---|---|---|",
        ]
        for layer, s in per_layer[-5:][::-1]:
            rate = 100.0 * s["hosted"] / max(1, s["total_dropped_neurons"])
            lines.append(
                f"| {layer} | {s['hosted']:,} | {s['dropped_below_thr']:,} | {rate:.1f}% | {s['sim_mean_hosted']:.3f} | {s['host_load_max']} |")
        lines.append("")

        max_load = max(s["host_load_max"] for s in plan.neuron_stats.values())
        all_hist: dict[str, int] = {"0": 0, "1": 0, "2": 0, "3": 0, ">=4": 0}
        for s in plan.neuron_stats.values():
            for bk, bv in s.get("host_load_histogram", {}).items():
                all_hist[bk] = all_hist.get(bk, 0) + int(bv)
        total_hist = sum(all_hist.values()) or 1
        lines += [
            "## Host load distribution (network-wide)", "",
            f"- Max dropped-neurons absorbed by a single host: **{max_load}**",
            f"- Hosts with 0 absorbed: {all_hist['0']:,} ({100.0 * all_hist['0'] / total_hist:.1f}%)",
            f"- Hosts with 1: {all_hist['1']:,} ({100.0 * all_hist['1'] / total_hist:.1f}%)",
            f"- Hosts with 2: {all_hist['2']:,}",
            f"- Hosts with 3: {all_hist['3']:,}",
            f"- Hosts with ≥4: {all_hist['>=4']:,}", ""
        ]

        warnings: list[str] = []
        for layer, s in plan.neuron_stats.items():
            rate = s["hosted"] / max(1, s["total_dropped_neurons"])
            if rate < 0.10 and s["total_dropped_neurons"] > 0:
                warnings.append(
                    f"- Layer {layer}: hosted-rate {100 * rate:.1f}% < 10% — consider lowering --neuron-sim-threshold.")
            if s["host_load_max"] >= 32:
                warnings.append(
                    f"- Layer {layer}: a single host absorbed {s['host_load_max']} dropped neurons — consider raising --neuron-sim-threshold.")
        if warnings:
            lines += ["## Warnings", ""] + warnings + [""]
        else:
            lines += ["## Warnings", "", "_None_", ""]

    elif plan.mode == "macro":
        lines += [
            "## Aggregate (macro)", "",
            f"- Layers covered: {len(plan.weights)}",
            f"- Alpha: {plan.alpha}", "",
            "Per-layer triples are written to `merge_plan.json`.", ""
        ]

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


__all__ = [
    "build_merge_plan",
    "build_neuron_merge_plan",
    "serialize_merge_plan",
    "write_merge_report",
]


# === Neuron-level (SwiGLU-aware) merge ====================================

def _build_super_vec(gate_up: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """
    Build normalized per-neuron super-vectors [E, I, 2H].
    【核心防御3】原位拼接直接投喂给 FP32，避免产生不必要的连续化大张量副本。
    """
    if gate_up.ndim != 3:
        raise ValueError(f"gate_up_proj must be rank-3 [E, 2I, H], got shape {tuple(gate_up.shape)}")

    e, two_i, h = gate_up.shape
    if two_i % 2 != 0:
        raise ValueError(f"gate_up_proj middle dim must be even (2*I), got {two_i}")

    i = two_i // 2

    # 直接开辟目标矩阵，彻底屏蔽显式 .contiguous() 副本堆积
    super_vec = torch.empty((e, i, 2 * h), dtype=torch.float32)
    super_vec[:, :, :h].copy_(gate_up[:, :i, :])
    super_vec[:, :, h:].copy_(gate_up[:, i:, :])

    norms = super_vec.norm(dim=2, keepdim=True).clamp_min_(1e-12)
    super_vec.div_(norms)
    del norms

    return super_vec, i, h


def _scatter_down_columns(
        bucket: torch.Tensor,  # [Nk, H, I]
        d_down: torch.Tensor,  # [H, I]
        host_kept_e: torch.Tensor,  # [Q] long, in [0, Nk)
        host_kept_n: torch.Tensor,  # [Q] long, in [0, I)
        src_neuron: torch.Tensor,  # [Q] long, in [0, I)
        scale: float,
) -> None:
    if host_kept_e.numel() == 0:
        return
    src = d_down.index_select(dim=1, index=src_neuron) * scale
    h = bucket.shape[1]
    h_idx = torch.arange(h, dtype=torch.long).view(h, 1).expand(h, host_kept_e.numel())
    e_idx = host_kept_e.view(1, -1).expand(h, host_kept_e.numel())
    n_idx = host_kept_n.view(1, -1).expand(h, host_kept_e.numel())
    bucket.index_put_((e_idx, h_idx, n_idx), src, accumulate=True)


def _quantile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return 0.0
    return float(torch.quantile(values.to(torch.float32), q).item())


def _build_host_load_histogram(host_load: torch.Tensor) -> dict[str, int]:
    flat = host_load.reshape(-1)
    h: dict[str, int] = {"0": 0, "1": 0, "2": 0, "3": 0, ">=4": 0}
    for v_t in flat:
        v = int(v_t.item())
        if v >= 4:
            h[">=4"] += 1
        else:
            h[str(v)] += 1
    return h


def _match_local_layer(
        super_norm: torch.Tensor,
        down: torch.Tensor,
        surviving: list[int],
        dropped: list[int],
        router_w: torch.Tensor,
        *,
        sim_threshold: float,
        alpha: float,
        expert_alpha_scale: dict[int, float] | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    nk = len(surviving)
    h = down.shape[1]
    i = down.shape[2]

    # 【核心防御4】Bucket 降级使用半精度，直接缩小一倍内存占用 (GB级别)
    bucket = torch.zeros(nk, h, i, dtype=torch.float16)
    host_load = torch.zeros(nk, i, dtype=torch.long)

    rw = router_w.to(torch.float32)
    rw_norm = rw / rw.norm(dim=1, keepdim=True).clamp_min(1e-12)
    rw_d = rw_norm[dropped]
    rw_k = rw_norm[surviving]
    sim_router = rw_d @ rw_k.t()
    host_kept_idx = sim_router.argmax(dim=1)
    host_pairs: dict[int, int] = {
        int(d): int(surviving[host_kept_idx[di].item()])
        for di, d in enumerate(dropped)
    }

    hosted_total = 0
    drop_below_thr_total = 0
    all_hosted_sims: list[torch.Tensor] = []

    for d_pos, d_id in enumerate(dropped):
        k_pos = int(host_kept_idx[d_pos].item())
        k_id = surviving[k_pos]
        d_super = super_norm[d_id]
        k_super = super_norm[k_id]
        sim = d_super @ k_super.t()
        best_val, best_n = sim.max(dim=1)
        mask = best_val >= sim_threshold
        hosted_total += int(mask.sum().item())
        drop_below_thr_total += int((~mask).sum().item())

        if mask.any():
            keep_d_neurons = mask.nonzero(as_tuple=False).flatten()
            best_n_kept = best_n.index_select(0, keep_d_neurons)

            # 【配套降级】传入半精度参与组装
            d_down_f = down[d_id].to(torch.float16)
            scale = alpha
            if expert_alpha_scale is not None:
                scale *= float(expert_alpha_scale.get(int(d_id), 1.0))
            host_e = torch.full_like(keep_d_neurons, k_pos)
            _scatter_down_columns(bucket, d_down_f, host_e, best_n_kept, keep_d_neurons, scale)
            host_load[k_pos].index_add_(0, best_n_kept, torch.ones_like(best_n_kept))
            all_hosted_sims.append(best_val.index_select(0, keep_d_neurons))

    sims = torch.cat(all_hosted_sims) if all_hosted_sims else torch.empty(0, dtype=torch.float32)
    stats: dict[str, Any] = {
        "total_dropped_neurons": len(dropped) * i,
        "hosted": hosted_total,
        "dropped_below_thr": drop_below_thr_total,
        "sim_mean_hosted": float(sims.mean().item()) if sims.numel() else 0.0,
        "sim_p10_hosted": _quantile(sims, 0.10),
        "sim_p90_hosted": _quantile(sims, 0.90),
        "host_load_max": int(host_load.max().item()) if host_load.numel() else 0,
        "host_load_histogram": _build_host_load_histogram(host_load),
        "host_pairs": host_pairs,
    }
    return bucket, stats


def _candidate_block_hungarian_assignment(
        top_sim: torch.Tensor,
        top_idx: torch.Tensor,
        num_cols: int,
        *,
        row_block_size: int = 2048,
        large: float = 1e6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GPT: Memory-safe block Hungarian on a sparse top-K candidate graph."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    n, k = top_idx.shape
    assignment = torch.full((n,), -1, dtype=torch.long)
    score = torch.full((n,), -1.0, dtype=torch.float32)
    claimed = torch.zeros(num_cols, dtype=torch.bool)

    row_order = torch.argsort(top_sim[:, 0], descending=True)

    for start in range(0, n, row_block_size):
        rows = row_order[start:start + row_block_size]
        if rows.numel() == 0:
            continue

        block_top_idx = top_idx.index_select(0, rows)
        block_top_sim = top_sim.index_select(0, rows)

        valid = block_top_idx >= 0
        if claimed.numel() > 0:
            valid = valid & (~claimed[block_top_idx.clamp_min(0)])
        if not valid.any():
            continue

        cand_cols = torch.unique(block_top_idx[valid])
        cand_cols = cand_cols[cand_cols >= 0]
        if cand_cols.numel() == 0:
            continue

        col_to_local = {int(c.item()): j for j, c in enumerate(cand_cols)}

        rb = rows.numel()
        cb = cand_cols.numel()

        sub_cost = np.full((rb, cb), large, dtype=np.float32)
        b_idx_np = block_top_idx.cpu().numpy()
        b_sim_np = block_top_sim.cpu().numpy()

        for local_r in range(rb):
            for slot in range(k):
                c = int(b_idx_np[local_r, slot])
                if c < 0 or bool(claimed[c].item()):
                    continue
                local_c = col_to_local.get(c)
                if local_c is None:
                    continue
                sub_cost[local_r, local_c] = -float(b_sim_np[local_r, slot])

        row_ind, col_ind = linear_sum_assignment(sub_cost)

        for rr, cc in zip(row_ind, col_ind):
            cost_val = float(sub_cost[rr, cc])
            if cost_val >= large - 1.0:
                continue
            global_row = int(rows[rr].item())
            global_col = int(cand_cols[cc].item())
            if bool(claimed[global_col].item()):
                continue

            assignment[global_row] = global_col
            score[global_row] = -cost_val
            claimed[global_col] = True

        del sub_cost

    return assignment, score


def _match_global_layer(
        super_norm: torch.Tensor,
        down: torch.Tensor,
        surviving: list[int],
        dropped: list[int],
        *,
        sim_threshold: float,
        alpha: float,
        top_k: int,
        expert_alpha_scale: dict[int, float] | None,
        expert_chunk: int,
        scratch_dir: Path | None,
        layer: int,
        block_threshold: int,
        hungarian_blocks: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    import gc
    nk, nd = len(surviving), len(dropped)
    e, i, two_h = super_norm.shape
    h = down.shape[1]

    kept_super = super_norm[surviving].reshape(nk * i, two_h).contiguous()

    nd_total = nd * i
    nk_total = nk * i

    top_sim = torch.full((nd_total, top_k), -1.0, dtype=torch.float32)
    top_idx = torch.full((nd_total, top_k), -1, dtype=torch.long)

    cache_path = (scratch_dir / f"neuron_match_layer{layer}.pt") if scratch_dir else None
    cached = (
        torch.load(cache_path, map_location="cpu", weights_only=True)
        if cache_path is not None and cache_path.is_file()
        else None
    )
    if cached is not None and cached.get("nd_total") == nd_total and cached.get("nk_total") == nk_total:
        top_sim = cached["top_sim"]
        top_idx = cached["top_idx"]
    else:
        chunk = max(1, min(expert_chunk, 4))
        for d_start in range(0, nd, chunk):
            d_end = min(nd, d_start + chunk)
            d_block = super_norm[[dropped[di] for di in range(d_start, d_end)]]
            d_flat = d_block.reshape((d_end - d_start) * i, two_h).contiguous()

            sim = d_flat @ kept_super.t()
            k_eff = min(top_k, sim.shape[1])
            vals, idx = sim.topk(k_eff, dim=1, largest=True, sorted=True)

            row0 = d_start * i
            row1 = d_end * i
            top_sim[row0:row1, :k_eff] = vals
            top_idx[row0:row1, :k_eff] = idx

            del sim, vals, idx, d_flat, d_block

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"nd_total": nd_total, "nk_total": nk_total, "top_sim": top_sim, "top_idx": top_idx}, cache_path)

    del kept_super
    gc.collect()

    try:
        from scipy.optimize import linear_sum_assignment
        scipy_installed = True
    except ImportError:
        scipy_installed = False

    if scipy_installed:
        assignment, assignment_score = _candidate_block_hungarian_assignment(
            top_sim,
            top_idx,
            num_cols=nk_total,
            row_block_size=1024,
        )
    else:
        assignment = torch.full((nd_total,), -1, dtype=torch.long)
        assignment_score = torch.full((nd_total,), -1.0, dtype=torch.float32)
        claimed = torch.zeros(nk_total, dtype=torch.bool)
        best_sim_per_row = top_sim[:, 0]
        order = torch.argsort(best_sim_per_row, descending=True).tolist()
        for r in order:
            for slot in range(top_k):
                cand = int(top_idx[r, slot].item())
                if cand < 0: continue
                if not bool(claimed[cand].item()):
                    assignment[r] = cand
                    assignment_score[r] = top_sim[r, slot].item()
                    claimed[cand] = True
                    break

    # 【核心防御4】生成半精度 Bucket
    bucket = torch.zeros(nk, h, i, dtype=torch.float16)
    host_load = torch.zeros(nk, i, dtype=torch.long)
    hosted_total = 0
    drop_below_thr_total = 0
    sims_hosted = []

    for d_pos in range(nd):
        d_id = dropped[d_pos]
        # 【配套降级】传入半精度源
        d_down_f = down[d_id].to(torch.float16)

        scale = alpha
        if expert_alpha_scale is not None:
            scale *= float(expert_alpha_scale.get(int(d_id), 1.0))

        row_lo = d_pos * i
        keep_e_buf, keep_n_buf, src_buf = [], [], []

        for src_n in range(i):
            global_row = row_lo + src_n
            assigned = int(assignment[global_row].item())

            if assigned < 0:
                drop_below_thr_total += 1
                continue

            sim_value = float(assignment_score[global_row].item())

            if sim_value < sim_threshold:
                drop_below_thr_total += 1
                continue

            kept_e = assigned // i
            kept_n = assigned % i
            keep_e_buf.append(kept_e)
            keep_n_buf.append(kept_n)
            src_buf.append(src_n)
            sims_hosted.append(sim_value)
            host_load[kept_e, kept_n] += 1

        if keep_e_buf:
            host_e = torch.tensor(keep_e_buf, dtype=torch.long)
            host_n = torch.tensor(keep_n_buf, dtype=torch.long)
            src = torch.tensor(src_buf, dtype=torch.long)
            _scatter_down_columns(bucket, d_down_f, host_e, host_n, src, scale)
            hosted_total += len(keep_e_buf)

            del keep_e_buf, keep_n_buf, src_buf, host_e, host_n, src

    sims_t = torch.tensor(sims_hosted, dtype=torch.float32) if sims_hosted else torch.empty(0, dtype=torch.float32)

    stats: dict[str, Any] = {
        "total_dropped_neurons": nd_total,
        "hosted": hosted_total,
        "dropped_below_thr": drop_below_thr_total,
        "sim_mean_hosted": float(sims_t.mean().item()) if sims_t.numel() else 0.0,
        "sim_p10_hosted": float(_quantile(sims_t, 0.10)),
        "sim_p90_hosted": float(_quantile(sims_t, 0.90)),
        "host_load_max": int(host_load.max().item()) if host_load.numel() else 0,
        "host_load_histogram": _build_host_load_histogram(host_load),
    }

    del top_sim, top_idx, assignment, assignment_score
    gc.collect()

    return bucket, stats


def _load_router_stats_scale(
        path: str | Path,
        num_layers: int,
) -> dict[int, dict[int, float]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[int, dict[int, float]] = {}
    for layer in range(num_layers):
        layer_data = payload.get("layers", {}).get(str(layer))
        if layer_data is None:
            continue
        usage = layer_data.get("usage_counts") or {}
        if not usage:
            continue
        freq_by_id = {int(eid): float(c) for eid, c in usage.items()}
        out[layer] = freq_by_id
    return out


def _normalize_alpha_scale_for_dropped(
        full_scale: dict[int, dict[int, float]],
        layer: int,
        dropped: list[int],
) -> dict[int, float] | None:
    layer_freq = full_scale.get(layer)
    if layer_freq is None:
        return None
    vals = [layer_freq.get(int(d), 0.0) for d in dropped]
    if not vals:
        return None
    mean = sum(vals) / len(vals)
    if mean <= 0.0:
        return {int(d): 1.0 for d in dropped}
    return {int(d): float(layer_freq.get(int(d), 0.0)) / mean for d in dropped}


def build_neuron_merge_plan(
        teacher_dir: str | Path,
        adapter: MoEAdapter,
        hf_config: dict[str, Any],
        surviving_per_layer: dict[int, list[int]],
        *,
        strategy: str,
        alpha: float = 0.5,
        sim_threshold: float = 0.5,
        top_k: int = 4,
        expert_chunk: int = 16,
        block_threshold: int = 80_000,
        hungarian_blocks: int = 8,
        router_stats_path: str | Path | None = None,
        scratch_dir: str | Path | None = None,
        bucket_dir: str | Path | None = None,
        log: logging.Logger | None = None,
) -> MergePlan:
    log = log or logging.getLogger("moe_prune_distill.expert_merge")
    if strategy not in ("neuron_swiglu_local", "neuron_swiglu_global"):
        raise ValueError(f"unknown neuron strategy: {strategy!r}")

    teacher_dir = Path(teacher_dir)
    num_layers = adapter.get_num_layers(hf_config)
    num_experts = adapter.get_num_experts(hf_config)

    weight_map = _build_weight_map(teacher_dir)
    per_layer_expert_keys = _layer_expert_keys(weight_map, num_layers)
    per_layer_router_keys = _layer_router_keys(weight_map, num_layers)

    full_alpha_scale: dict[int, dict[int, float]] | None = None
    if router_stats_path is not None:
        full_alpha_scale = _load_router_stats_scale(router_stats_path, num_layers)
        log.info(
            "neuron merge: loaded router stats from %s (layers=%d)",
            router_stats_path,
            len(full_alpha_scale),
        )

    scratch_path = Path(scratch_dir) if scratch_dir is not None else None
    bucket_path_root = Path(bucket_dir) if bucket_dir is not None else None
    if bucket_path_root is not None:
        bucket_path_root.mkdir(parents=True, exist_ok=True)

    plan = MergePlan(alpha=float(alpha), mode="neuron_swiglu")
    plan.surviving_per_layer = {l: list(v) for l, v in surviving_per_layer.items()}
    plan.dropped_per_layer = {
        l: [e for e in range(num_experts) if e not in set(surviving_per_layer.get(l, []))]
        for l in range(num_layers)
    }
    plan.neuron_meta = {
        "strategy": strategy,
        "alpha": float(alpha),
        "sim_threshold": float(sim_threshold),
        "top_k": int(top_k) if strategy == "neuron_swiglu_global" else None,
        "router_stats_used": bool(full_alpha_scale is not None),
    }

    skipped: list[tuple[int, str]] = []
    for layer in range(num_layers):
        surviving = list(surviving_per_layer.get(layer, []))
        dropped = plan.dropped_per_layer[layer]
        if not surviving or not dropped:
            continue
        if layer not in per_layer_expert_keys:
            skipped.append((layer, "no stacked expert keys"))
            continue
        if strategy == "neuron_swiglu_local" and layer not in per_layer_router_keys:
            skipped.append((layer, "no router gate key (local needs router cosine)"))
            continue

        _log_process_mem(f"layer {layer} before load", log)
        gate_up, down = _load_layer_expert_stack(
            teacher_dir, weight_map, per_layer_expert_keys[layer]
        )
        _log_process_mem(f"layer {layer} after load gate_up/down", log)

        try:
            super_norm, intermediate, hidden = _build_super_vec(gate_up)

            # 【核心防御5】提取完超级归一特征，原版超大 gate_up 当即处决！
            del gate_up
            gc.collect()
            _log_process_mem(f"layer {layer} after kill gate_up", log)

        except ValueError as exc:
            skipped.append((layer, f"gate_up shape: {exc}"))
            del gate_up, down
            continue

        layer_alpha_scale: dict[int, float] | None = None
        if full_alpha_scale is not None:
            layer_alpha_scale = _normalize_alpha_scale_for_dropped(
                full_alpha_scale, layer, dropped
            )

        if strategy == "neuron_swiglu_local":
            router_w = _load_layer_router(
                teacher_dir, weight_map, per_layer_router_keys[layer]
            )
            bucket, stats = _match_local_layer(
                super_norm,
                down,
                surviving,
                dropped,
                router_w,
                sim_threshold=sim_threshold,
                alpha=alpha,
                expert_alpha_scale=layer_alpha_scale,
            )
            _log_process_mem(f"layer {layer} after match bucket", log)
            del router_w
        else:
            bucket, stats = _match_global_layer(
                super_norm,
                down,
                surviving,
                dropped,
                sim_threshold=sim_threshold,
                alpha=alpha,
                top_k=top_k,
                expert_alpha_scale=layer_alpha_scale,
                expert_chunk=expert_chunk,
                scratch_dir=scratch_path,
                layer=layer,
                block_threshold=block_threshold,
                hungarian_blocks=hungarian_blocks,
            )
            _log_process_mem(f"layer {layer} after match bucket", log)

        if bucket_path_root is not None:
            bucket_file = bucket_path_root / f"layer{layer}.pt"
            # 【核心防御6】绝不用 .contiguous() 开缓冲辟新地址！直接保存当前内存片
            torch.save(bucket, bucket_file)
            plan.neuron_down_contrib_paths[layer] = str(bucket_file)
            del bucket
        else:
            # 去掉连续化需求
            plan.neuron_down_contrib[layer] = bucket

        _log_process_mem(f"layer {layer} after save/delete bucket", log)
        plan.neuron_stats[layer] = stats
        log.info(
            "merge layer %d: hosted=%d/%d (%.1f%%) sim_mean=%.3f host_load_max=%d",
            layer,
            stats["hosted"],
            stats["total_dropped_neurons"],
            100.0 * stats["hosted"] / max(1, stats["total_dropped_neurons"]),
            stats["sim_mean_hosted"],
            stats["host_load_max"],
        )
        del down, super_norm
        gc.collect()
        _log_process_mem(f"layer {layer} after cleanup", log)

    if skipped:
        preview = [f"{l}({why})" for l, why in skipped[:8]]
        if len(skipped) > 8:
            preview.append("...")
        log.warning(
            "neuron merge: %d layer(s) skipped (kept exprs index-select with no merge): %s",
            len(skipped),
            preview,
        )

    return plan
