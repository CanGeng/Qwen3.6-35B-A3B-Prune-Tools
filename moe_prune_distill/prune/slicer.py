"""Shard-aware MoE weight slicing with per-layer expert selection.

The pruning loop is **streaming**: each input shard is opened lazily via
``safetensors.safe_open``, every key is loaded one tensor at a time, sliced
or renamed in place, then written to a sibling output shard before the next
key is touched. Output shards are first staged as ``_part_*.safetensors``
and renamed to the final ``model-*-of-*.safetensors`` form once the total
non-empty shard count is known. Peak resident memory is therefore bounded
by the largest single tensor (typically a ``routed_expert_stack`` of order
1 GB), not by the sum of input shards.
"""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from moe_prune_distill.adapters.base import MoEAdapter
from moe_prune_distill.prune.expert_selector import surviving_experts_first_n


@dataclass
class MergePlan:
    """Per-layer dropped→kept expert merge weights.

    Two modes share this struct:

    * **macro** (default): ``weights[layer]`` is a
      ``[len(dropped), len(surviving)]`` row-stochastic matrix and the
      slicer applies a scaled add to every stacked expert tensor:

          kept_new[k] = kept_orig[k] + alpha * sum_d weights[d,k] * dropped[d]

    * **neuron_swiglu**: SwiGLU-aware neuron-level merge. ``weights`` is
      empty (per-row mixing would corrupt the gate/up activation
      boundary). Instead ``neuron_down_contrib[layer]`` is a pre-aggregated
      ``[num_kept, hidden, intermediate]`` bucket (already alpha-scaled)
      that the slicer adds onto the kept ``down_proj`` stack — and the
      ``gate_up_proj`` stack is left as a plain ``index_select`` of the
      kept experts (no mixing). See ``build_neuron_merge_plan``.
    """

    weights: dict[int, torch.Tensor] = field(default_factory=dict)
    surviving_per_layer: dict[int, list[int]] = field(default_factory=dict)
    dropped_per_layer: dict[int, list[int]] = field(default_factory=dict)
    alpha: float = 0.5
    mode: str = "macro"
    # neuron_swiglu mode: pre-aggregated down_proj contribution per layer,
    # shape [num_kept, hidden, intermediate], already multiplied by alpha.
    # Either ``neuron_down_contrib`` (in-memory) or
    # ``neuron_down_contrib_paths`` (per-layer .pt path) is populated, not
    # both. Path mode lets very large models (Qwen3.5 ~1.5 GB/layer) keep
    # only one bucket resident at a time during the slicer pass.
    neuron_down_contrib: dict[int, torch.Tensor] = field(default_factory=dict)
    neuron_down_contrib_paths: dict[int, str] = field(default_factory=dict)
    # neuron_swiglu mode: per-layer audit stats (hosted, dropped_below_thr,
    # sim_mean_hosted, sim_p10_hosted, sim_p90_hosted, host_load_histogram, ...).
    neuron_stats: dict[int, dict[str, Any]] = field(default_factory=dict)
    # neuron_swiglu mode: free-form metadata (strategy name, sim_threshold,
    # router_stats_used, top_k, ...). Surfaced verbatim into merge_plan.json
    # and merge_report.md.
    neuron_meta: dict[str, Any] = field(default_factory=dict)


def _build_weight_map(teacher_dir: Path) -> dict[str, str]:
    index_path = teacher_dir / "model.safetensors.index.json"
    if index_path.is_file():
        data = json.loads(index_path.read_text(encoding="utf-8"))
        return dict(data["weight_map"])
    st_files = sorted(p for p in teacher_dir.glob("*.safetensors") if p.is_file())
    if not st_files:
        raise FileNotFoundError(f"No safetensors under {teacher_dir}")
    if len(st_files) == 1:
        with safe_open(str(st_files[0]), framework="pt", device="cpu") as f:
            keys = list(f.keys())
        name = st_files[0].name
        return {k: name for k in keys}
    wm: dict[str, str] = {}
    for p in st_files:
        with safe_open(str(p), framework="pt", device="cpu") as f:
            for k in f.keys():
                wm[k] = p.name
    return wm


def _select_for_layer(
    surviving_per_layer: dict[int, list[int]] | None,
    fallback: list[int],
    layer: int | None,
) -> list[int]:
    if surviving_per_layer is None or layer is None:
        return fallback
    ids = surviving_per_layer.get(layer)
    if ids is None:
        return fallback
    return list(ids)


def _process_tensor(
    key: str,
    tensor: torch.Tensor,
    adapter: MoEAdapter,
    old_num_experts: int,
    target_num_experts: int,
    keep_shared_experts: bool,
    surviving_per_layer: dict[int, list[int]] | None = None,
    merge_plan: MergePlan | None = None,
) -> tuple[str, torch.Tensor] | None:
    info = adapter.parse_state_dict_key(key)
    fallback = surviving_experts_first_n(old_num_experts, target_num_experts)

    if info.type == "routed_expert" and info.expert_id is not None:
        keep = _select_for_layer(surviving_per_layer, fallback, info.layer)
        if info.expert_id not in keep:
            return None
        new_id = keep.index(info.expert_id)
        new_key = adapter.rename_expert_key(key, info.expert_id, new_id)
        return new_key, tensor

    if info.type == "routed_expert_stack":
        keep = _select_for_layer(surviving_per_layer, fallback, info.layer)
        if tensor.ndim < 1 or tensor.shape[0] != old_num_experts:
            return key, tensor
        keep_idx = torch.tensor(keep, dtype=torch.long)
        kept = tensor.index_select(0, keep_idx).contiguous()
        # Neuron-level (SwiGLU) merge: gate_up_proj is left untouched
        # (preserve the activation decision boundary); only down_proj
        # absorbs the pre-aggregated per-neuron contribution bucket.
        # Bucket may live in memory (``neuron_down_contrib``) or on disk
        # (``neuron_down_contrib_paths`` -> .pt file). Disk mode loads at
        # most one layer's bucket at a time and frees it before the next
        # tensor — required for big models (~1.5 GB/layer at Qwen3.5).
        is_neuron = (
            merge_plan is not None
            and merge_plan.mode == "neuron_swiglu"
            and info.layer is not None
        )
        if is_neuron and (
            info.layer in merge_plan.neuron_down_contrib
            or info.layer in merge_plan.neuron_down_contrib_paths
        ):
            if info.sub_key == "down_proj":
                if info.layer in merge_plan.neuron_down_contrib:
                    bucket = merge_plan.neuron_down_contrib[info.layer]
                else:
                    bucket = torch.load(
                        merge_plan.neuron_down_contrib_paths[info.layer],
                        map_location="cpu",
                        weights_only=True,
                    )
                if bucket.shape != kept.shape:
                    raise ValueError(
                        f"neuron_down_contrib shape {tuple(bucket.shape)} "
                        f"!= kept down_proj shape {tuple(kept.shape)} for layer {info.layer}"
                    )
                orig_dtype = kept.dtype
                kept = (
                    kept.to(torch.float32) + bucket.to(torch.float32)
                ).to(orig_dtype).contiguous()
                del bucket
            return key, kept
        # Macro merge: scaled add of full dropped expert tensors.
        if (
            merge_plan is not None
            and info.layer is not None
            and info.layer in merge_plan.weights
            and merge_plan.dropped_per_layer.get(info.layer)
            and merge_plan.alpha != 0.0
        ):
            dropped_ids = merge_plan.dropped_per_layer[info.layer]
            drop_idx = torch.tensor(dropped_ids, dtype=torch.long)
            dropped = tensor.index_select(0, drop_idx)             # [Nd, ...]
            w = merge_plan.weights[info.layer]                     # [Nd, Nk]
            assert w.shape[0] == dropped.shape[0], (
                f"merge_plan weights row count mismatch for layer {info.layer}"
            )
            assert w.shape[1] == kept.shape[0], (
                f"merge_plan weights col count mismatch for layer {info.layer}"
            )
            # Compute (alpha * w.T @ dropped) in fp32 for numerical safety,
            # then add back into kept in its native dtype.
            orig_dtype = kept.dtype
            d_flat = dropped.to(torch.float32).reshape(dropped.shape[0], -1)
            contrib = (w.to(torch.float32).t() @ d_flat).reshape(kept.shape[0], *kept.shape[1:])
            kept = (
                kept.to(torch.float32)
                + float(merge_plan.alpha) * contrib
            ).to(orig_dtype).contiguous()
        return key, kept

    if info.type == "router":
        keep = _select_for_layer(surviving_per_layer, fallback, info.layer)
        if tensor.ndim >= 1 and tensor.shape[0] == old_num_experts:
            idx = torch.tensor(keep, dtype=torch.long)
            return key, tensor.index_select(0, idx).contiguous()
        return key, tensor

    if info.type == "shared_expert" and not keep_shared_experts:
        return None
    return key, tensor


def _should_skip_without_load(
    info_type: str,
    expert_id: int | None,
    layer: int | None,
    fallback: list[int],
    surviving_per_layer: dict[int, list[int]] | None,
    keep_shared_experts: bool,
) -> bool:
    """True if we can decide to drop this key purely from its name."""
    if info_type == "routed_expert" and expert_id is not None:
        keep = _select_for_layer(surviving_per_layer, fallback, layer)
        return expert_id not in keep
    if info_type == "shared_expert" and not keep_shared_experts:
        return True
    return False


def build_expert_mapping_json(
    num_layers: int,
    old_num_experts: int,
    target_num_experts: int,
    surviving_per_layer: dict[int, list[int]] | None = None,
) -> dict[str, Any]:
    fallback = surviving_experts_first_n(old_num_experts, target_num_experts)
    out: dict[str, Any] = {}
    for layer in range(num_layers):
        if surviving_per_layer is not None and layer in surviving_per_layer:
            surv = list(surviving_per_layer[layer])
        else:
            surv = list(fallback)
        mapping = {str(old): new for new, old in enumerate(surv)}
        out[f"layer_{layer}"] = {
            "surviving_original_ids": surv,
            "mapping": mapping,
        }
    return out


def prune_state_dict_sharded(
    teacher_dir: Path,
    student_dir: Path,
    adapter: MoEAdapter,
    hf_config: dict[str, Any],
    target_num_experts: int,
    keep_shared_experts: bool,
    surviving_per_layer: dict[int, list[int]] | None = None,
    merge_plan: MergePlan | None = None,
) -> None:
    old_n = adapter.get_num_experts(hf_config)
    if target_num_experts > old_n:
        raise ValueError(f"target_num_experts {target_num_experts} > model num_experts {old_n}")
    if target_num_experts < 1:
        raise ValueError("target_num_experts must be >= 1")
    if surviving_per_layer is not None:
        for layer, ids in surviving_per_layer.items():
            if len(ids) != target_num_experts:
                raise ValueError(
                    f"surviving_per_layer[{layer}] has {len(ids)} ids; expected {target_num_experts}"
                )

    student_dir.mkdir(parents=True, exist_ok=True)
    weight_map = _build_weight_map(teacher_dir)

    keys_by_shard: dict[str, list[str]] = {}
    for k, sh in weight_map.items():
        keys_by_shard.setdefault(sh, []).append(k)
    shard_names = sorted(keys_by_shard.keys())

    fallback = surviving_experts_first_n(old_n, target_num_experts)
    staged_parts: list[tuple[Path, list[str]]] = []

    for shard_name in shard_names:
        shard_path = teacher_dir / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(shard_path)
        out_sd: dict[str, torch.Tensor] = {}
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for key in keys_by_shard[shard_name]:
                info = adapter.parse_state_dict_key(key)
                if _should_skip_without_load(
                    info.type,
                    info.expert_id,
                    info.layer,
                    fallback,
                    surviving_per_layer,
                    keep_shared_experts,
                ):
                    continue
                t = f.get_tensor(key)
                processed = _process_tensor(
                    key,
                    t,
                    adapter,
                    old_n,
                    target_num_experts,
                    keep_shared_experts,
                    surviving_per_layer=surviving_per_layer,
                    merge_plan=merge_plan,
                )
                del t
                if processed is None:
                    continue
                new_key, new_tensor = processed
                out_sd[new_key] = new_tensor.contiguous()
                del new_tensor

        if out_sd:
            part_idx = len(staged_parts) + 1
            tmp_path = student_dir / f"_part_{part_idx:05d}.safetensors"
            save_file(out_sd, str(tmp_path))
            staged_parts.append((tmp_path, list(out_sd.keys())))
        out_sd.clear()
        del out_sd
        gc.collect()

    total_parts = len(staged_parts)
    if total_parts == 0:
        raise RuntimeError("Pruning produced no tensors; check adapter / paths.")

    new_weight_map: dict[str, str] = {}
    written_files: list[Path] = []
    for idx, (tmp_path, keys) in enumerate(staged_parts, start=1):
        if total_parts == 1:
            new_name = "model.safetensors"
        else:
            new_name = f"model-{idx:05d}-of-{total_parts:05d}.safetensors"
        out_path = student_dir / new_name
        if out_path.exists():
            out_path.unlink()
        tmp_path.replace(out_path)
        written_files.append(out_path)
        for k in keys:
            new_weight_map[k] = new_name

    total_size = sum(p.stat().st_size for p in written_files)
    if total_parts > 1:
        index = {"metadata": {"total_size": total_size}, "weight_map": new_weight_map}
        (student_dir / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
