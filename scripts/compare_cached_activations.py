from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from moe_prune_distill.distill.teacher_cache import load_sample_cache


HIDDEN_KEY_HINTS = (
    "hidden",
    "hidden_state",
    "hidden_states",
    "activation",
    "activations",
    "teacher_hidden",
)

SKIP_KEY_HINTS = (
    "router",
    "router_logits",
    "logits",
    "input_ids",
    "attention_mask",
    "position_ids",
    "labels",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    seen = set()

    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def looks_like_hidden_key(key: str) -> bool:
    lk = key.lower()

    if any(x in lk for x in SKIP_KEY_HINTS):
        return False

    if any(x in lk for x in HIDDEN_KEY_HINTS):
        return True

    return False


def infer_layer_id_from_text(text: str) -> int | None:
    patterns = [
        r"cache[_\-]?layer[_\-]?(\d+)",
        r"layer[s]?[_\-.]?(\d+)",
        r"block[_\-.]?(\d+)",
        r"decoder[_\-.]?layer[_\-.]?(\d+)",
        r"^(\d+)$",
    ]

    for pat in patterns:
        m = re.search(pat, str(text), flags=re.IGNORECASE)
        if m:
            return int(m.group(1))

    return None


def infer_layer_id_from_path(path: tuple[Any, ...]) -> int | None:
    for x in reversed(path):
        layer = infer_layer_id_from_text(str(x))
        if layer is not None:
            return layer
    return None


def collect_sample_ids_from_index(index: Any) -> list[str]:
    """
    尽量兼容不同 cache_index.json schema。

    常见可能结构：

    1.
      {
        "samples": {
          "sample_id_1": {...},
          "sample_id_2": {...}
        }
      }

    2.
      {
        "sample_ids": ["sample_id_1", "sample_id_2"]
      }

    3.
      {
        "entries": [
          {"id": "..."},
          {"sample_id": "..."}
        ]
      }

    4.
      {
        "index": {
          "sample_id_1": {...}
        }
      }
    """
    sample_ids: set[str] = set()

    if isinstance(index, dict):
        for key in ("sample_ids", "samples_ids", "ids"):
            v = index.get(key)
            if isinstance(v, list):
                for x in v:
                    if isinstance(x, str | int):
                        sample_ids.add(str(x))

        for key in ("samples", "index", "sample_index", "cache"):
            v = index.get(key)
            if isinstance(v, dict):
                for sid in v.keys():
                    if isinstance(sid, str | int):
                        sample_ids.add(str(sid))

        for key in ("entries", "items", "records"):
            v = index.get(key)
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        for id_key in ("id", "sid", "sample_id", "sample"):
                            if id_key in item and isinstance(item[id_key], str | int):
                                sample_ids.add(str(item[id_key]))

    # 递归兜底：寻找像 sample id 的字段
    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                lk = str(k).lower()
                if lk in ("id", "sid", "sample_id", "sample"):
                    if isinstance(v, str | int):
                        sample_ids.add(str(v))
                walk(v)
        elif isinstance(obj, list):
            for x in obj:
                walk(x)

    walk(index)

    # 过滤明显不是 sample id 的东西
    bad_suffixes = (
        ".safetensors",
        ".json",
        ".pt",
        ".bin",
    )

    cleaned = []
    for sid in sample_ids:
        s = str(sid)
        if not s:
            continue
        if s.endswith(bad_suffixes):
            continue
        if "/" in s or "\\" in s:
            continue
        cleaned.append(s)

    return sorted(set(cleaned))


def call_load_sample_cache(cache_dir: Path, sample_id: str) -> Any:
    """
    兼容不同 load_sample_cache 签名。

    期望签名可能是：
      load_sample_cache(cache_dir, sample_id)
      load_sample_cache(str(cache_dir), sample_id)
      load_sample_cache(sample_id, cache_dir)
      load_sample_cache(cache_dir=..., sample_id=...)
      load_sample_cache(cache_dir=..., sid=...)
    """
    errors = []

    attempts = [
        lambda: load_sample_cache(cache_dir, sample_id),
        lambda: load_sample_cache(str(cache_dir), sample_id),
        lambda: load_sample_cache(sample_id, cache_dir),
        lambda: load_sample_cache(sample_id, str(cache_dir)),
        lambda: load_sample_cache(cache_dir=cache_dir, sample_id=sample_id),
        lambda: load_sample_cache(cache_dir=str(cache_dir), sample_id=sample_id),
        lambda: load_sample_cache(cache_dir=cache_dir, sid=sample_id),
        lambda: load_sample_cache(cache_dir=str(cache_dir), sid=sample_id),
    ]

    for fn in attempts:
        try:
            return fn()
        except TypeError as e:
            errors.append(str(e))

    sig = None
    try:
        sig = str(inspect.signature(load_sample_cache))
    except Exception:
        pass

    raise RuntimeError(
        f"Cannot call load_sample_cache for sample_id={sample_id}. "
        f"signature={sig}, errors={errors[:3]}"
    )


def is_candidate_hidden_tensor(path: tuple[Any, ...], x: torch.Tensor) -> bool:
    if x.ndim < 2:
        return False

    if x.shape[-1] < 16:
        return False

    joined = "/".join(str(p) for p in path)
    lower = joined.lower()

    if any(skip in lower for skip in SKIP_KEY_HINTS):
        return False

    # 如果名字里带 hidden，强烈认为是 hidden activation
    if any(h in lower for h in HIDDEN_KEY_HINTS):
        return True

    # 兜底：二维或三维且最后一维较大，也可能是 hidden
    if x.ndim in (2, 3):
        return True

    return False


def extract_layer_hidden_states(obj: Any) -> dict[int, torch.Tensor]:
    """
    从 load_sample_cache 返回对象中递归提取每层 hidden activation。

    支持返回结构例如：

      {
        0: tensor,
        1: tensor
      }

      {
        "0": {"hidden_states": tensor},
        "1": {"hidden_states": tensor}
      }

      {
        "layers": {
          "0": {"hidden": tensor}
        }
      }

      {
        "hidden_states": {
          "0": tensor
        }
      }
    """
    candidates: dict[int, list[tuple[int, tuple[Any, ...], torch.Tensor]]] = defaultdict(list)

    def score_candidate(path: tuple[Any, ...], x: torch.Tensor) -> int:
        joined = "/".join(str(p) for p in path).lower()
        score = 0

        if "hidden_states" in joined:
            score += 50
        elif "hidden_state" in joined:
            score += 45
        elif "hidden" in joined:
            score += 40
        elif "activation" in joined:
            score += 30

        if x.ndim == 2:
            score += 10
        elif x.ndim == 3:
            score += 5

        # router logits 通常最后一维是 num_experts，hidden 最后一维更大
        if x.shape[-1] >= 512:
            score += 10

        return score

    def walk(x: Any, path: tuple[Any, ...]) -> None:
        if isinstance(x, torch.Tensor):
            if not is_candidate_hidden_tensor(path, x):
                return

            layer = infer_layer_id_from_path(path)
            if layer is None:
                return

            candidates[layer].append((score_candidate(path, x), path, x))
            return

        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, path + (k,))
            return

        if isinstance(x, list | tuple):
            for i, v in enumerate(x):
                walk(v, path + (i,))
            return

    walk(obj, ())

    out: dict[int, torch.Tensor] = {}

    for layer, items in candidates.items():
        # 同一层如果找到多个 tensor，选最像 hidden 的
        items = sorted(items, key=lambda z: z[0], reverse=True)
        out[layer] = items[0][2]

    return out


def normalize_hidden(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    x = x.detach().cpu()

    # 单样本 cache 常见 shape:
    #   [1, seq, hidden]
    # 转成：
    #   [seq, hidden]
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]

    return x.to(dtype=dtype)


def finite_stats(x: torch.Tensor) -> dict[str, int]:
    return {
        "numel": int(x.numel()),
        "nan": int(torch.isnan(x).sum().item()),
        "posinf": int(torch.isposinf(x).sum().item()),
        "neginf": int(torch.isneginf(x).sum().item()),
    }


def safe_mean(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    return float(sum(xs) / len(xs))


def safe_std(xs: list[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    m = safe_mean(xs)
    return float(math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)))


def safe_min(xs: list[float]) -> float:
    return float(min(xs)) if xs else float("nan")


def safe_max(xs: list[float]) -> float:
    return float(max(xs)) if xs else float("nan")


def quantile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    idx = int(round(q * (len(ys) - 1)))
    idx = min(len(ys) - 1, max(0, idx))
    return float(ys[idx])


def compare_pair(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> dict[str, Any]:
    original_shape_a = list(a.shape)
    original_shape_b = list(b.shape)

    shape_mismatch = False

    if a.ndim != b.ndim:
        return {
            "shape_a": original_shape_a,
            "shape_b": original_shape_b,
            "shape_mismatch": True,
            "error": "rank mismatch",
        }

    if a.shape != b.shape:
        shape_mismatch = True
        slices = tuple(slice(0, min(sa, sb)) for sa, sb in zip(a.shape, b.shape))
        a = a[slices]
        b = b[slices]

    a = a.float()
    b = b.float()

    diff = b - a

    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    d_flat = diff.reshape(-1)

    a_norm = torch.linalg.vector_norm(a_flat).item()
    b_norm = torch.linalg.vector_norm(b_flat).item()
    delta_norm = torch.linalg.vector_norm(d_flat).item()

    cosine = torch.nn.functional.cosine_similarity(
        a_flat.unsqueeze(0),
        b_flat.unsqueeze(0),
        dim=-1,
        eps=eps,
    ).item()

    mse = torch.mean(d_flat * d_flat).item()
    mae = torch.mean(torch.abs(d_flat)).item()
    max_abs_diff = torch.max(torch.abs(d_flat)).item()

    token_delta_norms: list[float] = []
    token_relative_delta_norms: list[float] = []
    token_cosines: list[float] = []

    if a.ndim >= 2:
        a_tok = a.reshape(-1, a.shape[-1])
        b_tok = b.reshape(-1, b.shape[-1])
        d_tok = b_tok - a_tok

        tok_a_norm = torch.linalg.vector_norm(a_tok, dim=-1)
        tok_d_norm = torch.linalg.vector_norm(d_tok, dim=-1)
        tok_cos = torch.nn.functional.cosine_similarity(
            a_tok,
            b_tok,
            dim=-1,
            eps=eps,
        )

        token_delta_norms = tok_d_norm.tolist()
        token_relative_delta_norms = (
            tok_d_norm / torch.clamp(tok_a_norm, min=eps)
        ).tolist()
        token_cosines = tok_cos.tolist()

    return {
        "shape_a": original_shape_a,
        "shape_b": original_shape_b,
        "used_shape": list(a.shape),
        "shape_mismatch": shape_mismatch,

        "a_norm": float(a_norm),
        "b_norm": float(b_norm),
        "delta_norm": float(delta_norm),
        "relative_delta_vs_a": float(delta_norm / max(a_norm, eps)),
        "relative_delta_vs_b": float(delta_norm / max(b_norm, eps)),
        "cosine_similarity": float(cosine),
        "mse": float(mse),
        "mae": float(mae),
        "max_abs_diff": float(max_abs_diff),

        "a_finite": finite_stats(a),
        "b_finite": finite_stats(b),
        "diff_finite": finite_stats(diff),

        "token_delta_norm_mean": safe_mean(token_delta_norms),
        "token_delta_norm_p50": quantile(token_delta_norms, 0.50),
        "token_delta_norm_p90": quantile(token_delta_norms, 0.90),
        "token_delta_norm_p95": quantile(token_delta_norms, 0.95),
        "token_delta_norm_p99": quantile(token_delta_norms, 0.99),
        "token_delta_norm_max": safe_max(token_delta_norms),

        "token_relative_delta_mean": safe_mean(token_relative_delta_norms),
        "token_relative_delta_p50": quantile(token_relative_delta_norms, 0.50),
        "token_relative_delta_p90": quantile(token_relative_delta_norms, 0.90),
        "token_relative_delta_p95": quantile(token_relative_delta_norms, 0.95),
        "token_relative_delta_p99": quantile(token_relative_delta_norms, 0.99),
        "token_relative_delta_max": safe_max(token_relative_delta_norms),

        "token_cosine_mean": safe_mean(token_cosines),
        "token_cosine_p50": quantile(token_cosines, 0.50),
        "token_cosine_p10": quantile(token_cosines, 0.10),
        "token_cosine_p05": quantile(token_cosines, 0.05),
        "token_cosine_min": safe_min(token_cosines),
    }


def aggregate_metric(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    vals: list[float] = []

    for row in rows:
        v = row.get(key)
        if isinstance(v, int | float):
            v = float(v)
            if math.isfinite(v):
                vals.append(v)

    return {
        f"{key}_mean": safe_mean(vals),
        f"{key}_std": safe_std(vals),
        f"{key}_min": safe_min(vals),
        f"{key}_p50": quantile(vals, 0.50),
        f"{key}_p90": quantile(vals, 0.90),
        f"{key}_p95": quantile(vals, 0.95),
        f"{key}_p99": quantile(vals, 0.99),
        f"{key}_max": safe_max(vals),
    }


def format_float(x: Any, ndigits: int = 6) -> str:
    if not isinstance(x, int | float):
        return str(x)
    x = float(x)
    if math.isnan(x):
        return "nan"
    return f"{x:.{ndigits}g}"


def write_markdown_report(
    path: Path,
    cache_dir: Path,
    sample_ids: list[str],
    cached_layers: list[int],
    layer_summary: list[dict[str, Any]],
    sample_details: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    lines.append("# Cached Activation Layer Drift Report")
    lines.append("")
    lines.append(f"- Cache dir: `{cache_dir}`")
    lines.append(f"- Number of discovered samples: `{len(sample_ids)}`")
    lines.append(f"- Cached layers: `{cached_layers}`")
    lines.append(f"- Number of compared layer pairs: `{len(layer_summary)}`")
    lines.append(f"- Number of sample-pair comparisons: `{len(sample_details)}`")
    lines.append("")

    lines.append("## 1. Layer-pair Summary")
    lines.append("")
    lines.append(
        "| Layer Pair | Samples | Errors | Shape Mismatch | Cosine Mean | Rel Delta Mean | MSE Mean | MAE Mean | Max Abs Mean | Token Rel Delta P95 Mean | Token Cosine Mean |"
    )
    lines.append(
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )

    for r in layer_summary:
        lines.append(
            "| "
            f"{r['from_layer']} -> {r['to_layer']} | "
            f"{r['num_samples']} | "
            f"{r['num_errors']} | "
            f"{r['num_shape_mismatch']} | "
            f"{format_float(r.get('cosine_similarity_mean'))} | "
            f"{format_float(r.get('relative_delta_vs_a_mean'))} | "
            f"{format_float(r.get('mse_mean'))} | "
            f"{format_float(r.get('mae_mean'))} | "
            f"{format_float(r.get('max_abs_diff_mean'))} | "
            f"{format_float(r.get('token_relative_delta_p95_mean'))} | "
            f"{format_float(r.get('token_cosine_mean_mean'))} |"
        )

    lines.append("")
    lines.append("## 2. Potential Anomalies")
    lines.append("")

    anomalies: list[str] = []

    for r in layer_summary:
        pair = f"{r['from_layer']} -> {r['to_layer']}"

        cos = r.get("cosine_similarity_mean")
        rel = r.get("relative_delta_vs_a_mean")
        mse = r.get("mse_mean")

        if isinstance(cos, int | float) and math.isfinite(float(cos)) and float(cos) < 0.5:
            anomalies.append(f"- Low cosine similarity at layer `{pair}`: `{cos:.6g}`")

        if isinstance(rel, int | float) and math.isfinite(float(rel)) and float(rel) > 1.0:
            anomalies.append(f"- High relative delta at layer `{pair}`: `{rel:.6g}`")

        if isinstance(mse, int | float) and math.isfinite(float(mse)) and float(mse) > 1.0:
            anomalies.append(f"- High MSE at layer `{pair}`: `{mse:.6g}`")

        if r.get("num_shape_mismatch", 0) > 0:
            anomalies.append(
                f"- Shape mismatch found at layer `{pair}`: `{r.get('num_shape_mismatch')}` samples"
            )

        if r.get("num_errors", 0) > 0:
            anomalies.append(
                f"- Errors found at layer `{pair}`: `{r.get('num_errors')}` samples"
            )

    for row in sample_details:
        for side in ("a_finite", "b_finite", "diff_finite"):
            fs = row.get(side)
            if not isinstance(fs, dict):
                continue

            if fs.get("nan", 0) or fs.get("posinf", 0) or fs.get("neginf", 0):
                anomalies.append(
                    f"- Non-finite values: sample `{row.get('sample_id')}`, "
                    f"layer `{row.get('from_layer')} -> {row.get('to_layer')}`, "
                    f"{side}={fs}"
                )

    if anomalies:
        lines.extend(anomalies[:300])
        if len(anomalies) > 300:
            lines.append(f"- ... truncated, total anomalies: `{len(anomalies)}`")
    else:
        lines.append("No obvious anomaly detected by default thresholds.")

    lines.append("")
    lines.append("## 3. Metric Meaning")
    lines.append("")
    lines.append("- `cosine_similarity`: 两层 hidden state 展平后的方向相似度，越接近 1 越相似。")
    lines.append("- `relative_delta_vs_a`: `||hidden_next - hidden_prev|| / ||hidden_prev||`。")
    lines.append("- `mse`: 两层激活逐元素均方误差。")
    lines.append("- `mae`: 两层激活逐元素平均绝对误差。")
    lines.append("- `max_abs_diff`: 最大逐元素绝对差。")
    lines.append("- `token_relative_delta_p95`: token 级 relative delta 的 95 分位数。")
    lines.append("- `token_cosine_mean`: token 级平均 cosine similarity。")
    lines.append("")

    lines.append("## 4. Notes")
    lines.append("")
    lines.append("- 这个版本通过项目自带的 `load_sample_cache` 读取缓存，因此应该能正确处理 v2 batched layout。")
    lines.append("- 如果配置只缓存了部分层，那么这里比较的是相邻缓存层，不一定是相邻 Transformer block。")
    lines.append("- 如果仍然只能发现很少样本，请检查 `cache_index.json` 里的 sample id schema，或者用 `--sample-ids-file` 显式传入。")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compare cached hidden activations between teacher cache layers via load_sample_cache."
    )

    p.add_argument("--cache-dir", type=str, required=True)
    p.add_argument("--out-dir", type=str, default="./outputs/cache_activation_report")
    p.add_argument("--sample-limit", type=int, default=None)
    p.add_argument(
        "--sample-ids-file",
        type=str,
        default=None,
        help="Optional txt/json file containing sample ids. One id per line for txt, or a JSON list.",
    )
    p.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Optional comma-separated cached layers, e.g. 0,1,2,3.",
    )
    p.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float64"],
        default="float32",
    )
    p.add_argument(
        "--debug-first",
        action="store_true",
        help="Print first loaded sample cache structure and exit.",
    )

    args = p.parse_args()

    cache_dir = Path(args.cache_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    if args.sample_ids_file:
        sample_ids_path = Path(args.sample_ids_file)
        if sample_ids_path.suffix.lower() == ".json":
            sample_ids = [str(x) for x in read_json(sample_ids_path)]
        else:
            sample_ids = [
                line.strip()
                for line in sample_ids_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
    else:
        index_path = cache_dir / "cache_index.json"
        if not index_path.is_file():
            raise SystemExit(f"cache_index.json not found: {index_path}")

        index = read_json(index_path)
        sample_ids = collect_sample_ids_from_index(index)

    sample_ids = sorted(set(sample_ids))

    if args.sample_limit is not None:
        sample_ids = sample_ids[: args.sample_limit]

    if not sample_ids:
        raise SystemExit(
            "No sample ids found. "
            "Please provide --sample-ids-file, or paste cache_index.json structure for parser adjustment."
        )

    print(f"[INFO] cache_dir={cache_dir}")
    print(f"[INFO] discovered sample ids: {len(sample_ids)}")

    include_layers: set[int] | None = None
    if args.layers:
        include_layers = {int(x.strip()) for x in args.layers.split(",") if x.strip()}

    # debug first sample
    if args.debug_first:
        sid = sample_ids[0]
        obj = call_load_sample_cache(cache_dir, sid)

        print(f"[DEBUG] first sample id: {sid}")
        print(f"[DEBUG] returned type: {type(obj)}")

        def print_structure(x: Any, indent: int = 0, max_depth: int = 5) -> None:
            prefix = "  " * indent

            if indent > max_depth:
                print(prefix + "...")
                return

            if isinstance(x, torch.Tensor):
                print(prefix + f"Tensor shape={tuple(x.shape)} dtype={x.dtype}")
            elif isinstance(x, dict):
                print(prefix + f"dict keys={list(x.keys())[:20]}")
                for k, v in list(x.items())[:20]:
                    print(prefix + f"- key={k!r}:")
                    print_structure(v, indent + 1, max_depth)
            elif isinstance(x, list | tuple):
                print(prefix + f"{type(x).__name__} len={len(x)}")
                for i, v in enumerate(list(x)[:10]):
                    print(prefix + f"- idx={i}:")
                    print_structure(v, indent + 1, max_depth)
            else:
                print(prefix + f"{type(x).__name__}: {repr(x)[:200]}")

        print_structure(obj)
        hidden = extract_layer_hidden_states(obj)
        print(f"[DEBUG] extracted layers: {sorted(hidden.keys())}")
        for layer, t in sorted(hidden.items()):
            print(f"[DEBUG] layer {layer}: shape={tuple(t.shape)}, dtype={t.dtype}")
        return

    # 第一遍：确认所有可用层
    layers_per_sample: dict[str, list[int]] = {}
    cached_layers_set: set[int] = set()
    failed_loads: dict[str, str] = {}

    print("[INFO] scanning samples through load_sample_cache...")

    for i, sid in enumerate(sample_ids):
        if i % 100 == 0:
            print(f"[INFO] scanning {i}/{len(sample_ids)}")

        try:
            obj = call_load_sample_cache(cache_dir, sid)
            hiddens = extract_layer_hidden_states(obj)
        except Exception as e:
            failed_loads[sid] = str(e)
            continue

        layers = sorted(hiddens.keys())

        if include_layers is not None:
            layers = [x for x in layers if x in include_layers]

        layers_per_sample[sid] = layers
        cached_layers_set.update(layers)

    cached_layers = sorted(cached_layers_set)

    if len(cached_layers) < 2:
        raise SystemExit(
            f"Need at least 2 cached layers, found {cached_layers}. "
            f"Failed loads: {len(failed_loads)}. "
            f"Try --debug-first to inspect load_sample_cache output."
        )

    print(f"[INFO] cached layers discovered: {cached_layers}")
    print(f"[INFO] failed loads: {len(failed_loads)}")

    layer_pairs = list(zip(cached_layers[:-1], cached_layers[1:]))

    metric_keys = [
        "a_norm",
        "b_norm",
        "delta_norm",
        "relative_delta_vs_a",
        "relative_delta_vs_b",
        "cosine_similarity",
        "mse",
        "mae",
        "max_abs_diff",
        "token_delta_norm_mean",
        "token_delta_norm_p50",
        "token_delta_norm_p90",
        "token_delta_norm_p95",
        "token_delta_norm_p99",
        "token_delta_norm_max",
        "token_relative_delta_mean",
        "token_relative_delta_p50",
        "token_relative_delta_p90",
        "token_relative_delta_p95",
        "token_relative_delta_p99",
        "token_relative_delta_max",
        "token_cosine_mean",
        "token_cosine_p50",
        "token_cosine_p10",
        "token_cosine_p05",
        "token_cosine_min",
    ]

    sample_details: list[dict[str, Any]] = []
    rows_by_pair: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)

    print("[INFO] comparing activations...")

    for i, sid in enumerate(sample_ids):
        if i % 50 == 0:
            print(f"[INFO] comparing sample {i}/{len(sample_ids)}")

        try:
            obj = call_load_sample_cache(cache_dir, sid)
            hiddens = extract_layer_hidden_states(obj)
        except Exception as e:
            row = {
                "sample_id": sid,
                "error": f"load_sample_cache failed: {e}",
            }
            sample_details.append(row)
            continue

        if include_layers is not None:
            hiddens = {
                layer: value
                for layer, value in hiddens.items()
                if layer in include_layers
            }

        for from_layer, to_layer in layer_pairs:
            if from_layer not in hiddens or to_layer not in hiddens:
                continue

            try:
                a = normalize_hidden(hiddens[from_layer], dtype=dtype)
                b = normalize_hidden(hiddens[to_layer], dtype=dtype)
                metrics = compare_pair(a, b)
            except Exception as e:
                metrics = {
                    "error": str(e),
                }

            row = {
                "sample_id": sid,
                "from_layer": from_layer,
                "to_layer": to_layer,
                **metrics,
            }

            sample_details.append(row)
            rows_by_pair[(from_layer, to_layer)].append(row)

    layer_summary: list[dict[str, Any]] = []

    for from_layer, to_layer in layer_pairs:
        rows = rows_by_pair[(from_layer, to_layer)]
        valid_rows = [r for r in rows if "error" not in r]

        summary: dict[str, Any] = {
            "from_layer": from_layer,
            "to_layer": to_layer,
            "num_samples": len(rows),
            "num_errors": sum(1 for r in rows if "error" in r),
            "num_shape_mismatch": sum(1 for r in rows if r.get("shape_mismatch")),
        }

        for key in metric_keys:
            summary.update(aggregate_metric(valid_rows, key))

        layer_summary.append(summary)

    report = {
        "cache_dir": str(cache_dir),
        "num_sample_ids": len(sample_ids),
        "sample_ids_preview": sample_ids[:20],
        "cached_layers": cached_layers,
        "failed_loads": failed_loads,
        "layer_summary": layer_summary,
        "sample_details": sample_details,
    }

    write_json(out_dir / "activation_compare_report.json", report)
    write_csv(out_dir / "activation_layer_summary.csv", layer_summary)
    write_csv(out_dir / "activation_sample_details.csv", sample_details)

    write_markdown_report(
        out_dir / "activation_compare_report.md",
        cache_dir=cache_dir,
        sample_ids=sample_ids,
        cached_layers=cached_layers,
        layer_summary=layer_summary,
        sample_details=sample_details,
    )

    print("")
    print("[DONE] wrote reports:")
    print(f"  - {out_dir / 'activation_compare_report.md'}")
    print(f"  - {out_dir / 'activation_compare_report.json'}")
    print(f"  - {out_dir / 'activation_layer_summary.csv'}")
    print(f"  - {out_dir / 'activation_sample_details.csv'}")


if __name__ == "__main__":
    main()
