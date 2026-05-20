"""Step 5 (optional): export a trained safetensor model to quantized GGUF.

Pipeline (CPU-only, no VRAM required):

  1. Strip vision encoder + (default) MTP layers from ``input_dir`` into a
     temporary HF text-only directory under ``work_dir``.
  2. Run llama.cpp's ``convert_hf_to_gguf.py`` on that dir to produce a single
     BF16 GGUF.
  3. For each requested quant type, run ``llama-quantize`` to produce the
     final GGUF in ``output_dir``.
  4. Smoke-test the outputs via gguf-py and write ``export_report.json``.

External requirements (the script validates and refuses to bootstrap):

  - ``llama_cpp_src_dir``  contains ``convert_hf_to_gguf.py``
                           (clone https://github.com/ggml-org/llama.cpp)
  - ``llama_cpp_dir``      contains ``llama-quantize`` / ``llama-quantize.exe``
                           (download a CPU release zip from
                           https://github.com/ggml-org/llama.cpp/releases)

Both can point to the same directory if the user clones source AND drops the
prebuilt binaries inside it.

Notes:
  * llama.cpp PR #19435 added ``qwen3_5_moe`` text-only support; vision encoder
    must be stripped first (this script does that).
  * Issue #23033 reports a missing ``blk.40.ssm_conv1d.weight`` when MTP layers
    are present. ``drop_mtp=True`` avoids it; toggle off only with a llama.cpp
    build that contains the fix.
  * PR #23305 fixes ``in_proj_qkv`` quantization for the SSM tensors; if
    quantize stderr mentions that key, point the user there.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from moe_prune_distill.config import (  # type: ignore[import-not-found]
    _ALLOWED_GGUF_QUANTS,
    _DEFAULT_GGUF_QUANTS,
    ExportGGUFConfig,
    load_config,
)
from moe_prune_distill.utils.logging import get_logger

# llama.cpp PR #19435 expects "Qwen3_5MoeForCausalLM" for the text-only path.
# If a future llama.cpp main rename surfaces (e.g. "Qwen35MoeForCausalLM"),
# only this constant needs adjusting -- the work_dir is reusable.
_TEXT_ARCH_NAME = "Qwen3_5MoeForCausalLM"
_TEXT_MODEL_TYPE = "qwen3_5_moe_text"  # matches inner text_config.model_type

# Tensors to drop from the safetensor shards.
_VISION_KEY_PREFIXES: tuple[str, ...] = (
    "model.visual.",
    "model.vision_",
    "visual.",
)
_MTP_KEY_PREFIXES: tuple[str, ...] = ("mtp.",)

# Sidecar files to copy into the text-only dir. Mirrors scripts/prune.py
# minus the vision-only preprocessor configs.
_TOKENIZER_FILES: tuple[str, ...] = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "chat_template.jinja",
    "generation_config.json",
)
_VISION_ONLY_SIDECARS: tuple[str, ...] = (
    "preprocessor_config.json",
    "video_preprocessor_config.json",
)

# Coarse fraction-of-BF16 estimates for disk preflight. Numbers come from
# llama.cpp's own quant size tables; intentionally conservative.
_QUANT_SIZE_FRAC: dict[str, float] = {
    "BF16": 1.00,
    "F16": 1.00,
    "Q8_0": 0.53,
    "Q6_K": 0.41,
    "Q5_K_M": 0.31,
    "Q5_K_S": 0.30,
    "Q4_K_M": 0.27,
    "Q4_K_S": 0.25,
    "Q3_K_L": 0.22,
    "Q3_K_M": 0.20,
}

# Vision projector: gguf-py advertises Qwen3-VL family with this substring in
# the ``general.architecture`` (or per-arch) field; smoke test uses it as a hint.
_MMPROJ_ARCH_HINT = "qwen3vl"
_MMPROJ_OUTTYPES: tuple[str, ...] = ("bf16", "f16", "f32")
# BF16/F16 mmproj GGUFs for Qwen3.6 35B-A3B run ~10 GiB on disk; F32 doubles it.
_MMPROJ_SIZE_GIB: dict[str, float] = {"bf16": 11.0, "f16": 11.0, "f32": 22.0}


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def validate_llama_cpp(
    llama_cpp_dir: Path | None,
    src_dir: Path | None,
) -> tuple[Path, Path]:
    """Locate ``convert_hf_to_gguf.py`` and ``llama-quantize`` binary.

    Both inputs are optional and may point at the same directory. Raises with
    a clear remediation message if either piece is missing.
    """
    if llama_cpp_dir is None and src_dir is None:
        raise FileNotFoundError(
            "export_gguf.llama_cpp_dir / llama_cpp_src_dir is unset. "
            "Set at least one of them in your config or pass --llama_cpp_dir / "
            "--llama_cpp_src_dir on the CLI. See the script docstring for setup."
        )

    candidates_src = [d for d in (src_dir, llama_cpp_dir) if d is not None]
    convert_script: Path | None = None
    for d in candidates_src:
        for name in ("convert_hf_to_gguf.py", "convert-hf-to-gguf.py"):
            p = d / name
            if p.is_file():
                convert_script = p
                break
        if convert_script is not None:
            break
    if convert_script is None:
        raise FileNotFoundError(
            "convert_hf_to_gguf.py not found under "
            f"{[str(d) for d in candidates_src]}. Clone llama.cpp source: "
            "git clone --depth 1 https://github.com/ggml-org/llama.cpp <dir>"
        )

    candidates_bin = [d for d in (llama_cpp_dir, src_dir) if d is not None]
    quantize_bin: Path | None = None
    quant_names = (
        "llama-quantize.exe",
        "llama-quantize",
        "quantize.exe",
        "quantize",
    )
    for d in candidates_bin:
        for name in quant_names:
            for sub in ("", "build/bin", "bin"):
                p = (d / sub / name) if sub else (d / name)
                if p.is_file():
                    quantize_bin = p
                    break
            if quantize_bin is not None:
                break
        if quantize_bin is not None:
            break
    if quantize_bin is None:
        raise FileNotFoundError(
            "llama-quantize binary not found under "
            f"{[str(d) for d in candidates_bin]}. Download a CPU release zip "
            "from https://github.com/ggml-org/llama.cpp/releases (file name "
            "like llama-bXXXX-bin-win-cpu-x64.zip) and extract it next to the "
            "convert script, or build llama.cpp with cmake."
        )
    return convert_script, quantize_bin


# --------------------------------------------------------------------------
# Pure helpers (unit-testable, no I/O)
# --------------------------------------------------------------------------

def make_drop_predicate(drop_mtp: bool) -> Callable[[str], bool]:
    """Return a predicate that returns True if a tensor name should be dropped."""

    prefixes: tuple[str, ...] = _VISION_KEY_PREFIXES
    if drop_mtp:
        prefixes = prefixes + _MTP_KEY_PREFIXES

    def _pred(key: str) -> bool:
        return any(key.startswith(p) for p in prefixes)

    return _pred


def filter_config_json(src_cfg: dict[str, Any], drop_mtp: bool) -> dict[str, Any]:
    """Strip multimodal + (optionally) MTP fields and rewrite arch/model_type.

    Operates on a deep copy so the caller's dict is untouched.
    """
    cfg = deepcopy(src_cfg)
    for key in (
        "vision_config",
        "image_token_id",
        "video_token_id",
        "vision_start_token_id",
        "vision_end_token_id",
    ):
        cfg.pop(key, None)

    cfg["architectures"] = [_TEXT_ARCH_NAME]
    cfg["model_type"] = _TEXT_MODEL_TYPE

    text = cfg.get("text_config")
    if isinstance(text, dict) and drop_mtp:
        for k in list(text.keys()):
            if k.startswith("mtp_"):
                text.pop(k, None)
    return cfg


# --------------------------------------------------------------------------
# Shard rewrite (streaming, mirrors moe_prune_distill/prune/slicer.py:281-323)
# --------------------------------------------------------------------------

def _build_weight_map(input_dir: Path) -> dict[str, str]:
    idx_path = input_dir / "model.safetensors.index.json"
    if idx_path.is_file():
        idx = json.loads(idx_path.read_text(encoding="utf-8"))
        wm = idx.get("weight_map")
        if not isinstance(wm, dict) or not wm:
            raise ValueError(f"{idx_path} has no weight_map")
        return {str(k): str(v) for k, v in wm.items()}
    # Fallback: single-file safetensors model.
    single = input_dir / "model.safetensors"
    if not single.is_file():
        raise FileNotFoundError(
            f"Neither model.safetensors.index.json nor model.safetensors under {input_dir}"
        )
    with safe_open(str(single), framework="pt", device="cpu") as f:
        return {k: single.name for k in f.keys()}


def _filter_shards_streaming(
    input_dir: Path,
    out_dir: Path,
    drop_pred: Callable[[str], bool],
    log,
) -> tuple[int, int, int]:
    """Stream-rewrite each shard, dropping keys that match drop_pred.

    Returns (kept_count, dropped_count, output_shard_count).
    """
    weight_map = _build_weight_map(input_dir)
    keys_by_shard: dict[str, list[str]] = {}
    for k, sh in weight_map.items():
        keys_by_shard.setdefault(sh, []).append(k)
    shard_names = sorted(keys_by_shard.keys())

    out_dir.mkdir(parents=True, exist_ok=True)
    staged: list[tuple[Path, list[str]]] = []
    kept_total = 0
    dropped_total = 0

    for i, shard_name in enumerate(shard_names, start=1):
        shard_path = input_dir / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(shard_path)
        out_sd: dict[str, torch.Tensor] = {}
        kept = 0
        dropped = 0
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for key in keys_by_shard[shard_name]:
                if drop_pred(key):
                    dropped += 1
                    continue
                t = f.get_tensor(key)
                out_sd[key] = t.contiguous()
                kept += 1
        log.info(
            "[strip] shard %d/%d (%s): %d kept, %d dropped",
            i,
            len(shard_names),
            shard_name,
            kept,
            dropped,
        )
        kept_total += kept
        dropped_total += dropped
        if out_sd:
            tmp_path = out_dir / f"_part_{len(staged) + 1:05d}.safetensors"
            save_file(out_sd, str(tmp_path))
            staged.append((tmp_path, list(out_sd.keys())))
        out_sd.clear()
        del out_sd
        gc.collect()

    total_parts = len(staged)
    if total_parts == 0:
        raise RuntimeError("strip produced no tensors; check input dir / drop predicate")

    new_weight_map: dict[str, str] = {}
    total_size = 0
    for idx, (tmp, keys) in enumerate(staged, start=1):
        final_name = f"model-{idx:05d}-of-{total_parts:05d}.safetensors"
        final_path = out_dir / final_name
        if final_path.exists():
            final_path.unlink()
        os.replace(tmp, final_path)
        for k in keys:
            new_weight_map[k] = final_name
        total_size += final_path.stat().st_size

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": new_weight_map,
    }
    (out_dir / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True), encoding="utf-8"
    )
    return kept_total, dropped_total, total_parts


def _copy_tokenizer_assets(src: Path, dst: Path) -> int:
    n = 0
    for name in _TOKENIZER_FILES:
        sp = src / name
        if sp.is_file():
            shutil.copy2(sp, dst / name)
            n += 1
    return n


# --------------------------------------------------------------------------
# Top-level orchestration
# --------------------------------------------------------------------------

def prepare_text_only_dir(
    input_dir: Path,
    work_dir: Path,
    drop_mtp: bool,
    log,
) -> Path:
    """Materialize a vision-stripped HF dir at ``work_dir`` and return it."""
    work_dir.mkdir(parents=True, exist_ok=True)

    src_cfg_path = input_dir / "config.json"
    if not src_cfg_path.is_file():
        raise FileNotFoundError(src_cfg_path)
    src_cfg = json.loads(src_cfg_path.read_text(encoding="utf-8"))
    new_cfg = filter_config_json(src_cfg, drop_mtp=drop_mtp)
    (work_dir / "config.json").write_text(
        json.dumps(new_cfg, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info(
        "[strip] wrote config.json (architectures=%s model_type=%s drop_mtp=%s)",
        new_cfg["architectures"],
        new_cfg["model_type"],
        drop_mtp,
    )

    n_assets = _copy_tokenizer_assets(input_dir, work_dir)
    log.info("[strip] copied %d tokenizer/sidecar files", n_assets)
    skipped = [n for n in _VISION_ONLY_SIDECARS if (input_dir / n).is_file()]
    if skipped:
        log.info("[strip] skipped vision-only sidecars: %s", skipped)

    drop_pred = make_drop_predicate(drop_mtp)
    kept, dropped, n_shards = _filter_shards_streaming(input_dir, work_dir, drop_pred, log)
    log.info(
        "[strip] done. %d tensors kept, %d dropped, %d shards written",
        kept, dropped, n_shards,
    )

    if not drop_mtp:
        log.warning(
            "MTP layers retained. llama.cpp issue #23033 reports a missing "
            "blk.N.ssm_conv1d.weight in this case; pin a build that includes "
            "the fix or set drop_mtp: true.",
        )
    return work_dir


def preflight_disk(
    output_dir: Path,
    input_size_bytes: int,
    quants: list[str],
    keep_bf16: bool,
    mmproj_outtype: str | None = None,
) -> None:
    """Estimate required free space and abort early if disk is too tight.

    BF16 GGUF must exist on disk during quantize regardless of keep_bf16; we
    only delete it after all quants succeed. ``mmproj_outtype`` adds the
    standalone vision projector budget when not None.
    """
    bf16_est = int(input_size_bytes * 1.05)  # GGUF ≈ raw bf16 weights + header
    quant_est = sum(int(bf16_est * _QUANT_SIZE_FRAC.get(q, 0.55)) for q in quants)
    mmproj_est = 0
    if mmproj_outtype is not None:
        mmproj_est = int(_MMPROJ_SIZE_GIB[mmproj_outtype] * (1 << 30))
    needed = quant_est + bf16_est + mmproj_est
    free = shutil.disk_usage(output_dir).free
    gib = 1 << 30
    if free < needed:
        raise RuntimeError(
            f"Insufficient free space in {output_dir}: have "
            f"{free / gib:.1f} GiB, need ~{needed / gib:.1f} GiB "
            f"(BF16 GGUF {bf16_est / gib:.1f} + {len(quants)} quants "
            f"{quant_est / gib:.1f} + mmproj {mmproj_est / gib:.1f}). "
            f"Free up space or move output_dir."
        )
    _ = keep_bf16  # currently informational; final size after cleanup is smaller


def run_convert(
    convert_script: Path,
    text_only_dir: Path,
    out_bf16: Path,
    log,
) -> None:
    cmd = [
        sys.executable,
        str(convert_script),
        str(text_only_dir),
        "--outfile",
        str(out_bf16),
        "--outtype",
        "bf16",
    ]
    log.info("[convert] %s", " ".join(cmd))
    rc = subprocess.run(cmd, check=False)
    if rc.returncode != 0:
        raise RuntimeError(
            f"convert_hf_to_gguf.py failed (exit={rc.returncode}). If the error "
            f"mentions an unknown architecture, adjust _TEXT_ARCH_NAME in "
            f"scripts/export_gguf.py and re-run (the work_dir is reusable)."
        )
    if not out_bf16.is_file():
        raise RuntimeError(f"convert reported success but {out_bf16} is missing")
    log.info("[convert] wrote %s (%.2f GiB)", out_bf16, out_bf16.stat().st_size / (1 << 30))


def run_convert_mmproj(
    convert_script: Path,
    input_dir: Path,
    out_mmproj: Path,
    outtype: str,
    log,
) -> None:
    """Run convert_hf_to_gguf.py --mmproj against the ORIGINAL input dir.

    The mmproj path requires ``vision_config`` and ``model.visual.*`` tensors,
    so callers must pass the original (un-stripped) input dir, not the
    text-only ``work_dir``.
    """
    cmd = [
        sys.executable,
        str(convert_script),
        str(input_dir),
        "--mmproj",
        "--outfile",
        str(out_mmproj),
        "--outtype",
        outtype,
    ]
    log.info("[mmproj] %s", " ".join(cmd))
    rc = subprocess.run(cmd, check=False)
    if rc.returncode != 0:
        raise RuntimeError(
            f"convert_hf_to_gguf.py --mmproj failed (exit={rc.returncode}). "
            f"Check that {input_dir}/config.json still has vision_config and "
            f"that model.visual.* tensors are present (do NOT pass the "
            f"vision-stripped work_dir here)."
        )
    if not out_mmproj.is_file():
        raise RuntimeError(f"mmproj convert reported success but {out_mmproj} is missing")
    log.info(
        "[mmproj] wrote %s (%.2f GiB)",
        out_mmproj,
        out_mmproj.stat().st_size / (1 << 30),
    )


def run_quantize(
    quantize_bin: Path,
    in_gguf: Path,
    out_gguf: Path,
    qtype: str,
    log,
) -> None:
    cmd = [str(quantize_bin), str(in_gguf), str(out_gguf), qtype]
    log.info("[quantize] %s", " ".join(cmd))
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, encoding="utf-8")
    if proc.stdout:
        for line in proc.stdout.splitlines():
            log.info("[quantize:%s] %s", qtype, line)
    if proc.returncode != 0:
        if proc.stderr:
            for line in proc.stderr.splitlines():
                log.error("[quantize:%s] %s", qtype, line)
        if proc.stderr and ("in_proj_qkv" in proc.stderr or "unsupported tensor" in proc.stderr):
            log.error(
                "Hint: this looks like the SSM in_proj_qkv quant issue tracked in "
                "https://github.com/ggml-org/llama.cpp/pull/23305 -- pin a llama.cpp "
                "build that includes that fix."
            )
        raise RuntimeError(f"llama-quantize failed (exit={proc.returncode}, qtype={qtype})")
    if not out_gguf.is_file():
        raise RuntimeError(f"quantize reported success but {out_gguf} is missing")
    log.info(
        "[quantize] wrote %s (%.2f GiB)",
        out_gguf,
        out_gguf.stat().st_size / (1 << 30),
    )


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def smoke_test(
    out_gguf: Path,
    expected_arch_substr: str,
    log,
) -> dict[str, Any]:
    """gguf-py header read + cheap metadata checks. Soft-fails on old gguf."""
    info: dict[str, Any] = {
        "path": str(out_gguf),
        "size_bytes": out_gguf.stat().st_size,
    }
    try:
        from gguf import GGUFReader  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover - import-only failure
        log.warning("[smoke] gguf import failed (%s); skipping reader checks", e)
        info["smoke_test_skipped"] = "gguf import failed"
        return info
    try:
        reader = GGUFReader(str(out_gguf))
    except Exception as e:
        log.warning(
            "[smoke] GGUFReader failed on %s (%s). Likely the installed gguf is "
            "too old for qwen3_5_moe; upgrade to >=0.18.",
            out_gguf, e,
        )
        info["smoke_test_skipped"] = f"GGUFReader: {e}"
        return info
    arch = ""
    try:
        for k in reader.fields:
            if k.endswith("architecture") or k == "general.architecture":
                v = reader.fields[k].parts[reader.fields[k].data[0]]
                if hasattr(v, "tobytes"):
                    arch = bytes(v).decode("utf-8", errors="replace")
                else:
                    arch = str(v)
                break
    except Exception:  # pragma: no cover - best effort
        pass
    info["tensor_count"] = len(reader.tensors)
    info["architecture"] = arch
    log.info("[smoke] %s tensors=%d arch=%s", out_gguf.name, info["tensor_count"], arch or "?")
    if expected_arch_substr and arch and expected_arch_substr.lower() not in arch.lower():
        log.warning(
            "[smoke] architecture %r does not contain %r; metadata may be off",
            arch, expected_arch_substr,
        )
    return info


def _input_dir_size(p: Path) -> int:
    total = 0
    for f in p.glob("*.safetensors"):
        total += f.stat().st_size
    return total


def _resolve_path(value: str | None) -> Path | None:
    return Path(value).expanduser().resolve() if value else None


# --------------------------------------------------------------------------
# CLI entry
# --------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export trained safetensor model -> quantized GGUF (Qwen3.5 MoE text-only).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Disk: ~70 GiB BF16 GGUF + sum of quants (Q4_K_M ~25%, Q5_K_M ~31%, "
            "Q6_K ~41%, Q8_0 ~53% of BF16). CPU-only; 16 GiB VRAM is unused.\n"
            "Imatrix-based Q*_K_XL / IQ4_XS quants are out of scope for v1; "
            "see TODO(imatrix)."
        ),
    )
    p.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    p.add_argument("--input", dest="input_dir", type=str, default=None,
                   help="Override export_gguf.input_dir.")
    p.add_argument("--output", dest="output_dir", type=str, default=None,
                   help="Override export_gguf.output_dir.")
    p.add_argument("--quants", type=str, default=None,
                   help=f"Comma-separated quant types. Allowed: {sorted(_ALLOWED_GGUF_QUANTS)}. "
                        f"Default: {','.join(_DEFAULT_GGUF_QUANTS)}")
    p.add_argument("--keep_bf16", action="store_true",
                   help="Keep the intermediate BF16 GGUF after quantization.")
    p.add_argument("--no_strip_mtp", action="store_true",
                   help="Retain MTP layers (default drops them; see llama.cpp issue #23033).")
    p.add_argument("--llama_cpp_dir", type=str, default=None,
                   help="Dir containing llama-quantize(.exe).")
    p.add_argument("--llama_cpp_src_dir", type=str, default=None,
                   help="Dir containing convert_hf_to_gguf.py.")
    p.add_argument("--work_dir", type=str, default=None,
                   help="Where to write the vision-stripped HF dir.")
    p.add_argument("--skip_convert", action="store_true",
                   help="Skip the BF16 convert step (assume model-BF16.gguf already exists).")
    p.add_argument("--skip_quantize", action="store_true",
                   help="Stop after BF16 GGUF; do not quantize.")
    p.add_argument("--no_smoke_test", action="store_true",
                   help="Skip post-export smoke test.")
    p.add_argument("--no_mmproj", action="store_true",
                   help="Skip vision mmproj export (default emits mmproj-<TYPE>.gguf).")
    p.add_argument("--mmproj_outtype", choices=_MMPROJ_OUTTYPES, default=None,
                   help="mmproj float type: bf16 (default) | f16 | f32.")
    return p


def _resolve_settings(args: argparse.Namespace, app_export: ExportGGUFConfig | None) -> ExportGGUFConfig:
    """Merge CLI overrides on top of YAML; CLI wins when explicitly given."""
    if app_export is None and any(
        getattr(args, k) is None
        for k in ("input_dir", "output_dir")
    ):
        raise ValueError(
            "export_gguf section missing in config and --input/--output not provided. "
            "Either add the export_gguf block or pass both --input and --output."
        )
    base = app_export or ExportGGUFConfig(
        input_dir="",
        output_dir="",
    )
    quants = base.quant_types
    if args.quants is not None:
        quants = [q.strip() for q in args.quants.split(",") if q.strip()]
        bad = [q for q in quants if q not in _ALLOWED_GGUF_QUANTS]
        if bad:
            raise ValueError(
                f"--quants contains unsupported {bad}. Allowed: {sorted(_ALLOWED_GGUF_QUANTS)}"
            )
    return ExportGGUFConfig(
        input_dir=args.input_dir or base.input_dir,
        output_dir=args.output_dir or base.output_dir,
        llama_cpp_dir=args.llama_cpp_dir or base.llama_cpp_dir,
        llama_cpp_src_dir=args.llama_cpp_src_dir or base.llama_cpp_src_dir,
        quant_types=list(quants),
        drop_mtp=False if args.no_strip_mtp else base.drop_mtp,
        keep_bf16=args.keep_bf16 or base.keep_bf16,
        work_dir=args.work_dir or base.work_dir,
        smoke_test=False if args.no_smoke_test else base.smoke_test,
        export_mmproj=False if args.no_mmproj else base.export_mmproj,
        mmproj_outtype=(args.mmproj_outtype or base.mmproj_outtype).lower(),  # type: ignore[arg-type]
    )


def main() -> None:
    log = get_logger()
    args = _build_parser().parse_args()
    app = load_config(args.config)
    settings = _resolve_settings(args, app.export_gguf)

    input_dir = _resolve_path(settings.input_dir)
    output_dir = _resolve_path(settings.output_dir)
    if input_dir is None or output_dir is None:
        raise ValueError("export_gguf.input_dir and output_dir are required")
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    work_dir = _resolve_path(settings.work_dir) or (output_dir / "_work_text_only")
    llama_cpp_dir = _resolve_path(settings.llama_cpp_dir)
    src_dir = _resolve_path(settings.llama_cpp_src_dir)

    log.info(
        "[export_gguf] input=%s output=%s work=%s quants=%s drop_mtp=%s keep_bf16=%s",
        input_dir, output_dir, work_dir, settings.quant_types,
        settings.drop_mtp, settings.keep_bf16,
    )

    convert_script, quantize_bin = validate_llama_cpp(llama_cpp_dir, src_dir)
    log.info("[export_gguf] convert=%s quantize=%s", convert_script, quantize_bin)

    in_size = _input_dir_size(input_dir)
    if in_size > 0:
        preflight_disk(
            output_dir,
            in_size,
            settings.quant_types,
            settings.keep_bf16,
            mmproj_outtype=(settings.mmproj_outtype if settings.export_mmproj else None),
        )
        log.info("[export_gguf] input ~%.2f GiB; disk preflight OK", in_size / (1 << 30))

    out_bf16 = output_dir / "model-BF16.gguf"
    if not args.skip_convert:
        prepare_text_only_dir(input_dir, work_dir, settings.drop_mtp, log)
        run_convert(convert_script, work_dir, out_bf16, log)
    else:
        if not out_bf16.is_file():
            raise FileNotFoundError(
                f"--skip_convert set but {out_bf16} is missing"
            )
        log.info("[export_gguf] skipping convert; using existing %s", out_bf16)

    report: dict[str, Any] = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "drop_mtp": settings.drop_mtp,
        "bf16": str(out_bf16) if out_bf16.is_file() else None,
        "quants": {},
    }

    if not args.skip_quantize:
        for qtype in settings.quant_types:
            out_path = output_dir / f"model-{qtype}.gguf"
            run_quantize(quantize_bin, out_bf16, out_path, qtype, log)
            entry: dict[str, Any] = {"path": str(out_path)}
            if settings.smoke_test:
                entry.update(smoke_test(out_path, "qwen", log))
                entry["sha256"] = _sha256_file(out_path)
            report["quants"][qtype] = entry

        if not settings.keep_bf16 and out_bf16.is_file():
            log.info("[export_gguf] removing intermediate %s", out_bf16)
            out_bf16.unlink()
            report["bf16"] = None
    else:
        log.info("[export_gguf] skipping quantize (--skip_quantize)")

    if settings.export_mmproj and not args.skip_convert:
        out_mm = output_dir / f"mmproj-{settings.mmproj_outtype.upper()}.gguf"
        run_convert_mmproj(convert_script, input_dir, out_mm, settings.mmproj_outtype, log)
        mm_entry: dict[str, Any] = {"path": str(out_mm), "outtype": settings.mmproj_outtype}
        if settings.smoke_test:
            mm_entry.update(smoke_test(out_mm, _MMPROJ_ARCH_HINT, log))
            mm_entry["sha256"] = _sha256_file(out_mm)
        report["mmproj"] = mm_entry
    elif settings.export_mmproj and args.skip_convert:
        log.info("[export_gguf] --skip_convert set; skipping mmproj convert too")

    report_path = output_dir / "export_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("[export_gguf] wrote %s", report_path)
    log.info("[export_gguf] done.")


if __name__ == "__main__":
    main()
