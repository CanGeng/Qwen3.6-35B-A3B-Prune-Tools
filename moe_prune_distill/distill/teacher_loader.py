"""Inference-only teacher loading with 4bit quant + layer/CPU/disk offload.

Designed for 16GB GPUs running 30B-class MoE teachers. Uses ``device_map="auto"``
with an explicit ``max_memory`` budget so accelerate spreads layers across GPU,
CPU RAM, and an optional disk offload folder.

Vision branches are treated as a frozen "special embedding": when no images are
fed, the visual tower is never executed, so we let accelerate place it on CPU
and forget about it.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig, PreTrainedModel


def _gpu_max_memory(reserve_gb: float = 1.5) -> dict[int | str, str]:
    """Build ``max_memory`` dict for accelerate using all visible GPUs."""
    out: dict[int | str, str] = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total = torch.cuda.get_device_properties(i).total_memory
            usable = max(int(total / (1024 ** 3)) - reserve_gb, 1.0)
            out[i] = f"{usable:.1f}GiB"
    # generous CPU budget; accelerate clamps to actual RAM
    out["cpu"] = os.environ.get("MOE_CPU_MAX_MEMORY", "120GiB")
    return out


def load_teacher_for_inference(
    teacher_dir: str | Path,
    *,
    log: logging.Logger | None = None,
    offload_folder: str | Path | None = None,
    extra_max_memory: dict[int | str, str] | None = None,
    compute_dtype: torch.dtype = torch.bfloat16,
    no_split_modules: list[str] | None = None,
) -> PreTrainedModel:
    """Load HF model in 4bit with auto layer offload (GPU -> CPU -> disk)."""
    teacher_dir = Path(teacher_dir).resolve()
    log = log or logging.getLogger("moe_prune_distill")

    cfg = AutoConfig.from_pretrained(str(teacher_dir), trust_remote_code=True)
    arch0 = (cfg.architectures or [""])[0]

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    max_memory = _gpu_max_memory()
    if extra_max_memory:
        max_memory.update(extra_max_memory)

    if offload_folder:
        offload_folder = Path(offload_folder).resolve()
        offload_folder.mkdir(parents=True, exist_ok=True)

    common: dict[str, Any] = dict(
        trust_remote_code=True,
        quantization_config=bnb,
        device_map="auto",
        max_memory=max_memory,
        torch_dtype=compute_dtype,
    )
    if offload_folder:
        common["offload_folder"] = str(offload_folder)
        common["offload_state_dict"] = True
    if no_split_modules:
        common["no_split_module_classes"] = list(no_split_modules)

    log.info(
        "Loading teacher (4bit, device_map=auto, max_memory=%s, offload=%s)",
        max_memory,
        offload_folder,
    )

    def _try(attn: str | None) -> PreTrainedModel:
        kwargs = dict(common)
        if attn:
            kwargs["attn_implementation"] = attn
        if arch0 == "Qwen3_5MoeForConditionalGeneration":
            from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
                Qwen3_5MoeForConditionalGeneration,
            )

            return Qwen3_5MoeForConditionalGeneration.from_pretrained(
                str(teacher_dir), **kwargs
            )
        return AutoModelForCausalLM.from_pretrained(str(teacher_dir), **kwargs)

    try:
        return _try("sdpa")
    except Exception as e:
        log.warning("teacher load with sdpa failed (%s); retry with default attn", e)
        return _try(None)
