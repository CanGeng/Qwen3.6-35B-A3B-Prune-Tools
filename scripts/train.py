"""P1: distill training (4bit + LoRA + hidden MSE + router KL + SFT CE).

When ``teacher_cache.enabled`` is true and the cache directory contains files,
the trainer runs the full distill objective. Otherwise it falls back to plain
SFT CE (P0 behavior) for compatibility.

Visual / multimodal branches are treated as a frozen "special embedding": all
parameters under ``visual.``/``vision_tower.``/``image_*`` are forced
``requires_grad=False`` and excluded from LoRA target matching, which restricts
training to the language tower only (matches the design intent).
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
)

from moe_prune_distill.config import load_config
from moe_prune_distill.data.collator import SFTCollator
from moe_prune_distill.data.dataset import JsonlSFTDataset
from moe_prune_distill.distill.dataset import DistillJsonlDataset, collate_distill
from moe_prune_distill.distill.lr_scheduler import build_scheduler
from moe_prune_distill.distill.metrics import (
    batch_token_stats,
    hidden_metrics,
    router_diagnostics,
)
from moe_prune_distill.distill.teacher_cache import cache_layers_for
from moe_prune_distill.distill.trainer import (
    _hf_hidden_to_dict,
    _hf_router_to_dict,
    _restore_router_batch,
    compute_distill_loss,
    load_expert_mapping,
)
from moe_prune_distill.utils.log_format import format_metrics_row
from moe_prune_distill.utils.logging import get_logger
from moe_prune_distill.utils.metrics_log import JsonlMetricsWriter
from moe_prune_distill.utils.tensorboard import TensorBoardWriter


VISION_SUBSTRINGS = (
    "visual.",
    "vision_tower",
    "image_newline",
    "vision_model",
    "mm_projector",
)


# Only kwargs that ``Qwen3_5MoeForConditionalGeneration.forward`` actually
# accepts.  ``position_ids_3d`` is intentionally excluded — the model
# recomputes M-RoPE positions internally via ``compute_3d_position_ids`` when
# ``position_ids`` is None, given ``mm_token_type_ids`` + ``image_grid_thw``.
_VL_BATCH_KEYS = ("pixel_values", "image_grid_thw", "mm_token_type_ids")


def _vl_forward_kwargs(batch: dict, device) -> dict:
    """Pull VL tensors out of a batch and move them to ``device``.

    Returns an empty dict for text-only batches so we don't accidentally
    pass ``None`` into the model and trip the vision branch.
    """
    out: dict = {}
    for k in _VL_BATCH_KEYS:
        v = batch.get(k)
        if v is None:
            continue
        if hasattr(v, "to"):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def _torch_dtype(name: str) -> torch.dtype:
    n = name.lower()
    if n in ("bf16", "bfloat16"):
        return torch.bfloat16
    if n in ("fp16", "float16"):
        return torch.float16
    return torch.float32


def _is_vision_param(name: str) -> bool:
    return any(s in name for s in VISION_SUBSTRINGS)


def _set_router_trainable(model: torch.nn.Module, train: bool) -> int:
    n = 0
    for name, module in model.named_modules():
        if not name.endswith("mlp.gate"):
            continue
        if _is_vision_param(name):
            continue
        for p in module.parameters():
            p.requires_grad = train
            if train:
                n += p.numel()
    return n


def _set_named_trainable(model: torch.nn.Module, suffix: str, train: bool) -> None:
    for name, module in model.named_modules():
        if name.endswith(suffix) and not _is_vision_param(name):
            for p in module.parameters():
                p.requires_grad = train


def _freeze_vision(model: torch.nn.Module, log) -> int:
    frozen = 0
    for name, p in model.named_parameters():
        if _is_vision_param(name):
            p.requires_grad = False
            frozen += p.numel()
    if frozen:
        log.info("Vision branch frozen: %d params", frozen)
    return frozen


def _gpu_max_memory(reserve_gb: float = 1.5) -> dict[int | str, str]:
    out: dict[int | str, str] = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total_g = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            usable = max(total_g - reserve_gb, 1.0)
            out[i] = f"{usable:.1f}GiB"
    out["cpu"] = "40GiB"
    return out


def _load_student_model(
    model_path: Path,
    bnb_config: BitsAndBytesConfig,
    attn_impl: str,
    log,
    offload_folder: Path | None,
) -> PreTrainedModel:
    cfg = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
    arch0 = (cfg.architectures or [""])[0]
    common = dict(
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory=_gpu_max_memory(),
    )
    if offload_folder is not None:
        offload_folder.mkdir(parents=True, exist_ok=True)
        common["offload_folder"] = str(offload_folder)
        common["offload_state_dict"] = True

    def _try(attn: str) -> PreTrainedModel:
        if arch0 == "Qwen3_5MoeForConditionalGeneration":
            from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
                Qwen3_5MoeForConditionalGeneration,
            )

            return Qwen3_5MoeForConditionalGeneration.from_pretrained(
                str(model_path), attn_implementation=attn, **common
            )
        return AutoModelForCausalLM.from_pretrained(
            str(model_path), attn_implementation=attn, **common
        )

    try:
        return _try(attn_impl)
    except Exception as e:
        log.warning("Load with attn_implementation=%s failed (%s); retry sdpa", attn_impl, e)
        return _try("sdpa")


def _build_lora_target_regex(
    target_modules: list[str],
    train_layer_range: tuple[int, int] | None,
) -> str:
    """LoRA target regex restricted to language tower (no ``visual.``)."""
    mods = "|".join(re.escape(m) for m in target_modules)
    layer_pat = r"\d+"
    if train_layer_range is not None:
        lo, hi = train_layer_range
        # alternation across explicit indices to keep regex readable
        layer_pat = "(" + "|".join(str(i) for i in range(lo, hi)) + ")"
    return (
        rf"^(?!.*\bvisual\.)(?!.*vision_tower)(?!.*vision_model)"
        rf".*model\.(?:language_model\.)?layers\.{layer_pat}\..*({mods})$"
    )


def _evaluate(
    model,
    val_dl: DataLoader,
    device,
    *,
    distill_mode: bool,
    weights: dict[str, float],
    hidden_layer_weighting: str,
    router_temperature: float,
    surviving_by_layer: dict[int, list[int]],
    cache_layers: list[int] | None,
) -> dict[str, float]:
    """Run a no-grad pass over ``val_dl`` and return averaged metrics."""
    was_training = model.training
    model.eval()
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    n_batches = 0
    try:
        with torch.no_grad():
            for batch in val_dl:
                student_inputs = {
                    "input_ids": batch["input_ids"].to(device),
                    "attention_mask": batch["attention_mask"].to(device),
                    "labels": batch["labels"].to(device),
                }
                fwd_kwargs = dict(student_inputs)
                fwd_kwargs["use_cache"] = False
                fwd_kwargs.update(_vl_forward_kwargs(batch, device))
                if distill_mode:
                    fwd_kwargs["output_hidden_states"] = bool(weights["hidden_mse"] > 0)
                    fwd_kwargs["output_router_logits"] = bool(weights["router_kl"] > 0)
                    fwd_kwargs["labels"] = None

                out = model(**fwd_kwargs)
                row: dict[str, float] = {}
                if distill_mode:
                    full_batch = {
                        "input_ids": student_inputs["input_ids"],
                        "attention_mask": student_inputs["attention_mask"],
                        "labels": student_inputs["labels"],
                        "teacher_hidden": batch.get("teacher_hidden") or {},
                        "teacher_router": batch.get("teacher_router") or {},
                    }
                    loss, comps = compute_distill_loss(
                        out,
                        full_batch,
                        cache_layers=list((batch.get("teacher_hidden") or {}).keys())
                        or (cache_layers or []),
                        surviving_by_layer=surviving_by_layer,
                        weights=weights,
                        hidden_layer_weighting=hidden_layer_weighting,
                        router_temperature=router_temperature,
                    )
                    row["loss"] = float(loss.detach().cpu())
                    for k, v in comps.items():
                        row[k] = float(v)

                    teacher_hidden = batch.get("teacher_hidden") or {}
                    if teacher_hidden:
                        s_hidden = _hf_hidden_to_dict(
                            getattr(out, "hidden_states", None),
                            list(teacher_hidden.keys()),
                        )
                        if s_hidden:
                            row.update(
                                hidden_metrics(
                                    s_hidden,
                                    teacher_hidden,
                                    student_inputs["attention_mask"],
                                )
                            )
                    teacher_router = batch.get("teacher_router") or {}
                    s_router_raw = getattr(out, "router_logits", None)
                    s_router = _hf_router_to_dict(
                        s_router_raw, list(teacher_router.keys())
                    ) if teacher_router else {}
                    B = student_inputs["input_ids"].shape[0]
                    T = student_inputs["input_ids"].shape[1]
                    if s_router:
                        s_router = {
                            k: _restore_router_batch(v, B, T) for k, v in s_router.items()
                        }
                    row.update(
                        router_diagnostics(
                            s_router,
                            teacher_router,
                            surviving_by_layer,
                            student_inputs["attention_mask"],
                        )
                    )
                else:
                    row["loss"] = float(out.loss.detach().cpu())
                    row["sft_ce"] = row["loss"]

                row.update(batch_token_stats(student_inputs["attention_mask"]))

                for k, v in row.items():
                    sums[k] = sums.get(k, 0.0) + float(v)
                    counts[k] = counts.get(k, 0) + 1
                n_batches += 1
    finally:
        if was_training:
            model.train()
    if n_batches == 0:
        return {}
    return {k: sums[k] / counts[k] for k in sums}


def main() -> None:
    log = get_logger()
    p = argparse.ArgumentParser(description="P1 distill training (4bit + LoRA)")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--offload-folder", type=str, default=None)
    p.add_argument(
        "--student-dir-override",
        type=str,
        default=None,
        help="use this directory instead of prune.student_dir (e.g. ./models/student_layerwise)",
    )
    p.add_argument(
        "--train-layer-start",
        type=int,
        default=None,
        help="restrict LoRA targeting to layers [start, end)",
    )
    p.add_argument("--train-layer-end", type=int, default=None)
    p.add_argument(
        "--optimizer",
        type=str,
        default="muon_triton_batched",
        choices=["adamw", "muon_triton", "muon_triton_batched"],
        help="end-to-end optimizer (default adamw). Muon variants need Triton.",
    )
    args = p.parse_args()
    app = load_config(args.config)

    torch.manual_seed(app.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(app.train.seed)

    model_path = (
        Path(args.student_dir_override).resolve()
        if args.student_dir_override
        else Path(app.prune.student_dir).resolve()
    )
    log.info("Loading student from %s", model_path)
    tok = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    qcfg = app.train.quantization
    compute_dtype = _torch_dtype(qcfg.bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=qcfg.load_in_4bit,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=qcfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=True,
    )

    attn_impl = "flash_attention_2" if app.train.use_flash_attention else "sdpa"
    offload_folder = Path(args.offload_folder).resolve() if args.offload_folder else None
    model = _load_student_model(model_path, bnb_config, attn_impl, log, offload_folder)

    _freeze_vision(model, log)

    if app.train.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    model = prepare_model_for_kbit_training(model)
    _freeze_vision(model, log)  # prepare_model_for_kbit_training flips some grads back

    train_range = None
    if args.train_layer_start is not None and args.train_layer_end is not None:
        train_range = (int(args.train_layer_start), int(args.train_layer_end))
        log.info("LoRA training restricted to layers [%d, %d)", *train_range)

    target_pattern = _build_lora_target_regex(list(app.train.lora.target_modules), train_range)
    lora = LoraConfig(
        r=app.train.lora.r,
        lora_alpha=app.train.lora.alpha,
        lora_dropout=app.train.lora.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_pattern,
    )
    model = get_peft_model(model, lora)

    tr = app.train.trainable
    if tr.router == "full":
        n_router = _set_router_trainable(model, True)
        log.info("Router trainable params: %d", n_router)
    else:
        _set_router_trainable(model, False)
    _set_named_trainable(model, "embed_tokens", tr.embedding)
    _set_named_trainable(model, "lm_head", tr.lm_head)
    _freeze_vision(model, log)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters; check trainable.* and LoRA config")
    n_train = sum(p.numel() for p in trainable_params)
    log.info("Total trainable params: %d", n_train)

    # === decide between distill and plain SFT mode =====================
    distill_mode = app.teacher_cache.enabled and Path(app.teacher_cache.cache_dir).is_dir()
    if distill_mode:
        try:
            cfg_json = (model_path / "config.json").read_text(encoding="utf-8")
            from moe_prune_distill.adapters import detect_adapter
            import json as _json

            student_hf = _json.loads(cfg_json)
            adapter = detect_adapter(student_hf)
            num_layers = adapter.get_num_layers(student_hf)
        except Exception as e:
            log.warning("Could not detect adapter on student (%s); cache_layers=all", e)
            num_layers = 0
        cache_layers = (
            cache_layers_for(
                num_layers,
                app.teacher_cache.cache_layers,
                app.teacher_cache.cache_layer_interval,
            )
            if num_layers
            else None
        )
        log.info("Distill mode: cache_layers=%s", cache_layers)
        ds = DistillJsonlDataset(
            app.data.train_file,
            tok,
            max_seq_len=app.data.max_seq_len,
            cache_dir=app.teacher_cache.cache_dir,
            cache_layers=cache_layers,
            max_samples=app.data.max_samples,
            require_cache=False,
            split="train" if app.data.val_split > 0 else "all",
            val_split=app.data.val_split,
            teacher_dir=model_path,
        )
        val_ds = None
        if app.data.val_split > 0:
            val_ds = DistillJsonlDataset(
                app.data.train_file,
                tok,
                max_seq_len=app.data.max_seq_len,
                cache_dir=app.teacher_cache.cache_dir,
                cache_layers=cache_layers,
                max_samples=app.data.max_samples,
                require_cache=False,
                split="val",
                val_split=app.data.val_split,
                teacher_dir=model_path,
            )
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

        def collator(features):
            return collate_distill(features, pad_id=pad_id)

        surviving_by_layer = load_expert_mapping(model_path)
    else:
        log.info("teacher_cache disabled or missing -> plain SFT CE (P0 mode)")
        ds = JsonlSFTDataset(
            app.data.train_file,
            tok,
            max_seq_len=app.data.max_seq_len,
            max_samples=app.data.max_samples,
            split="train" if app.data.val_split > 0 else "all",
            val_split=app.data.val_split,
            teacher_dir=model_path,
        )
        val_ds = None
        if app.data.val_split > 0:
            val_ds = JsonlSFTDataset(
                app.data.train_file,
                tok,
                max_seq_len=app.data.max_seq_len,
                max_samples=app.data.max_samples,
                split="val",
                val_split=app.data.val_split,
                teacher_dir=model_path,
            )
        collator = SFTCollator(tok)
        cache_layers = None
        surviving_by_layer = {}

    dl = DataLoader(ds, batch_size=app.train.batch_size, shuffle=True, collate_fn=collator)
    val_dl = (
        DataLoader(val_ds, batch_size=app.train.batch_size, shuffle=False, collate_fn=collator)
        if val_ds is not None and len(val_ds) > 0
        else None
    )
    if val_dl is not None:
        log.info("Validation set: %d samples (val_split=%.3f)", len(val_ds), app.data.val_split)
    else:
        log.info("No validation set (val_split=0 or empty hash slice)")

    out_dir = Path(app.train.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.optimizer == "adamw":
        opt = AdamW(
            trainable_params,
            lr=app.train.learning_rate,
            weight_decay=app.train.weight_decay,
        )
    else:
        from moe_prune_distill.distill.muon_triton import (
            Muon,
            MuonBatched,
            partition_for_muon,
        )

        muon_p, adamw_p = partition_for_muon(
            (n, p) for n, p in model.named_parameters() if p.requires_grad
        )
        cls = MuonBatched if args.optimizer == "muon_triton_batched" else Muon
        log.info(
            "Optimizer=%s: %d matrix params (Muon) + %d 1D/embed params (AdamW)",
            args.optimizer, len(muon_p), len(adamw_p),
        )
        opt = cls(
            muon_p, adamw_p,
            lr=app.train.learning_rate,
            weight_decay=app.train.weight_decay,
            momentum=0.95,
            nesterov=True,
            ns_steps=5,
        )
    steps_per_epoch = max(1, math.ceil(len(dl) / app.train.gradient_accumulation_steps))
    total_steps = max(1, app.train.epochs * steps_per_epoch)
    warmup = int(total_steps * app.train.warmup_ratio)
    sched = build_scheduler(
        opt,
        type_=app.train.lr_scheduler.type,
        num_warmup=warmup,
        num_training=total_steps,
        min_lr_ratio=app.train.lr_scheduler.min_lr_ratio,
    )
    log.info(
        "LR schedule: %s warmup=%d total=%d min_lr_ratio=%.3f",
        app.train.lr_scheduler.type,
        warmup,
        total_steps,
        app.train.lr_scheduler.min_lr_ratio,
    )

    eval_steps = app.train.eval_steps if app.train.eval_steps > 0 else app.train.save_steps
    # Periodic console row independent of tqdm postfix, so a non-TTY run
    # (CI, redirected output) still shows per-step progress.
    train_console_every = max(1, eval_steps // 10)
    train_log = JsonlMetricsWriter(out_dir / "train_log.jsonl")
    val_log = JsonlMetricsWriter(out_dir / "val_log.jsonl")
    tb_root = (
        Path(app.train.tensorboard.log_dir).resolve() / "end_to_end"
        if app.train.tensorboard.enabled
        else None
    )
    tb_train = TensorBoardWriter(
        tb_root, enabled=app.train.tensorboard.enabled, namespace="train"
    )
    tb_val = TensorBoardWriter(
        tb_root,
        enabled=app.train.tensorboard.enabled and val_dl is not None,
        namespace="val",
    )

    weights = {
        "hidden_mse": app.train.losses.hidden_mse,
        "router_kl": app.train.losses.router_kl,
        "sft_ce": app.train.losses.sft_ce,
    }
    hidden_layer_weighting = app.train.losses.hidden_layer_weighting
    router_temperature = app.train.losses.router_kl_temperature

    model.train()
    accum = app.train.gradient_accumulation_steps
    global_step = 0
    device = next(model.parameters()).device
    ema_h: float | None = None
    ema_beta = 0.95

    for epoch in range(app.train.epochs):
        bar = tqdm(dl, desc=f"epoch {epoch+1}/{app.train.epochs}")
        opt.zero_grad(set_to_none=True)
        micro = 0
        loss_accum = 0.0
        comp_accum = {"hidden_mse": 0.0, "router_kl": 0.0, "sft_ce": 0.0}
        valid_tokens_accum = 0
        seq_len_sum = 0.0
        seq_len_count = 0
        for batch in bar:
            student_inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "labels": batch["labels"].to(device),
            }
            forward_kwargs = dict(student_inputs)
            forward_kwargs["use_cache"] = False
            forward_kwargs.update(_vl_forward_kwargs(batch, device))
            if distill_mode:
                forward_kwargs["output_hidden_states"] = bool(weights["hidden_mse"] > 0)
                forward_kwargs["output_router_logits"] = bool(weights["router_kl"] > 0)
                # HF expects labels=None when we override CE; we still want logits returned
                forward_kwargs["labels"] = None

            out = model(**forward_kwargs)

            if distill_mode:
                full_batch = {
                    "input_ids": student_inputs["input_ids"],
                    "attention_mask": student_inputs["attention_mask"],
                    "labels": student_inputs["labels"],
                    "teacher_hidden": batch.get("teacher_hidden") or {},
                    "teacher_router": batch.get("teacher_router") or {},
                }
                loss, comps = compute_distill_loss(
                    out,
                    full_batch,
                    cache_layers=list((batch.get("teacher_hidden") or {}).keys()) or (cache_layers or []),
                    surviving_by_layer=surviving_by_layer,
                    weights=weights,
                    hidden_layer_weighting=hidden_layer_weighting,
                    router_temperature=router_temperature,
                )
                for k, v in comps.items():
                    comp_accum[k] = comp_accum.get(k, 0.0) + v
            else:
                loss = out.loss
                comps = {"sft_ce": float(loss.detach().cpu())}
                comp_accum["sft_ce"] += comps["sft_ce"]

            tok_stats = batch_token_stats(student_inputs["attention_mask"])
            valid_tokens_accum += int(tok_stats.get("valid_tokens", 0))
            seq_len_sum += float(tok_stats.get("mean_seq_len", 0.0))
            seq_len_count += 1

            loss = loss / accum
            loss.backward()
            loss_accum += float(loss.detach().cpu()) * accum
            micro += 1
            if micro % accum == 0:
                grad_norm_t = torch.nn.utils.clip_grad_norm_(
                    trainable_params, app.train.max_grad_norm
                )
                grad_norm = float(grad_norm_t) if torch.is_tensor(grad_norm_t) else float(grad_norm_t)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1
                avg = loss_accum / accum
                avg_comps = {k: v / accum for k, v in comp_accum.items()}
                if avg_comps.get("hidden_mse", 0.0) > 0:
                    h = avg_comps["hidden_mse"]
                    ema_h = h if ema_h is None else ema_beta * ema_h + (1 - ema_beta) * h
                lr_now = float(sched.get_last_lr()[0])
                postfix = {
                    "loss": f"{avg:.4f}",
                    "lr": f"{lr_now:.2e}",
                    "gn": f"{grad_norm:.2f}",
                }
                for k, v in avg_comps.items():
                    if v > 0:
                        postfix[k] = f"{v:.3f}"
                if ema_h is not None:
                    postfix["ema_h"] = f"{ema_h:.4f}"
                bar.set_postfix(**postfix)

                row = {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "loss": avg,
                    "lr": lr_now,
                    "grad_norm": grad_norm,
                    "valid_tokens": valid_tokens_accum,
                    "mean_seq_len": seq_len_sum / max(1, seq_len_count),
                }
                for k, v in avg_comps.items():
                    if v != 0.0:
                        row[k] = v
                if ema_h is not None:
                    row["ema_h"] = ema_h
                train_log.log(row)
                tb_train.log(row, step=global_step)

                if global_step % train_console_every == 0:
                    train_scalars = {
                        "loss": avg,
                        "lr": lr_now,
                        "gn": grad_norm,
                        "valid_tokens": valid_tokens_accum,
                        "mean_seq_len": seq_len_sum / max(1, seq_len_count),
                    }
                    for k, v in avg_comps.items():
                        if v != 0.0:
                            train_scalars[k] = float(v)
                    if ema_h is not None:
                        train_scalars["ema_h"] = float(ema_h)
                    log.info(
                        format_metrics_row(
                            prefix={
                                "epoch": f"{epoch + 1}/{app.train.epochs}",
                                "step": f"{global_step:>6d}",
                                "mode": "train",
                            },
                            scalars=train_scalars,
                        )
                    )

                loss_accum = 0.0
                comp_accum = {"hidden_mse": 0.0, "router_kl": 0.0, "sft_ce": 0.0}
                valid_tokens_accum = 0
                seq_len_sum = 0.0
                seq_len_count = 0

                if val_dl is not None and global_step % eval_steps == 0:
                    val_metrics = _evaluate(
                        model,
                        val_dl,
                        device,
                        distill_mode=distill_mode,
                        weights=weights,
                        hidden_layer_weighting=hidden_layer_weighting,
                        router_temperature=router_temperature,
                        surviving_by_layer=surviving_by_layer,
                        cache_layers=cache_layers,
                    )
                    if val_metrics:
                        val_metrics = {"step": global_step, **val_metrics}
                        val_log.log(val_metrics)
                        tb_val.log(val_metrics, step=global_step)
                        val_scalars = {"lr": lr_now, **{
                            k: float(v) for k, v in val_metrics.items()
                            if k != "step" and isinstance(v, (int, float))
                        }}
                        log.info(
                            format_metrics_row(
                                prefix={
                                    "epoch": f"{epoch + 1}/{app.train.epochs}",
                                    "step": f"{global_step:>6d}",
                                    "mode": "  val",
                                },
                                scalars=val_scalars,
                            )
                        )

                if global_step % app.train.save_steps == 0:
                    save_path = out_dir / f"checkpoint-{global_step}"
                    model.save_pretrained(save_path)
                    tok.save_pretrained(save_path)
                    log.info("Saved %s", save_path)

        if micro % accum != 0:
            grad_norm_t = torch.nn.utils.clip_grad_norm_(
                trainable_params, app.train.max_grad_norm
            )
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            global_step += 1

    train_log.close()
    val_log.close()
    tb_train.close()
    tb_val.close()
    final = out_dir / "final"
    model.save_pretrained(final)
    tok.save_pretrained(final)
    log.info("Training complete: %s", final)


if __name__ == "__main__":
    main()
