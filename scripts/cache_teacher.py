"""Step 4: cache teacher hidden states + router logits to disk.

Per-sample safetensors files keyed by ``hidden.layer_<i>`` and ``router.layer_<i>``
matching the design doc. The teacher itself is loaded with 4bit quantisation and
``device_map="auto"`` so accelerate offloads layers it cannot fit on GPU; the
visual branch is treated as an inert frozen embedding and never invoked here.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from moe_prune_distill.adapters import detect_adapter
from moe_prune_distill.config import load_config
from moe_prune_distill.data.dataset import JsonlSFTDataset
from moe_prune_distill.distill.teacher_cache import (
    cache_exists,
    cache_layers_for,
    parse_cache_dtype,
    save_sample_cache,
)
from moe_prune_distill.distill.teacher_loader import load_teacher_for_inference
from moe_prune_distill.utils.logging import get_logger


def _select_router_logits(out, num_layers: int) -> dict[int, torch.Tensor]:
    raw = getattr(out, "router_logits", None)
    if raw is None:
        return {}
    cleaned: dict[int, torch.Tensor] = {}
    for layer, r in enumerate(raw):
        if layer >= num_layers:
            break
        if isinstance(r, torch.Tensor) and r.ndim >= 2:
            cleaned[layer] = r
    return cleaned


def _select_hidden_states(out, layers: list[int]) -> dict[int, torch.Tensor]:
    hs = getattr(out, "hidden_states", None)
    if hs is None:
        return {}
    # HF returns L+1 tensors: index 0 is embeddings, 1..L are post-block.
    sel: dict[int, torch.Tensor] = {}
    for layer in layers:
        idx = layer + 1
        if idx < len(hs):
            sel[layer] = hs[idx]
    return sel


def main() -> None:
    log = get_logger()
    p = argparse.ArgumentParser(description="Cache teacher hidden / router")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--offload-folder", type=str, default=None)
    args = p.parse_args()
    app = load_config(args.config)

    if not app.teacher_cache.enabled:
        raise SystemExit("teacher_cache.enabled=false; nothing to do")

    teacher_dir = Path(app.download.local_dir).resolve()
    cfg_path = teacher_dir / "config.json"
    hf = json.loads(cfg_path.read_text(encoding="utf-8"))
    adapter = detect_adapter(hf)
    num_layers = adapter.get_num_layers(hf)

    layers = cache_layers_for(
        num_layers,
        app.teacher_cache.cache_layers,
        app.teacher_cache.cache_layer_interval,
    )
    log.info("Caching layers (%d/%d): %s", len(layers), num_layers, layers)

    cache_dir = Path(app.teacher_cache.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    dtype = parse_cache_dtype(app.teacher_cache.cache_dtype)

    tok = AutoTokenizer.from_pretrained(str(teacher_dir), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = JsonlSFTDataset(
        app.data.train_file,
        tok,
        max_seq_len=app.data.max_seq_len,
        max_samples=app.data.max_samples,
    )

    todo: list[int] = [i for i in range(len(ds)) if not cache_exists(cache_dir, ds.samples[i].id)]
    log.info("Samples: total=%d to-process=%d", len(ds), len(todo))
    if not todo:
        log.info("All samples already cached.")
        return

    model = load_teacher_for_inference(
        teacher_dir,
        log=log,
        offload_folder=args.offload_folder,
    )
    model.eval()

    bar = tqdm(todo, desc="cache-teacher")
    with torch.inference_mode():
        for i in bar:
            sample = ds[i]
            sid = sample["id"]
            input_ids_list = sample["input_ids"]
            attn_list = sample["attention_mask"]
            input_ids = torch.tensor([input_ids_list], dtype=torch.long)
            attn = torch.tensor([attn_list], dtype=torch.long)

            try:
                out = model(
                    input_ids=input_ids,
                    attention_mask=attn,
                    output_hidden_states=True,
                    output_router_logits=app.teacher_cache.cache_router_logits,
                    use_cache=False,
                    return_dict=True,
                )
            except torch.cuda.OutOfMemoryError:
                log.warning("OOM at sample %s; skipping", sid)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            hiddens = _select_hidden_states(out, layers)
            routers = (
                _select_router_logits(out, num_layers)
                if app.teacher_cache.cache_router_logits
                else None
            )
            if routers:
                routers = {k: v for k, v in routers.items() if k in set(layers)}

            # squeeze batch dim and move to cpu before saving
            def _sq(d: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
                return {k: v.squeeze(0).detach().to("cpu") for k, v in d.items()}

            save_sample_cache(
                cache_dir,
                sid,
                input_ids.squeeze(0),
                attn.squeeze(0),
                _sq(hiddens),
                _sq(routers) if routers else None,
                dtype=dtype,
            )

            del out, hiddens, routers
            if i % 8 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    log.info("Cache complete: %s", cache_dir)


if __name__ == "__main__":
    main()
