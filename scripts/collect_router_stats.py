"""Step 2: collect router top-K usage statistics from the teacher.

Runs the teacher with output_router_logits=True over the first N samples and
counts how often each routed expert is among the top-k for each layer.
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
from moe_prune_distill.distill.router_stats import write_router_stats
from moe_prune_distill.distill.teacher_loader import load_teacher_for_inference
from moe_prune_distill.utils.logging import get_logger


def _extract_router_logits(out) -> tuple[torch.Tensor, ...]:
    raw = getattr(out, "router_logits", None)
    if raw is None:
        return ()
    cleaned: list[torch.Tensor] = []
    for r in raw:
        if isinstance(r, torch.Tensor) and r.ndim >= 2:
            cleaned.append(r)
    return tuple(cleaned)


def main() -> None:
    log = get_logger()
    p = argparse.ArgumentParser(description="Collect router top-k stats")
    p.add_argument("--config", type=str, required=True)
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="output path for router_stats.json (defaults to <student_dir>/router_stats.json)",
    )
    args = p.parse_args()
    app = load_config(args.config)

    teacher_dir = Path(app.download.local_dir).resolve()
    cfg_path = teacher_dir / "config.json"
    hf = json.loads(cfg_path.read_text(encoding="utf-8"))
    adapter = detect_adapter(hf)
    num_layers = adapter.get_num_layers(hf)
    num_experts = adapter.get_num_experts(hf)
    topk = adapter.get_num_experts_per_tok(hf)

    log.info(
        "Teacher: layers=%d experts=%d top_k=%d (samples=%d)",
        num_layers,
        num_experts,
        topk,
        app.prune.router_stat_samples,
    )

    tok = AutoTokenizer.from_pretrained(str(teacher_dir), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # NB: dataset truncation length covers prompt; router stats need only the input pass.
    ds = JsonlSFTDataset(
        app.data.train_file,
        tok,
        max_seq_len=app.data.max_seq_len,
        max_samples=app.prune.router_stat_samples,
    )

    model = load_teacher_for_inference(teacher_dir, log=log)
    model.eval()
    device = next(model.parameters()).device

    counts = torch.zeros(num_layers, num_experts, dtype=torch.long)

    pbar = tqdm(range(len(ds)), desc="router-stats")
    with torch.inference_mode():
        for i in pbar:
            sample = ds[i]
            input_ids = torch.tensor([sample["input_ids"]], dtype=torch.long, device=device)
            attn = torch.tensor([sample["attention_mask"]], dtype=torch.long, device=device)
            out = model(
                input_ids=input_ids,
                attention_mask=attn,
                output_router_logits=True,
                use_cache=False,
                return_dict=True,
            )
            rls = _extract_router_logits(out)
            if not rls:
                continue
            for layer, rl in enumerate(rls):
                if layer >= num_layers:
                    break
                if rl.shape[-1] != num_experts:
                    continue
                k = min(topk, num_experts)
                _, topi = rl.float().topk(k=k, dim=-1)
                flat = topi.reshape(-1).to("cpu")
                counts[layer].index_add_(0, flat, torch.ones_like(flat))
            del out, rls
            if i % 16 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    out_path = (
        Path(args.output).resolve()
        if args.output
        else Path(app.prune.student_dir).resolve().parent / "router_stats.json"
    )
    write_router_stats(
        out_path,
        model_id=app.download.model_id,
        num_samples=len(ds),
        num_layers=num_layers,
        num_experts=num_experts,
        top_k=topk,
        counts=counts,
        target_num_experts=app.prune.target_num_experts,
    )
    log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
