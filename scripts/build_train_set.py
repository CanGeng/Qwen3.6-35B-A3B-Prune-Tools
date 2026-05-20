"""Build train.jsonl from a registry of HuggingFace data sources.

Streams each source lazily, applies its transform, filters by token length,
balances across length buckets, deduplicates, and writes a single jsonl
plus accompanying images for VL samples.

Usage:
    python -m scripts.build_train_set --config configs/data_sources.yaml
    python -m scripts.build_train_set --config configs/data_sources.yaml --smoke
    python -m scripts.build_train_set --config configs/data_sources.yaml \
        --only fineweb_edu_zh,tulu3_sft

In --smoke mode every source's token quota is capped at 500 tokens so the
full pipeline can be exercised end-to-end in well under a minute — enough
to validate schema and bucket flow without pulling real volume.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from moe_prune_distill.data.sources import (  # noqa: E402
    SourceSpec,
    _install_endpoint_redirect,
    apply_transform,
    iter_source,
)


@dataclass
class BuilderConfig:
    output_jsonl: Path
    output_image_dir: Path
    hf_endpoint: str | None
    tokenizer_path: str
    max_tokens: int
    bucket_edges: list[int]
    bucket_balance: bool
    dedup_prefix_chars: int
    shuffle_seed: int


def load_config(path: Path) -> tuple[BuilderConfig, list[SourceSpec]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    b = raw["builder"]
    cfg = BuilderConfig(
        output_jsonl=Path(b["output_jsonl"]),
        output_image_dir=Path(b["output_image_dir"]),
        hf_endpoint=b.get("hf_endpoint"),
        tokenizer_path=b["tokenizer_path"],
        max_tokens=int(b["max_tokens"]),
        bucket_edges=list(b["bucket_edges"]),
        bucket_balance=bool(b.get("bucket_balance", True)),
        dedup_prefix_chars=int(b.get("dedup_prefix_chars", 256)),
        shuffle_seed=int(b.get("shuffle_seed", 42)),
    )
    sources = []
    for s in raw["sources"]:
        sources.append(
            SourceSpec(
                name=s["name"],
                type=s["type"],
                hf_dataset=s["hf_dataset"],
                hf_config=s.get("hf_config"),
                split=s.get("split", "train"),
                transform=s["transform"],
                quota=int(s["quota"]),
                enabled=bool(s.get("enabled", True)),
                extra=s.get("extra") or {},
            )
        )
    return cfg, sources


def load_tokenizer(path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def count_tokens_text(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=True)["input_ids"])


def _strip_image_blocks(messages: list[dict]) -> list[dict]:
    """For length counting only — replace image blocks with the literal '<image>'."""
    out = []
    for m in messages:
        c = m["content"]
        if isinstance(c, str):
            out.append({"role": m["role"], "content": c})
        else:
            parts = []
            for blk in c:
                if blk.get("type") == "text":
                    parts.append(blk.get("text", ""))
                else:
                    parts.append("<image>")
            out.append({"role": m["role"], "content": "\n".join(parts)})
    return out


def count_tokens_messages(tokenizer, messages: list[dict]) -> int:
    msgs = _strip_image_blocks(messages)
    try:
        ids = tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=False
        )
    except Exception:
        joined = "\n".join(f"{m['role']}: {m['content']}" for m in msgs)
        ids = tokenizer(joined, add_special_tokens=True)["input_ids"]
    return len(ids) if isinstance(ids, list) else len(ids[0])


def bucket_of(n: int, edges: list[int]) -> int | None:
    """Return bucket index in [0, len(edges)-2], or None if above max.

    A length below edges[0] still falls in bucket 0 — we no longer reject
    short samples, only over-long ones.
    """
    if n <= 0 or n > edges[-1]:
        return None
    for i in range(len(edges) - 1):
        if n <= edges[i + 1]:
            return i
    return None


def _image_dedup_token(img) -> str:
    """Cheap perceptual key for VL dedup: 32x32 thumbnail bytes → 16-char sha1.

    Cauldron sources like ocrvqa have heavily templated prompts ("Who wrote
    this book?") on thousands of *distinct* book covers — text-only dedup
    collapses them to ~30 unique keys and drops 99% of the dataset. Hashing
    on the first image's content keeps each cover as a separate sample.
    """
    if not hasattr(img, "resize"):
        return ""
    try:
        thumb = img.convert("L").resize((32, 32))
        return hashlib.sha1(thumb.tobytes()).hexdigest()[:16]
    except Exception:
        return ""


def content_hash(record: dict, prefix_chars: int) -> str:
    if "text" in record:
        key = record["text"][:prefix_chars]
    else:
        msgs = record["messages"]
        # Prefer the first non-system turn for the dedup key. Sources like
        # Glaive recycle ~30 system templates across all rows; if the first
        # message is system the prefix collapses to a handful of values and
        # 99% of samples get rejected as duplicates.
        first = ""
        candidates = [m for m in msgs if m.get("role") != "system"] or list(msgs)
        for m in candidates:
            c = m["content"]
            if isinstance(c, str):
                first = c
                break
            for blk in c:
                if blk.get("type") == "text":
                    first = blk.get("text", "")
                    break
            if first:
                break
        key = first[:prefix_chars]
        pil_imgs = record.get("_pil_images") or []
        if pil_imgs:
            key = f"{key}|img:{_image_dedup_token(pil_imgs[0])}"
    return hashlib.sha1(key.encode("utf-8", errors="ignore")).hexdigest()


def save_pil_image(img, out_dir: Path) -> str:
    """Persist a PIL image to out_dir/{sha8}.jpg, return absolute file path."""
    if not hasattr(img, "save"):
        from PIL import Image  # type: ignore

        if isinstance(img, dict) and "bytes" in img:
            img = Image.open(io.BytesIO(img["bytes"]))
        else:
            return ""
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    data = buf.getvalue()
    sha = hashlib.sha1(data).hexdigest()[:16]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{sha}.jpg"
    if not path.exists():
        path.write_bytes(data)
    return str(path.resolve())


def materialize_images(record: dict, image_dir: Path) -> dict | None:
    """Replace _pil_images + __IMG_n__ markers with file:// URIs. Drop on failure."""
    pil_imgs = record.pop("_pil_images", None)
    if not pil_imgs:
        return record
    paths = []
    for img in pil_imgs:
        p = save_pil_image(img, image_dir)
        if not p:
            return None
        paths.append(p)
    new_msgs = []
    for m in record["messages"]:
        c = m["content"]
        if isinstance(c, str):
            new_msgs.append(m)
            continue
        new_blocks = []
        for blk in c:
            if blk.get("type") == "image":
                marker = blk.get("image", "")
                if marker.startswith("__IMG_") and marker.endswith("__"):
                    idx = int(marker[6:-2])
                    if idx < len(paths):
                        new_blocks.append({"type": "image", "image": f"file://{paths[idx]}"})
                        continue
                new_blocks.append(blk)
            else:
                new_blocks.append(blk)
        new_msgs.append({"role": m["role"], "content": new_blocks})
    record["messages"] = new_msgs
    record["images"] = paths
    return record


def harvest_source(
    spec: SourceSpec,
    cfg: BuilderConfig,
    tokenizer,
    seen_hashes: set[str],
    image_dir: Path,
    rng: random.Random,
    token: bool | str | None = None
) -> list[dict]:
    """Pull from one source until its TOKEN quota is met.

    `spec.quota` is interpreted as a target token count (sum of input_ids
    lengths across accepted samples). Length filter only enforces an upper
    cap (max_tokens / spec.extra.max_tokens) — short samples are welcome.
    Bucket balance acts as a soft preference: once a bucket exceeds 50% of
    the budget, further samples in it are rejected unless other buckets are
    also full.
    """
    accepted: list[dict] = []
    accepted_tokens = 0
    bucket_tokens = [0] * (len(cfg.bucket_edges) - 1)
    # per-bucket soft cap: split quota across N buckets with 50% headroom.
    # When quota is small (e.g. smoke mode) one long sample alone would exceed
    # quota//2 and freeze the source at 1 sample — disable the cap below the
    # threshold where balance is even meaningful.
    n_buckets = len(bucket_tokens)
    per_bucket_cap = spec.quota / n_buckets * 1.5
    bucket_balance_active = (
        cfg.bucket_balance and spec.quota >= cfg.bucket_edges[-1] * n_buckets
    )
    seen_in_source = 0
    rejected_streak = 0
    rejected_streak_cap = max(2000, spec.quota // 50)

    max_tokens = int(spec.extra.get("max_tokens", cfg.max_tokens))

    print(f"[{spec.name}] streaming, token_quota={spec.quota}, type={spec.type}")
    it = iter_source(spec, hf_endpoint=cfg.hf_endpoint, streaming=True, token=token)

    while True:
        try:
            raw = next(it)
        except StopIteration:
            break
        except Exception as e:
            print(f"[{spec.name}] FAILED while streaming: {type(e).__name__}: {str(e)[:200]}")
            return accepted

        seen_in_source += 1
        if accepted_tokens >= spec.quota:
            break
        # bail out of dud sources fast (e.g. malformed schema → 100% reject)
        if rejected_streak >= rejected_streak_cap and not accepted:
            print(f"[{spec.name}] giving up after {seen_in_source} rejected rows (no schema match)")
            break

        try:
            record = apply_transform(raw, spec)
        except Exception:
            rejected_streak += 1
            continue
        if record is None:
            rejected_streak += 1
            continue

        if "text" in record:
            n_tok = count_tokens_text(tokenizer, record["text"])
        else:
            n_tok = count_tokens_messages(tokenizer, record["messages"])
        if n_tok <= 0 or n_tok > max_tokens:
            rejected_streak += 1
            continue

        b = bucket_of(n_tok, cfg.bucket_edges)
        if b is None:
            rejected_streak += 1
            continue
        if bucket_balance_active and bucket_tokens[b] > per_bucket_cap:
            # bucket dominant — skip unless every bucket is already saturated
            if not (min(bucket_tokens) == 0 and max(bucket_tokens) >= 10000):
                if not all(bt > per_bucket_cap for bt in bucket_tokens):
                    rejected_streak += 1
                    continue

        h = content_hash(record, cfg.dedup_prefix_chars)
        if h in seen_hashes:
            rejected_streak += 1
            continue
        seen_hashes.add(h)

        record = materialize_images(record, image_dir)
        if record is None:
            rejected_streak += 1
            continue

        record["id"] = f"{spec.name}_{len(accepted):06d}"
        record["source"] = spec.name
        record["_n_tokens"] = n_tok  # stripped before write
        bucket_tokens[b] += n_tok
        accepted_tokens += n_tok
        accepted.append(record)
        rejected_streak = 0

        if len(accepted) % 200 == 0:
            print(
                f"[{spec.name}]   {len(accepted)} samples, "
                f"{accepted_tokens}/{spec.quota} tokens, buckets={bucket_tokens}"
            )

    print(
        f"[{spec.name}] done: {len(accepted)} samples, "
        f"{accepted_tokens} tokens, scanned {seen_in_source}"
    )
    return accepted


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_sources.yaml")
    ap.add_argument("--smoke", action="store_true", help="cap each source to 500 tokens")
    ap.add_argument("--only", default="", help="comma-separated source names to include")
    ap.add_argument("--output", default="", help="override output_jsonl path")
    args = ap.parse_args()

    cfg, sources = load_config(Path(args.config))
    if args.output:
        cfg.output_jsonl = Path(args.output)

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    if only:
        sources = [s for s in sources if s.name in only]
    sources = [s for s in sources if s.enabled]
    if args.smoke:
        # Smoke caps each source at ~500 tokens — enough to verify the schema
        # and a couple of buckets, fast enough to finish in under a minute.
        for s in sources:
            s.quota = min(s.quota, 5000)

    if cfg.hf_endpoint:
        os.environ["HF_ENDPOINT"] = cfg.hf_endpoint
        # Install the httpx/requests/aiohttp monkeypatches BEFORE the tokenizer
        # load. transformers eagerly imports huggingface_hub which can create
        # long-lived aiohttp connectors caching the un-rewritten URL — patch
        # the class methods here so every later request hits the mirror.
        _install_endpoint_redirect(cfg.hf_endpoint)
        print(f"HF_ENDPOINT={cfg.hf_endpoint}")

    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    hf_token = (
        raw["builder"].get("hf_token")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if isinstance(hf_token, str):
        hf_token = hf_token.strip() or None
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        print("HF_TOKEN active (from yaml or env) — gated repos will use it")
    print(f"loading tokenizer from {cfg.tokenizer_path}")
    tokenizer = load_tokenizer(cfg.tokenizer_path)

    cfg.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_image_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(cfg.shuffle_seed)
    seen_hashes: set[str] = set()
    all_records: list[dict] = []

    for spec in sources:
        recs = harvest_source(spec, cfg, tokenizer, seen_hashes, cfg.output_image_dir, rng, hf_token)
        all_records.extend(recs)

    rng.shuffle(all_records)

    # final id rewrite to be globally unique and sequential
    total_tokens = 0
    tokens_by_source: dict[str, int] = {}
    with cfg.output_jsonl.open("w", encoding="utf-8") as f:
        for i, rec in enumerate(all_records):
            rec["id"] = f"{rec['source']}_{i:06d}"
            n = rec.pop("_n_tokens", 0)
            total_tokens += n
            tokens_by_source[rec["source"]] = tokens_by_source.get(rec["source"], 0) + n
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nwrote {len(all_records)} samples / {total_tokens} tokens to {cfg.output_jsonl}")
    by_source: dict[str, int] = {}
    for r in all_records:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    for k in sorted(by_source):
        print(f"  {k:30s} {by_source[k]:>6} samples  {tokens_by_source.get(k,0):>10} tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
