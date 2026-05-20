"""Single-source streaming probe.

Bypasses build_train_set to isolate why a given source yields zero rows.
Reports where exactly the iterator dies: load_dataset(), iter() startup,
or the first network fetch.

Usage:
    python -m scripts.probe_source fineweb_edu_en
    python -m scripts.probe_source fineweb_edu_en --reset-cache
    python -m scripts.probe_source fineweb_edu_en --no-streaming
    python -m scripts.probe_source fineweb_edu_en --raw-tree
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import traceback
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _raw_tree_check(spec: dict, endpoint: str) -> None:
    """Hit the Hub tree API directly, see what hf-mirror returns for the
    config's expected file path. Tells us if mirror has the parquet shards."""
    import requests

    repo = spec["hf_dataset"]
    cfg = spec.get("hf_config")
    base = endpoint.rstrip("/")

    paths = ["", "data"]
    if cfg:
        # Common patterns in dataset YAML configs
        paths.extend([cfg, cfg.replace("-", "/"), f"data/{cfg}"])
    print(f"\n>>> raw-tree probe on {repo} (configs to scan: {paths})")
    for sub in paths:
        url = f"{base}/api/datasets/{repo}/tree/main"
        if sub:
            url += f"/{sub}"
        url += "?recursive=false&expand=false"
        try:
            r = requests.get(url, timeout=20)
            text = r.text
            n = text.count('"path"') if r.ok else 0
            print(f"  {url}\n    -> {r.status_code}, {len(text)} bytes, ~{n} entries")
            if r.ok and n > 0:
                # show first 3 entries
                import re
                for m in list(re.finditer(r'"path"\s*:\s*"([^"]+)"', text))[:3]:
                    print(f"      • {m.group(1)}")
        except Exception as e:
            print(f"  {url}\n    -> {type(e).__name__}: {e}")


def _scan_tree(spec: dict, endpoint: str) -> None:
    """Recursively explore a repo to find where parquet/json* files live.

    Useful to figure out the right `subdir` + `glob` for parquet_glob loader
    on a source that returns empty under streaming.
    """
    import re

    import requests

    repo = spec["hf_dataset"]
    base = endpoint.rstrip("/")
    print(f"\n>>> scan-tree on {repo}")

    # 1. List root recursively (capped at first page)
    url = f"{base}/api/datasets/{repo}/tree/main?recursive=true&expand=false"
    r = requests.get(url, timeout=30)
    if not r.ok:
        print(f"  root tree GET -> {r.status_code}: {r.text[:200]}")
        return
    rows = r.json()
    if isinstance(rows, dict):
        print(f"  root tree returned envelope: {rows}")
        return
    files = [row["path"] for row in rows if row.get("type") == "file"]
    print(f"  root recursive: {len(files)} files (first page; tree API caps at 1000)")
    # bucket by extension and top-level dir
    by_dir: dict[str, list[str]] = {}
    by_ext: dict[str, int] = {}
    for p in files:
        d = p.split("/", 1)[0] if "/" in p else "."
        by_dir.setdefault(d, []).append(p)
        ext = re.search(r"\.[a-z0-9]+(?:\.gz)?$", p.lower())
        if ext:
            by_ext[ext.group(0)] = by_ext.get(ext.group(0), 0) + 1
    print(f"  extensions: {by_ext}")
    print(f"  top-level entries: {sorted(by_dir.keys())}")
    for d, paths in sorted(by_dir.items()):
        # show 3 samples per top-level dir
        sample = paths[:3]
        print(f"    {d}/  ({len(paths)} files)")
        for p in sample:
            print(f"      • {p}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="source name from data_sources.yaml")
    ap.add_argument("--config", default="configs/data_sources.yaml")
    ap.add_argument("--reset-cache", action="store_true",
                    help="rm the dataset's cache dir before probing")
    ap.add_argument("--no-streaming", action="store_true",
                    help="try non-streaming load (downloads all shards)")
    ap.add_argument("--raw-tree", action="store_true",
                    help="hit the Hub tree API directly to see if mirror has files")
    ap.add_argument("--scan-tree", action="store_true",
                    help="recursively scan repo and bucket files by directory + extension")
    args = ap.parse_args()

    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    endpoint = raw["builder"].get("hf_endpoint")
    spec = next((s for s in raw["sources"] if s["name"] == args.source), None)
    if spec is None:
        print(f"source '{args.source}' not in {args.config}")
        return 2

    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
        from moe_prune_distill.data.sources import _install_endpoint_redirect
        _install_endpoint_redirect(endpoint)
        print(f"HF_ENDPOINT={endpoint}")

    if args.reset_cache:
        cache_root = Path(
            os.environ.get("HF_DATASETS_CACHE")
            or (Path.home() / ".cache" / "huggingface" / "datasets")
        )
        ds_slug = spec["hf_dataset"].replace("/", "___")
        for p in cache_root.rglob(f"*{ds_slug}*"):
            print(f"rm {p}")
            shutil.rmtree(p, ignore_errors=True)

    if args.scan_tree:
        _scan_tree(spec, endpoint or "https://huggingface.co")
        return 0
    if args.raw_tree:
        _raw_tree_check(spec, endpoint or "https://huggingface.co")
        return 0

    print(f"\n>>> dataset={spec['hf_dataset']} config={spec.get('hf_config')} "
          f"split={spec.get('split', 'train')} streaming={not args.no_streaming}")

    from datasets import load_dataset

    try:
        kwargs = {"split": spec.get("split", "train"),
                  "streaming": not args.no_streaming}
        if spec.get("hf_config"):
            kwargs["name"] = spec["hf_config"]
        ds = load_dataset(spec["hf_dataset"], **kwargs)
        print(f"load_dataset() returned: {type(ds).__name__}")
    except Exception:
        print("load_dataset() failed:")
        traceback.print_exc()
        return 1

    try:
        it = iter(ds)
        print(f"iter(ds) returned: {type(it).__name__}")
    except Exception:
        print("iter(ds) failed:")
        traceback.print_exc()
        return 1

    for i in range(3):
        try:
            row = next(it)
        except StopIteration:
            print(f"StopIteration after {i} rows — iterator empty")
            return 1
        except Exception:
            print(f"next() raised after {i} rows:")
            traceback.print_exc()
            return 1
        keys = list(row.keys()) if isinstance(row, dict) else type(row).__name__
        sample = str(row)[:240]
        print(f"row {i}: keys={keys}\n  preview: {sample}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
