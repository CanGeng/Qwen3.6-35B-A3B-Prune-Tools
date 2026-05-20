"""Pre-download HF datasets referenced by data_sources.yaml.

Streaming mode caches files lazily on first read; for sources backed by a
single multi-GB blob (e.g. teknium/OpenHermes-2.5 @ ~6GB JSON) the first
smoke run blocks until that download finishes. Run this once after a fresh
clone to warm `~/.cache/huggingface` so iterative smoke builds stay fast.

After prefetch the loader auto-detects the local snapshot
(`huggingface_hub.snapshot_download(local_files_only=True)`) and reads
files off disk, skipping the mirror entirely on subsequent builds — for
both parquet_glob sources and plain `load_dataset` sources.

Gated sources (xlam_function_calling, the_stack_smol) require HF_TOKEN.
Provide it via either:
    - builder.hf_token in configs/data_sources.yaml
    - HF_TOKEN env var
    - HUGGING_FACE_HUB_TOKEN env var
YAML wins over env so smoke iteration with a one-shot token doesn't need
shell rituals. Without a token gated sources fail with 401 here; other
sources are unaffected.

Verification: after each snapshot_download, re-list the repo via the Hub
tree API (with size metadata) and compare against local file sizes. Any
data file (parquet/json/jsonl/csv/arrow) that is missing or short
(interrupted previous download, partial blob) is force-redownloaded with
`hf_hub_download(force_download=True)`. This catches "snapshot exists
but is incomplete" — `snapshot_download` checks etag, not byte size, so
a half-written blob can fool it on re-runs.

Usage:
    python -m scripts.prefetch_datasets --config configs/data_sources.yaml
    python -m scripts.prefetch_datasets --config configs/data_sources.yaml \
        --only openhermes_2p5,ultrachat_200k
    python -m scripts.prefetch_datasets --config configs/data_sources.yaml \
        --only glaive_function_calling,magicoder_evol_instruct
    python -m scripts.prefetch_datasets --config configs/data_sources.yaml \
        --only openhermes_2p5 --verify-only       # skip download, only check
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from moe_prune_distill.data.sources import _install_endpoint_redirect  # noqa: E402


_DATA_EXTS = (".parquet", ".jsonl", ".json", ".json.gz", ".csv", ".arrow",
              ".tar", ".tar.gz")


def _is_data_file(path: str) -> bool:
    low = path.lower()
    return any(low.endswith(ext) for ext in _DATA_EXTS)


def _list_repo_files_with_sizes(
    repo: str, endpoint: str, branch: str = "main", token: str | None = None
) -> list[tuple[str, int]]:
    """Return [(repo-relative path, byte size)] from the Hub tree API.

    Sizes for LFS files come from `lfs.size`; small files report `size`
    directly. Files without a size go in as -1.

    Robust to mirror 3xx flapping: disable auto-redirect and retry — the
    huggingface.co<->hf-mirror redirect loop blows past `requests`'s 30-hop
    cap otherwise (see _list_repo_files in sources.py for the same fix).
    """
    import re as _re
    import time

    import requests

    url = f"{endpoint.rstrip('/')}/api/datasets/{repo}/tree/{branch}?recursive=true&expand=true"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    out: list[tuple[str, int]] = []
    cursor = None
    while True:
        u = url + (f"&cursor={cursor}" if cursor else "")
        rows = None
        link_hdr = ""
        last_err = ""
        for attempt in range(5):
            try:
                r = requests.get(u, timeout=30, headers=headers, allow_redirects=False)
                if r.status_code in (301, 302, 303, 307, 308):
                    last_err = f"HTTP {r.status_code}"
                    time.sleep(0.5 * (attempt + 1))
                    continue
                r.raise_for_status()
                rows = r.json()
                link_hdr = r.headers.get("Link", "")
                break
            except (
                requests.exceptions.TooManyRedirects,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError,
            ) as e:
                last_err = f"{type(e).__name__}: {str(e)[:120]}"
                time.sleep(0.5 * (attempt + 1))
                continue
        if rows is None:
            raise RuntimeError(f"tree API failed after retries: {last_err}")
        if isinstance(rows, dict):
            break
        for row in rows:
            if row.get("type") != "file":
                continue
            lfs = row.get("lfs") or {}
            sz = lfs.get("size") or row.get("size") or -1
            out.append((row["path"], int(sz)))
        m = _re.search(r"cursor=([^&>]+)", link_hdr) if 'rel="next"' in link_hdr else None
        if not m:
            break
        cursor = m.group(1)
    return out


def _verify_and_repair(
    repo: str, branch: str, endpoint: str, token: str | None
) -> tuple[int, int, int]:
    """Compare remote tree (with sizes) against the local snapshot.

    Returns (ok, repaired, failed). Force-redownloads any data file that is
    missing locally or whose size is below the remote LFS-reported size.
    Non-data files (.gitattributes, README) are left alone — `snapshot_download`
    handles those because they're tiny and etag-correct.
    """
    from huggingface_hub import hf_hub_download, snapshot_download

    try:
        local_root = Path(snapshot_download(
            repo_id=repo, repo_type="dataset", revision=branch,
            local_files_only=True,
        ))
    except Exception:
        return 0, 0, 0

    try:
        remote = _list_repo_files_with_sizes(repo, endpoint, branch=branch, token=token)
    except Exception as e:
        print(f"  verify: tree API failed ({type(e).__name__}: {str(e)[:120]}); skipping verify")
        return 0, 0, 0

    ok = repaired = failed = 0
    for rel_path, remote_size in remote:
        if not _is_data_file(rel_path):
            ok += 1
            continue
        local = local_root / rel_path
        local_size = local.stat().st_size if local.is_file() else -1
        if remote_size > 0 and local_size >= remote_size:
            ok += 1
            continue
        why = "missing" if local_size < 0 else f"short ({local_size}/{remote_size})"
        print(f"  repair: {rel_path}  [{why}] -> force_download")
        try:
            hf_hub_download(
                repo_id=repo, filename=rel_path, repo_type="dataset",
                revision=branch, force_download=True, token=token,
            )
            new_size = (local_root / rel_path).stat().st_size if (local_root / rel_path).is_file() else -1
            if remote_size > 0 and new_size < remote_size:
                print(f"    still short after redownload: {new_size}/{remote_size}")
                failed += 1
            else:
                repaired += 1
        except Exception as e:
            print(f"    redownload FAILED: {type(e).__name__}: {str(e)[:200]}")
            failed += 1
    return ok, repaired, failed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_sources.yaml")
    ap.add_argument(
        "--only",
        default="",
        help="comma-separated source names; default = every enabled source",
    )
    ap.add_argument(
        "--verify-only",
        action="store_true",
        help="skip snapshot_download (assume cache exists); only run the size verify pass",
    )
    args = ap.parse_args()

    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    builder = raw.get("builder", {})
    endpoint = builder.get("hf_endpoint") or "https://huggingface.co"
    os.environ["HF_ENDPOINT"] = endpoint
    _install_endpoint_redirect(endpoint)
    print(f"HF_ENDPOINT={endpoint}")

    token = (
        builder.get("hf_token")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if token and isinstance(token, str) and token.strip():
        token = token.strip()
        os.environ["HF_TOKEN"] = token
        print("HF_TOKEN detected — gated repos will use it for download")
    else:
        token = None
        print("HF_TOKEN not set — gated repos (e.g. xlam, the_stack_smol) will 401")

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    sources = [
        s for s in raw["sources"]
        if s.get("enabled", True) and (not only or s["name"] in only)
    ]

    from huggingface_hub import snapshot_download

    for s in sources:
        repo = s["hf_dataset"]
        branch = (s.get("extra") or {}).get("branch", "main")
        if not args.verify_only:
            print(f"[{s['name']}] snapshot_download {repo} ...")
            try:
                local = snapshot_download(repo_id=repo, repo_type="dataset",
                                          revision=branch, token=token)
                print(f"[{s['name']}] cached at {local}")
            except Exception as e:
                print(f"[{s['name']}] snapshot FAILED: {type(e).__name__}: {str(e)[:240]}")
                continue
        else:
            print(f"[{s['name']}] verify-only on {repo} ...")
        ok, repaired, failed = _verify_and_repair(repo, branch, endpoint, token)
        if repaired or failed:
            print(f"[{s['name']}] verify: ok={ok} repaired={repaired} failed={failed}")
        else:
            print(f"[{s['name']}] verify: all {ok} files OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
