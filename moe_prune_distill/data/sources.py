"""Streaming data source registry for train.jsonl synthesis.

Each transform converts a raw row from a HuggingFace dataset into the unified
JSONL schema:

    {"id": ..., "source": ..., "text": ...}                       # pretrain-style
    {"id": ..., "source": ..., "messages": [...]}                 # chat-style
    {"id": ..., "source": ..., "messages": [...], "images": [...]} # VL chat-style

For VL samples the `images` field carries absolute paths *on disk* written
by the builder; messages contain Qwen3-VL style content blocks referencing
those paths via "file://...". Image saving is the builder's job; transforms
only return PIL Image objects via the optional "_pil_images" key (consumed
and stripped before write).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal


SourceType = Literal["text", "chat", "vl"]


@dataclass
class SourceSpec:
    name: str
    type: SourceType
    hf_dataset: str
    transform: str
    quota: int
    hf_config: str | None = None
    split: str = "train"
    enabled: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


TransformFn = Callable[[dict, SourceSpec], dict | None]
REGISTRY: dict[str, TransformFn] = {}


def register(name: str) -> Callable[[TransformFn], TransformFn]:
    def deco(fn: TransformFn) -> TransformFn:
        REGISTRY[name] = fn
        return fn

    return deco


# ---------- text transforms ----------


@register("plain_text")
def t_plain_text(row: dict, spec: SourceSpec) -> dict | None:
    field_name = spec.extra.get("text_field", "text")
    text = row.get(field_name)
    if not isinstance(text, str):
        return None
    text = text.strip()
    if not text:
        return None
    return {"text": text}


@register("fineweb2_text")
def t_fineweb2_text(row: dict, spec: SourceSpec) -> dict | None:
    text = row.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    quality = row.get("language_score")
    if quality is not None and quality < 0.85:
        return None
    return {"text": text.strip()}


# ---------- chat transforms ----------


def _norm_messages(msgs: list[dict]) -> list[dict] | None:
    out = []
    for m in msgs:
        role = m.get("role") or m.get("from")
        content = m.get("content") or m.get("value")
        if role is None:
            continue
        if role in ("human", "user"):
            role = "user"
        elif role in ("gpt", "assistant", "bot"):
            role = "assistant"
        elif role == "system":
            role = "system"
        else:
            continue
        # Skip blank turns (e.g. empty system prompts) instead of rejecting
        # the whole sample — common in OpenHermes/ShareGPT exports.
        if not isinstance(content, str) or not content.strip():
            continue
        out.append({"role": role, "content": content.strip()})
    if not any(m["role"] == "assistant" for m in out):
        return None
    return out


@register("messages_passthrough")
def t_messages_passthrough(row: dict, spec: SourceSpec) -> dict | None:
    field_name = spec.extra.get("messages_field", "messages")
    msgs = row.get(field_name)
    if not isinstance(msgs, list):
        return None
    norm = _norm_messages(msgs)
    return {"messages": norm} if norm else None


@register("conversations_sharegpt")
def t_conversations_sharegpt(row: dict, spec: SourceSpec) -> dict | None:
    msgs = row.get("conversations") or row.get("conversation")
    # Some HF datasets store the list as a JSON string; parse on the fly.
    if isinstance(msgs, str):
        import json as _json

        try:
            msgs = _json.loads(msgs.replace("'", '"'))
        except Exception:
            try:
                import ast

                msgs = ast.literal_eval(msgs)
            except Exception:
                return None
    if not isinstance(msgs, list):
        return None
    norm = _norm_messages(msgs)
    return {"messages": norm} if norm else None


@register("conversations_humanassistant")
def t_conversations_humanassistant(row: dict, spec: SourceSpec) -> dict | None:
    """ShareGPT-style where each turn is {'human': ..., 'assistant': ...}."""
    msgs = row.get("conversation") or row.get("conversations")
    if isinstance(msgs, str):
        import ast

        try:
            msgs = ast.literal_eval(msgs)
        except Exception:
            return None
    if not isinstance(msgs, list):
        return None
    flat: list[dict] = []
    for turn in msgs:
        if not isinstance(turn, dict):
            return None
        h = turn.get("human") or turn.get("user")
        a = turn.get("assistant") or turn.get("gpt") or turn.get("bot")
        if isinstance(h, str) and h.strip():
            flat.append({"role": "user", "content": h.strip()})
        if isinstance(a, str) and a.strip():
            flat.append({"role": "assistant", "content": a.strip()})
    norm = _norm_messages(flat)
    return {"messages": norm} if norm else None


@register("instruction_io")
def t_instruction_io(row: dict, spec: SourceSpec) -> dict | None:
    instr = row.get("instruction") or row.get("question") or row.get("prompt") or row.get("query")
    out = row.get("output") or row.get("response") or row.get("answer")
    if not (isinstance(instr, str) and isinstance(out, str)):
        return None
    instr, out = instr.strip(), out.strip()
    if not instr or not out:
        return None
    user = instr
    inp = row.get("input") or row.get("context")
    if isinstance(inp, str) and inp.strip():
        user = f"{instr}\n\n{inp.strip()}"
    return {"messages": [
        {"role": "user", "content": user},
        {"role": "assistant", "content": out},
    ]}


@register("mmlu_mc")
def t_mmlu_mc(row: dict, spec: SourceSpec) -> dict | None:
    """MMLU: question + 4 choices + label (0..3). Render as plain MC chat.

    The `auxiliary_train` parquet on cais/mmlu wraps every record inside a
    top-level `train` struct (`{"train": {"question", "choices", "answer",
    "subject"}}`); the `all` config exposes those fields at the row root.
    Unwrap the struct when present so one transform covers both layouts.
    """
    if isinstance(row.get("train"), dict) and "question" in row["train"]:
        row = row["train"]
    q = row.get("question")
    choices = row.get("choices")
    answer_idx = row.get("answer")
    if not (isinstance(q, str) and isinstance(choices, list) and isinstance(answer_idx, int)):
        return None
    if len(choices) < 2 or not (0 <= answer_idx < len(choices)):
        return None
    letters = "ABCDEFGH"[: len(choices)]
    body = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices))
    user = f"{q.strip()}\n\n{body}\n\nAnswer with the letter only."
    asst = letters[answer_idx]
    return {"messages": [
        {"role": "user", "content": user},
        {"role": "assistant", "content": asst},
    ]}


@register("glaive_chat")
def t_glaive_chat(row: dict, spec: SourceSpec) -> dict | None:
    """Glaive function-calling-v2: `{system, chat}` where `chat` is one big
    string with `USER:` / `ASSISTANT: ... <|endoftext|>` / `FUNCTION RESPONSE:`
    sections separated by `\\n\\n\\n`. Preserves `<functioncall>` markers
    inside assistant turns since that's the target format the model must learn.
    """
    chat = row.get("chat")
    system = row.get("system") or ""
    if not isinstance(chat, str) or not chat.strip():
        return None
    EOT = "<|endoftext|>"
    msgs: list[dict] = []
    if isinstance(system, str):
        s = system.strip()
        if s.startswith("SYSTEM:"):
            s = s[len("SYSTEM:"):].strip()
        if s:
            msgs.append({"role": "system", "content": s})

    def _peel(chunk: str, marker: str) -> str:
        body = chunk[len(marker):].strip()
        if body.endswith(EOT):
            body = body[: -len(EOT)].rstrip()
        return body

    for chunk in chat.split("\n\n\n"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if chunk.startswith("USER:"):
            body = _peel(chunk, "USER:")
            if body:
                msgs.append({"role": "user", "content": body})
        elif chunk.startswith("ASSISTANT:"):
            body = _peel(chunk, "ASSISTANT:")
            if body:
                msgs.append({"role": "assistant", "content": body})
        elif chunk.startswith("FUNCTION RESPONSE:"):
            body = _peel(chunk, "FUNCTION RESPONSE:")
            if body:
                # Re-attach the prefix so the model sees the tool-response
                # convention; route it as `user` to satisfy _norm_messages.
                msgs.append({"role": "user", "content": f"FUNCTION RESPONSE: {body}"})
        # silently drop unrecognized prefixes
    norm = _norm_messages(msgs)
    return {"messages": norm} if norm else None


@register("xlam_function_call")
def t_xlam_function_call(row: dict, spec: SourceSpec) -> dict | None:
    """xLAM-function-calling-60k: `{id, query, tools, answers}` where `tools`
    and `answers` are JSON-encoded strings. Renders as system (tools) + user
    (query) + assistant (answers JSON).
    """
    import json as _json

    query = row.get("query")
    tools = row.get("tools")
    answers = row.get("answers")
    if not (isinstance(query, str) and query.strip()):
        return None
    if isinstance(tools, str):
        try:
            tools = _json.loads(tools)
        except Exception:
            return None
    if isinstance(answers, str):
        try:
            answers = _json.loads(answers)
        except Exception:
            return None
    if not isinstance(tools, list) or not isinstance(answers, list):
        return None
    tools_str = _json.dumps(tools, ensure_ascii=False)
    answers_str = _json.dumps(answers, ensure_ascii=False)
    return {"messages": [
        {"role": "system", "content": f"You have access to the following tools:\n{tools_str}"},
        {"role": "user", "content": query.strip()},
        {"role": "assistant", "content": answers_str},
    ]}


@register("hellaswag_mc")
def t_hellaswag_mc(row: dict, spec: SourceSpec) -> dict | None:
    """HellaSwag: ctx + 4 endings + string-typed label."""
    ctx = row.get("ctx")
    endings = row.get("endings")
    label = row.get("label")
    if isinstance(endings, str):
        try:
            import ast as _ast

            endings = _ast.literal_eval(endings)
        except Exception:
            return None
    if not (isinstance(ctx, str) and isinstance(endings, list) and label not in (None, "")):
        return None
    try:
        idx = int(label)
    except (TypeError, ValueError):
        return None
    if not (0 <= idx < len(endings)):
        return None
    letters = "ABCD"[: len(endings)]
    body = "\n".join(f"{letters[i]}. {e}" for i, e in enumerate(endings))
    user = (
        f"Choose the ending that best continues the passage.\n\n"
        f"{ctx.strip()}\n\n{body}\n\nAnswer with the letter only."
    )
    return {"messages": [
        {"role": "user", "content": user},
        {"role": "assistant", "content": letters[idx]},
    ]}


# ---------- vision-language transforms ----------


def _vl_message(image_marker: str, user_text: str, assistant_text: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_marker},
                {"type": "text", "text": user_text},
            ],
        },
        {"role": "assistant", "content": assistant_text},
    ]


@register("the_cauldron")
def t_the_cauldron(row: dict, spec: SourceSpec) -> dict | None:
    images = row.get("images")
    texts = row.get("texts")
    if not images or not texts:
        return None
    pil_image = images[0]
    qa = texts[0]
    user_text = (qa.get("user") or "").strip()
    assistant_text = (qa.get("assistant") or "").strip()
    if not user_text or not assistant_text:
        return None
    return {
        "messages": _vl_message("__IMG_0__", user_text, assistant_text),
        "_pil_images": [pil_image],
    }


@register("vl_image_qa")
def t_vl_image_qa(row: dict, spec: SourceSpec) -> dict | None:
    img_field = spec.extra.get("image_field", "image")
    q_field = spec.extra.get("question_field", "question")
    a_field = spec.extra.get("answer_field", "answer")
    img = row.get(img_field)
    q = row.get(q_field)
    a = row.get(a_field)
    if img is None or not isinstance(q, str) or not isinstance(a, str):
        return None
    q, a = q.strip(), a.strip()
    if not q or not a:
        return None
    return {
        "messages": _vl_message("__IMG_0__", q, a),
        "_pil_images": [img],
    }


# ---------- iteration ----------


def _install_endpoint_redirect(target: str) -> None:
    """Force every request to huggingface.co to hit `target` instead.

    `huggingface_hub.paginate` builds absolute URLs from prior responses that
    point back to huggingface.co, ignoring HF_ENDPOINT. `datasets` streaming
    of parquet/json shards goes through `requests` (via fsspec) AND
    `aiohttp` (via huggingface_hub's async file-download path), both of
    which embed redirected absolute URLs back to huggingface.co. We patch
    all three so every layer behind the GFW lands on the mirror.

    Set MPD_DEBUG_REDIRECT=1 to print every URL the patch sees, marking
    which ones actually got rewritten.
    """
    import os as _os

    debug = bool(_os.environ.get("MPD_DEBUG_REDIRECT"))
    host = target.replace("https://", "").replace("http://", "").rstrip("/")

    def _rewrite(url, layer):
        if isinstance(url, str):
            new = url.replace("huggingface.co", host) if "huggingface.co" in url else url
        else:
            s = str(url)
            new = s.replace("huggingface.co", host) if "huggingface.co" in s else s
        if debug:
            tag = "REWROTE" if (isinstance(url, str) and new != url) or (not isinstance(url, str) and new != str(url)) else "passthru"
            try:
                print(f"  [{layer}] {tag}: {url} -> {new}", flush=True)
            except Exception:
                pass
        return new

    try:
        import httpx
    except ImportError:
        httpx = None
    if httpx is not None and not getattr(httpx.Client, "_mpd_patched", False):
        orig_build = httpx.Client.build_request

        def patched_httpx(self, method, url, *args, **kwargs):
            return orig_build(self, method, _rewrite(url, "httpx"), *args, **kwargs)

        httpx.Client.build_request = patched_httpx
        httpx.Client._mpd_patched = True

    try:
        import requests
    except ImportError:
        requests = None
    if requests is not None and not getattr(requests.Session, "_mpd_patched", False):
        orig_send = requests.Session.send

        def patched_send(self, request, **kwargs):
            new_url = _rewrite(request.url, "requests")
            if new_url != request.url:
                request.url = new_url
            return orig_send(self, request, **kwargs)

        requests.Session.send = patched_send
        requests.Session._mpd_patched = True

    try:
        import aiohttp
        from yarl import URL as _YURL
    except ImportError:
        aiohttp = None
    if aiohttp is not None and not getattr(aiohttp.ClientSession, "_mpd_patched", False):
        orig_request = aiohttp.ClientSession._request

        async def patched_request(self, method, str_or_url, *args, **kwargs):
            if isinstance(str_or_url, str):
                str_or_url = _rewrite(str_or_url, "aiohttp")
            else:
                s = str(str_or_url)
                new = _rewrite(s, "aiohttp")
                if new != s:
                    str_or_url = _YURL(new)
            return await orig_request(self, method, str_or_url, *args, **kwargs)

        aiohttp.ClientSession._request = patched_request
        aiohttp.ClientSession._mpd_patched = True


def _list_repo_files(repo: str, subdir: str, endpoint: str, branch: str = "main",
                     token: str | None = None) -> list[str]:
    """List parquet/jsonl/etc files under <endpoint>/<repo>/<subdir> via the
    Hub tree API. Returns absolute mirror URLs ready for `data_files=`.

    If `token` is set, sends `Authorization: Bearer <token>` so gated repos
    (xLAM, the-stack-smol etc.) return file lists instead of 401.

    Robustness: hf-mirror sometimes returns 302 -> huggingface.co on a tree
    API request (CDN cache miss). With our requests-layer monkeypatch
    rewriting huggingface.co back to hf-mirror, the redirect loops until
    requests bails with TooManyRedirects. We disable auto-redirect on this
    call and retry a handful of times on 3xx / TooManyRedirects / network
    errors with backoff."""
    import time

    import requests

    url = f"{endpoint.rstrip('/')}/api/datasets/{repo}/tree/{branch}"
    if subdir:
        url += f"/{subdir.strip('/')}"
    url += "?recursive=true&expand=false"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    files: list[str] = []
    cursor = None
    while True:
        u = url + (f"&cursor={cursor}" if cursor else "")
        rows = None
        last_err = ""
        for attempt in range(5):
            try:
                r = requests.get(u, timeout=30, headers=headers,
                                 allow_redirects=False)
                if r.status_code in (301, 302, 303, 307, 308):
                    last_err = f"HTTP {r.status_code} -> {r.headers.get('Location', '?')[:80]}"
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
            # error envelope
            break
        for row in rows:
            if row.get("type") == "file":
                files.append(row["path"])
        # tree API uses Link header for pagination
        m = None
        if 'rel="next"' in link_hdr:
            import re
            m = re.search(r"cursor=([^&>]+)", link_hdr)
        if not m:
            break
        cursor = m.group(1)
    base = f"{endpoint.rstrip('/')}/datasets/{repo}/resolve/{branch}"
    return [f"{base}/{p}" for p in files]


_DATA_EXTS = {".parquet": "parquet", ".jsonl": "json", ".json": "json",
              ".gz": "json", ".csv": "csv", ".arrow": "arrow"}


def _detect_fmt(paths: list[str]) -> str:
    """Pick the loader format that covers the majority of `paths` by extension."""
    counts: dict[str, int] = {}
    for p in paths:
        low = p.lower()
        for ext, fmt in _DATA_EXTS.items():
            if low.endswith(ext):
                counts[fmt] = counts.get(fmt, 0) + 1
                break
    if not counts:
        return "parquet"
    return max(counts.items(), key=lambda x: x[1])[0]


def _local_snapshot_files(
    repo: str,
    branch: str,
    subdir: str,
    patterns: list[str] | None,
) -> tuple[list[str], str] | None:
    """Probe the HF cache for an existing snapshot of `repo`. Returns
    `(file_paths, fmt)` if at least one matching data file is present locally,
    otherwise None.

    Patterns are matched against subdir-relative paths (e.g. "python/data.json"
    when subdir="data" matches an absolute file "<root>/data/python/data.json").
    Setting `patterns=None` disables filtering and returns every data-file-shaped
    file under `subdir`.

    This is the local-first fast path; works for any source, not just
    parquet_glob — if a source has no `subdir`/`patterns` configured, we accept
    any data-shaped extension under the snapshot root.
    """
    import fnmatch
    from pathlib import Path as _P

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return None
    try:
        local_root = snapshot_download(
            repo_id=repo,
            repo_type="dataset",
            revision=branch,
            local_files_only=True,
        )
    except Exception:
        return None

    sub_root = _P(local_root) / subdir.strip("/") if subdir else _P(local_root)
    if not sub_root.is_dir():
        return None

    found: list[str] = []
    for f in sub_root.rglob("*"):
        if not f.is_file():
            continue
        rel = f.relative_to(sub_root).as_posix()
        if patterns:
            if not any(fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(f.name, p)
                       for p in patterns):
                continue
        else:
            low = f.name.lower()
            if not any(low.endswith(ext) for ext in _DATA_EXTS):
                continue
        found.append(str(f))
    if not found:
        return None
    fmt = _detect_fmt(found)
    return found, fmt


def _iter_via_parquet_glob(
    spec: SourceSpec,
    hf_endpoint: str,
    streaming: bool,
    token: bool | str | None = None
) -> Iterable[dict]:
    """Bypass datasets auto-resolution by enumerating files via tree API and
    feeding absolute URLs straight to the `parquet` loader. Necessary for
    repos where the script/YAML resolver mis-targets endpoints under hf-mirror
    (e.g. HuggingFaceFW/fineweb-edu hits /resolve/sample/10BT/ instead of
    /api/.../tree/sample/10BT and gets 404).

    Local-cache fast path lives in `iter_source` upstream; by the time we get
    here we know the cache has nothing usable for this source, so go to the
    network.
    """
    import fnmatch

    extra = spec.extra
    subdir = extra.get("subdir", "")
    pat_one = extra.get("glob")
    pats_many = extra.get("globs")
    patterns: list[str] = list(pats_many) if isinstance(pats_many, list) else (
        [pat_one] if isinstance(pat_one, str) else ["*.parquet"]
    )
    repo = spec.hf_dataset
    branch = extra.get("branch", "main")
    fmt = extra.get("format", "parquet")
    sd_norm = subdir.strip("/") + "/" if subdir else ""

    if extra.get("requires_token") and not token:
        print(f"[{spec.name}] parquet_glob: requires_token=true but no HF_TOKEN — likely 401 ahead")
    try:
        all_files = _list_repo_files(repo, subdir, hf_endpoint, branch=branch, token=token)
    except Exception as e:
        print(f"[{spec.name}] parquet_glob: tree API failed: {type(e).__name__}: {str(e)[:200]}")
        return

    def _rel(u: str) -> str:
        # strip everything up to and including the branch segment, then strip
        # the configured subdir so user patterns are written subdir-relative.
        rel = u.split(f"/{branch}/", 1)[-1] if f"/{branch}/" in u else u
        if sd_norm and rel.startswith(sd_norm):
            rel = rel[len(sd_norm):]
        return rel

    matched = [u for u in all_files if any(
        fnmatch.fnmatch(_rel(u), p) or fnmatch.fnmatch(u.rsplit("/", 1)[-1], p)
        for p in patterns
    )]
    if not matched:
        print(f"[{spec.name}] parquet_glob: 0 files matched repo={repo} subdir='{subdir}' patterns={patterns}")
        return
    print(f"[{spec.name}] parquet_glob: {len(matched)} files matched")

    from datasets import load_dataset

    ds = load_dataset(fmt, data_files=matched, split=spec.split or "train", streaming=streaming)
    yield from ds


def iter_source(
    spec: SourceSpec,
    hf_endpoint: str | None = None,
    streaming: bool = True,
    token: bool | str | None = None
) -> Iterable[dict]:
    """Yield raw rows from a HF dataset. Lazy import of `datasets`.

    Order of preference:
    1. Local HF cache (any source) — `snapshot_download(local_files_only=True)`
       finds the repo under `~/.cache/huggingface/hub/datasets--<repo>--main`
       and we hand the file paths straight to `load_dataset(<fmt>, data_files=...)`.
       This skips the mirror entirely.
    2. parquet_glob via mirror tree API — for repos where the auto-resolver
       fails under hf-mirror.
    3. plain `load_dataset(repo, streaming=True)` — last resort.
    """
    import os

    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint
        _install_endpoint_redirect(hf_endpoint)

    extra = spec.extra or {}
    branch = extra.get("branch", "main")
    subdir = extra.get("subdir", "")
    pat_one = extra.get("glob")
    pats_many = extra.get("globs")
    patterns: list[str] | None
    if isinstance(pats_many, list):
        patterns = list(pats_many)
    elif isinstance(pat_one, str):
        patterns = [pat_one]
    else:
        patterns = None

    cached = _local_snapshot_files(spec.hf_dataset, branch, subdir, patterns)
    if cached is not None:
        local_paths, detected_fmt = cached
        fmt = extra.get("format", detected_fmt)
        print(f"[{spec.name}] local cache hit: {len(local_paths)} files (fmt={fmt})")
        from datasets import load_dataset

        ds = load_dataset(
            fmt, data_files=local_paths, split=spec.split or "train", streaming=streaming
        )
        yield from ds
        return

    if extra.get("loader") == "parquet_glob":
        if not hf_endpoint:
            raise ValueError(f"[{spec.name}] parquet_glob loader requires hf_endpoint")
        yield from _iter_via_parquet_glob(spec, hf_endpoint, streaming, token)
        return

    from datasets import load_dataset

    kwargs: dict[str, Any] = {"split": spec.split, "streaming": streaming}
    if spec.hf_config:
        kwargs["name"] = spec.hf_config
    ds = load_dataset(spec.hf_dataset, token=token, **kwargs)
    it = iter(ds)
    try:
        first = next(it)
    except StopIteration:
        print(
            f"[{spec.name}] WARNING: streaming iterator empty on first read "
            f"(dataset={spec.hf_dataset}, config={spec.hf_config}, split={spec.split})"
        )
        return
    yield first
    for row in it:
        yield row


def apply_transform(row: dict, spec: SourceSpec) -> dict | None:
    fn = REGISTRY.get(spec.transform)
    if fn is None:
        raise KeyError(f"Unknown transform: {spec.transform}")
    return fn(row, spec)
