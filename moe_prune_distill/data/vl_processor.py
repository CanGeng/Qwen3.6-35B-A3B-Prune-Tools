"""Vision-language preprocessing for Qwen3.5 MoE samples.

Given a chat-style sample (``messages`` with image blocks) plus a list of
image file paths, produce the full set of tensors the teacher's
``Qwen3_5MoeForConditionalGeneration`` consumes:

* ``input_ids`` / ``attention_mask`` / ``labels`` -- with each
  ``<|image_pad|>`` token already expanded into the per-image run of
  ``thw_t * thw_h * thw_w / merge_size**2`` placeholders.
* ``pixel_values`` -- vision-tower input.
* ``image_grid_thw`` -- one (T, H, W) row per image.
* ``mm_token_type_ids`` -- 0 (text) / 1 (image) / 2 (video) per token; needed
  for ``get_rope_index``-style 3D position computation.
* ``position_ids_3d`` -- the (3, S) M-RoPE positions, computed up front so
  layer-streamed inference and the layerwise trainer can drive the rotary
  table without instantiating the full HF model.

The text path goes through ``tokenizer.apply_chat_template`` directly (so
``return_assistant_tokens_mask`` works for SFT label masking), and image-pad
expansion is done in lockstep with the assistant mask. The image processor
is invoked separately on PIL images. Loading is lazy and cached per worker
process.
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import torch
from PIL import Image


_log = logging.getLogger("moe_prune_distill.vl_processor")

_TOKENIZER_CACHE: dict[str, Any] = {}
_IMAGE_PROC_CACHE: dict[str, Any] = {}


def _resolve_local_path(p: str) -> str:
    """Convert ``file://...`` URLs (with Windows quirks) to a local path."""
    if p.startswith("file://"):
        parsed = urlparse(p)
        local = parsed.path
        if local.startswith("/") and len(local) > 2 and local[2] == ":":
            local = local[1:]
        return local
    return p


def get_tokenizer(teacher_dir: str | Path):
    key = str(Path(teacher_dir).resolve())
    tok = _TOKENIZER_CACHE.get(key)
    if tok is not None:
        return tok
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(key, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    _TOKENIZER_CACHE[key] = tok
    return tok


def get_image_processor(teacher_dir: str | Path):
    key = str(Path(teacher_dir).resolve())
    ip = _IMAGE_PROC_CACHE.get(key)
    if ip is not None:
        return ip
    from transformers import AutoImageProcessor

    ip = AutoImageProcessor.from_pretrained(key)
    _IMAGE_PROC_CACHE[key] = ip
    return ip


def load_pil_images(paths: list[str]) -> list[Image.Image] | None:
    """Open each path, RGB-convert, return list. ``None`` on any failure."""
    out: list[Image.Image] = []
    for p in paths:
        local = _resolve_local_path(p)
        try:
            img = Image.open(local)
            img.load()
        except Exception as e:
            _log.warning("vl_processor: failed to open %s: %s", local, e)
            return None
        if img.mode != "RGB":
            img = img.convert("RGB")
        out.append(img)
    return out


def _peek_image_size(path: str) -> tuple[int, int] | None:
    """Read (height, width) from PIL header without decoding pixels.

    Used by the lazy-pixel path: we need ``image_grid_thw`` upfront to expand
    ``<|image_pad|>`` tokens, but don't want to pay full image_processor cost
    (resize + normalize + patchify) — those run later inside the streamer's
    embed pass, one microbatch at a time, so the per-sample pixel tensor never
    accumulates in CPU RAM.
    """
    local = _resolve_local_path(path)
    try:
        with Image.open(local) as img:
            w, h = img.size
        return int(h), int(w)
    except Exception as e:
        _log.warning("vl_processor: failed to peek %s: %s", local, e)
        return None


def cheap_image_grid_thw(
    image_paths: list[str], image_processor: Any
) -> torch.Tensor | None:
    """Compute ``image_grid_thw`` without loading or resizing pixel data.

    Reads PIL headers for image dimensions, applies the processor's own
    ``smart_resize`` math (factor = patch_size * merge_size), and returns
    a ``(n_images, 3)`` long tensor — same shape and values that running the
    full image_processor would produce.
    """
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

    patch_size = int(getattr(image_processor, "patch_size", 14))
    merge_size = int(getattr(image_processor, "merge_size", 2))
    factor = patch_size * merge_size
    size = getattr(image_processor, "size", None) or {}
    min_pixels = int(size.get("shortest_edge", 56 * 56))
    max_pixels = int(size.get("longest_edge", 28 * 28 * 1280))

    rows: list[list[int]] = []
    for p in image_paths:
        hw = _peek_image_size(p)
        if hw is None:
            return None
        h, w = hw
        try:
            rh, rw = smart_resize(
                h, w, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels
            )
        except ValueError as e:
            _log.warning("vl_processor: smart_resize rejected %s (%dx%d): %s", p, h, w, e)
            return None
        rows.append([1, rh // patch_size, rw // patch_size])
    return torch.tensor(rows, dtype=torch.long)


def build_pixel_values(
    image_paths: list[str], teacher_dir: str | Path
) -> torch.Tensor | None:
    """Run the full image_processor on one sample's images.

    Used by the lazy-pixel path inside ``LayerStreamer._embed_pass`` — the
    streamer calls this just before forwarding the microbatch through the
    vision tower, then immediately discards the result. Returns the same
    ``pixel_values`` tensor the full ``build_vl_inputs`` would have produced.
    """
    pil_imgs = load_pil_images(image_paths)
    if pil_imgs is None or not pil_imgs:
        return None
    image_proc = get_image_processor(teacher_dir)
    try:
        ip_out = image_proc(images=pil_imgs, return_tensors="pt")
    except Exception as e:
        _log.warning("vl_processor: image_processor failed: %s", e)
        return None
    return ip_out["pixel_values"].contiguous()


def _build_mm_token_type_ids(
    input_ids: list[int], image_token_id: int, video_token_id: int
) -> list[int]:
    """0 = text, 1 = image_pad, 2 = video_pad."""
    out = []
    for t in input_ids:
        if t == image_token_id:
            out.append(1)
        elif t == video_token_id:
            out.append(2)
        else:
            out.append(0)
    return out


def get_rope_index_single(
    input_ids: list[int],
    mm_token_type_ids: list[int],
    image_grid_thw: torch.Tensor | None,
    video_grid_thw: torch.Tensor | None,
    spatial_merge_size: int,
) -> list[list[int]]:
    """Compute the (3, S) M-RoPE positions for one sample.

    Replicates ``Qwen3_5MoeModel.get_rope_index`` (single-sample slice) so
    callers don't have to instantiate the conditional-generation model just
    to grab position_ids. Returns three parallel int lists of length ``S``
    (T, H, W axes).
    """
    seq_len = len(input_ids)
    if seq_len == 0:
        return [[], [], []]
    grid_iters = {
        1: iter(image_grid_thw)
        if image_grid_thw is not None and len(image_grid_thw) > 0
        else None,
        2: iter(video_grid_thw)
        if video_grid_thw is not None and len(video_grid_thw) > 0
        else None,
    }
    flat: list[list[int]] = [[], [], []]
    current_pos = 0
    for modality, group in itertools.groupby(
        enumerate(mm_token_type_ids), lambda x: x[1]
    ):
        idxs = list(group)
        start = idxs[0][0]
        end = idxs[-1][0] + 1
        if modality == 0:
            run_len = end - start
            for axis in range(3):
                flat[axis].extend(range(current_pos, current_pos + run_len))
            current_pos += run_len
        else:
            grid_thw = next(grid_iters[modality])
            t = int(grid_thw[0])
            h = int(grid_thw[1])
            w = int(grid_thw[2])
            llm_t = t  # temp_merge_size = 1
            llm_h = h // spatial_merge_size
            llm_w = w // spatial_merge_size
            # Replicate the reference exactly:
            #   width = arange(llm_w) + start_pos, then repeat by h*t
            #   height = arange(llm_h) + start_pos, then repeat_interleave(w) then repeat(t)
            #   temporal = arange(llm_t) (after time_interval=1), repeat_interleave(h*w), then + start_pos
            width_base = [wi + current_pos for wi in range(llm_w)]
            height_base = [hi + current_pos for hi in range(llm_h)]
            temporal_base = list(range(llm_t))
            width_rep = width_base * (llm_h * llm_t)
            height_rep_inner: list[int] = []
            for hi in height_base:
                height_rep_inner.extend([hi] * llm_w)
            height_rep = height_rep_inner * llm_t
            temporal_rep: list[int] = []
            for ti in temporal_base:
                temporal_rep.extend([ti + current_pos] * (llm_h * llm_w))
            flat[0].extend(temporal_rep)
            flat[1].extend(height_rep)
            flat[2].extend(width_rep)
            # Original code advances by max(grid_thw[1], grid_thw[2]) // spatial_merge_size,
            # which is max(llm_h, llm_w). The image *consumes* llm_t*llm_h*llm_w token slots
            # but contributes a smaller delta to the running text position so that the next
            # text segment continues just past the image's spatial extent.
            current_pos += max(llm_h, llm_w)
    # Defensive: if rounding ever leaves a short tail (e.g. extra pad tokens at
    # the end), fill it with the current position so the tensor stays seq_len.
    for axis in range(3):
        if len(flat[axis]) < seq_len:
            flat[axis].extend([current_pos] * (seq_len - len(flat[axis])))
        flat[axis] = flat[axis][:seq_len]
    return flat


def _expand_image_pad(
    text_ids: list[int],
    attn: list[int],
    asst_mask: list[int] | None,
    image_grid_thw: torch.Tensor,  # (n_images, 3)
    image_token_id: int,
    merge_size: int,
) -> tuple[list[int], list[int], list[int] | None]:
    """Replace each ``<|image_pad|>`` token in ``text_ids`` with the per-image
    run of placeholder tokens whose length matches the vision tower's
    flattened patch count. ``attn`` and ``asst_mask`` are expanded in
    lockstep (image-run positions get attn=1 and asst_mask=0).
    """
    merge_length = merge_size * merge_size
    new_ids: list[int] = []
    new_attn: list[int] = []
    new_asst: list[int] | None = [] if asst_mask is not None else None
    img_idx = 0
    for i, tid in enumerate(text_ids):
        if tid == image_token_id:
            if img_idx >= image_grid_thw.shape[0]:
                # More image_pad tokens than grids — shouldn't happen with a
                # well-formed sample. Fall back to a single placeholder.
                run = 1
            else:
                grid = image_grid_thw[img_idx]
                run = int(grid.prod().item() // merge_length)
                img_idx += 1
            new_ids.extend([image_token_id] * run)
            new_attn.extend([attn[i]] * run)
            if new_asst is not None and asst_mask is not None:
                new_asst.extend([0] * run)
        else:
            new_ids.append(tid)
            new_attn.append(attn[i])
            if new_asst is not None and asst_mask is not None:
                new_asst.append(asst_mask[i])
    return new_ids, new_attn, new_asst


def build_vl_inputs(
    messages: list[dict],
    image_paths: list[str],
    teacher_dir: str | Path,
    max_seq_len: int,
    *,
    spatial_merge_size: int = 2,
    image_token_id: int | None = None,
    video_token_id: int | None = None,
) -> dict | None:
    """Tokenise a VL chat sample end-to-end. Returns ``None`` on any failure
    (image load, oversize after expansion, etc.) so the caller can drop it.

    Output dict keys (every value is a python list / torch tensor on CPU):

    * ``input_ids`` (list[int])
    * ``attention_mask`` (list[int])
    * ``labels`` (list[int]) -- assistant-only when ``return_assistant_tokens_mask``
      is supported, else last assistant span.
    * ``pixel_values`` (Tensor)
    * ``image_grid_thw`` (Tensor, (n_images, 3))
    * ``mm_token_type_ids`` (list[int])
    * ``position_ids_3d`` (list[list[int]], length 3)
    """
    pil_imgs = load_pil_images(image_paths)
    if pil_imgs is None or not pil_imgs:
        return None

    tok = get_tokenizer(teacher_dir)
    if image_token_id is None:
        image_token_id = tok.convert_tokens_to_ids("<|image_pad|>")
    if video_token_id is None:
        video_token_id = tok.convert_tokens_to_ids("<|video_pad|>")

    try:
        enc = tok.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=False,
            return_assistant_tokens_mask=True,
        )
        asst_mask_text = enc.get("assistant_masks")
    except TypeError:
        enc = tok.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=False,
        )
        asst_mask_text = None
    except Exception as e:
        _log.warning("vl_processor: chat_template failed: %s", e)
        return None

    text_ids = list(enc["input_ids"])
    text_attn = list(enc.get("attention_mask") or [1] * len(text_ids))
    if asst_mask_text is not None:
        asst_mask_text = list(asst_mask_text)

    # Sanity: number of image_pad tokens in the rendered text must match the
    # number of images we were given. If not, drop -- the chat template likely
    # ignored some images (or the sample is malformed).
    n_img_pads = sum(1 for t in text_ids if t == image_token_id)
    if n_img_pads != len(pil_imgs):
        _log.warning(
            "vl_processor: %d image_pad tokens in template vs %d images; skipping",
            n_img_pads,
            len(pil_imgs),
        )
        return None

    image_proc = get_image_processor(teacher_dir)
    try:
        ip_out = image_proc(images=pil_imgs, return_tensors="pt")
    except Exception as e:
        _log.warning("vl_processor: image_processor failed: %s", e)
        return None
    pixel_values = ip_out["pixel_values"]
    image_grid_thw = ip_out["image_grid_thw"]
    if not torch.is_tensor(image_grid_thw):
        image_grid_thw = torch.tensor(image_grid_thw, dtype=torch.long)
    else:
        image_grid_thw = image_grid_thw.to(torch.long)

    merge_size = int(getattr(image_proc, "merge_size", spatial_merge_size))
    new_ids, new_attn, new_asst = _expand_image_pad(
        text_ids,
        text_attn,
        asst_mask_text,
        image_grid_thw,
        image_token_id,
        merge_size,
    )
    if len(new_ids) > max_seq_len:
        # Truncating a VL sample mid-image-run would tear the placeholder
        # against the vision tower's flat patch count, so we drop instead.
        return None

    if new_asst is not None:
        labels = [tid if m else -100 for tid, m in zip(new_ids, new_asst, strict=True)]
    else:
        # Fallback: -100 everywhere except the last assistant span we can
        # locate by re-tokenising the assistant text (best-effort).
        assistant_text = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "assistant"),
            "",
        )
        if isinstance(assistant_text, list):
            assistant_text = " ".join(
                blk.get("text", "") for blk in assistant_text if blk.get("type") == "text"
            )
        a_ids = tok(assistant_text, add_special_tokens=False)["input_ids"]
        n = min(len(a_ids), len(new_ids))
        labels = [-100] * len(new_ids)
        if n > 0:
            labels[-n:] = new_ids[-n:]

    mm_type = _build_mm_token_type_ids(new_ids, image_token_id, video_token_id)
    pos3d = get_rope_index_single(
        new_ids,
        mm_type,
        image_grid_thw=image_grid_thw,
        video_grid_thw=None,
        spatial_merge_size=spatial_merge_size,
    )

    return {
        "input_ids": new_ids,
        "attention_mask": new_attn,
        "labels": labels,
        "pixel_values": pixel_values.contiguous(),
        "image_grid_thw": image_grid_thw,
        "mm_token_type_ids": mm_type,
        "position_ids_3d": pos3d,
    }


def build_vl_inputs_lazy(
    messages: list[dict],
    image_paths: list[str],
    teacher_dir: str | Path,
    max_seq_len: int,
    *,
    spatial_merge_size: int = 2,
    image_token_id: int | None = None,
    video_token_id: int | None = None,
) -> dict | None:
    """Same as :func:`build_vl_inputs` but defers ``pixel_values`` computation.

    Computes ``image_grid_thw`` cheaply from PIL header dimensions (via
    :func:`cheap_image_grid_thw`) and produces every text-side tensor the
    streamer needs (``input_ids`` / ``attention_mask`` / ``labels`` /
    ``mm_token_type_ids`` / ``position_ids_3d``). The caller is expected to
    invoke :func:`build_pixel_values` later — typically just before the
    embedding forward — so the multi-MB ``pixel_values`` tensor never sits
    in CPU RAM across all N samples.

    Returns ``None`` on any failure (image header unreadable, oversize after
    expansion, chat_template error, etc.) — same drop semantics as the eager
    builder.
    """
    tok = get_tokenizer(teacher_dir)
    image_proc = get_image_processor(teacher_dir)
    if image_token_id is None:
        image_token_id = tok.convert_tokens_to_ids("<|image_pad|>")
    if video_token_id is None:
        video_token_id = tok.convert_tokens_to_ids("<|video_pad|>")

    image_grid_thw = cheap_image_grid_thw(image_paths, image_proc)
    if image_grid_thw is None or image_grid_thw.shape[0] != len(image_paths):
        return None

    try:
        enc = tok.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=False,
            return_assistant_tokens_mask=True,
        )
        asst_mask_text = enc.get("assistant_masks")
    except TypeError:
        enc = tok.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=False,
        )
        asst_mask_text = None
    except Exception as e:
        _log.warning("vl_processor: chat_template failed: %s", e)
        return None

    text_ids = list(enc["input_ids"])
    text_attn = list(enc.get("attention_mask") or [1] * len(text_ids))
    if asst_mask_text is not None:
        asst_mask_text = list(asst_mask_text)

    n_img_pads = sum(1 for t in text_ids if t == image_token_id)
    if n_img_pads != len(image_paths):
        _log.warning(
            "vl_processor (lazy): %d image_pad tokens vs %d images; skipping",
            n_img_pads,
            len(image_paths),
        )
        return None

    merge_size = int(getattr(image_proc, "merge_size", spatial_merge_size))
    new_ids, new_attn, new_asst = _expand_image_pad(
        text_ids,
        text_attn,
        asst_mask_text,
        image_grid_thw,
        image_token_id,
        merge_size,
    )
    if len(new_ids) > max_seq_len:
        return None

    if new_asst is not None:
        labels = [tid if m else -100 for tid, m in zip(new_ids, new_asst, strict=True)]
    else:
        assistant_text = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "assistant"),
            "",
        )
        if isinstance(assistant_text, list):
            assistant_text = " ".join(
                blk.get("text", "") for blk in assistant_text if blk.get("type") == "text"
            )
        a_ids = tok(assistant_text, add_special_tokens=False)["input_ids"]
        n = min(len(a_ids), len(new_ids))
        labels = [-100] * len(new_ids)
        if n > 0:
            labels[-n:] = new_ids[-n:]

    mm_type = _build_mm_token_type_ids(new_ids, image_token_id, video_token_id)
    pos3d = get_rope_index_single(
        new_ids,
        mm_type,
        image_grid_thw=image_grid_thw,
        video_grid_thw=None,
        spatial_merge_size=spatial_merge_size,
    )

    return {
        "input_ids": new_ids,
        "attention_mask": new_attn,
        "labels": labels,
        "image_paths": list(image_paths),
        "image_grid_thw": image_grid_thw,
        "mm_token_type_ids": mm_type,
        "position_ids_3d": pos3d,
    }


__all__ = [
    "build_pixel_values",
    "build_vl_inputs",
    "build_vl_inputs_lazy",
    "cheap_image_grid_thw",
    "get_image_processor",
    "get_rope_index_single",
    "get_tokenizer",
    "load_pil_images",
]
