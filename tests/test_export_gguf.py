"""Unit tests for scripts/export_gguf pure helpers (no I/O, no llama.cpp needed)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from scripts.export_gguf import (
    _TEXT_ARCH_NAME,
    _TEXT_MODEL_TYPE,
    _filter_shards_streaming,
    filter_config_json,
    make_drop_predicate,
)


def test_drop_predicate_drops_vision_keeps_lm():
    pred = make_drop_predicate(drop_mtp=True)
    assert pred("model.visual.blocks.0.attn.qkv.weight")
    assert pred("model.vision_anything.weight")
    assert pred("visual.merger.weight")
    assert pred("mtp.layers.0.input_layernorm.weight")
    assert pred("mtp.fc.weight")
    assert not pred("model.language_model.layers.0.self_attn.q_proj.weight")
    assert not pred("lm_head.weight")


def test_drop_predicate_keeps_mtp_when_disabled():
    pred = make_drop_predicate(drop_mtp=False)
    assert pred("model.visual.blocks.0.weight")
    assert not pred("mtp.layers.0.input_layernorm.weight")
    assert not pred("mtp.fc.weight")


def test_filter_config_json_strips_vision_and_mtp():
    src = {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "image_token_id": 248056,
        "video_token_id": 248057,
        "vision_start_token_id": 248053,
        "vision_end_token_id": 248054,
        "vision_config": {"depth": 27, "hidden_size": 1152},
        "text_config": {
            "hidden_size": 2048,
            "num_experts": 256,
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": False,
            "model_type": "qwen3_5_moe_text",
        },
    }
    out = filter_config_json(src, drop_mtp=True)

    # Multimodal fields all gone.
    for k in (
        "vision_config",
        "image_token_id",
        "video_token_id",
        "vision_start_token_id",
        "vision_end_token_id",
    ):
        assert k not in out, f"{k} should be stripped"

    # Architecture rewritten.
    assert out["architectures"] == [_TEXT_ARCH_NAME]
    assert out["model_type"] == _TEXT_MODEL_TYPE

    # MTP fields stripped from text_config.
    text = out["text_config"]
    assert "mtp_num_hidden_layers" not in text
    assert "mtp_use_dedicated_embeddings" not in text

    # Non-MTP text fields preserved.
    assert text["hidden_size"] == 2048
    assert text["num_experts"] == 256

    # Source dict not mutated.
    assert "vision_config" in src
    assert "mtp_num_hidden_layers" in src["text_config"]


def test_filter_config_json_keeps_mtp_when_drop_mtp_false():
    src = {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "vision_config": {"depth": 27},
        "text_config": {
            "hidden_size": 2048,
            "mtp_num_hidden_layers": 1,
        },
    }
    out = filter_config_json(src, drop_mtp=False)
    assert "vision_config" not in out
    assert out["text_config"]["mtp_num_hidden_layers"] == 1


def test_filter_shards_streaming_drops_vision_keys(tmp_path: Path):
    """End-to-end shard rewrite: tiny synthetic tensors, real safetensors files."""
    src = tmp_path / "src"
    src.mkdir()
    # Two shards, mixing kept and dropped keys.
    shard_a = {
        "model.language_model.layers.0.input_layernorm.weight": torch.zeros(4),
        "model.visual.blocks.0.attn.qkv.weight": torch.zeros(8),
    }
    shard_b = {
        "model.language_model.layers.0.self_attn.q_proj.weight": torch.zeros(4, 4),
        "mtp.fc.weight": torch.zeros(4, 4),
        "lm_head.weight": torch.zeros(4, 8),
    }
    save_file(shard_a, str(src / "model-00001-of-00002.safetensors"))
    save_file(shard_b, str(src / "model-00002-of-00002.safetensors"))
    weight_map = {
        **{k: "model-00001-of-00002.safetensors" for k in shard_a},
        **{k: "model-00002-of-00002.safetensors" for k in shard_b},
    }
    (src / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 999}, "weight_map": weight_map}),
        encoding="utf-8",
    )

    dst = tmp_path / "dst"

    class _Log:
        def info(self, *a, **kw):
            pass

        def warning(self, *a, **kw):
            pass

    pred = make_drop_predicate(drop_mtp=True)
    kept, dropped, n_shards = _filter_shards_streaming(src, dst, pred, _Log())

    assert kept == 3, f"expected 3 kept (2 LM + lm_head), got {kept}"
    assert dropped == 2, f"expected 2 dropped (visual + mtp), got {dropped}"
    assert n_shards >= 1

    new_index = json.loads((dst / "model.safetensors.index.json").read_text(encoding="utf-8"))
    keys = set(new_index["weight_map"].keys())
    assert "model.visual.blocks.0.attn.qkv.weight" not in keys
    assert "mtp.fc.weight" not in keys
    assert "lm_head.weight" in keys
    assert "model.language_model.layers.0.self_attn.q_proj.weight" in keys


def test_filter_config_json_does_not_mutate_src_for_mmproj_path():
    """The mmproj path passes the ORIGINAL config.json to llama.cpp; this guards
    against a future refactor that mutates src in place and breaks vision
    convert."""
    src = {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "vision_config": {"depth": 27, "hidden_size": 1152},
        "text_config": {"hidden_size": 2048, "mtp_num_hidden_layers": 1},
    }
    snapshot = json.dumps(src, sort_keys=True)
    _ = filter_config_json(src, drop_mtp=True)
    assert json.dumps(src, sort_keys=True) == snapshot

