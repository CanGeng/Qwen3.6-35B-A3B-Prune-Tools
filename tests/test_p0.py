import json
from pathlib import Path

import torch

from moe_prune_distill.adapters.qwen_moe import QwenMoeAdapter
from moe_prune_distill.prune.slicer import _process_tensor, build_expert_mapping_json


def test_qwen_adapter_parse_router_and_expert():
    a = QwenMoeAdapter()
    k_gate = "model.layers.3.mlp.gate.weight"
    info = a.parse_state_dict_key(k_gate)
    assert info.type == "router" and info.layer == 3
    k_exp = "model.layers.2.mlp.experts.7.gate_proj.weight"
    info2 = a.parse_state_dict_key(k_exp)
    assert info2.type == "routed_expert" and info2.expert_id == 7
    assert a.rename_expert_key(k_exp, 7, 0) == "model.layers.2.mlp.experts.0.gate_proj.weight"


def test_qwen_modify_config():
    a = QwenMoeAdapter()
    cfg = {"num_experts": 60, "num_experts_per_tok": 4, "num_hidden_layers": 24}
    out = a.modify_config(cfg, 30, 2)
    assert out["num_experts"] == 30 and out["num_experts_per_tok"] == 2


def test_qwen35_nested_config_and_stack_slice():
    a = QwenMoeAdapter()
    hf = {
        "model_type": "qwen3_5_moe",
        "text_config": {
            "num_hidden_layers": 40,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "shared_expert_intermediate_size": 512,
        },
    }
    assert a.detect(hf)
    assert a.get_num_experts(hf) == 256
    assert a.get_num_layers(hf) == 40
    k = "model.language_model.layers.0.mlp.experts.down_proj"
    assert a.parse_state_dict_key(k).type == "routed_expert_stack"
    t = torch.randn(256, 2, 2)
    out = _process_tensor(k, t, a, 256, 64, True)
    assert out is not None and out[1].shape == (64, 2, 2)
    g = torch.randn(256, 8)
    og = _process_tensor("model.language_model.layers.1.mlp.gate.weight", g, a, 256, 64, True)
    assert og is not None and og[1].shape == (64, 8)


def test_qwen35_modify_nested_text_config():
    a = QwenMoeAdapter()
    hf = {
        "model_type": "qwen3_5_moe",
        "text_config": {
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "num_hidden_layers": 1,
            "moe_topk": 8,
        },
    }
    out = a.modify_config(hf, 64, 4)
    assert out["text_config"]["num_experts"] == 64
    assert out["text_config"]["num_experts_per_tok"] == 4
    assert out["text_config"]["moe_topk"] == 4


def test_slicer_router_slice_and_drop_expert():
    a = QwenMoeAdapter()
    old_n = 8
    target = 4
    w = torch.randn(old_n, 16)
    out = _process_tensor("model.layers.0.mlp.gate.weight", w, a, old_n, target, True)
    assert out is not None and out[1].shape == (4, 16)
    dropped = _process_tensor(
        "model.layers.0.mlp.experts.6.gate_proj.weight",
        torch.randn(3, 3),
        a,
        old_n,
        target,
        True,
    )
    assert dropped is None


def test_expert_mapping_json():
    m = build_expert_mapping_json(num_layers=2, old_num_experts=8, target_num_experts=3)
    assert "layer_0" in m and m["layer_0"]["surviving_original_ids"] == [0, 1, 2]


def test_prune_sharded_roundtrip(tmp_path: Path):
    from moe_prune_distill.prune.slicer import prune_state_dict_sharded

    teacher = tmp_path / "teacher"
    teacher.mkdir()
    hf = {
        "model_type": "qwen2_moe",
        "num_hidden_layers": 2,
        "num_experts": 4,
        "num_experts_per_tok": 2,
    }
    (teacher / "config.json").write_text(json.dumps(hf), encoding="utf-8")
    shard = {
        "model.layers.0.mlp.gate.weight": torch.randn(4, 8),
        "model.layers.0.mlp.experts.0.gate_proj.weight": torch.randn(2, 2),
        "model.layers.0.mlp.experts.3.gate_proj.weight": torch.randn(2, 2),
        "model.embed_tokens.weight": torch.randn(10, 8),
    }
    from safetensors.torch import save_file

    save_file(shard, str(teacher / "model.safetensors"))

    student = tmp_path / "student"
    a = QwenMoeAdapter()
    prune_state_dict_sharded(teacher, student, a, hf, target_num_experts=2, keep_shared_experts=True)
    from safetensors.torch import load_file

    out = load_file(str(student / "model.safetensors"))
    assert out["model.layers.0.mlp.gate.weight"].shape[0] == 2
    assert "model.layers.0.mlp.experts.0.gate_proj.weight" in out
    assert "model.layers.0.mlp.experts.3.gate_proj.weight" not in out
