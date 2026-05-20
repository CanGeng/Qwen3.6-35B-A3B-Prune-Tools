"""Qwen MoE adapters: Qwen2 MoE (``model.layers``) + Qwen3.5 MoE VL (``model.language_model`` + stacked experts)."""

from __future__ import annotations

import copy
import re

from moe_prune_distill.adapters.base import MoEAdapter, TensorInfo

# --- Legacy Qwen2 / Qwen1.5 MoE (per-expert keys) ---
_RE_LAYER = re.compile(r"^model\.layers\.(\d+)\.")
_RE_ROUTER = re.compile(r"^model\.layers\.(\d+)\.mlp\.gate\.(weight|bias)$")
_RE_EXPERT = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(.+)$")
_RE_SHARED = re.compile(r"^model\.layers\.(\d+)\.mlp\.shared_expert\.(.+)$")

# --- Qwen3.5 MoE (stacked expert tensors under language_model) ---
_RE_LM_ROUTER = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.gate\.(weight|bias)$"
)
_RE_LM_EXPERT_STACK = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(down_proj|gate_up_proj)$"
)
_RE_LM_SHARED = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.shared_expert\.(.+)$"
)
_RE_LM_LAYER = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)$")


class QwenMoeAdapter(MoEAdapter):
    """HF Qwen2 MoE (CausalLM) + Qwen3.5 MoE (multimodal, stacked expert weights)."""

    @staticmethod
    def _flat_moe_config(config: dict) -> dict:
        """Return the dict that holds ``num_experts`` / ``num_hidden_layers`` (``text_config`` or root)."""
        tc = config.get("text_config")
        if isinstance(tc, dict) and tc.get("num_experts") is not None:
            return tc
        return config

    def detect(self, config: dict) -> bool:
        mt = config.get("model_type")
        if mt in ("qwen2_moe", "qwen_moe"):
            return bool(config.get("num_experts"))
        if mt in ("qwen3_5_moe", "qwen3_5_moe_text"):
            return bool(self._flat_moe_config(config).get("num_experts"))
        arch = config.get("architectures") or []
        names = (
            "Qwen2MoeForCausalLM",
            "Qwen2MoeModel",
            "Qwen1.5MoEForCausalLM",
            "Qwen3_5MoeForConditionalGeneration",
            "Qwen3_5MoeModel",
        )
        return any(name in arch for name in names)

    def get_num_layers(self, config: dict) -> int:
        return int(self._flat_moe_config(config)["num_hidden_layers"])

    def get_num_experts(self, config: dict) -> int:
        return int(self._flat_moe_config(config)["num_experts"])

    def get_num_experts_per_tok(self, config: dict) -> int:
        c = self._flat_moe_config(config)
        v = c.get("num_experts_per_tok")
        if v is not None:
            return int(v)
        v2 = c.get("moe_topk")
        if v2 is not None:
            return int(v2)
        return 1

    def has_shared_expert(self, config: dict) -> bool:
        c = self._flat_moe_config(config)
        if c.get("shared_expert_intermediate_size"):
            return int(c["shared_expert_intermediate_size"]) > 0
        return bool(c.get("use_shared_expert", False))

    def get_router_key_pattern(self) -> str:
        return (
            r"(model\.language_model\.layers\.\d+\.mlp\.gate\.(weight|bias)|"
            r"model\.layers\.\d+\.mlp\.gate\.(weight|bias))"
        )

    def parse_state_dict_key(self, key: str) -> TensorInfo:
        # --- Qwen3.5 multimodal language trunk ---
        if key == "model.language_model.embed_tokens.weight":
            return TensorInfo(None, "embedding", None, None)
        if key == "lm_head.weight":
            return TensorInfo(None, "lm_head", None, None)

        m = _RE_LM_ROUTER.match(key)
        if m:
            return TensorInfo(int(m.group(1)), "router", None, m.group(2))

        m = _RE_LM_EXPERT_STACK.match(key)
        if m:
            return TensorInfo(
                int(m.group(1)), "routed_expert_stack", None, m.group(2)
            )

        m = _RE_LM_SHARED.match(key)
        if m:
            return TensorInfo(int(m.group(1)), "shared_expert", None, m.group(2))

        m = _RE_LM_LAYER.match(key)
        if m:
            layer = int(m.group(1))
            tail = m.group(2)
            if tail.startswith("self_attn") or tail.startswith("linear_attn"):
                return TensorInfo(layer, "attention", None, tail)
            if "norm" in tail or "layernorm" in tail:
                return TensorInfo(layer, "layernorm", None, tail)
            return TensorInfo(layer, "other", None, tail)

        # --- Legacy Qwen2 MoE ---
        if key == "model.embed_tokens.weight":
            return TensorInfo(None, "embedding", None, None)

        m = _RE_ROUTER.match(key)
        if m:
            layer = int(m.group(1))
            return TensorInfo(layer, "router", None, m.group(2))

        m = _RE_EXPERT.match(key)
        if m:
            layer = int(m.group(1))
            eid = int(m.group(2))
            return TensorInfo(layer, "routed_expert", eid, m.group(3))

        m = _RE_SHARED.match(key)
        if m:
            layer = int(m.group(1))
            return TensorInfo(layer, "shared_expert", None, m.group(2))

        m = _RE_LAYER.match(key)
        if m:
            layer = int(m.group(1))
            rest = key.split(".", 3)[-1]
            if "self_attn" in key or "attention" in key:
                return TensorInfo(layer, "attention", None, rest)
            if "norm" in key or "ln" in key:
                return TensorInfo(layer, "layernorm", None, rest)
            return TensorInfo(layer, "other", None, rest)

        if "norm" in key:
            return TensorInfo(None, "layernorm", None, key)
        return TensorInfo(None, "other", None, key)

    def rename_expert_key(self, key: str, old_id: int, new_id: int) -> str:
        return key.replace(f".experts.{old_id}.", f".experts.{new_id}.")

    def modify_config(self, config: dict, num_experts: int, num_experts_per_tok: int) -> dict:
        out = copy.deepcopy(config)
        if isinstance(out.get("text_config"), dict):
            tc = out["text_config"]
            tc["num_experts"] = int(num_experts)
            tc["num_experts_per_tok"] = int(num_experts_per_tok)
            if "moe_topk" in tc:
                tc["moe_topk"] = int(num_experts_per_tok)
        else:
            out["num_experts"] = int(num_experts)
            out["num_experts_per_tok"] = int(num_experts_per_tok)
            if "moe_topk" in out:
                out["moe_topk"] = int(num_experts_per_tok)
        return out
