"""MoE architecture adapters."""

from moe_prune_distill.adapters.base import MoEAdapter, TensorInfo
from moe_prune_distill.adapters.qwen_moe import QwenMoeAdapter

_ADAPTERS: list[type[MoEAdapter]] = [QwenMoeAdapter]


def register_adapter(cls: type[MoEAdapter]) -> None:
    if cls not in _ADAPTERS:
        _ADAPTERS.append(cls)


def detect_adapter(config: dict) -> MoEAdapter:
    for cls in _ADAPTERS:
        a = cls()
        if a.detect(config):
            return a
    raise ValueError("Unsupported MoE architecture: no adapter matched config.json")


__all__ = ["MoEAdapter", "TensorInfo", "QwenMoeAdapter", "detect_adapter", "register_adapter"]
