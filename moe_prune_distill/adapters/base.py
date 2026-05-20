from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal


@dataclass
class TensorInfo:
    layer: int | None
    type: Literal[
        "embedding",
        "lm_head",
        "layernorm",
        "attention",
        "router",
        "routed_expert",
        "routed_expert_stack",
        "shared_expert",
        "other",
    ]
    expert_id: int | None = None
    sub_key: str | None = None


class MoEAdapter(ABC):
    """Each MoE family implements one adapter."""

    @abstractmethod
    def detect(self, config: dict) -> bool:
        """Return True if config.json belongs to this architecture."""

    @abstractmethod
    def get_num_layers(self, config: dict) -> int:
        ...

    @abstractmethod
    def get_num_experts(self, config: dict) -> int:
        ...

    @abstractmethod
    def get_num_experts_per_tok(self, config: dict) -> int:
        ...

    @abstractmethod
    def has_shared_expert(self, config: dict) -> bool:
        ...

    @abstractmethod
    def parse_state_dict_key(self, key: str) -> TensorInfo:
        ...

    @abstractmethod
    def rename_expert_key(self, key: str, old_id: int, new_id: int) -> str:
        ...

    @abstractmethod
    def get_router_key_pattern(self) -> str:
        ...

    @abstractmethod
    def modify_config(self, config: dict, num_experts: int, num_experts_per_tok: int) -> dict:
        """Return a new config dict with expert-related fields updated."""
