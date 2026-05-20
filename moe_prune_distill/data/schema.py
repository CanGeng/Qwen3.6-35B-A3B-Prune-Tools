"""JSONL training sample schema."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TrainSample:
    id: str
    source: str | None
    messages: list[dict[str, str]] | None
    text: str | None
    images: list[str] | None = None

    @staticmethod
    def from_dict(row: dict[str, Any]) -> "TrainSample":
        sid = str(row.get("id") or row.get("sample_id") or "")
        if not sid:
            raise ValueError("Each sample must have a unique string 'id'")
        messages = row.get("messages")
        text = row.get("text")
        if messages is None and not text:
            raise ValueError("Sample must have 'messages' or 'text'")
        if messages is not None and text:
            raise ValueError("Sample cannot have both 'messages' and 'text'")
        if messages is not None:
            if not isinstance(messages, list):
                raise ValueError("messages must be a list")
            for m in messages:
                if not isinstance(m, dict) or "role" not in m or "content" not in m:
                    raise ValueError("Each message needs role and content")
        images = row.get("images")
        if images is not None:
            if not isinstance(images, list) or not all(isinstance(p, str) for p in images):
                raise ValueError("images must be a list of file paths")
            if not images:
                images = None
        return TrainSample(
            id=sid,
            source=row.get("source"),
            messages=messages,
            text=text,
            images=images,
        )
