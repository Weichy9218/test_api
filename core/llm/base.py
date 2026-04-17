"""
Base abstractions shared by the lightweight LLM clients.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LLMResponse:
    """Normalized LLM response used by the smoke-test scripts."""

    content: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    usage: Dict[str, int] = field(
        default_factory=lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    )
    model: str = ""
    finish_reason: str = ""
    message: Optional[dict] = None

    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    def get_total_tokens(self) -> int:
        return self.usage.get("total_tokens", 0)

    def is_truncated(self) -> bool:
        return self.finish_reason == "length"


class BaseLLMClient(ABC):
    """Shared interface for lightweight LLM clients."""

    def __init__(self, model: str, temperature: Optional[float] = 0.7, **kwargs: Any):
        self.model = model
        self.temperature = temperature
        self.extra_params = kwargs
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_requests = 0

    def get_usage_stats(self) -> Dict[str, int]:
        return {
            "total_prompt_tokens": self._total_prompt_tokens,
            "total_completion_tokens": self._total_completion_tokens,
            "total_tokens": self._total_prompt_tokens + self._total_completion_tokens,
            "total_requests": self._total_requests,
        }

    def _update_usage_stats(self, usage: Dict[str, int]) -> None:
        self._total_prompt_tokens += usage.get("prompt_tokens", 0)
        self._total_completion_tokens += usage.get("completion_tokens", 0)
        self._total_requests += 1

    def estimate_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Use a simple character heuristic for quick token estimates."""
        total_chars = 0
        for msg in messages:
            try:
                total_chars += len(json.dumps(msg, ensure_ascii=False))
            except Exception:
                total_chars += len(str(msg))
        return total_chars // 4

    async def aclose(self) -> None:
        """Allow subclasses to release network resources."""
        return None

    @abstractmethod
    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send chat messages to the backend and return a normalized response."""

    @abstractmethod
    async def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ):
        """Yield normalized streaming chunks from the backend."""

    def format_system_message(self, content: str) -> Dict[str, str]:
        return {"role": "system", "content": content}

    def format_user_message(self, content: str) -> Dict[str, str]:
        return {"role": "user", "content": content}

    def format_assistant_message(self, content: str) -> Dict[str, str]:
        return {"role": "assistant", "content": content}

    def supports_response_format(self) -> bool:
        return False


LLM_CLIENTS: Dict[str, type] = {}


def register_llm_client_class(name: str):
    """Register a client class under a simple string identifier."""

    def register_cls(cls):
        LLM_CLIENTS[name] = cls
        return cls

    return register_cls
