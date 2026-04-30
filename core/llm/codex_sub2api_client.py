"""
Sub2API / codex_sub2api Responses API client.

OpenAI Responses-compatible client for the Sub2API gateway.
Reads credentials from env vars (priority order):
  API key : SUB2API_API_KEY > HAOXIANG_OPENAI_API_KEY > HAOXIANG_API_KEY
  Base URL: SUB2API_BASE_URL > HAOXIANG_BASE_URL > DEFAULT_BASE_URL
"""

from __future__ import annotations

from typing import Any, Optional

from .base import register_llm_client
from .env_utils import load_env, resolve_client_setting
from .base import LLMResponse
from .openai_client import OpenAIClient

load_env()


@register_llm_client("codex_sub2api", aliases=("CodexSub2API", "sub2api", "Sub2API"))
class CodexSub2APIClient(OpenAIClient):
    """OpenAI Responses-compatible client for the Sub2API gateway."""

    DEFAULT_BASE_URL = "https://ie-crs.haoxiang.ai/v1"
    DEFAULT_MODEL = "gpt-5.4"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: Optional[float] = 0.2,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key_env: Optional[str] = None,
        base_url_env: Optional[str] = None,
        max_tokens: Optional[int] = 4096,
        reasoning_effort: Optional[str] = "high",
        async_mode: bool = True,
        **kwargs: Any,
    ) -> None:
        resolved_api_key, _ = resolve_client_setting(
            api_key,
            preferred_env=api_key_env,
            fallback_envs=(
                "SUB2API_API_KEY",
                "HAOXIANG_OPENAI_API_KEY",
                "HAOXIANG_API_KEY",
            ),
        )
        resolved_base_url, _ = resolve_client_setting(
            base_url,
            preferred_env=base_url_env,
            fallback_envs=(
                "SUB2API_BASE_URL",
                "HAOXIANG_BASE_URL",
            ),
            default=self.DEFAULT_BASE_URL,
        )
        if not resolved_api_key:
            raise ValueError(
                "CodexSub2APIClient requires an API key. "
                "Set SUB2API_API_KEY in your .env file."
            )

        super().__init__(
            model=model,
            temperature=temperature,
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            async_mode=async_mode,
            **kwargs,
        )
