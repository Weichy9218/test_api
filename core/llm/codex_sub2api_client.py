"""
codex_sub2api Responses API client.

Minimal OpenAI Responses-compatible client wired to the codex_sub2api gateway.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from .base import LLMResponse, register_llm_client_class
from .env_utils import load_env
from .openai_client import OpenAIClient

load_env()


@register_llm_client_class("codex_sub2api")
class CodexSub2APIClient(OpenAIClient):
    """OpenAI Responses-compatible client for the codex_sub2api GPT-5.4 route."""

    DEFAULT_BASE_URL = "https://ie-crs.haoxiang.ai/v1"
    DEFAULT_MODEL = "gpt-5.4"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: Optional[float] = 0.2,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: Optional[int] = 4096,
        reasoning_effort: Optional[str] = "high",
        async_mode: bool = True,
        **kwargs: Any,
    ) -> None:
        resolved_api_key = api_key or os.getenv("HAOXIANG_OPENAI_API_KEY") or os.getenv("HAOXIANG_API_KEY")
        resolved_base_url = base_url or os.getenv("HAOXIANG_BASE_URL") or self.DEFAULT_BASE_URL
        if not resolved_api_key:
            raise ValueError("HAOXIANG_OPENAI_API_KEY or HAOXIANG_API_KEY must be set for CodexSub2APIClient")

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

    def __repr__(self) -> str:
        return f"CodexSub2APIClient(model={self.model}, temperature={self.temperature})"

    @staticmethod
    def _event_get(obj: Any, key: str) -> Any:
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    async def chat(
        self,
        messages,
        tools=None,
        response_format=None,
        **kwargs,
    ) -> LLMResponse:
        """
        Use the Responses streaming path because this backend can return empty
        non-streaming payloads while still emitting correct stream deltas.
        """
        if not self.async_mode:
            return await super().chat(messages=messages, tools=tools, response_format=response_format, **kwargs)

        tool_choice = kwargs.pop("tool_choice", None)
        reasoning_effort = self._normalize_reasoning_effort(kwargs.pop("reasoning_effort", None))
        max_completion_tokens = kwargs.pop("max_completion_tokens", None)
        if max_completion_tokens is not None and "max_output_tokens" not in kwargs:
            kwargs["max_output_tokens"] = max_completion_tokens

        params = {
            "model": self.model,
            "input": self._normalize_input_messages(messages),
            **self.extra_params,
            **kwargs,
        }
        if self.max_tokens is not None and "max_output_tokens" not in params:
            params["max_output_tokens"] = self.max_tokens
        if self.temperature is not None and self._supports_temperature():
            params["temperature"] = self.temperature
        if "reasoning" not in params:
            effort = reasoning_effort or self.reasoning_effort
            if effort:
                params["reasoning"] = {"effort": effort}

        merged_tools = self._build_tools(tools)
        if merged_tools:
            params["tools"] = merged_tools
            params["tool_choice"] = tool_choice or "auto"
        if response_format is not None:
            params["response_format"] = response_format

        text_chunks: list[str] = []
        tool_calls_by_item_id: dict[str, dict[str, Any]] = {}

        async with self.client.responses.stream(**params) as stream:
            async for event in stream:
                event_type = self._event_get(event, "type")

                if event_type == "response.output_text.delta":
                    delta = self._event_get(event, "delta")
                    if delta:
                        text_chunks.append(str(delta))
                    continue

                if event_type == "response.output_item.added":
                    item = self._event_get(event, "item")
                    if self._event_get(item, "type") != "function_call":
                        continue
                    item_id = str(self._event_get(item, "id") or "")
                    tool_calls_by_item_id[item_id] = {
                        "id": self._event_get(item, "call_id") or item_id,
                        "name": self._event_get(item, "name") or "",
                        "arguments": str(self._event_get(item, "arguments") or ""),
                    }
                    continue

                if event_type == "response.function_call_arguments.delta":
                    item_id = str(self._event_get(event, "item_id") or "")
                    if not item_id:
                        continue
                    record = tool_calls_by_item_id.setdefault(item_id, {"id": item_id, "name": "", "arguments": ""})
                    record["arguments"] += str(self._event_get(event, "delta") or "")
                    continue

                if event_type == "response.function_call_arguments.done":
                    item_id = str(self._event_get(event, "item_id") or "")
                    if not item_id:
                        continue
                    record = tool_calls_by_item_id.setdefault(item_id, {"id": item_id, "name": "", "arguments": ""})
                    final_args = self._event_get(event, "arguments")
                    if final_args is not None:
                        record["arguments"] = str(final_args)
                    continue

                if event_type == "response.output_item.done":
                    item = self._event_get(event, "item")
                    if self._event_get(item, "type") != "function_call":
                        continue
                    item_id = str(self._event_get(item, "id") or "")
                    record = tool_calls_by_item_id.setdefault(item_id, {"id": item_id, "name": "", "arguments": ""})
                    record["id"] = self._event_get(item, "call_id") or record["id"]
                    record["name"] = self._event_get(item, "name") or record["name"]
                    item_args = self._event_get(item, "arguments")
                    if item_args is not None:
                        record["arguments"] = str(item_args)

            final_response = await stream.get_final_response()

        content = "".join(text_chunks)
        if not content:
            content = self._extract_text(final_response)

        tool_calls = []
        for record in tool_calls_by_item_id.values():
            if not record.get("name"):
                continue
            tool_calls.append(
                {
                    "id": record["id"],
                    "type": "function",
                    "function": {
                        "name": record["name"],
                        "arguments": record.get("arguments", ""),
                    },
                }
            )
        if not tool_calls:
            tool_calls = self._extract_tool_calls(final_response)

        usage = self._extract_usage(final_response)
        self._update_usage_stats(usage)

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            model=getattr(final_response, "model", self.model),
            finish_reason=getattr(final_response, "status", ""),
            message={
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls or None,
            },
        )
