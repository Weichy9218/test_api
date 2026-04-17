"""
OpenAI Responses API client used as the base for the codex_sub2api adapter.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from openai import APITimeoutError, AsyncOpenAI, OpenAI

from .base import BaseLLMClient, LLMResponse, register_llm_client_class
from .env_utils import load_env

load_env()


@register_llm_client_class("OpenAI")
class OpenAIClient(BaseLLMClient):
    """Thin wrapper around the OpenAI SDK Responses API."""

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        temperature: Optional[float] = 0.7,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: Optional[int] = 4096,
        use_web_search: bool = False,
        web_search_type: str = "web_search",
        web_search_options: Optional[Dict[str, Any]] = None,
        reasoning_effort: Optional[str] = None,
        async_mode: bool = True,
        **kwargs: Any,
    ):
        super().__init__(model, temperature, **kwargs)

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.max_tokens = max_tokens
        self.async_mode = async_mode
        self.use_web_search = use_web_search
        self.web_search_type = web_search_type
        self.web_search_options = web_search_options
        self.reasoning_effort = self._normalize_reasoning_effort(reasoning_effort)

        if async_mode:
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, timeout=180)
        else:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=180)

    def _build_web_search_tool(self) -> Dict[str, Any]:
        tool: Dict[str, Any] = {"type": self.web_search_type}
        if isinstance(self.web_search_options, dict):
            for key, value in self.web_search_options.items():
                if key != "type":
                    tool[key] = value
        return tool

    @staticmethod
    def _normalize_input_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role") or "user"
            if role == "tool":
                role = "user"
            content = msg.get("content")
            target_type = "output_text" if role in {"assistant", "tool"} else "input_text"

            base: Dict[str, Any] = {"role": role}
            if isinstance(content, str):
                base["content"] = [{"type": target_type, "text": content}]
                normalized.append(base)
                continue

            if isinstance(content, list):
                parts: List[Dict[str, Any]] = []
                for part in content:
                    if isinstance(part, dict):
                        part_type = part.get("type")
                        if part_type == "input_text" and target_type == "output_text":
                            part = dict(part)
                            part["type"] = "output_text"
                        elif part_type == "output_text" and target_type == "input_text":
                            part = dict(part)
                            part["type"] = "input_text"
                        parts.append(part)
                    elif isinstance(part, str):
                        parts.append({"type": target_type, "text": part})
                base["content"] = parts
                normalized.append(base)
                continue

            base["content"] = content
            normalized.append(base)
        return normalized

    def _build_tools(self, tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        merged_tools: List[Dict[str, Any]] = []
        if tools:
            merged_tools.extend(tools)
        if self.use_web_search:
            existing = None
            for tool in merged_tools:
                if isinstance(tool, dict) and tool.get("type") == self.web_search_type:
                    existing = tool
                    break
            if existing is None:
                merged_tools.append(self._build_web_search_tool())
            elif isinstance(self.web_search_options, dict):
                for key, value in self.web_search_options.items():
                    if key != "type" and key not in existing:
                        existing[key] = value
        return self._normalize_tools_for_responses(merged_tools) if merged_tools else None

    @staticmethod
    def _normalize_tools_for_responses(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert chat-completions style function tools to Responses API format."""
        normalized: List[Dict[str, Any]] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            tool_type = tool.get("type")
            if tool_type and tool_type != "function":
                normalized.append(tool)
                continue
            if "name" in tool:
                normalized.append(tool)
                continue

            function = tool.get("function")
            if isinstance(function, dict):
                name = function.get("name")
                if not name:
                    continue
                normalized.append(
                    {
                        "type": "function",
                        "name": name,
                        "description": function.get("description"),
                        "parameters": function.get("parameters") or {},
                    }
                )
                continue

            normalized.append(tool)
        return normalized

    @staticmethod
    def _normalize_reasoning_effort(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        effort = str(value).strip().lower()
        if not effort or effort in {"none", "off", "disabled"}:
            return None
        return effort

    def _extract_text(self, response: Any) -> str:
        output_text = getattr(response, "output_text", None)
        if output_text is not None:
            return output_text

        output = getattr(response, "output", None) or []
        chunks: List[str] = []
        for item in output:
            item_type = getattr(item, "type", None) or (item.get("type") if isinstance(item, dict) else None)
            if item_type in {"output_text", "text", "refusal"}:
                text = getattr(item, "text", None) or (item.get("text") if isinstance(item, dict) else None)
                if isinstance(text, dict):
                    text = text.get("value")
                if text:
                    chunks.append(text)
                continue
            if item_type != "message":
                continue
            content = getattr(item, "content", None) or (item.get("content") if isinstance(item, dict) else None) or []
            for part in content:
                part_type = getattr(part, "type", None) or (part.get("type") if isinstance(part, dict) else None)
                if part_type not in {"output_text", "text", "refusal"}:
                    continue
                text = getattr(part, "text", None) or (part.get("text") if isinstance(part, dict) else None)
                if isinstance(text, dict):
                    text = text.get("value")
                if text:
                    chunks.append(text)
        return "".join(chunks)

    def _extract_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        tool_calls: List[Dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()

        def _get(obj: Any, key: str) -> Any:
            if isinstance(obj, dict):
                return obj.get(key)
            return getattr(obj, key, None)

        def _record(name: Optional[str], arguments: Any, call_id: Optional[str]) -> None:
            if not name:
                return
            dedupe_key = (call_id or "", name, str(arguments))
            if dedupe_key in seen:
                return
            seen.add(dedupe_key)
            tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments,
                    },
                }
            )

        def _extract_from_obj(obj: Any) -> None:
            if obj is None:
                return

            tool_calls_field = _get(obj, "tool_calls")
            if tool_calls_field:
                for tool_call in tool_calls_field:
                    function = _get(tool_call, "function")
                    name = _get(function, "name") or _get(tool_call, "name") or _get(tool_call, "tool_name")
                    arguments = _get(function, "arguments") if function else (_get(tool_call, "arguments") or _get(tool_call, "input"))
                    call_id = _get(tool_call, "id") or _get(tool_call, "call_id")
                    _record(name, arguments, call_id)

            item_type = _get(obj, "type")
            name = _get(obj, "name") or _get(obj, "tool_name")
            arguments = _get(obj, "arguments") or _get(obj, "input")
            call_id = _get(obj, "id") or _get(obj, "call_id")
            if item_type in {"tool_call", "function_call"} or (name and arguments is not None and item_type is None):
                _record(name, arguments, call_id)

            content = _get(obj, "content")
            if isinstance(content, list):
                for part in content:
                    _extract_from_obj(part)

        output = getattr(response, "output", None)
        if output is None and isinstance(response, dict):
            output = response.get("output")
        if not output:
            return tool_calls

        for item in output:
            _extract_from_obj(item)

        return tool_calls

    def _extract_usage(self, response: Any) -> Dict[str, int]:
        usage_obj = getattr(response, "usage", None)
        prompt_tokens = getattr(usage_obj, "input_tokens", 0) if usage_obj else 0
        completion_tokens = getattr(usage_obj, "output_tokens", 0) if usage_obj else 0
        total_tokens = getattr(usage_obj, "total_tokens", 0) if usage_obj else 0
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    def supports_response_format(self) -> bool:
        return True

    def _supports_temperature(self) -> bool:
        model_name = str(self.model or "").strip().lower()
        return not model_name.startswith("gpt-5")

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
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

        response = None
        last_exc: Optional[Exception] = None
        for _ in range(3):
            try:
                if self.async_mode:
                    response = await self.client.responses.create(**params)
                else:
                    response = self.client.responses.create(**params)
                break
            except APITimeoutError:
                raise
            except TypeError as exc:
                if "response_format" not in str(exc):
                    raise
                params.pop("response_format", None)
                last_exc = exc
                continue
            except Exception as exc:
                if "reasoning.effort" not in str(exc) and "reasoning" not in str(exc):
                    raise
                params.pop("reasoning", None)
                last_exc = exc
                continue

        if response is None:
            raise last_exc or RuntimeError("LLM returned empty response")

        content = self._extract_text(response)
        tool_calls = self._extract_tool_calls(response)
        usage = self._extract_usage(response)
        self._update_usage_stats(usage)

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            model=getattr(response, "model", self.model),
            finish_reason=getattr(response, "status", ""),
            message={
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls or None,
            },
        )

    async def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ):
        yield await self.chat(messages=messages, tools=tools, **kwargs)

    async def aclose(self) -> None:
        if not getattr(self, "client", None):
            return
        if self.async_mode and hasattr(self.client, "close"):
            await self.client.close()
        elif hasattr(self.client, "close"):
            self.client.close()
        self.client = None

    def __repr__(self) -> str:
        return f"OpenAIClient(model={self.model}, temperature={self.temperature})"
