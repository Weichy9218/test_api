"""
OpenAI Client

OpenAI SDK client using the Responses API.
"""

import os
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI, OpenAI, APITimeoutError

from .base import LLMResponse, OpenAISDKClient, register_llm_client
from .env_utils import load_env

# Load environment variables
load_env()


@register_llm_client("openai", aliases=("OpenAI",))
class OpenAIClient(OpenAISDKClient):
    """
    OpenAI client using the official SDK.

    Features:
    - Responses API
    - Optional tool calling (including web_search)
    - Streaming support (single-chunk fallback)
    """

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
        **kwargs
    ):
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        resolved_base_url = base_url or os.getenv("OPENAI_BASE_URL")
        super().__init__(
            model=model,
            temperature=temperature,
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            async_mode=async_mode,
            timeout_seconds=180,
            **kwargs,
        )
        self.max_tokens = max_tokens
        self.use_web_search = use_web_search
        self.web_search_type = web_search_type
        self.web_search_options = web_search_options
        self.reasoning_effort = self._normalize_reasoning_effort(reasoning_effort)

    def _sdk_client_classes(self):
        return AsyncOpenAI, OpenAI

    def _build_web_search_tool(self) -> Dict[str, Any]:
        tool: Dict[str, Any] = {"type": self.web_search_type}
        if isinstance(self.web_search_options, dict):
            for key, value in self.web_search_options.items():
                if key == "type":
                    continue
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
            if role == "tool":
                tool_call_id = msg.get("tool_call_id")
                if tool_call_id:
                    base["tool_call_id"] = tool_call_id
                name = msg.get("name")
                if name:
                    base["name"] = name

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

    def _build_tools(
        self,
        tools: Optional[List[Dict[str, Any]]]
    ) -> Optional[List[Dict[str, Any]]]:
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
                    if key == "type" or key in existing:
                        continue
                    existing[key] = value
        return self._normalize_tools_for_responses(merged_tools) if merged_tools else None

    @staticmethod
    def _normalize_tool_choice_for_responses(tool_choice: Any) -> Any:
        """Accept Chat Completions-style forced function choices on Responses clients."""
        if not isinstance(tool_choice, dict):
            return tool_choice
        if tool_choice.get("type") != "function" or "name" in tool_choice:
            return tool_choice
        function = tool_choice.get("function")
        if not isinstance(function, dict):
            return tool_choice
        name = function.get("name")
        if not name:
            return tool_choice
        return {"type": "function", "name": name}

    @staticmethod
    def _normalize_tools_for_responses(
        tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Normalize tool schema to Responses API format.

        OpenAI Responses API expects function tools with top-level name/description/parameters.
        """
        normalized: List[Dict[str, Any]] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            tool_type = tool.get("type")
            # Keep web_search or other non-function tools as-is
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

    def _extract_text(self, response: Any) -> str:
        output_text = getattr(response, "output_text", None)
        if output_text is not None:
            return output_text

        output = getattr(response, "output", None) or []
        chunks: List[str] = []
        for item in output:
            item_type = self.get_field_value(item, "type")
            if item_type in {"output_text", "text", "refusal"}:
                text = self.get_field_value(item, "text")
                if isinstance(text, dict):
                    text = text.get("value")
                if text:
                    chunks.append(text)
                continue
            if item_type != "message":
                continue
            content = self.get_field_value(item, "content") or []
            for part in content:
                part_type = self.get_field_value(part, "type")
                if part_type not in {"output_text", "text", "refusal"}:
                    continue
                text = self.get_field_value(part, "text")
                if isinstance(text, dict):
                    text = text.get("value")
                if text:
                    chunks.append(text)
        return "".join(chunks)

    def _extract_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        tool_calls: List[Dict[str, Any]] = []
        seen: set = set()

        def _record(name: Optional[str], arguments: Any, call_id: Optional[str]) -> None:
            if not name:
                return
            key = (call_id or "", name, str(arguments))
            if key in seen:
                return
            seen.add(key)
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

            tool_calls_field = self.get_field_value(obj, "tool_calls")
            if tool_calls_field:
                for tc in tool_calls_field:
                    fn = self.get_field_value(tc, "function")
                    name = self.get_field_value(fn, "name") or self.get_field_value(tc, "name") or self.get_field_value(tc, "tool_name")
                    args = self.get_field_value(fn, "arguments") if fn else (self.get_field_value(tc, "arguments") or self.get_field_value(tc, "input"))
                    call_id = self.get_field_value(tc, "id") or self.get_field_value(tc, "call_id")
                    _record(name, args, call_id)

            item_type = self.get_field_value(obj, "type")
            name = self.get_field_value(obj, "name") or self.get_field_value(obj, "tool_name")
            args = self.get_field_value(obj, "arguments") or self.get_field_value(obj, "input")
            call_id = self.get_field_value(obj, "id") or self.get_field_value(obj, "call_id")
            if item_type in {"tool_call", "function_call"} or (name and args is not None and item_type is None):
                _record(name, args, call_id)

            content = self.get_field_value(obj, "content")
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

    def _build_responses_request_params(
        self,
        *,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Assemble one Responses API request."""
        tool_choice = kwargs.pop("tool_choice", None)
        tool_choice = self._normalize_tool_choice_for_responses(tool_choice)
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
        return params

    async def _create_response(self, params: Dict[str, Any]):
        """Create one Responses API call with compatibility fallbacks."""
        response = None
        last_exc: Optional[Exception] = None
        for _ in range(3):
            try:
                if self.async_mode:
                    response = await self.client.responses.create(**params)
                else:
                    response = self.client.responses.create(**params)
                break
            except APITimeoutError as exc:
                import logging

                logger = logging.getLogger(__name__)
                logger.error(f"OpenAI API timeout: {exc}")
                logger.error(f"Model: {self.model}")
                raise
            except TypeError as exc:
                if "response_format" not in str(exc):
                    raise
                params = dict(params)
                params.pop("response_format", None)
                last_exc = exc
                continue
            except Exception as exc:
                message = str(exc)
                if "reasoning.effort" not in message and "reasoning" not in message:
                    raise
                params = dict(params)
                params.pop("reasoning", None)
                last_exc = exc
                continue

        if response is None:
            raise last_exc or Exception("LLM returned empty response")
        return response

    def _build_responses_llm_response(
        self,
        response: Any,
        *,
        content: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Normalize a Responses API payload into the repo's shared response shape."""
        resolved_content = self._extract_text(response) if content is None else content
        resolved_tool_calls = self._extract_tool_calls(response) if tool_calls is None else tool_calls
        usage = self._extract_usage(response)
        self._update_usage_stats(usage)
        return LLMResponse(
            content=resolved_content,
            tool_calls=resolved_tool_calls,
            usage=usage,
            model=getattr(response, "model", self.model),
            finish_reason=getattr(response, "status", ""),
            message={
                "role": "assistant",
                "content": resolved_content,
                "tool_calls": resolved_tool_calls or None,
            },
        )

    async def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> LLMResponse:
        params = self._build_responses_request_params(
            messages=messages,
            tools=tools,
            response_format=response_format,
            **kwargs,
        )
        response = await self._create_response(params)
        return self._build_responses_llm_response(response)

    async def call_json(
        self,
        messages: List[Dict[str, Any]],
        schema: Optional[Dict[str, Any]] = None,
        strict_json: bool = True,
        **kwargs
    ) -> LLMResponse:
        """
        Call the model with an optional JSON schema response_format.
        """
        if strict_json and schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "schema": schema,
                    "strict": True,
                }
            }
        return await self.chat(messages=messages, **kwargs)

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ):
        response = await self.chat(messages=messages, tools=tools, **kwargs)
        yield response
