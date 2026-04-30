"""
Base LLM Client Interface

Abstract base class for all LLM clients.
"""

from abc import ABC, abstractmethod
import json
from typing import Callable, List, Dict, Any, Optional, Tuple, Type, Union
from dataclasses import dataclass, field
import re


@dataclass
class LLMResponse:
    """
    Response from LLM

    Attributes:
        content: The text content of the response
        tool_calls: List of tool calls made by the LLM
        usage: Token usage statistics
        model: Model name used for generation
        finish_reason: Reason for completion (stop, length, etc.)
        message: The original message of the output
    """
    content: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=lambda: {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0
    })
    model: str = ""
    finish_reason: str = ""
    message: Optional[dict] = None

    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls"""
        return len(self.tool_calls) > 0

    def get_total_tokens(self) -> int:
        """Get total token count"""
        return self.usage.get("total_tokens", 0)

    def is_truncated(self) -> bool:
        """Check if response was truncated due to length"""
        return self.finish_reason == "length"


class BaseLLMClient(ABC):
    """
    Abstract base class for LLM clients

    All LLM providers (OpenAI, Anthropic, etc.) should implement this interface.
    """

    def __init__(self, model: str, temperature: Optional[float] = 0.7, **kwargs):
        self.model = model
        self.temperature = temperature
        self.extra_params = kwargs
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_requests = 0
        self._total_cached_tokens = 0

    def get_usage_stats(self) -> Dict[str, int]:
        """Get cumulative usage statistics"""
        return {
            "total_prompt_tokens": self._total_prompt_tokens,
            "total_completion_tokens": self._total_completion_tokens,
            "total_tokens": self._total_prompt_tokens + self._total_completion_tokens,
            "total_requests": self._total_requests,
            "cached_tokens": self._total_cached_tokens,
        }

    def _update_usage_stats(self, usage: Dict[str, int]):
        """Update usage statistics"""
        self._total_prompt_tokens += usage.get("prompt_tokens", 0)
        self._total_completion_tokens += usage.get("completion_tokens", 0)
        self._total_requests += 1
        self._total_cached_tokens += usage.get("cached_tokens", 0)

    def estimate_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Estimate token count for a list of messages."""
        total_chars = 0
        for msg in messages:
            try:
                total_chars += len(json.dumps(msg, ensure_ascii=False))
            except Exception:
                total_chars += len(str(msg))
        return total_chars // 4

    async def aclose(self) -> None:
        """
        Cleanup hook for async clients.

        Override in subclasses that hold network resources (e.g., AsyncOpenAI).
        """
        return None

    @abstractmethod
    async def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Send chat messages to LLM and get response

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of tool definitions for function calling
            **kwargs: Additional parameters

        Returns:
            LLMResponse object
        """
        pass

    @abstractmethod
    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ):
        """
        Stream chat response from LLM

        Args:
            messages: List of message dicts
            tools: Optional tool definitions
            **kwargs: Additional parameters

        Yields:
            Chunks of LLMResponse
        """
        pass

    def format_system_message(self, content: str) -> Dict[str, str]:
        """Format system message"""
        return {"role": "system", "content": content}

    def format_user_message(self, content: str) -> Dict[str, str]:
        """Format user message"""
        return {"role": "user", "content": content}

    def format_assistant_message(self, content: str) -> Dict[str, str]:
        """Format assistant message"""
        return {"role": "assistant", "content": content}

    def supports_response_format(self) -> bool:
        """
        Whether the client supports OpenAI-style response_format (e.g. json_schema).
        Override in subclasses when supported.
        """
        return False

    @staticmethod
    def get_field_value(obj: Any, key: str, default: Any = None) -> Any:
        """Read one field from either an object or dict payload."""
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model}, temperature={self.temperature})"


class OpenAISDKClient(BaseLLMClient):
    """Shared OpenAI SDK bootstrap and lifecycle helpers for provider adapters."""

    def __init__(
        self,
        model: str,
        temperature: Optional[float] = 0.7,
        *,
        api_key: Optional[str],
        base_url: Optional[str],
        async_mode: bool = True,
        extra_headers: Optional[Dict[str, str]] = None,
        timeout_seconds: int = 180,
        **kwargs,
    ):
        super().__init__(model, temperature, **kwargs)
        self.api_key = api_key
        self.base_url = base_url
        self.async_mode = async_mode
        self.default_headers = dict(extra_headers or {})
        self.client = self._build_sdk_client(timeout_seconds=timeout_seconds)

    def _sdk_client_classes(self) -> Tuple[Type[Any], Type[Any]]:
        """Return async/sync OpenAI SDK client classes."""
        from openai import AsyncOpenAI, OpenAI

        return AsyncOpenAI, OpenAI

    def _build_sdk_client(self, *, timeout_seconds: int):
        """Create the underlying OpenAI-compatible SDK client."""
        async_cls, sync_cls = self._sdk_client_classes()
        client_cls = async_cls if self.async_mode else sync_cls
        return client_cls(
            api_key=self.api_key,
            base_url=self.base_url,
            default_headers=self.default_headers or None,
            timeout=timeout_seconds,
        )

    @staticmethod
    def _normalize_reasoning_effort(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        effort = str(value).strip().lower()
        if not effort or effort in {"none", "off", "disabled"}:
            return None
        return effort

    async def aclose(self) -> None:
        """Close underlying SDK resources when present."""
        if not getattr(self, "client", None):
            return

        if self.async_mode and hasattr(self.client, "close"):
            await self.client.close()
        elif hasattr(self.client, "close"):
            self.client.close()

        self.client = None

    def _chat_completion_message_has_payload(self, message: Any) -> bool:
        return bool(getattr(message, "content", None)) or bool(getattr(message, "tool_calls", None))

    def _extract_chat_completion_tool_calls(self, message: Any) -> List[Dict[str, Any]]:
        tool_calls: List[Dict[str, Any]] = []
        for tool_call in getattr(message, "tool_calls", None) or []:
            function = getattr(tool_call, "function", None)
            tool_calls.append(
                {
                    "id": getattr(tool_call, "id", None),
                    "type": getattr(tool_call, "type", None),
                    "function": {
                        "name": getattr(function, "name", None),
                        "arguments": getattr(function, "arguments", None),
                    },
                }
            )
        return tool_calls

    def _extract_chat_completion_usage(self, response: Any) -> Dict[str, int]:
        usage = getattr(response, "usage", None)
        cached_tokens = 0
        if usage:
            details = getattr(usage, "prompt_tokens_details", None)
            if details is not None:
                cached_tokens = int(getattr(details, "cached_tokens", 0) or 0)
            if cached_tokens == 0:
                cached_tokens = int(getattr(usage, "cached_tokens", 0) or 0)
        return {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
            "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
            "cached_tokens": cached_tokens,
        }

    def _parse_chat_completion_response(
        self,
        response: Any,
        *,
        content_transform: Optional[Callable[[str], str]] = None,
        context_limit_error_cls: Optional[Type[Exception]] = None,
    ) -> LLMResponse:
        """Normalize one chat-completions response into the shared response shape."""
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise ValueError("LLM returned empty response")

        choice = choices[0]
        message = getattr(choice, "message", None)
        finish_reason = getattr(choice, "finish_reason", "")
        if message is None:
            raise ValueError("LLM response missing message payload")

        if finish_reason == "length" and not self._chat_completion_message_has_payload(message):
            if context_limit_error_cls is not None:
                raise context_limit_error_cls("Response truncated due to maximum context length")
            raise ValueError("LLM returned empty truncated response")
        if finish_reason == "stop" and not self._chat_completion_message_has_payload(message):
            raise ValueError("LLM returned empty content with finish_reason='stop'")

        content = getattr(message, "content", None) or ""
        if content and content_transform is not None:
            content = content_transform(content)

        return LLMResponse(
            content=content,
            tool_calls=self._extract_chat_completion_tool_calls(message),
            usage=self._extract_chat_completion_usage(response),
            model=getattr(response, "model", ""),
            finish_reason=finish_reason,
            message=message,
        )

    async def _iter_chat_completion_chunks(self, params: Dict[str, Any]):
        """Yield normalized content chunks from a chat-completions stream."""
        if self.async_mode:
            stream = await self.client.chat.completions.create(**params)
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if getattr(delta, "content", None):
                    yield LLMResponse(
                        content=delta.content,
                        model=chunk.model,
                        finish_reason=chunk.choices[0].finish_reason or "",
                    )
            return

        stream = self.client.chat.completions.create(**params)
        for chunk in stream:
            delta = chunk.choices[0].delta
            if getattr(delta, "content", None):
                yield LLMResponse(
                    content=delta.content,
                    model=chunk.model,
                    finish_reason=chunk.choices[0].finish_reason or "",
                )


LLM_CLIENTS: Dict[str, Type[BaseLLMClient]] = {}
LLM_CLIENT_ALIASES: Dict[str, str] = {}
_CLIENT_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def _normalize_registry_lookup_key(name: str) -> str:
    return str(name or "").strip().lower()


def resolve_llm_client_name(name: str) -> str:
    """Resolve an input client name to the canonical registry key when possible."""
    candidate = str(name or "").strip()
    if not candidate:
        raise KeyError("LLM client name must be a non-empty string")
    if candidate in LLM_CLIENTS:
        return candidate
    alias_key = _normalize_registry_lookup_key(candidate)
    return LLM_CLIENT_ALIASES.get(alias_key, candidate)


def get_llm_client_class(name: Union[str, Type[BaseLLMClient]]) -> Type[BaseLLMClient]:
    """Resolve one registered client class by canonical name or legacy alias."""
    if isinstance(name, type) and issubclass(name, BaseLLMClient):
        return name

    candidate = resolve_llm_client_name(str(name))
    if candidate in LLM_CLIENTS:
        return LLM_CLIENTS[candidate]
    known = ", ".join(sorted(LLM_CLIENTS))
    raise KeyError(f"Unknown LLM client '{name}'. Known clients: {known}")


def instantiate_llm_client(
    llm_client: Union[BaseLLMClient, str, Type[BaseLLMClient], Any],
    llm_client_args: Optional[Dict[str, Any]] = None,
) -> BaseLLMClient:
    """Instantiate a client from either a ready instance or a registered name."""
    if isinstance(llm_client, BaseLLMClient):
        return llm_client
    if not isinstance(llm_client, str) and not (
        isinstance(llm_client, type) and issubclass(llm_client, BaseLLMClient)
    ):
        return llm_client
    client_cls = get_llm_client_class(llm_client)
    return client_cls(**(llm_client_args or {}))


def register_llm_client(name, *, aliases: Optional[Tuple[str, ...]] = None):
    def register_cls(cls):
        canonical_name = str(name).strip()
        if not _CLIENT_NAME_PATTERN.match(canonical_name):
            raise ValueError(
                f"Canonical LLM client name '{canonical_name}' must be lowercase snake_case"
            )
        LLM_CLIENTS[canonical_name] = cls
        LLM_CLIENT_ALIASES[_normalize_registry_lookup_key(canonical_name)] = canonical_name
        for alias in aliases or ():
            alias_name = str(alias).strip()
            if not alias_name or alias_name == canonical_name:
                continue
            LLM_CLIENT_ALIASES[_normalize_registry_lookup_key(alias_name)] = canonical_name
        return cls
    return register_cls


# Backward-compat alias for older smoke-test code
def register_llm_client_class(name: str):
    return register_llm_client(name)
