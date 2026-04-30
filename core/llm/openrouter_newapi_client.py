"""
OpenRouter NewAPI LLM client.

This client keeps the repository's existing chat-completions integration while
making the OpenRouter-specific behavior explicit:
- OpenAI-compatible request/response handling
- OpenRouter provider routing via ``extra_body``
- context-limit normalization and transient retry
- lightweight debug logging for hard-to-reproduce request issues
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

import tiktoken
from openai import AsyncOpenAI, OpenAI

from .base import LLMResponse, OpenAISDKClient, register_llm_client
from .env_utils import load_env, resolve_client_setting


load_env()

logger = logging.getLogger(__name__)

_OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
_OPENROUTER_BASE_URL_ENV = "OPENROUTER_BASE_URL"
_DEFAULT_TIMEOUT_SECONDS = 180
_DEFAULT_MAX_RETRIES = 5
_FULL_DEBUG_MESSAGE_THRESHOLD = 8
_UNSUPPORTED_MESSAGE_FIELDS = (
    "cache_control",
    "annotations",
    "audio",
    "refusal",
    "reasoning_details",
)
_CONTEXT_LIMIT_ERROR_PATTERNS = (
    "Input is too long for requested model",
    "input length and `max_tokens` exceed context limit",
    "maximum context length",
    "prompt is too long",
    "exceeds the maximum length",
    "exceeds the maximum allowed length",
    "Input tokens exceed the configured limit",
)
_ENDPOINT_KIND_OPENROUTER = "openrouter"
_ENDPOINT_KIND_OPENAI_COMPATIBLE = "openai_compatible"
_MODEL_ALIAS_MODE_PREFIXED = "prefixed"
_MODEL_ALIAS_MODE_BARE = "bare"
def _normalize_base_url(base_url: Optional[str]) -> str:
    return str(base_url or "").strip().lower()


def _is_openrouter_endpoint(base_url: Optional[str]) -> bool:
    """Treat the official OpenRouter host as the OpenRouter-specific route."""
    return "openrouter.ai" in _normalize_base_url(base_url)


def _resolve_openrouter_extras_enabled(
    *,
    base_url: Optional[str],
    explicit_value: Optional[bool],
) -> bool:
    if explicit_value is not None:
        return bool(explicit_value)
    return _is_openrouter_endpoint(base_url)


def _resolve_model_alias_mode(
    *,
    base_url: Optional[str],
    explicit_value: Optional[str],
) -> str:
    if explicit_value is not None:
        candidate = str(explicit_value).strip().lower()
        if candidate not in {_MODEL_ALIAS_MODE_PREFIXED, _MODEL_ALIAS_MODE_BARE}:
            raise ValueError(
                "model_alias_mode must be 'prefixed', 'bare', or omitted for auto resolution"
            )
        return candidate
    if _is_openrouter_endpoint(base_url):
        return _MODEL_ALIAS_MODE_PREFIXED
    return _MODEL_ALIAS_MODE_BARE


def _strip_provider_prefix(model: Optional[str]) -> Optional[str]:
    if not model or "/" not in model:
        return model
    provider, candidate = model.split("/", 1)
    if provider.lower() in {"openai", "qwen"} and candidate:
        return candidate
    return model


def _is_context_limit_error(error_str: str) -> bool:
    return any(pattern in error_str for pattern in _CONTEXT_LIMIT_ERROR_PATTERNS)


def _is_model_not_found_error(exc: Exception) -> bool:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error", {})
        if isinstance(error, dict) and error.get("code") == "model_not_found":
            return True
    return "model_not_found" in str(exc)


def _clean_user_content_from_response(text: str) -> str:
    """Drop leaked prompt echoes that occasionally appear in tool-heavy runs."""
    pattern = r"\n\nUser:.*?(?=<use_mcp_tool>|$)"
    return re.sub(pattern, "", text, flags=re.MULTILINE | re.DOTALL)


def _next_retry_wait_seconds(retry_count: int) -> int:
    return min(2 * (2 ** min(retry_count - 1, 2)), 20)


class ContextLimitError(Exception):
    """Raised when the upstream provider reports a prompt/context limit error."""


@register_llm_client(
    "openrouter_newapi",
    aliases=("OpenRouter", "openrouter", "OpenRouterNewAPI"),
)
class OpenRouterNewAPIClient(OpenAISDKClient):
    """
    OpenRouter chat-completions client built on the official OpenAI SDK.

    The public surface stays aligned with the repo's generic LLM interface while
    the OpenRouter-specific request shaping is isolated in small helpers.
    """

    def __init__(
        self,
        model: str = "openai/gpt-5.4",
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key_env: Optional[str] = None,
        base_url_env: Optional[str] = None,
        max_tokens: int = 128000,
        reasoning_effort: str = "medium",
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        disable_cache_control: bool = False,
        openrouter_provider: Optional[str] = None,
        enable_openrouter_extras: Optional[bool] = None,
        model_alias_mode: Optional[str] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        async_mode: bool = True,
        timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
        **kwargs,
    ):
        """
        Initialize the OpenRouter client.

        Explicit ``api_key``/``base_url`` arguments win. If omitted, the client
        resolves ``api_key_env`` / ``base_url_env`` first, then falls back to
        ``OPENROUTER_API_KEY`` and ``OPENROUTER_BASE_URL``.
        """
        resolved_api_key, self.api_key_source = resolve_client_setting(
            api_key,
            preferred_env=api_key_env,
            fallback_envs=(_OPENROUTER_API_KEY_ENV,),
        )
        resolved_base_url, self.base_url_source = resolve_client_setting(
            base_url,
            preferred_env=base_url_env,
            fallback_envs=(_OPENROUTER_BASE_URL_ENV,),
        )
        if not resolved_api_key or not resolved_base_url:
            raise ValueError(
                "OpenRouterNewAPIClient requires api_key/base_url, api_key_env/base_url_env, "
                f"or {_OPENROUTER_API_KEY_ENV}/{_OPENROUTER_BASE_URL_ENV}"
            )
        super().__init__(
            model=model,
            temperature=temperature,
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            async_mode=async_mode,
            extra_headers=extra_headers,
            timeout_seconds=timeout_seconds,
            **kwargs,
        )

        self.max_tokens = max_tokens
        self.reasoning_effort = self._normalize_reasoning_effort(reasoning_effort)
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.repetition_penalty = repetition_penalty
        self.disable_cache_control = disable_cache_control
        self.openrouter_provider = openrouter_provider
        self.extra_body = dict(extra_body or {})
        self.endpoint_kind = (
            _ENDPOINT_KIND_OPENROUTER
            if _is_openrouter_endpoint(self.base_url)
            else _ENDPOINT_KIND_OPENAI_COMPATIBLE
        )
        self.enable_openrouter_extras = _resolve_openrouter_extras_enabled(
            base_url=self.base_url,
            explicit_value=enable_openrouter_extras,
        )
        self.model_alias_mode = _resolve_model_alias_mode(
            base_url=self.base_url,
            explicit_value=model_alias_mode,
        )
        self._init_tokenizer()

    def _sdk_client_classes(self):
        return AsyncOpenAI, OpenAI

    def _init_tokenizer(self) -> None:
        """Prefer the newest tokenizer, but keep a safe fallback for old envs."""
        try:
            self.encoding = tiktoken.get_encoding("o200k_base")
        except Exception:
            try:
                self.encoding = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self.encoding = None

    def supports_response_format(self) -> bool:
        # OpenRouter chat completions do not reliably support json_schema.
        return False

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count conservatively when exact provider counts are absent."""
        if self.encoding:
            try:
                return len(self.encoding.encode(text))
            except Exception:
                pass
        return len(text) // 4

    def estimate_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Estimate token count for a list of messages."""
        total_tokens = 0
        for msg in messages:
            try:
                text = json.dumps(msg, ensure_ascii=False)
            except Exception:
                text = str(msg)
            total_tokens += self._estimate_tokens(text)
        return total_tokens

    def _apply_cache_control(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Preserve the existing hook without mutating requests.

        OpenRouter's OpenAI-compatible chat-completions route does not support
        Anthropic-style ``cache_control`` payloads, so the effective behavior is
        a no-op even if ``disable_cache_control`` is false.
        """
        return messages

    def _sanitize_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Strip fields the OpenAI-compatible endpoint rejects or ignores."""
        sanitized: List[Dict[str, Any]] = []
        for msg in messages:
            if not isinstance(msg, dict):
                sanitized.append(msg)
                continue
            cleaned = dict(msg)
            for field in _UNSUPPORTED_MESSAGE_FIELDS:
                cleaned.pop(field, None)
            sanitized.append(cleaned)
        return sanitized

    def _resolve_request_model(self, model: Optional[str] = None) -> str:
        """Map repo-facing model names to the endpoint's expected naming style."""
        requested_model = str(self.model if model is None else model).strip()
        if self.model_alias_mode == _MODEL_ALIAS_MODE_BARE:
            return _strip_provider_prefix(requested_model) or requested_model
        return requested_model

    def _resolve_model_not_found_retry(self, request_model: Optional[str]) -> Optional[str]:
        """Keep one defensive alias fallback for misconfigured generic gateways."""
        fallback_model = _strip_provider_prefix(request_model)
        if not fallback_model or fallback_model == request_model:
            return None
        return fallback_model

    def _build_extra_body(self) -> Dict[str, Any]:
        """Build the OpenRouter-specific ``extra_body`` payload."""
        if not self.enable_openrouter_extras:
            return {}

        extra_body: Dict[str, Any] = {}

        provider_config = (self.openrouter_provider or "").strip().lower()
        if provider_config == "google":
            extra_body["provider"] = {
                "only": ["google-vertex/us", "google-vertex/europe", "google-vertex/global"]
            }
        elif provider_config == "anthropic":
            extra_body["provider"] = {"only": ["anthropic"]}
        elif provider_config == "amazon":
            extra_body["provider"] = {"only": ["amazon-bedrock"]}
        elif provider_config:
            extra_body["provider"] = {"only": [provider_config]}

        if self.top_k != -1:
            extra_body["top_k"] = self.top_k
        if self.min_p != 0.0:
            extra_body["min_p"] = self.min_p
        if self.repetition_penalty != 1.0:
            extra_body["repetition_penalty"] = self.repetition_penalty

        # Excluding reasoning tokens keeps logs readable and avoids opaque blobs.
        extra_body["reasoning"] = {"exclude": True}
        return extra_body

    def _build_request_params(
        self,
        *,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        response_format: Optional[Any],
        stream: bool,
        tool_choice: Optional[str],
        **kwargs,
    ) -> Dict[str, Any]:
        """Assemble one OpenAI-compatible chat-completions request."""
        processed_messages = self._sanitize_messages(self._apply_cache_control(messages))

        params: Dict[str, Any] = {
            "model": self._resolve_request_model(),
            "messages": processed_messages,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_tokens,
            "stream": stream,
            # OpenRouter chat completions do not support reasoning_effort directly.
            **self.extra_params,
            **kwargs,
        }

        if self.top_p != 1.0:
            params["top_p"] = self.top_p

        inline_extra_body = params.pop("extra_body", None)
        extra_body = self._build_extra_body()
        if self.extra_body:
            extra_body.update(self.extra_body)
        if isinstance(inline_extra_body, dict):
            extra_body.update(inline_extra_body)
        if extra_body:
            params["extra_body"] = extra_body

        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice or "auto"
        else:
            # OpenRouter rejects parallel_tool_calls when no tools are attached.
            params.pop("parallel_tool_calls", None)

        if response_format is not None:
            params["response_format"] = response_format

        return params

    def _log_request_debug(self, params: Dict[str, Any]) -> None:
        """Emit concise debug logs and opt-in full dumps for pathological prompts."""
        logger.debug(
            "LLM API request params: %s",
            {
                "configured_model": self.model,
                "request_model": params.get("model"),
                "endpoint_kind": self.endpoint_kind,
                "model_alias_mode": self.model_alias_mode,
                "api_key_source": self.api_key_source,
                "base_url_source": self.base_url_source,
                "temperature": params.get("temperature"),
                "max_completion_tokens": params.get("max_completion_tokens"),
                "stream": params.get("stream"),
                "top_p": params.get("top_p"),
                "message_count": len(params.get("messages", [])),
                "tool_count": len(params.get("tools", [])) if params.get("tools") else 0,
                "has_extra_body": bool(params.get("extra_body")),
                "has_tool_choice": "tool_choice" in params,
            },
        )

        if params.get("extra_body"):
            logger.info("OpenRouter extra_body params: %s", params["extra_body"])

        if params.get("tools"):
            tool_names = [
                t.get("function", {}).get("name")
                for t in params["tools"]
                if isinstance(t, dict)
            ]
            logger.debug("Tool definitions: %s", tool_names)

        debug_enabled = os.getenv("DEBUG_API_REQUESTS", "false").lower() == "true"
        if not debug_enabled or len(params.get("messages", [])) < _FULL_DEBUG_MESSAGE_THRESHOLD:
            return

        print("\n" + "=" * 80)
        print(f"FULL API REQUEST DEBUG (Messages: {len(params.get('messages', []))})")
        print("=" * 80)
        for index, msg in enumerate(params.get("messages", []), start=1):
            print(f"\n--- Message {index} ---")
            msg_copy = dict(msg) if isinstance(msg, dict) else {"raw": str(msg)}
            content = msg_copy.get("content")
            if isinstance(content, str) and len(content) > 300:
                msg_copy["content"] = f"{content[:300]}... (truncated, {len(content)} chars total)"
            print(json.dumps(msg_copy, indent=2, ensure_ascii=False))
        print("\n" + "=" * 80 + "\n")

    async def _create_completion(
        self,
        params: Dict[str, Any],
        *,
        is_async: bool,
    ):
        """Call the upstream chat-completions endpoint with retry normalization."""
        from openai import AuthenticationError, BadRequestError, PermissionDeniedError

        retry_count = 0
        while retry_count < _DEFAULT_MAX_RETRIES:
            try:
                if is_async:
                    return await self.client.chat.completions.create(**params)
                return self.client.chat.completions.create(**params)
            except (BadRequestError, AuthenticationError, PermissionDeniedError) as exc:
                logger.error("Client error (will not retry): %s", exc)
                logger.error("Error type: %s", type(exc).__name__)
                logger.error("Model: %s", params.get("model"))
                logger.error("Messages count: %s", len(params.get("messages", [])))
                logger.error(
                    "Tools count: %s",
                    len(params.get("tools", [])) if params.get("tools") else 0,
                )
                raise
            except Exception as exc:
                error_str = str(exc)
                if _is_context_limit_error(error_str):
                    raise ContextLimitError(f"Context limit exceeded: {error_str}") from exc

                fallback_model = None
                if _is_model_not_found_error(exc):
                    fallback_model = self._resolve_model_not_found_retry(params.get("model"))
                if fallback_model:
                    logger.info(
                        "Model %s not found on %s, retrying with alias %s",
                        params.get("model"),
                        self.base_url,
                        fallback_model,
                    )
                    params = dict(params)
                    params["model"] = fallback_model
                    continue

                retry_count += 1
                if retry_count >= _DEFAULT_MAX_RETRIES:
                    logger.error(
                        "API call failed after %s attempts: %s",
                        _DEFAULT_MAX_RETRIES,
                        error_str,
                    )
                    raise

                wait_time = _next_retry_wait_seconds(retry_count)
                logger.warning(
                    "API call failed (attempt %s/%s), retrying in %ss: %s",
                    retry_count,
                    _DEFAULT_MAX_RETRIES,
                    wait_time,
                    error_str,
                )

                if is_async:
                    await asyncio.sleep(wait_time)
                else:
                    time.sleep(wait_time)

        raise RuntimeError(f"API call failed after {_DEFAULT_MAX_RETRIES} attempts")

    def _build_llm_response(self, response: Any) -> LLMResponse:
        """Normalize one chat-completions response into the repo's LLMResponse."""
        llm_response = self._parse_chat_completion_response(
            response,
            content_transform=_clean_user_content_from_response,
            context_limit_error_cls=ContextLimitError,
        )
        if llm_response.finish_reason == "length" and llm_response.content:
            logger.warning("Response has finish_reason='length' but content exists, continuing...")
        return llm_response

    async def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Any] = None,
        **kwargs,
    ) -> LLMResponse:
        """Send one non-streaming chat-completions request."""
        tool_choice = kwargs.pop("tool_choice", None)
        params = self._build_request_params(
            messages=messages,
            tools=tools,
            response_format=response_format,
            stream=False,
            tool_choice=tool_choice,
            **kwargs,
        )
        self._log_request_debug(params)

        try:
            response = await self._create_completion(params, is_async=self.async_mode)
            llm_response = self._build_llm_response(response)
            self._update_usage_stats(llm_response.usage)
            return llm_response
        except ContextLimitError as exc:
            logger.warning("LLM API context limit: %s", exc)
            raise
        except asyncio.CancelledError:
            logger.warning("LLM API call was cancelled (likely due to user interrupt)")
            raise
        except KeyboardInterrupt:
            logger.warning("LLM API call interrupted by user")
            raise
        except Exception as exc:
            logger.error("LLM API error: %s", exc)
            logger.error("Model: %s", self.model)
            logger.error("Message count: %s", len(messages))
            if tools:
                logger.error("Tool count: %s", len(tools))
            raise

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ):
        """Stream chat-completions chunks while keeping sync and async SDK modes working."""
        tool_choice = kwargs.pop("tool_choice", None)
        params = self._build_request_params(
            messages=messages,
            tools=tools,
            response_format=None,
            stream=True,
            tool_choice=tool_choice,
            **kwargs,
        )
        async for chunk in self._iter_chat_completion_chunks(params):
            yield chunk
