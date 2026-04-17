from __future__ import annotations

import asyncio
from types import SimpleNamespace

from core.llm.codex_sub2api_client import CodexSub2APIClient


def test_codex_sub2api_client_prefers_explicit_haoxiang_env(monkeypatch):
    monkeypatch.setenv("HAOXIANG_OPENAI_API_KEY", "hx-test-key")
    monkeypatch.delenv("HAOXIANG_API_KEY", raising=False)
    monkeypatch.delenv("HAOXIANG_BASE_URL", raising=False)

    client = CodexSub2APIClient(async_mode=False)

    assert client.api_key == "hx-test-key"
    assert client.base_url == "https://ie-crs.haoxiang.ai/v1"
    assert client.model == "gpt-5.4"


def test_codex_sub2api_client_recovers_text_and_tool_calls_from_stream(monkeypatch):
    monkeypatch.setenv("HAOXIANG_OPENAI_API_KEY", "hx-test-key")
    monkeypatch.delenv("HAOXIANG_API_KEY", raising=False)
    monkeypatch.delenv("HAOXIANG_BASE_URL", raising=False)

    client = CodexSub2APIClient()

    class _FakeEvent(SimpleNamespace):
        pass

    class _FakeStream:
        def __init__(self, events, final_response):
            self._events = list(events)
            self._final_response = final_response

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def __aiter__(self):
            self._index = 0
            return self

        async def __anext__(self):
            if self._index >= len(self._events):
                raise StopAsyncIteration
            event = self._events[self._index]
            self._index += 1
            return event

        async def get_final_response(self):
            return self._final_response

    class _FakeResponses:
        def __init__(self, events, final_response):
            self._events = events
            self._final_response = final_response

        def stream(self, **params):
            return _FakeStream(self._events, self._final_response)

    final_response = SimpleNamespace(
        model="gpt-5.4",
        status="completed",
        usage=SimpleNamespace(input_tokens=11, output_tokens=7, total_tokens=18),
        output_text="",
        output=[],
    )
    events = [
        _FakeEvent(type="response.created"),
        _FakeEvent(
            type="response.output_item.added",
            item=SimpleNamespace(type="function_call", id="item-1", call_id="call-1", name="demo_tool", arguments=""),
        ),
        _FakeEvent(type="response.function_call_arguments.delta", item_id="item-1", delta='{"x"'),
        _FakeEvent(type="response.function_call_arguments.delta", item_id="item-1", delta=':1}'),
        _FakeEvent(type="response.output_text.delta", delta="hello"),
        _FakeEvent(
            type="response.output_item.done",
            item=SimpleNamespace(type="function_call", id="item-1", call_id="call-1", name="demo_tool", arguments='{"x":1}'),
        ),
    ]

    client.client = SimpleNamespace(responses=_FakeResponses(events, final_response))

    response = asyncio.run(client.chat([{"role": "user", "content": "hi"}]))

    assert response.content == "hello"
    assert response.tool_calls == [
        {
            "id": "call-1",
            "type": "function",
            "function": {
                "name": "demo_tool",
                "arguments": '{"x":1}',
            },
        }
    ]
    assert response.usage["total_tokens"] == 18
