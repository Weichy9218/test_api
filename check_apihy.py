"""Smoke test for apihy gateway (apihy_API_KEY + apihy_BASE_URL)."""

from __future__ import annotations

import argparse
import asyncio
import sys

from openai import OpenAI

from core.llm.env_utils import load_env
from core.llm.openrouter_newapi_client import OpenRouterNewAPIClient

_API_KEY_ENV = "apihy_API_KEY"
_BASE_URL_ENV = "apihy_BASE_URL"
_DEFAULT_MODEL = "gpt-5.4"


def _mask(key: str) -> str:
    return f"{key[:4]}...{key[-4:]}" if len(key) > 8 else "****"


async def run(model: str, retries: int) -> int:
    client = OpenRouterNewAPIClient(
        model=model,
        api_key_env=_API_KEY_ENV,
        base_url_env=_BASE_URL_ENV,
        max_tokens=64,
        reasoning_effort=None,
    )

    print(f"Base URL: {client.base_url}")
    print(f"Model:    {client.model}")
    print(f"API key:  {_mask(client.api_key or '')}")
    print()

    # Model list
    sync_client = OpenAI(api_key=client.api_key, base_url=client.base_url, timeout=30)
    models = [m.id for m in sync_client.models.list().data]
    print(f"Models ({len(models)}):")
    for m in models:
        print(f"  {m}{' ◀' if m == model else ''}")
    print()

    # Chat test
    prompt = "Reply with exactly one word: OK"
    for attempt in range(1, retries + 2):
        try:
            resp = await client.chat([{"role": "user", "content": prompt}])
            break
        except Exception as exc:
            if attempt > retries:
                print(f"Failed after {attempt} attempt(s): {exc}", file=sys.stderr)
                await client.aclose()
                return 1
            print(f"Attempt {attempt} failed, retrying: {exc}", file=sys.stderr)
            await asyncio.sleep(2)

    await client.aclose()
    print(f"Chat reply: {resp.content!r}")
    print("Usage:", client.get_usage_stats())
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("env_path", nargs="?", default=".env")
    parser.add_argument("--model", default=_DEFAULT_MODEL)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()
    load_env(env_path=args.env_path, override=True)
    return asyncio.run(run(args.model, args.retries))


if __name__ == "__main__":
    raise SystemExit(main())
