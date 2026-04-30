"""Smoke test for the Sub2API Responses client (SUB2API_API_KEY + SUB2API_BASE_URL)."""

from __future__ import annotations

import argparse
import asyncio
import sys

from core.llm.codex_sub2api_client import CodexSub2APIClient
from core.llm.env_utils import load_env


def _mask(key: str) -> str:
    return f"{key[:4]}...{key[-4:]}" if len(key) > 8 else "****"


async def run(model: str | None, retries: int) -> int:
    client = CodexSub2APIClient(model=model or CodexSub2APIClient.DEFAULT_MODEL, max_tokens=128)
    prompt = "Reply with OK and one short sentence about streaming responses."

    print(f"Base URL: {client.base_url}")
    print(f"Model:    {client.model}")
    print(f"API key:  {_mask(client.api_key or '')}")
    print()

    for attempt in range(1, retries + 2):
        try:
            response = await client.chat([{"role": "user", "content": prompt}])
            break
        except Exception as exc:
            if attempt > retries:
                print(f"Failed after {attempt} attempt(s): {exc}", file=sys.stderr)
                await client.aclose()
                return 1
            print(f"Attempt {attempt} failed, retrying: {exc}", file=sys.stderr)
            await asyncio.sleep(2)

    await client.aclose()
    print("Reply:")
    print(response.content or "<empty>")
    print("\nUsage:", response.usage)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("env_path", nargs="?", default=".env")
    parser.add_argument("--model", default=None)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()
    load_env(env_path=args.env_path, override=True)
    return asyncio.run(run(args.model, args.retries))


if __name__ == "__main__":
    raise SystemExit(main())
