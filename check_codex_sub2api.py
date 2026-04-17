#!/usr/bin/env python3
"""
Smoke-test script for the codex_sub2api Responses client.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import Optional

from core.llm.codex_sub2api_client import CodexSub2APIClient
from core.llm.env_utils import load_env


def mask_key(key: str) -> str:
    if len(key) <= 8:
        return "****"
    return f"{key[:4]}...{key[-4:]}"


async def run_smoke_test(prompt: str, retries: int, retry_delay: float, model: Optional[str]) -> int:
    last_error: Optional[Exception] = None

    for attempt in range(1, retries + 2):
        client = CodexSub2APIClient(model=model or CodexSub2APIClient.DEFAULT_MODEL, max_tokens=128)
        try:
            response = await client.chat(
                [{"role": "user", "content": prompt}],
            )
            break
        except Exception as exc:
            last_error = exc
            if attempt > retries:
                print(f"Request failed after {attempt} attempt(s): {exc}", file=sys.stderr)
                return 1
            print(f"Attempt {attempt} failed: {exc}", file=sys.stderr)
            print(f"Retrying in {retry_delay:.1f}s...", file=sys.stderr)
            await asyncio.sleep(retry_delay)
        finally:
            await client.aclose()
    else:
        if last_error is not None:
            print(f"Request failed: {last_error}", file=sys.stderr)
        return 1

    print("=== codex_sub2api smoke test ===")
    print(f"Base URL: {client.base_url}")
    print(f"Model:    {client.model}")
    print(f"API key:  {mask_key(client.api_key or '')}")
    print()
    print("Reply:")
    print(response.content or "<empty>")
    print()
    print("Usage:")
    print(response.usage)
    if response.tool_calls:
        print()
        print("Tool calls:")
        print(response.tool_calls)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a live codex_sub2api Responses API smoke test.")
    parser.add_argument(
        "env_path",
        nargs="?",
        default=".env",
        help="Path to the .env file that contains HAOXIANG_OPENAI_API_KEY / HAOXIANG_BASE_URL",
    )
    parser.add_argument(
        "--prompt",
        default="Reply with OK and one short sentence about streaming responses.",
        help="Prompt to send to the backend.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="How many times to retry transient backend failures.",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=2.0,
        help="Seconds to wait between retries.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model override. Defaults to the client's gpt-5.4 route.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.env_path):
        print(f"Missing .env file: {args.env_path}", file=sys.stderr)
        return 1

    load_env(env_path=args.env_path, override=True)
    return asyncio.run(run_smoke_test(args.prompt, args.retries, args.retry_delay, args.model))


if __name__ == "__main__":
    raise SystemExit(main())
