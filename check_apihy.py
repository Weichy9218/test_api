"""
check_apihy.py — 测试 apihy 网关（OpenAI 兼容中转）的连通性、模型列表和 chat 能力。

读取凭证：
  apihy_API_KEY   — API key
  apihy_BASE_URL  — 网关地址，例如 https://zgc.apihy.com/v1

用法：
  uv run python check_apihy.py
  uv run python check_apihy.py /path/to/.env
  uv run python check_apihy.py --model gpt-5.4 --retries 2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Optional

from core.llm.env_utils import load_env
from core.llm.openrouter_newapi_client import OpenRouterNewAPIClient


_API_KEY_ENV = "apihy_API_KEY"
_BASE_URL_ENV = "apihy_BASE_URL"
_DEFAULT_MODEL = "gpt-5.4"
_TEST_MESSAGE = "Reply with exactly one word: OK"


def _mask_key(key: str) -> str:
    if len(key) <= 8:
        return "****"
    return f"{key[:4]}...{key[-4:]}"


def _build_client(model: str) -> OpenRouterNewAPIClient:
    return OpenRouterNewAPIClient(
        model=model,
        api_key_env=_API_KEY_ENV,
        base_url_env=_BASE_URL_ENV,
        max_tokens=64,
        reasoning_effort=None,
        async_mode=True,
    )


async def _list_models(client: OpenRouterNewAPIClient) -> list[str]:
    from openai import OpenAI

    sync_client = OpenAI(
        api_key=client.api_key,
        base_url=client.base_url,
        timeout=30,
    )
    response = sync_client.models.list()
    return [getattr(item, "id", None) for item in (response.data or []) if getattr(item, "id", None)]


async def _chat_test(client: OpenRouterNewAPIClient) -> str:
    resp = await client.chat([{"role": "user", "content": _TEST_MESSAGE}])
    return resp.content


async def main(env_path: Optional[str], model: str, max_retries: int) -> int:
    load_env(override=True, env_path=env_path)

    import os
    api_key = os.getenv(_API_KEY_ENV, "")
    base_url = os.getenv(_BASE_URL_ENV, "")

    print("=== apihy 连通性测试 ===")
    print(f"Base URL : {base_url}")
    print(f"Model    : {model}")
    print(f"API key  : {_mask_key(api_key) if api_key else '(未设置)'}")
    print()

    if not api_key or not base_url:
        print(f"错误：请在 .env 中设置 {_API_KEY_ENV} 和 {_BASE_URL_ENV}", file=sys.stderr)
        return 1

    try:
        client = _build_client(model)
    except ValueError as exc:
        print(f"初始化失败：{exc}", file=sys.stderr)
        return 1

    # ── 1. 模型列表 ───────────────────────────────────────────────────────────
    print("1. 获取模型列表 ...")
    try:
        models = await _list_models(client)
        print(f"   可用模型（{len(models)} 个）：")
        for m in models:
            marker = " ◀ 当前" if m == model else ""
            print(f"     {m}{marker}")
    except Exception as exc:
        print(f"   ✗ 模型列表失败：{exc}")

    print()

    # ── 2. Chat 测试 ──────────────────────────────────────────────────────────
    print(f"2. Chat 测试（{_TEST_MESSAGE!r}）...")
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            reply = await _chat_test(client)
            print(f"   ✓ 回复：{reply!r}")
            usage = client.get_usage_stats()
            print(f"   Token 用量：{usage}")
            break
        except Exception as exc:
            last_exc = exc
            print(f"   ✗ 第 {attempt} 次失败：{exc}")
            if attempt < max_retries:
                await asyncio.sleep(2)
    else:
        print(f"   所有 {max_retries} 次尝试均失败", file=sys.stderr)
        return 1

    await client.aclose()
    print("\n=== 测试通过 ===")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="测试 apihy 网关连通性")
    parser.add_argument("env_path", nargs="?", help=".env 文件路径（默认自动查找）")
    parser.add_argument("--model", default=_DEFAULT_MODEL, help=f"测试用模型（默认 {_DEFAULT_MODEL}）")
    parser.add_argument("--retries", type=int, default=2, dest="max_retries", help="chat 测试最大重试次数")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sys.exit(asyncio.run(main(args.env_path, args.model, args.max_retries)))
