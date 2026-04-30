"""Check OpenRouter connectivity, credits, and model list (OPENROUTER_API_KEY)."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request

from openai import OpenAI

from core.llm.env_utils import load_env, resolve_env_value

_API_KEY_ENV = "OPENROUTER_API_KEY"
_BASE_URL_ENV = "OPENROUTER_BASE_URL"
_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


def _mask(key: str) -> str:
    return f"{key[:10]}...{key[-4:]}" if len(key) > 14 else f"{key[:4]}***"


def _api_get(url: str, api_key: str) -> dict:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("env_path", nargs="?", default=".env")
    args = parser.parse_args()
    load_env(env_path=args.env_path, override=True)

    api_key = resolve_env_value(_API_KEY_ENV) or ""
    base_url = resolve_env_value(_BASE_URL_ENV) or _DEFAULT_BASE_URL

    if not api_key:
        print(f"Error: {_API_KEY_ENV} not set", file=sys.stderr)
        return 1

    print(f"Key:  {_mask(api_key)}")
    print(f"Base: {base_url}")
    print()

    # Credits
    try:
        data = _api_get(f"{base_url}/auth/key", api_key)["data"]
        print("Credits:")
        print(f"  Limit:     ${data['limit']:.2f}")
        print(f"  Remaining: ${data['limit_remaining']:.2f}")
        print(f"  Used:      ${data['usage']:.4f}")
        rl = data.get("rate_limit", {})
        print(f"  Rate:      {rl.get('requests', '?')} req/{rl.get('interval', '?')}")
    except Exception as exc:
        print(f"Credits check failed: {exc}", file=sys.stderr)
    print()

    # Models
    try:
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=15)
        models = [m.id for m in client.models.list().data]
        print(f"Models ({len(models)}):")
        for m in models[:8]:
            print(f"  {m}")
        if len(models) > 8:
            print(f"  ... and {len(models) - 8} more")
    except Exception as exc:
        print(f"Model list failed: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
