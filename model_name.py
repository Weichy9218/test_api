"""
model_name.py — list available models from a Sub2API-compatible endpoint.

Usage:
    python model_name.py                          # auto-detect credentials from .env
    python model_name.py --api-key-env MY_KEY_ENV --base-url-env MY_URL_ENV
"""

from __future__ import annotations

import argparse
import json
import os

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI


# Sub2API env var names (fall back to legacy names for backward compatibility)
DEFAULT_API_KEY_ENVS = (
    "SUB2API_API_KEY",
    "HAOXIANG_OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
)
DEFAULT_BASE_URL_ENVS = (
    "SUB2API_BASE_URL",
    "HAOXIANG_BASE_URL",
    "OPENROUTER_BASE_URL",
)
SELECTOR_API_KEY_ENV = "MODEL_NAME_API_KEY_ENV"
SELECTOR_BASE_URL_ENV = "MODEL_NAME_BASE_URL_ENV"


load_dotenv(find_dotenv(usecwd=True), override=False)


def _read_env(env_name: str) -> str:
    return os.getenv(env_name, "").strip()


def _pick_env_name(
    *,
    explicit_env: str | None,
    selector_env: str,
    fallback_envs: tuple[str, ...],
) -> str:
    candidate = str(explicit_env or "").strip()
    if candidate:
        return candidate

    selector_value = _read_env(selector_env)
    if selector_value:
        return selector_value

    for env_name in fallback_envs:
        if _read_env(env_name):
            return env_name
    return fallback_envs[0]


def resolve_credentials(
    *,
    api_key_env: str | None = None,
    base_url_env: str | None = None,
) -> tuple[str, str, str, str]:
    resolved_api_key_env = _pick_env_name(
        explicit_env=api_key_env,
        selector_env=SELECTOR_API_KEY_ENV,
        fallback_envs=DEFAULT_API_KEY_ENVS,
    )
    resolved_base_url_env = _pick_env_name(
        explicit_env=base_url_env,
        selector_env=SELECTOR_BASE_URL_ENV,
        fallback_envs=DEFAULT_BASE_URL_ENVS,
    )

    api_key = _read_env(resolved_api_key_env)
    base_url = _read_env(resolved_base_url_env)
    if not api_key or not base_url:
        raise ValueError(
            "Missing credentials. "
            f"Tried api key env `{resolved_api_key_env}` and base url env `{resolved_base_url_env}`.\n"
            "Set SUB2API_API_KEY and SUB2API_BASE_URL in your .env file."
        )
    return api_key, base_url, resolved_api_key_env, resolved_base_url_env


def list_model_names(
    *,
    api_key_env: str | None = None,
    base_url_env: str | None = None,
) -> list[str]:
    api_key, base_url, _, _ = resolve_credentials(
        api_key_env=api_key_env,
        base_url_env=base_url_env,
    )
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=60)
    response = client.models.list()
    return [
        model_id
        for item in getattr(response, "data", [])
        if (model_id := getattr(item, "id", None))
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List models available through a Sub2API / OpenAI-compatible endpoint."
    )
    parser.add_argument("--api-key-env", help="Env var name for the API key (default: SUB2API_API_KEY)")
    parser.add_argument("--base-url-env", help="Env var name for the base URL (default: SUB2API_BASE_URL)")
    args = parser.parse_args()

    _, _, resolved_api_key_env, resolved_base_url_env = resolve_credentials(
        api_key_env=args.api_key_env,
        base_url_env=args.base_url_env,
    )
    model_names = list_model_names(
        api_key_env=resolved_api_key_env,
        base_url_env=resolved_base_url_env,
    )
    print(
        json.dumps(
            {
                "api_key_env": resolved_api_key_env,
                "base_url_env": resolved_base_url_env,
                "model_count": len(model_names),
                "models": model_names,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
