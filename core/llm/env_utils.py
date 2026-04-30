import os
from typing import Optional

from dotenv import find_dotenv, load_dotenv


def load_env(override: bool = False, env_path: Optional[str] = None) -> None:
    if env_path:
        load_dotenv(env_path, override=override)
    else:
        load_dotenv(find_dotenv(usecwd=True), override=override)
    _sync_proxy_env()


def resolve_env_value(env_name: Optional[str]) -> Optional[str]:
    candidate = str(env_name or "").strip()
    if not candidate:
        return None
    value = os.getenv(candidate)
    if value is None:
        return None
    value = value.strip()
    return value or None


def resolve_client_setting(
    explicit_value: Optional[str],
    *,
    preferred_env: Optional[str] = None,
    fallback_envs: tuple[str, ...] = (),
    default: Optional[str] = None,
) -> tuple[Optional[str], str]:
    if explicit_value is not None:
        candidate = str(explicit_value).strip()
        if candidate:
            return candidate, "explicit"
        return default, "default" if default is not None else "missing"

    env_candidates = ((preferred_env,) if preferred_env else tuple()) + tuple(fallback_envs)
    for env_name in env_candidates:
        value = resolve_env_value(env_name)
        if value:
            return value, env_name

    if default is not None:
        return default, "default"
    return None, "missing"


def _sync_proxy_env() -> None:
    mappings = [
        ("http_proxy", "HTTP_PROXY"),
        ("https_proxy", "HTTPS_PROXY"),
        ("all_proxy", "ALL_PROXY"),
        ("no_proxy", "NO_PROXY"),
    ]
    for lower, upper in mappings:
        if os.getenv(upper) is None:
            value = os.getenv(lower)
            if value:
                os.environ[upper] = value
