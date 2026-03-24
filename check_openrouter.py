#!/usr/bin/env python3
"""
check_openrouter.py — OpenRouter API connectivity & credits checker

功能:
    1. 从 .env 文件加载 OPENROUTER_API_KEY
    2. 调用 /auth/key 验证 token 有效性并获取余额信息
    3. 调用 /models 列出可用模型（连通性二次验证）

用法:
    python3 check_openrouter.py [path/to/.env]
    默认读取 /home/dataset-local/env/.env

依赖:
    - requests (推荐) 或内置 urllib
    - Python 3.8+
"""

import json
import sys
from pathlib import Path

# ---- HTTP 后端选择: 优先 requests，回退到 urllib ----
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    import urllib.request
    HAS_REQUESTS = False


def load_env(path: str) -> dict:
    """
    从 .env 文件解析 key=value 对。
    支持 # 注释行、空行、以及引号包裹的值。
    """
    env = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                env[key.strip()] = val.strip().strip("'\"")
    return env


def api_get(url: str, api_key: str) -> dict:
    """
    发送 GET 请求到 OpenRouter API。
    自动附加 Authorization header，返回 JSON dict。
    """
    headers = {"Authorization": f"Bearer {api_key}"}

    if HAS_REQUESTS:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        return resp.json()
    else:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())


def mask_key(key: str) -> str:
    """脱敏显示 API key，只展示首 10 位和末 4 位。"""
    if len(key) <= 14:
        return key[:4] + "***"
    return f"{key[:10]}...{key[-4:]}"


def main():
    # ---- 加载配置 ----
    env_path = sys.argv[1] if len(sys.argv) > 1 else "/home/dataset-local/env/.env"

    if not Path(env_path).exists():
        print(f"❌ .env not found: {env_path}")
        sys.exit(1)

    env = load_env(env_path)
    api_key = env.get("OPENROUTER_API_KEY", "")
    base = env.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    if not api_key:
        print("❌ OPENROUTER_API_KEY missing from .env")
        sys.exit(1)

    print(f"=== OpenRouter API Check ===")
    print(f"Key:  {mask_key(api_key)}")
    print(f"Base: {base}\n")

    # ---- 1. Auth & Credits (认证 + 余额) ----
    print("--- Auth & Credits ---")
    try:
        data = api_get(f"{base}/auth/key", api_key)["data"]
    except Exception as e:
        print(f"❌ Auth request failed: {e}")
        sys.exit(1)

    print(f"  Status:        ✅ authenticated")
    print(f"  Total limit:   ${data['limit']:.2f}")
    print(f"  Remaining:     ${data['limit_remaining']:.2f}")
    print(f"  Usage (all):   ${data['usage']:.4f}")
    print(f"  Usage (month): ${data['usage_monthly']:.4f}")
    print(f"  Free tier:     {data['is_free_tier']}")

    rl = data.get("rate_limit", {})
    print(f"  Rate limit:    {rl.get('requests', '?')} req / {rl.get('interval', '?')}")
    print()

    # ---- 2. Models list (模型列表，二次验证连通性) ----
    print("--- Models (first 5) ---")
    try:
        models = api_get(f"{base}/models", api_key)["data"]
        for m in models[:5]:
            print(f"  {m['id']}")
        print(f"  ... ({len(models)} models total)")
    except Exception as e:
        print(f"  ⚠️ Could not fetch model list: {e}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
