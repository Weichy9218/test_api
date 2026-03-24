#!/usr/bin/env python3
"""
检查 .env 中 NEW_BASE_URL / NEW_API_KEY 的连通性。

功能:
  1. 检查 API 连通性（/v1/models 列表）
  2. 列出所有可用模型（按 provider 分组）
  3. 发送一条最短测试请求，验证 chat completion 实际可用

用法:
  python3 check_new_api.py                    # 默认读 /home/dataset-local/env/.env
  python3 check_new_api.py /path/to/other.env # 指定 .env 路径
"""

import json
import os
import sys
import urllib.request
import urllib.error


def load_env(env_path: str) -> dict:
    """从 .env 文件加载变量，只取 NEW_* 开头的。"""
    env = {}
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key in ("NEW_BASE_URL", "NEW_API_KEY"):
                    env[key] = value
    return env


def mask_key(key: str) -> str:
    """脱敏显示 API key，只保留首尾各 4 位。"""
    if len(key) <= 8:
        return "****"
    return f"{key[:4]}...{key[-4:]}"


def api_get(base_url: str, api_key: str, path: str) -> dict:
    """发起 GET 请求到 API。"""
    url = f"{base_url.rstrip('/')}{path}"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {api_key}",
    })
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def test_chat(base_url: str, api_key: str, model: str) -> dict:
    """发送一条最短 chat completion 测试请求。"""
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "Say hi"}],
        "max_tokens": 5,
    }).encode()
    req = urllib.request.Request(url, data=payload, headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    })
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def group_models(models: list) -> dict:
    """按 owned_by 分组模型。"""
    groups = {}
    for m in models:
        owner = m.get("owned_by", "unknown")
        groups.setdefault(owner, []).append(m["id"])
    return dict(sorted(groups.items()))


def main():
    env_path = sys.argv[1] if len(sys.argv) > 1 else "/home/dataset-local/env/.env"

    if not os.path.exists(env_path):
        print(f"❌ 找不到 .env 文件: {env_path}")
        sys.exit(1)

    env = load_env(env_path)
    base_url = env.get("NEW_BASE_URL", "")
    api_key = env.get("NEW_API_KEY", "")

    if not base_url or not api_key:
        print("❌ .env 中缺少 NEW_BASE_URL 或 NEW_API_KEY")
        sys.exit(1)

    print(f"📡 Base URL : {base_url}")
    print(f"🔑 API Key  : {mask_key(api_key)}")
    print()

    # --- Step 1: 连通性检测 ---
    print("=" * 50)
    print("Step 1: 检查 API 连通性 (GET /v1/models)")
    print("=" * 50)
    try:
        data = api_get(base_url, api_key, "/v1/models")
        models = data.get("data", [])
        print(f"✅ 连通正常，共 {len(models)} 个可用模型\n")
    except Exception as e:
        print(f"❌ 连通失败: {e}")
        sys.exit(1)

    # --- Step 2: 模型列表（按 provider 分组）---
    print("=" * 50)
    print("Step 2: 可用模型列表（按 provider 分组）")
    print("=" * 50)
    groups = group_models(models)
    for owner, ids in groups.items():
        print(f"\n▸ {owner} ({len(ids)} 个)")
        for mid in sorted(ids):
            print(f"    {mid}")

    # --- Step 3: Chat Completion 测试 ---
    print()
    print("=" * 50)
    print("Step 3: Chat Completion 测试")
    print("=" * 50)
    # 优先用 gpt-4o-mini，回退到第一个可用模型
    test_candidates = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-3.5-turbo", "deepseek-chat"]
    test_model = None
    available_ids = {m["id"] for m in models}
    for c in test_candidates:
        if c in available_ids:
            test_model = c
            break
    if not test_model and models:
        test_model = models[0]["id"]

    if not test_model:
        print("⚠️  无可用模型，跳过 chat 测试")
    else:
        try:
            print(f"使用模型: {test_model}")
            result = test_chat(base_url, api_key, test_model)
            reply = result["choices"][0]["message"]["content"]
            print(f"✅ Chat 正常，回复: {reply!r}")
        except Exception as e:
            print(f"❌ Chat 测试失败: {e}")

    print()
    print("=" * 50)
    print("完成")
    print("=" * 50)
    print("⚠️  余额查询: new-api 不提供公开余额接口，请登录 dashboard 查看。")


if __name__ == "__main__":
    main()
