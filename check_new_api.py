#!/usr/bin/env python3
"""
检查 new-api / OpenRouter 等 OpenAI 兼容 API 的连通性、模型列表和 chat 测试。

功能:
  1. 检查 API 连通性（GET /v1/models）
  2. 检查指定模型是否存在
  3. 发送 chat completion 测试请求，给出可用模型和具体使用实例
  4. 按 provider 分组展示模型

用法:
  python3 check_new_api.py [env_path] [model_name]

  # 默认读 /home/dataset-local/env/.env，自动选模型测试
  python3 check_new_api.py

  # 指定 .env 路径
  python3 check_new_api.py /path/to/.env

  # 检查指定模型名是否存在并测试
  python3 check_new_api.py /path/to/.env gpt-4o-mini

  # 测试多个模型
  python3 check_new_api.py "" gpt-4o-mini claude-sonnet-4.5 xiaomi/mimo-v2-pro
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
    """发送 chat completion 测试请求。"""
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "Say hi in one word"}],
        "max_tokens": 64,
    }).encode()
    req = urllib.request.Request(url, data=payload, headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    })
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def check_model_exists(model_name: str, models: list) -> bool:
    """检查模型名是否在可用列表中。"""
    available_ids = {m["id"] for m in models}
    return model_name in available_ids


def group_models(models: list) -> dict:
    """按 owned_by 分组模型。"""
    groups = {}
    for m in models:
        owner = m.get("owned_by", "unknown")
        groups.setdefault(owner, []).append(m["id"])
    return dict(sorted(groups.items()))


def print_usage_example(base_url: str, model: str):
    """打印该模型的 curl / Python 使用实例。"""
    print(f"\n📌 使用实例 — {model}")
    print("-" * 50)

    # curl 示例
    print(f"\ncurl:")
    print(f'  curl -s {base_url.rstrip("/")}/v1/chat/completions \\')
    print(f'    -H "Authorization: Bearer $API_KEY" \\')
    print(f'    -H "Content-Type: application/json" \\')
    print(f"    -d '{{")
    print(f'      "model": "{model}",')
    print(f'      "messages": [{{"role": "user", "content": "Hello"}}],')
    print(f'      "max_tokens": 256')
    print(f"    }}'")

    # Python 示例
    print(f"\nPython (requests):")
    print(f'  import requests')
    print(f'  ')
    print(f'  resp = requests.post(')
    print(f'      "{base_url.rstrip("/")}/v1/chat/completions",')
    print(f'      headers={{"Authorization": f"Bearer {{API_KEY}}"}},')
    print(f'      json={{')
    print(f'          "model": "{model}",')
    print(f'          "messages": [{{"role": "user", "content": "Hello"}}],')
    print(f'          "max_tokens": 256,')
    print(f'      }},')
    print(f'  )')
    print(f'  print(resp.json()["choices"][0]["message"]["content"])')


def main():
    args = sys.argv[1:]

    # 解析 env 路径
    env_path = args[0] if args and not args[0].startswith("-") else "/home/dataset-local/env/.env"
    if env_path == "":
        env_path = "/home/dataset-local/env/.env"

    # 解析要测试的模型列表
    test_models = [a for a in args[1:] if a] if len(args) > 1 else []

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
        print(f"✅ 连通正常，平台共注册 {len(models)} 个模型\n")
    except Exception as e:
        print(f"❌ 连通失败: {e}")
        sys.exit(1)

    # --- Step 2: 模型名检查 ---
    if test_models:
        print("=" * 50)
        print("Step 2: 模型名检查")
        print("=" * 50)
        for m in test_models:
            if check_model_exists(m, models):
                print(f"  ✅ {m} — 存在于模型列表")
            else:
                # 模糊匹配
                similar = [x["id"] for x in models if m.lower() in x["id"].lower()]
                print(f"  ❌ {m} — 未找到")
                if similar:
                    print(f"     相似模型: {', '.join(similar[:5])}")
        print()

    # --- Step 3: 模型列表（按 provider 分组）---
    print("=" * 50)
    print("Step 3: 可用模型列表（按 provider 分组）")
    print("=" * 50)
    groups = group_models(models)
    for owner, ids in groups.items():
        print(f"\n▸ {owner} ({len(ids)} 个)")
        for mid in sorted(ids):
            print(f"    {mid}")

    # --- Step 4: Chat Completion 测试 ---
    print()
    print("=" * 50)
    print("Step 4: Chat Completion 测试")
    print("=" * 50)

    # 确定要测试的模型
    if test_models:
        candidates = test_models
    else:
        candidates = [
            "gpt-4o-mini", "gpt-4.1-mini",
            "anthropic/claude-sonnet-4.5", "xiaomi/mimo-v2-pro",
            "gpt-3.5-turbo", "deepseek-chat",
        ]

    available_ids = {m["id"] for m in models}
    tested_any = False

    for model in candidates:
        if model not in available_ids:
            print(f"\n⚠️  {model} — 不在模型列表中，跳过")
            continue

        tested_any = True
        try:
            print(f"\n🧪 测试: {model}")
            result = test_chat(base_url, api_key, model)
            reply = result["choices"][0]["message"]["content"]
            usage = result.get("usage", {})
            cost = usage.get("cost", 0)
            print(f"   ✅ 成功")
            print(f"   💬 回复: {reply!r}")
            if cost:
                print(f"   💰 费用: ${cost:.6f}")

            # 打印使用实例
            print_usage_example(base_url, model)
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            try:
                err = json.loads(body).get("error", {})
                msg = err.get("message", body[:200])
            except:
                msg = body[:200]
            print(f"   ❌ 失败: {msg}")
        except Exception as e:
            print(f"   ❌ 失败: {e}")

    if not tested_any:
        print("⚠️  无指定模型可测试")
        # 回退到第一个可用模型
        if models:
            fallback = models[0]["id"]
            print(f"   尝试回退模型: {fallback}")
            try:
                result = test_chat(base_url, api_key, fallback)
                reply = result["choices"][0]["message"]["content"]
                print(f"   ✅ {fallback} 回复: {reply!r}")
                print_usage_example(base_url, fallback)
            except Exception as e:
                print(f"   ❌ 失败: {e}")

    print()
    print("=" * 50)
    print("完成")
    print("=" * 50)
    print("⚠️  余额查询: new-api 不提供公开余额接口，请登录 dashboard 查看。")


if __name__ == "__main__":
    main()
