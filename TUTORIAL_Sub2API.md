# Sub2API 使用教程

> 本教程展示如何通过 **Sub2API**（OpenAI 兼容网关）在本地调用大模型。  
> 所有示例均通过 `SUB2API_API_KEY` 和 `SUB2API_BASE_URL` 两个环境变量读取凭证，**不需要把 key 写进代码**。

---

## 目录

1. [准备工作](#1-准备工作)
2. [配置环境变量](#2-配置环境变量)
3. [快速验证：直接用 openai SDK](#3-快速验证直接用-openai-sdk)
4. [列出可用模型](#4-列出可用模型)
5. [使用封装好的 Sub2API 客户端](#5-使用封装好的-sub2api-客户端)
6. [异步调用 / Responses API](#6-异步调用--responses-api)
7. [Streaming 流式输出](#7-streaming-流式输出)
8. [常见问题](#8-常见问题)

---

## 1. 准备工作

**依赖 Python ≥ 3.10**

```bash
# 克隆仓库
git clone git@github.com:Weichy9218/test_api.git
cd test_api

# 安装依赖（推荐 uv，也可以用 pip）
uv sync
# 或
pip install -e .
```

---

## 2. 配置环境变量

复制模板，填入你拿到的凭证：

```bash
cp .env.example .env
```

编辑 `.env`：

```dotenv
SUB2API_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx   # 你的 API key
SUB2API_BASE_URL=https://ie-crs.haoxiang.ai/v1  # 网关地址（向发放 key 的人确认）
```

> `.env` 已被 `.gitignore` 排除，不会意外提交到仓库。

---

## 3. 快速验证：直接用 openai SDK

Sub2API 完全兼容 OpenAI SDK，只需把 `base_url` 指向网关即可。

```python
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # 从 .env 加载环境变量

client = OpenAI(
    api_key=os.environ["SUB2API_API_KEY"],
    base_url=os.environ["SUB2API_BASE_URL"],
)

response = client.chat.completions.create(
    model="gpt-4.1-mini",   # 替换为你想用的模型
    messages=[{"role": "user", "content": "你好，介绍一下你自己"}],
    max_tokens=200,
)
print(response.choices[0].message.content)
```

运行：

```bash
uv run python -c "
import os; from dotenv import load_dotenv; load_dotenv()
from openai import OpenAI
client = OpenAI(api_key=os.environ['SUB2API_API_KEY'], base_url=os.environ['SUB2API_BASE_URL'])
r = client.chat.completions.create(model='gpt-4.1-mini', messages=[{'role':'user','content':'hi'}], max_tokens=20)
print(r.choices[0].message.content)
"
```

---

## 4. 列出可用模型

用 `model_name.py` 查询当前 API key 下有哪些可用模型：

```bash
uv run python model_name.py
```

输出示例：

```json
{
  "api_key_env": "SUB2API_API_KEY",
  "base_url_env": "SUB2API_BASE_URL",
  "model_count": 12,
  "models": [
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-5.4",
    "claude-3-7-sonnet-20250219",
    ...
  ]
}
```

也可以手动指定其他环境变量名：

```bash
uv run python model_name.py --api-key-env MY_KEY --base-url-env MY_URL
```

---

## 5. 使用封装好的 Sub2API 客户端

`core/llm/codex_sub2api_client.py` 提供了一个封装层，支持 Responses API（OpenAI 最新接口）、reasoning effort、工具调用等特性。

### 5.1 同步调用

```python
import asyncio
from core.llm import CodexSub2APIClient

client = CodexSub2APIClient(
    model="gpt-5.4",          # 或其他可用模型
    max_tokens=1024,
    reasoning_effort="high",  # None / "low" / "medium" / "high"
    async_mode=True,
)

async def main():
    response = await client.chat(
        messages=[
            {"role": "system", "content": "你是一个有帮助的助手。"},
            {"role": "user", "content": "用一句话解释什么是大语言模型"},
        ]
    )
    print(response.content)
    print(f"Token 用量: {response.usage}")

asyncio.run(main())
```

### 5.2 凭证来源优先级

客户端按以下顺序查找凭证，第一个非空的值生效：

| 优先级 | API Key 来源 | Base URL 来源 |
|--------|-------------|--------------|
| 1 (最高) | 构造函数 `api_key=` 参数 | 构造函数 `base_url=` 参数 |
| 2 | `api_key_env=` 指定的环境变量名 | `base_url_env=` 指定的环境变量名 |
| 3 | `SUB2API_API_KEY` | `SUB2API_BASE_URL` |
| 4 | `HAOXIANG_OPENAI_API_KEY` (旧名) | `HAOXIANG_BASE_URL` (旧名) |
| 5 (最低) | — | 内置默认地址 |

---

## 6. 异步调用 / Responses API

`CodexSub2APIClient` 默认走 **Responses API**（OpenAI `client.responses.create`），相比 Chat Completions 支持更丰富的推理控制。

```python
import asyncio
from core.llm import CodexSub2APIClient

client = CodexSub2APIClient(model="gpt-5.4", reasoning_effort="medium")

async def ask(question: str) -> str:
    resp = await client.chat([{"role": "user", "content": question}])
    return resp.content

result = asyncio.run(ask("1+1 等于几？请一步步推理"))
print(result)
```

---

## 7. Streaming 流式输出

使用原生 `openai` SDK 的 streaming 接口（适用于所有 OpenAI 兼容端点）：

```python
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["SUB2API_API_KEY"],
    base_url=os.environ["SUB2API_BASE_URL"],
)

with client.chat.completions.stream(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": "写一首关于秋天的短诗"}],
    max_tokens=200,
) as stream:
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="", flush=True)
print()  # 换行
```

---

## 8. 常见问题

**Q: 报错 `SUB2API_API_KEY not set`**  
确认 `.env` 文件在项目根目录，且 key 拼写正确（注意大小写）。

**Q: 报错 `Connection refused` / `SSL error`**  
确认 `SUB2API_BASE_URL` 末尾有 `/v1`，格式为 `https://your-host/v1`。

**Q: 不知道有哪些可用模型**  
运行 `uv run python model_name.py` 查询当前 key 下的全部模型。

**Q: 想用不同的 model 但不改代码**  
直接在 `.env` 里加一行 `DEFAULT_MODEL=gpt-4.1`，然后在代码里读 `os.environ.get("DEFAULT_MODEL", "gpt-4.1-mini")`。

**Q: 如何测试连通性**  
```bash
uv run python check_codex_sub2api.py
```
这个脚本会走一次完整的 Responses API streaming，打印模型回复和 token 用量。
