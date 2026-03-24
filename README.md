# check_api

检查 API 连通性和可用模型的工具集。

## 文件列表

| 文件 | 说明 |
|---|---|
| `check_openrouter.py` | 检查 OpenRouter API（余额 + 模型列表） |
| `check_openrouter.sh` | Bash 版 OpenRouter 检查 |
| `check_new_api.py` | 检查 new-api 中转（连通性 + 模型列表 + chat 测试） |
| `check_new_api.sh` | Bash 版 new-api 检查 |

## 环境变量

脚本从 `/home/dataset-local/env/.env` 读取配置：

```
# OpenRouter（check_openrouter.*）
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# new-api 中转（check_new_api.*）
NEW_API_KEY=sk-...
NEW_BASE_URL=https://new-api.haoxiang.ai
```

## 用法

```bash
# Python 版（推荐）
python3 check_new_api.py                          # 默认读 /home/dataset-local/env/.env
python3 check_new_api.py /path/to/other.env       # 指定 .env 路径

# Bash 版
bash check_new_api.sh
bash check_new_api.sh /path/to/other.env

# OpenRouter 同理
python3 check_openrouter.py
bash check_openrouter.sh
```

## check_new_api 功能

1. **连通性检测** — 调用 `GET /v1/models`，确认 API key 有效且网络可达
2. **模型列表** — 按 provider 分组展示所有可用模型
3. **Chat 测试** — 发送最短请求（"Say hi"，max_tokens=5），验证 chat completion 实际可用

## 余额说明

- **OpenRouter**: 支持余额查询（`/api/v1/auth/key`）
- **new-api 中转**: 不提供公开余额接口，需登录 dashboard 查看

## 实现方式

- Python 版使用标准库（`urllib`），无第三方依赖
- API key 脱敏显示（只保留首尾各 4 位）
- chat 测试优先使用 `gpt-4o-mini`，自动回退到其他轻量模型
