#!/usr/bin/env bash
# ==============================================================================
# check_openrouter.sh — OpenRouter API connectivity & credits checker
#
# 功能:
#   1. 从 .env 文件加载 OPENROUTER_API_KEY
#   2. 调用 /auth/key 验证 token 有效性并获取余额信息
#   3. 调用 /models 列出可用模型（连通性二次验证）
#
# 用法:
#   bash check_openrouter.sh [path/to/.env]
#   默认读取 /home/dataset-local/env/.env
# ==============================================================================

set -euo pipefail  # -e: 出错即退出  -u: 未定义变量报错  -o pipefail: 管道中任一命令失败则整体失败

# ---- 配置 ----
ENV_FILE="${1:-/home/dataset-local/env/.env}"  # 第一个参数或默认路径

# ---- 加载 .env ----
if [[ ! -f "$ENV_FILE" ]]; then
  echo "❌ .env file not found: $ENV_FILE"
  exit 1
fi

# set -a 自动 export 后续 source 中定义的所有变量
set -a
source "$ENV_FILE"
set +a

# ---- 校验必需变量 ----
if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "❌ OPENROUTER_API_KEY not set in $ENV_FILE"
  exit 1
fi

BASE_URL="${OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}"
MASKED="${OPENROUTER_API_KEY:0:10}...${OPENROUTER_API_KEY: -4}"  # 仅显示首尾几位，避免泄露

echo "=== OpenRouter API Check ==="
echo "Key:  $MASKED"
echo "Base: $BASE_URL"
echo ""

# ---- 1. Auth & Credits (认证 + 余额) ----
echo "--- Auth & Credits ---"
RESP=$(curl -sf "$BASE_URL/auth/key" \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" 2>&1) || {
  echo "❌ Auth request failed — token 无效或网络不通"
  exit 1
}

# 用 python3 解析 JSON，格式化输出关键字段
echo "$RESP" | python3 -c "
import sys, json
d = json.load(sys.stdin)['data']
print(f\"  Status:        ✅ authenticated\")
print(f\"  Total limit:   \${d['limit']:.2f}\")
print(f\"  Remaining:     \${d['limit_remaining']:.2f}\")
print(f\"  Usage (all):   \${d['usage']:.4f}\")
print(f\"  Usage (month): \${d['usage_monthly']:.4f}\")
print(f\"  Free tier:     {d['is_free_tier']}\")
rl = d.get('rate_limit', {})
print(f\"  Rate limit:    {rl.get('requests', '?')} req / {rl.get('interval', '?')}\")
"
echo ""

# ---- 2. Models list (模型列表，二次验证连通性) ----
echo "--- Models (first 5) ---"
curl -sf "$BASE_URL/models" \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" 2>/dev/null | \
  python3 -c "
import sys, json
data = json.load(sys.stdin)['data']
for m in data[:5]:
    print(f\"  {m['id']}\")
print(f\"  ... ({len(data)} models total)\")
" || echo "  ⚠️ Could not fetch model list"

echo ""
echo "=== Done ==="
