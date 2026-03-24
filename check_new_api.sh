#!/usr/bin/env bash
# 检查 .env 中 NEW_BASE_URL / NEW_API_KEY 的连通性（Bash 版）
# 用法: bash check_new_api.sh [env_path]

set -euo pipefail

ENV_FILE="${1:-/home/dataset-local/env/.env}"

if [ ! -f "$ENV_FILE" ]; then
    echo "❌ 找不到 .env 文件: $ENV_FILE"
    exit 1
fi

# 加载 NEW_* 变量
BASE_URL=$(grep '^NEW_BASE_URL=' "$ENV_FILE" | cut -d= -f2- | tr -d '"' | tr -d "'")
API_KEY=$(grep '^NEW_API_KEY=' "$ENV_FILE" | cut -d= -f2- | tr -d '"' | tr -d "'")

if [ -z "$BASE_URL" ] || [ -z "$API_KEY" ]; then
    echo "❌ .env 中缺少 NEW_BASE_URL 或 NEW_API_KEY"
    exit 1
fi

# 脱敏显示 key
MASKED="${API_KEY:0:4}...${API_KEY: -4}"
echo "📡 Base URL : $BASE_URL"
echo "🔑 API Key  : $MASKED"
echo ""

echo "=================================================="
echo "Step 1: 检查 API 连通性 (GET /v1/models)"
echo "=================================================="
HTTP_CODE=$(curl -s -o /tmp/_newapi_models.json -w "%{http_code}" \
    -H "Authorization: Bearer $API_KEY" \
    "$BASE_URL/v1/models" 2>&1)

if [ "$HTTP_CODE" -ne 200 ]; then
    echo "❌ 连通失败，HTTP $HTTP_CODE"
    cat /tmp/_newapi_models.json 2>/dev/null || true
    exit 1
fi

MODEL_COUNT=$(python3 -c "import json; d=json.load(open('/tmp/_newapi_models.json')); print(len(d.get('data',[])))")
echo "✅ 连通正常，共 $MODEL_COUNT 个可用模型"
echo ""

echo "=================================================="
echo "Step 2: 可用模型列表（按 provider 分组）"
echo "=================================================="
python3 -c "
import json
data = json.load(open('/tmp/_newapi_models.json'))
models = data.get('data', [])
groups = {}
for m in models:
    groups.setdefault(m.get('owned_by','unknown'), []).append(m['id'])
for owner in sorted(groups):
    print(f'\n▸ {owner} ({len(groups[owner])} 个)')
    for mid in sorted(groups[owner]):
        print(f'    {mid}')
"

echo ""
echo "=================================================="
echo "Step 3: Chat Completion 测试"
echo "=================================================="
# 尝试 gpt-4o-mini, 回退到第一个可用模型
TEST_MODEL=$(python3 -c "
import json
data = json.load(open('/tmp/_newapi_models.json'))
ids = {m['id'] for m in data.get('data',[])}
for c in ['gpt-4o-mini','gpt-4.1-mini','anthropic/claude-sonnet-4.5','xiaomi/mimo-v2-pro','gpt-3.5-turbo','deepseek-chat']:
    if c in ids: print(c); exit()
models = data.get('data',[])
if models: print(models[0]['id'])
")

if [ -z "$TEST_MODEL" ]; then
    echo "⚠️  无可用模型，跳过 chat 测试"
else
    echo "使用模型: $TEST_MODEL"
    CHAT_RESULT=$(curl -s --max-time 30 \
        -H "Authorization: Bearer $API_KEY" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$TEST_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi\"}],\"max_tokens\":64}" \
        "$BASE_URL/v1/chat/completions" 2>&1)

    # 提取回复内容
    REPLY=$(echo "$CHAT_RESULT" | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
    print(d['choices'][0]['message']['content'])
except: print('解析失败')
" 2>/dev/null || echo "请求失败")

    if [ "$REPLY" = "请求失败" ] || [ "$REPLY" = "解析失败" ]; then
        echo "❌ Chat 测试失败"
        echo "$CHAT_RESULT" | head -5
    else
        echo "✅ Chat 正常，回复: $REPLY"
    fi
fi

echo ""
echo "=================================================="
echo "完成"
echo "=================================================="
echo "⚠️  余额查询: new-api 不提供公开余额接口，请登录 dashboard 查看。"
