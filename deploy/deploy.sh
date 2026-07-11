#!/usr/bin/env bash
# ============================================================
# 财务RAG系统 — 日常升级脚本（在内网服务器上执行）
# 作用：只更新代码，自动备份并保留 data/ 和 .env，然后重启服务
# 用法：bash deploy/deploy.sh
# ============================================================
set -euo pipefail

APP_DIR="/opt/finance-rag"          # ← 改成你的实际部署目录
SERVICE="finance-rag"               # systemd 服务名
BACKUP_DIR="$APP_DIR/backups"

cd "$APP_DIR"

echo "[1/5] 备份数据与配置（data/ + .env）..."
mkdir -p "$BACKUP_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
tar czf "$BACKUP_DIR/data_$STAMP.tar.gz" data .env 2>/dev/null || true
echo "    已备份到 $BACKUP_DIR/data_$STAMP.tar.gz"

echo "[2/5] 拉取/更新代码..."
if [ -d .git ]; then
    git pull
else
    echo "    非git目录：请先手动用新代码覆盖 *.py（勿动 data/ 和 .env），然后重跑本脚本"
fi

echo "[3/5] 更新依赖（如有变化）..."
.venv/bin/pip install -r requirements.txt --quiet

echo "[4/5] 重启服务..."
sudo systemctl restart "$SERVICE"

echo "[5/5] 检查状态..."
sleep 3
sudo systemctl --no-pager status "$SERVICE" | head -8

echo ""
echo "✅ 升级完成。如需回滚数据：tar xzf $BACKUP_DIR/data_$STAMP.tar.gz"
