#!/usr/bin/env bash
# AgentFrame 一键部署脚本 (本机 systemd)
set -e

FRAMEWORK_DIR="/root/.openclaw/workspace/agentframe"
INSTALL_DIR="/opt/agentframe"
STATE_DIR="/var/lib/agentframe"

echo "📦 AgentFrame 部署脚本"
echo "======================"

# 1. 复制到 /opt
if [ ! -d "$INSTALL_DIR" ]; then
    mkdir -p "$INSTALL_DIR"; cp -r "$FRAMEWORK_DIR" "$INSTALL_DIR/agentframe"
    echo "✅ 已复制到 $INSTALL_DIR"
else
    cp -r "$FRAMEWORK_DIR" "$INSTALL_DIR/"
    echo "✅ 已更新 $INSTALL_DIR"
fi

# 2. 状态目录
mkdir -p "$STATE_DIR"
echo "✅ 状态目录: $STATE_DIR"

# 3. systemd 服务
cp "$FRAMEWORK_DIR/deploy/agentframe.service" /etc/systemd/system/agentframe.service
# 修正工作目录
sed -i "s|WorkingDirectory=/opt/agentframe|WorkingDirectory=$INSTALL_DIR/agentframe|" /etc/systemd/system/agentframe.service
systemctl daemon-reload
systemctl enable agentframe
echo "✅ systemd 服务已注册"

# 4. API key 检查
if [ -z "$AGENTFRAME_API_KEY" ]; then
    echo "⚠️ 未设置 AGENTFRAME_API_KEY, 服务将使用 mock (离线模式)"
    echo "   设置: export AGENTFRAME_API_KEY=sk-xxx 后重启服务"
else
    sed -i "s|Environment=\"AGENTFRAME_API_KEY=.*|Environment=\"AGENTFRAME_API_KEY=$AGENTFRAME_API_KEY\"|" /etc/systemd/system/agentframe.service
    systemctl daemon-reload
    echo "✅ API key 已配置"
fi

# 5. 启动
systemctl restart agentframe || true
sleep 2
if systemctl is-active agentframe >/dev/null 2>&1; then
    echo "✅ 服务运行中: http://0.0.0.0:8090/health"
else
    echo "⚠️ 服务启动失败, 查看日志: journalctl -u agentframe -n 50"
fi

echo "完成!"
