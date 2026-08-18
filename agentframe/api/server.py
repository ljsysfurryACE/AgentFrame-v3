"""
AgentFrame REST API v1
=======================
多会话管理: 每个 session 独立 ContextEngine 实例

端点:
  GET    /health                         健康检查
  POST   /v1/sessions                    创建会话
  GET    /v1/sessions                    列出会话
  DELETE /v1/sessions/<sid>              删除会话
  POST   /v1/sessions/<sid>/ingest       摄入知识
  POST   /v1/sessions/<sid>/ingest_batch 批量摄入
  POST   /v1/sessions/<sid>/ask          查询+生成
  POST   /v1/sessions/<sid>/ask_hands    查询+生成(带工具)
  GET    /v1/sessions/<sid>/stats        会话状态
  POST   /v1/sessions/<sid>/forget       触发遗忘
  POST   /v1/sessions/<sid>/save         保存快照
  POST   /v1/sessions/<sid>/load         恢复快照
"""
import os
import uuid

from flask import Flask, request, jsonify

from ..config import AgentFrameConfig
from ..core.engine import ContextEngine


class AgentFrameAPI:
    """Flask 应用工厂"""

    def __init__(self, config: AgentFrameConfig = None, state_dir: str = None):
        self.config = config or AgentFrameConfig.from_env()
        self.state_dir = state_dir or os.environ.get(
            "AGENTFRAME_STATE_DIR", "/var/lib/agentframe")
        os.makedirs(self.state_dir, exist_ok=True)

        self.token = (self.config.api.token or "").strip()
        if not self.token:
            print("⚠️ 未配置 AGENTFRAME_API_TOKEN — 仅允许本机回环访问 (127.0.0.1)")
            print("   生产部署请设置: export AGENTFRAME_API_TOKEN=<随机长字符串>")

        self.sessions = {}   # sid -> ContextEngine

        self.app = Flask("agentframe")
        self.app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 请求体上限 2MB
        self._register_routes()

    # ============ 认证 ============

    def _authorize(self) -> bool:
        """
        认证策略:
        - 配置了 token → 必须携带 Authorization: Bearer <token>
        - 未配置 token → 仅接受本机回环来源 (生产应配置 token)
        """
        from flask import request as _req
        if self.token:
            auth = _req.headers.get("Authorization", "")
            return auth == f"Bearer {self.token}"
        # 无 token 模式: 仅回环
        return _req.remote_addr in ("127.0.0.1", "::1", "localhost")

    # ============ 会话管理 ============

    def _get_session(self, sid: str) -> ContextEngine:
        if sid not in self.sessions:
            raise KeyError(f"会话不存在: {sid}")
        return self.sessions[sid]

    def _create_session(self) -> str:
        sid = uuid.uuid4().hex[:12]
        self.sessions[sid] = ContextEngine(self.config)
        return sid

    def _restore_sessions(self):
        """启动时自动恢复已保存的会话快照"""
        if not os.path.isdir(self.state_dir):
            return
        for fname in sorted(os.listdir(self.state_dir)):
            if not fname.endswith(".json"):
                continue
            sid = fname[:-5]
            if sid in self.sessions:
                continue
            eng = ContextEngine(self.config)
            try:
                if eng.load(os.path.join(self.state_dir, fname)):
                    self.sessions[sid] = eng
                    print(f"   ↻ 恢复会话 {sid}")
            except Exception as e:
                print(f"   ⚠️ 恢复失败 {sid}: {e}")

    # ============ 路由 ============

    def _register_routes(self):
        app = self.app
        self._restore_sessions()

        @app.before_request
        def _auth_guard():
            """除 /health 外全部端点要求认证"""
            if request.path == "/health":
                return None
            if not self._authorize():
                return jsonify({"error": "未授权: 缺少有效 Bearer token"}), 401

        @app.get("/health")
        def health():
            return jsonify({"status": "ok", "version": "4.2.0",
                            "sessions": len(self.sessions),
                            "system": "AgentFrame Context API"})

        @app.post("/v1/sessions")
        def create_session():
            sid = self._create_session()
            return jsonify({"session_id": sid, "ok": True}), 201

        @app.get("/v1/sessions")
        def list_sessions():
            return jsonify({
                "sessions": [
                    {"id": sid, "chunks": len(eng.agent.chunk_meta),
                     "turns": len(eng.chat_history) // 2}
                    for sid, eng in self.sessions.items()
                ],
                "total": len(self.sessions),
            })

        @app.delete("/v1/sessions/<sid>")
        def delete_session(sid):
            if sid in self.sessions:
                del self.sessions[sid]
                return jsonify({"ok": True, "deleted": sid})
            return jsonify({"error": "会话不存在"}), 404

        @app.post("/v1/sessions/<sid>/ingest")
        def ingest(sid):
            try:
                eng = self._get_session(sid)
            except KeyError as e:
                return jsonify({"error": str(e)}), 404
            data = request.get_json(force=True, silent=True) or {}
            text = data.get("text", "")
            if not text:
                return jsonify({"error": "缺少 text 字段"}), 400
            cid = eng.ingest(text, data.get("tags", []), data.get("source"))
            return jsonify({"ok": True, "chunk_id": cid, "state": eng.stats()})

        @app.post("/v1/sessions/<sid>/ingest_batch")
        def ingest_batch(sid):
            try:
                eng = self._get_session(sid)
            except KeyError as e:
                return jsonify({"error": str(e)}), 404
            data = request.get_json(force=True, silent=True) or {}
            chunks = data.get("chunks", [])
            if not chunks:
                return jsonify({"error": "缺少 chunks 字段"}), 400
            ids = eng.ingest_batch(chunks)
            return jsonify({"ok": True, "chunk_ids": ids})

        @app.post("/v1/sessions/<sid>/ask")
        def ask(sid):
            try:
                eng = self._get_session(sid)
            except KeyError as e:
                return jsonify({"error": str(e)}), 404
            data = request.get_json(force=True, silent=True) or {}
            q = data.get("query", "")
            if not q:
                return jsonify({"error": "缺少 query 字段"}), 400
            result = eng.ask(q, chat=data.get("chat", True),
                             use_fast=data.get("use_fast", False))
            return jsonify({"ok": True, **result.to_dict()})

        @app.post("/v1/sessions/<sid>/ask_hands")
        def ask_hands(sid):
            try:
                eng = self._get_session(sid)
            except KeyError as e:
                return jsonify({"error": str(e)}), 404
            data = request.get_json(force=True, silent=True) or {}
            q = data.get("query", "")
            if not q:
                return jsonify({"error": "缺少 query 字段"}), 400
            result = eng.ask_with_hands(q)
            return jsonify({"ok": True, **result.to_dict()})

        @app.post("/v1/sessions/<sid>/autopilot")
        def autopilot(sid):
            """自主记忆管理: 分析对话, 自动记/忘"""
            try:
                eng = self._get_session(sid)
            except KeyError as e:
                return jsonify({"error": str(e)}), 404
            data = request.get_json(force=True, silent=True) or {}
            q = data.get("query", "")
            if not q:
                return jsonify({"error": "缺少 query 字段"}), 400
            # 用最近一次回答做记忆分析
            answer = ""
            if eng.chat_history and eng.chat_history[-1].get("role") == "assistant":
                answer = eng.chat_history[-1]["content"]
            decision = eng.autopilot_memory(q, answer)
            return jsonify({"ok": True, **decision})

        @app.get("/v1/sessions/<sid>/stats")
        def stats(sid):
            try:
                eng = self._get_session(sid)
            except KeyError as e:
                return jsonify({"error": str(e)}), 404
            return jsonify(eng.stats())

        @app.post("/v1/sessions/<sid>/forget")
        def forget(sid):
            try:
                eng = self._get_session(sid)
            except KeyError as e:
                return jsonify({"error": str(e)}), 404
            data = request.get_json(force=True, silent=True) or {}
            threshold = float(data.get("threshold", 0.15))
            victims = eng.forget(threshold)
            return jsonify({"ok": True, "forgotten": victims,
                            "remaining": len(eng.agent.chunk_meta)})

        @app.post("/v1/sessions/<sid>/save")
        def save(sid):
            try:
                eng = self._get_session(sid)
            except KeyError as e:
                return jsonify({"error": str(e)}), 404
            path = os.path.join(self.state_dir, f"{sid}.json")
            eng.save(path)
            return jsonify({"ok": True, "path": path})

        @app.post("/v1/sessions/<sid>/load")
        def load(sid):
            try:
                eng = self._get_session(sid)
            except KeyError as e:
                return jsonify({"error": str(e)}), 404
            path = os.path.join(self.state_dir, f"{sid}.json")
            ok = eng.load(path)
            return jsonify({"ok": ok, "path": path})


def create_app(config: AgentFrameConfig = None, state_dir: str = None) -> Flask:
    """应用工厂 (供 gunicorn/flask run 使用)"""
    return AgentFrameAPI(config, state_dir).app


def main():
    """直接运行: python -m agentframe.api.server [port]"""
    import sys
    config = AgentFrameConfig.from_env()
    port = int(sys.argv[1]) if len(sys.argv) > 1 else config.api.port
    api = AgentFrameAPI(config)
    print(f"🚀 AgentFrame API v4.2.0")
    print(f"   四层上下文保持: 认知×路由×存储×物理")
    print(f"   会话: POST /v1/sessions | 摄入: /ingest | 查询: /ask")
    print(f"   监听: {config.api.host}:{port}")
    api.app.run(host=config.api.host, port=port, debug=config.api.debug)


if __name__ == "__main__":
    main()
