#!/usr/bin/env python3
"""
AgentFrame CLI — 命令行工具
============================
用法:
  agentframe serve [--port 8090] [--config config.json]
  agentframe ingest "文本" [--tags a,b] [--session <sid>] [--state <path>]
  agentframe ask "问题" [--no-chat] [--hands] [--state <path>]
  agentframe stats [--state <path>]
  agentframe forget [--threshold 0.15] [--state <path>]
  agentframe config [--output config.json]
  agentframe demo
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agentframe.config import AgentFrameConfig
from agentframe.core.engine import ContextEngine

DEFAULT_STATE = "/var/lib/agentframe/cli.json"


def load_engine(config_path: str = None, state_path: str = None) -> ContextEngine:
    config = AgentFrameConfig.from_file(config_path) if config_path else AgentFrameConfig.from_env()
    eng = ContextEngine(config)
    sp = state_path or DEFAULT_STATE
    if os.path.exists(sp):
        eng.load(sp)
    return eng


def cmd_serve(args):
    from agentframe.api.server import AgentFrameAPI
    config = AgentFrameConfig.from_file(args.config) if args.config else AgentFrameConfig.from_env()
    api = AgentFrameAPI(config)
    print(f"🚀 AgentFrame API — http://{config.api.host}:{args.port or config.api.port}")
    api.app.run(host=config.api.host, port=args.port or config.api.port,
                debug=config.api.debug)


def cmd_ingest(args):
    eng = load_engine(args.config, args.state)
    tags = args.tags.split(",") if args.tags else []
    cid = eng.ingest(args.text, tags)
    sp = args.state or DEFAULT_STATE
    os.makedirs(os.path.dirname(sp), exist_ok=True)
    eng.save(sp)
    print(f"✅ 摄入 chunk_{cid}: {args.text[:50]}...")
    print(f"   标签: {tags} | 已存: {sp}")


def cmd_ask(args):
    eng = load_engine(args.config, args.state)
    if args.hands:
        result = eng.ask_with_hands(args.query)
    else:
        result = eng.ask(args.query, chat=not args.no_chat)
    print(f"\n🧠 检索到 {len(result.retrieved)} 块:")
    for cid, prev in result.retrieved:
        print(f"   chunk_{cid}: {prev}...")
    if result.answer:
        print(f"\n💬 回答:\n{result.answer}")
    sp = args.state or DEFAULT_STATE
    eng.save(sp)


def cmd_stats(args):
    eng = load_engine(args.config, args.state)
    for k, v in eng.stats().items():
        print(f"  {k}: {v}")


def cmd_forget(args):
    eng = load_engine(args.config, args.state)
    victims = eng.forget(args.threshold)
    sp = args.state or DEFAULT_STATE
    eng.save(sp)
    print(f"🧹 遗忘 {len(victims)} 块: {victims}")
    print(f"   剩余: {len(eng.agent.chunk_meta)} 块")


def cmd_config(args):
    config = AgentFrameConfig.from_env()
    out = args.output or "agentframe.config.json"
    config.save(out)
    print(f"✅ 配置已写入: {out}")
    print(f"   LLM: {config.llm.provider} / {config.llm.model}")
    print(f"   Memory: {config.memory.n_layers}层 INT{config.memory.quant_bits} top-k={config.memory.top_k}")
    print(f"   API: {config.api.host}:{config.api.port}")


def cmd_demo(args):
    """离线演示: 摄入 → 查询 → 遗忘 (无 API key 也可跑)"""
    config = AgentFrameConfig.from_env()
    config.llm.provider = "mock"
    eng = ContextEngine(config)

    print("=" * 56)
    print("AgentFrame 离线演示 (Mock LLM)")
    print("=" * 56)

    chunks = [
        {"text": "KV 缓存 270KB/token 展开存储", "tags": ["data"]},
        {"text": "吸收式 MLA 缓存 576 维潜在向量", "tags": ["method"]},
        {"text": "L40S 实测 35.6x 压缩", "tags": ["data", "result"]},
        {"text": "注意力分数不等于任务重要性", "tags": ["agent"]},
    ]
    for c in chunks:
        cid = eng.ingest(c["text"], c["tags"])
        print(f"  📥 摄入 chunk_{cid}: {c['text']}")

    print(f"\n  [存储] {eng.stats()['layers']}")
    print(f"  [压缩] {eng.stats()['bytes_per_token']} B/token")

    r = eng.ask("KV压缩能到多少倍？", chat=True)
    print(f"\n  🧠 检索: {r.retrieved}")
    print(f"  💬 回答: {r.answer}")

    v = eng.forget(0.3)
    print(f"\n  🧹 遗忘: {len(v)} 块 → 剩余 {len(eng.agent.chunk_meta)} 块")
    print("\n✅ 演示完成")


def main():
    p = argparse.ArgumentParser(prog="agentframe", description="AgentFrame 上下文保持框架")
    sub = p.add_subparsers(dest="cmd")

    pserve = sub.add_parser("serve", help="启动 API 服务")
    pserve.add_argument("--port", type=int, default=None)
    pserve.add_argument("--config", default=None)

    pi = sub.add_parser("ingest", help="摄入知识")
    pi.add_argument("text")
    pi.add_argument("--tags", default=None)
    pi.add_argument("--config", default=None)
    pi.add_argument("--state", default=None)

    pa = sub.add_parser("ask", help="查询")
    pa.add_argument("query")
    pa.add_argument("--no-chat", action="store_true")
    pa.add_argument("--hands", action="store_true")
    pa.add_argument("--config", default=None)
    pa.add_argument("--state", default=None)

    pst = sub.add_parser("stats", help="状态")
    pst.add_argument("--config", default=None)
    pst.add_argument("--state", default=None)

    pf = sub.add_parser("forget", help="遗忘")
    pf.add_argument("--threshold", type=float, default=0.15)
    pf.add_argument("--config", default=None)
    pf.add_argument("--state", default=None)

    pc = sub.add_parser("config", help="生成配置")
    pc.add_argument("--output", default=None)

    sub.add_parser("demo", help="离线演示")

    args = p.parse_args()
    if args.cmd == "serve":
        cmd_serve(args)
    elif args.cmd == "ingest":
        cmd_ingest(args)
    elif args.cmd == "ask":
        cmd_ask(args)
    elif args.cmd == "stats":
        cmd_stats(args)
    elif args.cmd == "forget":
        cmd_forget(args)
    elif args.cmd == "config":
        cmd_config(args)
    elif args.cmd == "demo":
        cmd_demo(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
