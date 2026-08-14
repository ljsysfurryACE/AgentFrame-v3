"""AgentFrame 核心测试 (离线, 无需 API key)"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentframe.config import AgentFrameConfig
from agentframe.core.engine import ContextEngine
from agentframe.embed.provider import HashEmbedding


def make_engine() -> ContextEngine:
    cfg = AgentFrameConfig.from_env()
    cfg.llm.provider = "mock"   # 离线
    return ContextEngine(cfg)


def test_ingest_and_retrieve():
    eng = make_engine()
    eng.ingest("KV 缓存压缩 35.6x 实测", ["kv"])
    eng.ingest("注意力分数不等于任务重要性", ["agent"])
    eng.ingest("Minecraft 1.20.4 逆向完成", ["mc"])
    assert len(eng.agent.chunk_meta) == 3
    r = eng.ask("KV 压缩多少倍？", chat=False)
    assert len(r.retrieved) >= 1
    print("✅ test_ingest_and_retrieve")


def test_similar_text_retrieval():
    eng = make_engine()
    eng.ingest("吸收式 MLA 缓存 576 维潜在向量", ["method"])
    eng.ingest("HiLS 分层软max 端到端块选择", ["method"])
    eng.ingest("今天天气很好适合出去玩", ["life"])
    # 查询与第一条相关
    r = eng.ask("MLA 潜在向量维度是多少？", chat=False)
    tops = [cid for cid, _ in r.retrieved]
    assert 0 in tops, f"期望命中 chunk_0, 实际 {tops}"
    print(f"✅ test_similar_text_retrieval (top: {tops[:3]})")


def test_forget_curve():
    eng = make_engine()
    eng.ingest("A", ["x"])
    eng.ingest("B", ["x"])
    eng.ingest("C", ["x"])
    # 时间推进 (默认半衰期 100, 500 轮后 decay=2^-5=0.031)
    for _ in range(500):
        eng.now += 1
    # 阈值设 0.1: strength = 0.5*0.031 + 0 ≈ 0.016 < 0.1 → 应遗忘
    victims = eng.forget(0.1)
    assert len(victims) >= 2, f"长时间不访问应遗忘, 实际 {len(victims)}"
    print(f"✅ test_forget_curve (遗忘 {len(victims)}/3)")


def test_save_load():
    eng = make_engine()
    eng.ingest("持久化测试内容", ["test"])
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    eng.save(path)
    eng2 = make_engine()
    ok = eng2.load(path)
    assert ok
    assert len(eng2.agent.chunk_meta) == 1
    os.unlink(path)
    print("✅ test_save_load")


def test_hash_embedding_deterministic():
    emb = HashEmbedding(dim=576)
    v1 = emb.embed("同一句话")
    v2 = emb.embed("同一句话")
    v3 = emb.embed("完全不同的话")
    assert (v1 == v2).all()
    sim_same = float(v1 @ v2)
    sim_diff = float(v1 @ v3)
    assert sim_same > sim_diff, f"{sim_same} vs {sim_diff}"
    print(f"✅ test_hash_embedding (同句相似度 {sim_same:.3f} > 异句 {sim_diff:.3f})")


def test_tool_exec():
    eng = make_engine()
    out = eng._exec_tool("print(6*7)")
    assert "42" in out
    bad = eng._exec_tool("print(undefined_var)")
    assert "Traceback" in bad or "Error" in bad
    print("✅ test_tool_exec")


if __name__ == "__main__":
    test_ingest_and_retrieve()
    test_similar_text_retrieval()
    test_forget_curve()
    test_save_load()
    test_hash_embedding_deterministic()
    test_tool_exec()
    print("\n🎉 全部核心测试通过!")
