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
    eng.ingest("KV 缓存压缩测试数据", ["kv"])
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


def test_lfru_hysteresis():
    """LFRU 滞回驱逐: 历史热块获得保护, 信用随时间衰减 (colibrì #441/#497)"""
    import numpy as np
    from agentframe.core.quad import KVPager, CompressedKV

    def mk(cid, max_heat, acc):
        return CompressedKV(
            chunk_id=cid, latent=np.zeros(576, dtype=np.float32),
            quant_bits=4, heat=max_heat, size_bytes=1000,
            importance=0.5, access_count=acc, max_heat=max_heat,
            last_access=0)

    # 冷却 200 轮 (2 个半衰期): 历史热块应被保护
    p = KVPager(vram_limit_mb=0.006, ram_limit_mb=64)
    for kv in [mk(0, 0.95, 0), mk(1, 0.4, 0), mk(2, 0.5, 0),
               mk(3, 0.5, 0), mk(4, 0.5, 0)]:
        p.vram[kv.chunk_id] = kv
        p.vram_used += kv.size_bytes
    victim = p.evict(200.0, "vram")
    assert victim == 1, f"历史热块0应被保护, 实际驱逐 {victim}"

    # 完全冷透 (2000 轮): 信用衰减殆尽, 照常驱逐
    p2 = KVPager(vram_limit_mb=0.006, ram_limit_mb=64)
    for kv in [mk(0, 0.95, 0), mk(1, 0.4, 0), mk(2, 0.5, 0),
               mk(3, 0.5, 0), mk(4, 0.5, 0)]:
        p2.vram[kv.chunk_id] = kv
        p2.vram_used += kv.size_bytes
    assert p2.evict(2000.0, "vram") is not None
    print("✅ test_lfru_hysteresis (滞回保护 + 信用衰减)")


def test_tool_safety():
    """工具执行安全: 危险命令/超长代码应被拦截"""
    eng = make_engine()
    out = eng._exec_tool("import os; os.system('rm -rf /tmp/x')")
    assert "安全拦截" in out, out
    out2 = eng._exec_tool("import subprocess; subprocess.run(['shutdown'])")
    assert "安全拦截" in out2, out2
    print("✅ test_tool_safety (危险命令拦截)")


def test_api_auth():
    """API 认证: token 模式 + 回环模式"""
    import os
    from agentframe.api.server import AgentFrameAPI
    from agentframe.config import AgentFrameConfig
    os.environ["AGENTFRAME_STATE_DIR"] = "/tmp/af_test_auth"
    cfg = AgentFrameConfig.from_env()
    cfg.api.token = "test-token"
    api = AgentFrameAPI(cfg)
    client = api.app.test_client()
    assert client.get("/v1/sessions").status_code == 401
    ok = client.get("/v1/sessions",
                    headers={"Authorization": "Bearer test-token"})
    assert ok.status_code == 200
    print("✅ test_api_auth (Bearer 认证)")


def test_couple_prefetch():
    """跨轮共现预取: 学习共现 → 预测 → disk→RAM 提升 (colibrì couple 移植)"""
    import numpy as np
    from agentframe.core.couple import CouplePrefetcher
    from agentframe.core.quad import KVPager, CompressedKV

    def mk(cid):
        return CompressedKV(chunk_id=cid, latent=np.zeros(576, dtype=np.float32),
                            quant_bits=4, heat=0.1, size_bytes=1000,
                            importance=0.5)

    # 1. 共现学习: 轮1检索{0,1}, 轮2检索{1,2}, 轮3检索{2,3}
    cp = CouplePrefetcher(top_k=8)
    cp.record([0, 1])
    cp.record([1, 2])
    cp.record([2, 3])
    # 轮2后: (0,2)共现1, (1,2)共现1; 轮3后: (1,2)再+1, (1,3)共现1, (2,3)共现1
    assert cp.cooccur[0][2] == 1
    assert cp.cooccur[1][2] == 2  # 轮2(1在prev,2在cur) + 轮3(1在prev,2在cur)
    assert cp.cooccur[2][3] == 1
    assert cp.cooccur[1][3] == 1
    # 2. 预测: 当前检索{1} → 预测 2 (共现1)
    pred = cp.predict([1])
    assert 2 in pred, f"应预测到块2, 实际 {pred}"
    assert 1 not in pred, "预测不应包含当前集合中的块"
    print(f"✅ test_couple_prefetch (学习+预测: {pred[:4]})")

    # 3. 预取落地: 块2在 disk, prefetch 后应到 RAM (不进 VRAM)
    p = KVPager(vram_limit_mb=64, ram_limit_mb=0.006)  # 6KB RAM
    for cid in (0, 1, 3):
        p.disk[cid] = mk(cid)
    p.disk[2] = mk(2)  # 目标块在 disk
    moved = p.prefetch([2], now=1.0)
    assert 2 in moved
    assert 2 in p.ram and 2 not in p.vram and 2 not in p.disk
    print("✅ test_couple_prefetch (预取 disk→RAM, 不占 VRAM)")

    # 4. RAM 满时腾位: 塞满 RAM 后预取新块
    p2 = KVPager(vram_limit_mb=64, ram_limit_mb=0.005)  # 5KB RAM (最多5块)
    for cid in range(5):
        p2.ram[cid] = mk(cid)
        p2.ram_used += 1000
    p2.disk[99] = mk(99)
    moved2 = p2.prefetch([99], now=1.0)
    assert 99 in p2.ram
    assert len(p2.ram) == 5  # 腾位后仍不超限
    print("✅ test_couple_prefetch (RAM 满时 LFRU 腾位)")


def test_int4_packing():
    """真 INT4 打包 (colibrì quant.h 移植): 往返精度 + 真实压缩比"""
    import numpy as np
    from agentframe.core.quad import ReversibleQuantizer, AbsorbedMLA

    rng = np.random.default_rng(42)
    latent = np.tanh(rng.normal(0, 1, 576).astype(np.float32))
    q4, scales, size = ReversibleQuantizer.quantize_int4(latent, n_ch=16)
    deq = ReversibleQuantizer.dequant_int4(q4, scales, 576)
    # 往返精度: 余弦相似度 > 0.99
    sim = float(deq @ latent) / (np.linalg.norm(deq) * np.linalg.norm(latent))
    assert sim > 0.99, f"INT4 往返相似度 {sim:.4f}"
    # 真实压缩: 352B vs 原始 2304B
    assert size == 288 + 64, f"打包大小 {size} != 352"
    # 每 token 压缩比对齐 L40S 实测 28.4x
    m = AbsorbedMLA(n_layers=27, quant_bits=4, n_ch=16)
    ratio = 270 * 1024 / m.bytes_per_token()
    assert ratio > 25, f"压缩比 {ratio:.1f}x 不足"
    print(f"✅ test_int4_packing (cos={sim:.4f}, {ratio:.1f}x)")


def test_incremental_persist():
    """增量持久化 (colibrì kv_persist 移植): append + crash-safe 恢复"""
    import tempfile, os
    from agentframe.memory.incremental import IncrementalKVStore

    path = tempfile.mktemp(suffix=".kv")
    store = IncrementalKVStore(path)
    # 追加 3 条
    import numpy as np
    n1 = store.append(0, np.zeros(288, dtype=np.uint8), np.ones(16, dtype=np.float32),
                      np.zeros(576, dtype=np.float32), 4, 352, {"text": "A"})
    n2 = store.append(1, np.ones(288, dtype=np.uint8), np.ones(16, dtype=np.float32) * 2,
                      np.ones(576, dtype=np.float32), 4, 352, {"text": "B"})
    n3 = store.append(2, np.zeros(288, dtype=np.uint8), np.ones(16, dtype=np.float32),
                      np.zeros(576, dtype=np.float32), 4, 352, {"text": "C"})
    assert n3 == 3
    # 模拟崩溃: 写入半行垃圾
    with open(path, "a") as f:
        f.write('{"magic": "AFKV1", "chunk_id": 99, "q4": "zz')
    recs = store.load()
    assert len(recs) == 3, f"坏行应被跳过, 实际 {len(recs)}"
    assert recs[1]["meta"]["text"] == "B"
    print("✅ test_incremental_persist (append + crash-safe 坏行跳过)")
    os.unlink(path)


def test_prefix_reuse():
    """查询前缀复用 (colibrì kv_prefix 移植)"""
    eng = make_engine()
    eng.ingest("KV 压缩 29 倍", ["kv"])
    eng.ingest("LFRU 滞回驱逐", ["method"])
    eng.ingest("Couple 预取", ["method"])
    # 第一轮: 正常检索
    r1 = eng.ask("KV 压缩是多少倍", chat=False)
    first = r1.retrieved
    # 第二轮: 相同前缀 → 复用
    r2 = eng.ask("KV 压缩是多少倍呢", chat=False)
    assert eng._prefix_hits == 1, "应触发前缀复用"
    assert [c for c, _ in r2.retrieved] == [c for c, _ in first]
    # 第三轮: 完全不同 → 不复用
    r3 = eng.ask("今天天气怎么样", chat=False)
    assert eng._prefix_hits == 1, "不同 query 不应复用"
    print("✅ test_prefix_reuse (前缀命中复用, 不同 query 不复用)")


def test_incremental_engine():
    """引擎级增量恢复: ingest → enable → load_incremental 重建"""
    import tempfile, os
    path = tempfile.mktemp(suffix=".kv")
    eng = make_engine()
    eng.enable_incremental(path)
    eng.ingest("增量持久化内容 A", ["a"])
    eng.ingest("增量持久化内容 B", ["b"])
    # 新引擎从日志恢复
    eng2 = make_engine()
    n = eng2.load_incremental(path)
    assert n == 2, f"应恢复 2 块, 实际 {n}"
    r = eng2.ask("增量持久化", chat=False)
    assert len(r.retrieved) >= 1
    os.unlink(path)
    print("✅ test_incremental_engine (引擎级增量恢复 + 检索)")


if __name__ == "__main__":
    test_ingest_and_retrieve()
    test_similar_text_retrieval()
    test_forget_curve()
    test_save_load()
    test_hash_embedding_deterministic()
    test_tool_exec()
    test_lfru_hysteresis()
    test_tool_safety()
    test_api_auth()
    test_couple_prefetch()
    test_int4_packing()
    test_incremental_persist()
    test_prefix_reuse()
    test_incremental_engine()
    print("\n🎉 全部核心测试通过!")
