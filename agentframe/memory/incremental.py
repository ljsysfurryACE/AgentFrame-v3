"""
IncrementalKVStore — 增量追加持久化 (colibrì kv_persist.h 移植)
=================================================================
colibrì 做法: 每轮对话后把压缩 MLA KV 增量 append 到 .coli_kv,
nrec 计数最后写 = crash-safe, 重启直接恢复不用重新 prefill。

AgentFrame 移植:
  - 每次 ingest 的新 chunk 追加为一行 JSON (不重写全量快照)
  - 行级独立 = 崩溃时最多丢最后一条, 坏行跳过 (crash-safe)
  - load() 读全部记录重建 chunk + 自动重建 landmark 摘要
  - 与全量快照 (StateStore) 互补: 增量管内容, 快照管运行时热度状态
"""
import json
import os
import numpy as np


class IncrementalKVStore:
    """增量追加 KV 日志 (colibrì kv_persist 思想)"""

    MAGIC = "AFKV1"

    def __init__(self, path: str):
        self.path = path

    # ============ 写入 ============

    def append(self, chunk_id: int, q4: np.ndarray, scales: np.ndarray,
               latent: np.ndarray, quant_bits: int, size_bytes: int,
               meta: dict) -> int:
        """
        追加一条 chunk 记录 (每行一个 JSON, 追加模式).
        返回当前记录数 (类似 colibrì 的 nrec).
        """
        rec = {
            "magic": self.MAGIC,
            "chunk_id": int(chunk_id),
            "quant_bits": int(quant_bits),
            "size_bytes": int(size_bytes),
            "q4": q4.tobytes().hex() if q4 is not None else None,
            "scales": scales.tolist() if scales is not None else None,
            "latent": latent.tolist(),
            "meta": meta,
        }
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return self.count()

    def count(self) -> int:
        """统计有效记录数 (跳过坏行)"""
        return len(self._read_raw())

    # ============ 读取 ============

    def _read_raw(self) -> list:
        """读全部行, 跳过损坏/不完整行 (crash-safe 核心)"""
        if not os.path.exists(self.path):
            return []
        recs = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("magic") != self.MAGIC:
                        continue
                    recs.append(rec)
                except Exception:
                    continue  # 崩溃残留的半行: 跳过
        return recs

    def load(self) -> list:
        """
        读全部有效记录, 反序列化为可重建的数据结构.
        返回: [{chunk_id, q4, scales, latent, quant_bits, size_bytes, meta}, ...]
        """
        out = []
        for rec in self._read_raw():
            q4 = None
            if rec.get("q4"):
                q4 = np.frombuffer(bytes.fromhex(rec["q4"]), dtype=np.uint8)
            scales = None
            if rec.get("scales"):
                scales = np.asarray(rec["scales"], dtype=np.float32)
            out.append({
                "chunk_id": int(rec["chunk_id"]),
                "quant_bits": int(rec.get("quant_bits", 4)),
                "size_bytes": int(rec.get("size_bytes", 0)),
                "q4": q4,
                "scales": scales,
                "latent": np.asarray(rec.get("latent", [0.0]), dtype=np.float32),
                "meta": rec.get("meta", {}),
            })
        return out

    def exists(self) -> bool:
        return os.path.exists(self.path)
