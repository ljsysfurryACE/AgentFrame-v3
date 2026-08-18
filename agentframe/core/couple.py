"""
CouplePrefetcher — 跨轮共现预取 (colibrì couple_prefetch 移植)
================================================================
colibrì 原理: MoE 路由有跨层共现 — "L 层选了专家 e, L+dL 层大概率选 f"。
用离线统计的 .coli_pairs 表在运行时打分预取 (实测 +3.6~+9.4pp 召回)。

AgentFrame 没有"层"概念, 但有时序: 一轮查询检索到的知识块集合 → 下一轮
查询检索到的集合。同样的共现结构存在: "检索到 A 后, 下一轮往往也检索 B"。

移植:
  cooccur[cid] -> {other_cid: count}    # 相邻两轮检索共现计数
  record(chunk_ids)                     # 每轮检索后记录, 更新共现表
  predict(chunk_ids) -> top-K           # 给定当前检索集, 预测下轮需要的块
  prefetch 由 KVPager 执行 (disk→ram 温层提升, 不占 vram 热层)
"""
from collections import defaultdict


class CouplePrefetcher:
    """跨轮共现预取器 (colibrì couple_prefetch 思想)"""

    def __init__(self, top_k: int = 8, max_entries: int = 256):
        self.top_k = top_k              # 每轮预取上限 (COUPLE_K 对应)
        self.max_entries = max_entries  # 共现表条目上限 (防膨胀)
        self.cooccur = defaultdict(lambda: defaultdict(int))  # cid -> {other: count}
        self.prev_set = None            # 上一轮检索到的块集合
        self.n_recorded = 0             # 已记录轮次

    def record(self, chunk_ids: list):
        """
        记录一轮检索结果, 更新跨轮共现表。
        时序: 上一轮集合 × 本轮集合 = 共现计数 (dL=1 的 AgentFrame 版)。
        """
        cur = set(chunk_ids)
        if self.prev_set is not None:
            for a in self.prev_set:
                for b in cur:
                    if a == b:
                        continue
                    self.cooccur[a][b] += 1
        self.prev_set = cur
        self.n_recorded += 1
        self._prune()

    def predict(self, chunk_ids: list) -> list:
        """
        给定当前检索集, 预测下一轮可能需要的块 (分数 = 共现计数累加)。
        返回 top-K 个不在当前集合中的块 id。
        """
        cur = set(chunk_ids)
        scores = defaultdict(int)
        for a in cur:
            for b, cnt in self.cooccur.get(a, {}).items():
                if b not in cur:
                    scores[b] += cnt
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return [cid for cid, _ in ranked[:self.top_k]]

    def _prune(self):
        """裁剪: 只保留共现次数最高的条目, 防表膨胀"""
        if len(self.cooccur) <= self.max_entries:
            return
        # 按总共现次数排序, 截断
        total = {cid: sum(cnts.values()) for cid, cnts in self.cooccur.items()}
        keep = set(sorted(total, key=total.get, reverse=True)[:self.max_entries])
        for cid in list(self.cooccur.keys()):
            if cid not in keep:
                del self.cooccur[cid]

    def stats(self) -> dict:
        n_pairs = sum(len(v) for v in self.cooccur.values())
        return {
            "recorded_rounds": self.n_recorded,
            "condition_entries": len(self.cooccur),
            "cooccur_pairs": n_pairs,
            "top_k": self.top_k,
        }
