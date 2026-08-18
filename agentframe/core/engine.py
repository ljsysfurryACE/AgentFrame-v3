"""
ContextEngine — AgentFrame 核心引擎
====================================
整合四层上下文保持 + LLM + Embedding:

  L1 MetaCog       认知层: 任务分解 / 信息缺口 / 检索指令
  L2 LandmarkRouter 路由层: landmark 摘要检索 top-k
  L3 AbsorbedMLA   存储层: 吸收式 MLA 压缩 (L40S 实测 28.4x)
  L4 KVPager       物理层: 三级换页 + Aura 遗忘曲线

对外接口 (引擎级, 与 API/CLI 解耦):
  ingest(text, tags)        → chunk_id
  ask(query, chat=True)     → AnswerResult
  stats()                   → 状态 dict
  forget(threshold)         → 遗忘列表
  save(path) / load(path)   → 持久化
"""
import json
import numpy as np

from ..config import AgentFrameConfig, MemoryConfig
from ..llm import create_llm, LLMProvider
from ..embed import create_embedding, EmbeddingProvider
from ..memory.store import StateStore
from .quad import QuadLayerAgent
from .couple import CouplePrefetcher

# DeepSeek 工具定义 (给 Agent 的"手")
RUN_PYTHON_TOOL = [{
    "type": "function",
    "function": {
        "name": "run_python",
        "description": "执行一段 Python 代码并返回 stdout/stderr。写代码前先调用此工具验证。",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "要执行的 Python 代码"}
            },
            "required": ["code"]
        }
    }
}]


class AnswerResult:
    """ask() 的返回"""
    def __init__(self, answer: str, retrieved: list, scores: list,
                 context_dim: int, directive=None, meta: dict = None):
        self.answer = answer
        self.retrieved = retrieved      # [(chunk_id, text_preview)]
        self.scores = scores
        self.context_dim = context_dim
        self.directive = directive
        self.meta = meta or {}

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "retrieved": self.retrieved,
            "scores": self.scores,
            "context_dim": self.context_dim,
            "meta": self.meta,
        }


MEMORY_DIRECTOR_SYSTEM = (
    "你是记忆主管 (MemoryDirector)。分析一段对话, 只输出 JSON 记忆决策:\n"
    "{\"remember\": [\"值得长期记住的新信息(具体、可复用)\", ...], "
    "\"forget\": [\"已过时/无关紧要/可丢弃的信息\", ...], "
    "\"promote_importance\": 0-1, \"reason\": \"一句话理由\"}\n"
    "规则: remember 只收对未来有复用价值的事实(偏好/身份/参数/数据); "
    "forget 收闲聊/一次性内容/已被替代信息; 宁可少记不可乱记; "
    "promote_importance 表示这段对话整体重要性。"
)


# ============================================================
# 工具执行安全防护 (P0: LLM 生成代码不可信)
# ============================================================
# 危险命令黑名单 (子串匹配, 命中即拒绝执行)
DANGEROUS_PATTERNS = [
    # 破坏性文件操作
    "rm -rf", "rm -fr", "rmdir /", "mkfs", "dd if=", "shred",
    "> /dev/sd", "mkfs.ext", "fdisk", "parted",
    # 系统级操作
    "shutdown", "reboot", "init 0", "init 6", "poweroff", "halt",
    "systemctl stop", "systemctl disable", "kill -9 1",
    # 提权/敏感读取
    "sudo ", "su -", "chmod 777 /", "chown -R",
    "cat /etc/shadow", "cat /etc/passwd", "cat ~/.ssh", "cat .ssh",
    "cat /root/.ssh", "cat /root/.openclaw", "cat /etc/ssl/private",
    "base64 -d /etc", "openssl genrsa", "ssh-keygen",
    # 反弹 shell / 外联
    "nc -e", "ncat -e", "bash -i >&", "/dev/tcp/", "curl | sh", "curl|sh", "wget | sh", "wget|sh",
    "curl -o /etc", "wget -o /etc", "python -c 'import os; os.system",
    # 挂载/防火墙/网络篡改
    "mount ", "umount", "iptables", "nft ", "ip link set", "tcpdump",
]

MAX_TOOL_CODE_LEN = 4096   # 单次工具调用代码长度上限


class ToolExecutionError(Exception):
    """工具执行被安全策略拦截"""
    pass


def _check_code_safety(code: str) -> None:
    """工具代码安全检查: 长度 + 危险模式黑名单"""
    if len(code) > MAX_TOOL_CODE_LEN:
        raise ToolExecutionError(f"代码过长 ({len(code)} > {MAX_TOOL_CODE_LEN} 字符), 已拦截")
    for pat in DANGEROUS_PATTERNS:
        if pat in code:
            raise ToolExecutionError(f"命中危险模式 '{pat}', 已拦截")


class ContextEngine:
    """四层上下文保持引擎"""

    def __init__(self, config: AgentFrameConfig = None, llm: LLMProvider = None,
                 embedder: EmbeddingProvider = None):
        self.config = config or AgentFrameConfig.from_env()
        m: MemoryConfig = self.config.memory

        # 四层核心
        self.agent = QuadLayerAgent(
            n_layers=m.n_layers, quant_bits=m.quant_bits,
            top_k=m.top_k, seed=m.seed, reversible=m.reversible)

        # 外部能力
        self.llm = llm or create_llm(
            provider=self.config.llm.provider,
            api_key=self.config.llm.api_key,
            base_url=self.config.llm.base_url,
            model=self.config.llm.model,
            fast_model=self.config.llm.fast_model,
            timeout=self.config.llm.timeout)
        self.embedder = embedder or create_embedding("hash", dim=self.agent.store.DIM)

        # 跨轮共现预取 (colibrì couple_prefetch 移植)
        self.couple = CouplePrefetcher(top_k=m.couple_k)

        # 会话状态
        self.chat_history = []
        self.max_history_turns = self.config.api.max_history_turns
        self.now = 0.0
        self._chunk_counter = 0

    # ============ 摄入 ============

    def ingest(self, text: str, tags: list = None, source: str = None) -> int:
        """摄入知识块 → 压缩进 KV 缓存"""
        tags = tags or []
        # 文本 → 向量 (真实嵌入)
        vec = self.embedder.embed(text)
        # 四层摄取: 用向量作为 hidden states 输入
        kv = self.agent.store.encode(vec)
        kv.heat = 0.9
        # 新知识 = 热: force_hot 优先进 VRAM (修复 heat 设置无效问题)
        self.agent.pager.place(kv, self.now, force_hot=True)
        # 建 landmark 摘要
        k_prime, bias = self.agent.router.build_summary(kv.latent.reshape(1, -1))
        cid = kv.chunk_id
        self.agent.summaries[cid] = (k_prime, bias)
        self.agent.chunk_meta[cid] = {
            "tags": tags, "text": text, "source": source or "",
            "created_at": self.now,
        }
        self._chunk_counter += 1
        return cid

    def ingest_batch(self, chunks: list) -> list:
        """批量摄入: [{"text":..., "tags":[...]}, ...]"""
        return [self.ingest(c["text"], c.get("tags", []), c.get("source"))
                for c in chunks]

    # ============ 查询 ============

    def _exec_tool(self, code: str, timeout: int = 15) -> str:
        """执行 run_python 工具 (带安全防护)"""
        import subprocess
        import sys
        try:
            _check_code_safety(code)
        except ToolExecutionError as e:
            return f"[安全拦截] {e}"
        try:
            r = subprocess.run([sys.executable, "-c", code],
                               capture_output=True, text=True, timeout=timeout,
                               cwd="/tmp")
            out = (r.stdout or "") + (("\n[stderr] " + r.stderr) if r.stderr else "")
            if r.returncode != 0:
                out = f"[退出码 {r.returncode}] " + out
            return (out or "(无输出) 执行成功")[:800]
        except subprocess.TimeoutExpired:
            return f"[超时 >{timeout}s]"
        except Exception as e:
            return f"[执行异常] {e}"

    def _llm_with_tools(self, system: str, user: str, max_rounds: int = 4,
                        thinking: dict = None, use_fast: bool = False) -> str:
        """带工具循环的 LLM 调用 (脑+手)"""
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        for _ in range(max_rounds + 1):
            resp = self.llm.chat(messages, max_tokens=2000, temperature=0.8,
                                 thinking=thinking, tools=RUN_PYTHON_TOOL,
                                 use_fast=use_fast)
            tool_calls = resp.get("tool_calls")
            if not tool_calls:
                return resp.get("content") or ""
            # 执行工具并回传
            messages.append({"role": "assistant", "content": resp.get("content") or "",
                             "tool_calls": tool_calls})
            for tc in tool_calls:
                fn = tc["function"]
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except Exception:
                    args = {}
                result = self._exec_tool(args.get("code", ""))
                messages.append({"role": "tool", "tool_call_id": tc.get("id", ""),
                                 "content": result})
        return messages[-1].get("content", "") if messages else ""

    def ask(self, query: str, chat: bool = True, use_fast: bool = False,
            thinking: dict = None) -> AnswerResult:
        """查询 + 生成"""
        # 1. 认知层: 分解 + 检索指令
        subtasks = self.agent.metacog.decompose(query)
        directive = self.agent.metacog.build_directive(subtasks[0], self.agent.chunk_meta)

        # 2. 路由层: landmark 检索
        q_vec = self.embedder.embed(query)
        selection = self.agent.router.route(q_vec, q_vec, self.agent.summaries)

        # 2.5 跨轮共现预取: 记录本轮检索 + 预测下轮 → 提升冷块到 RAM
        self.couple.record(selection.chunk_ids)
        predicted = self.couple.predict(selection.chunk_ids)
        if predicted:
            self.agent.pager.prefetch(predicted, self.now)

        # 3. 物理层: 访问更新 (遗忘曲线刷新)
        for cid, score in zip(selection.chunk_ids, selection.scores):
            self.agent.pager.access(cid, abs(score), self.now)
            self.agent.metacog.promote(cid, abs(score))
        self.now += 1

        # 4. 存储层: 聚合检索结果
        gathered, hit_ids = [], []
        for cid in selection.chunk_ids:
            kv = (self.agent.pager.vram.get(cid)
                  or self.agent.pager.ram.get(cid)
                  or self.agent.pager.disk.get(cid))
            if kv is not None:
                gathered.append(kv.latent)
                meta = self.agent.chunk_meta.get(cid, {})
                hit_ids.append((cid, meta.get("text", "")[:40]))
        context_dim = len(gathered[0]) if gathered else 0

        # 5. 生成回答
        answer = ""
        if chat:
            ctx_texts = []
            for cid in selection.chunk_ids:
                meta = self.agent.chunk_meta.get(cid, {})
                if meta.get("text"):
                    ctx_texts.append(f"[知识块 {cid}] {meta['text']}")
            ctx_block = "\n".join(ctx_texts) if ctx_texts else "(无检索到相关知识块)"

            system = (
                "你是 AgentFrame 上下文保持系统。下面是从压缩缓存中检索到的知识块，"
                "回答时优先使用这些知识。如果知识不足就如实说明。"
                f"\n\n【检索到的知识】\n{ctx_block}")
            messages = [{"role": "system", "content": system}]
            messages.extend(self.chat_history[-self.max_history_turns * 2:])
            messages.append({"role": "user", "content": query})

            resp = self.llm.chat(messages, max_tokens=self.config.llm.max_tokens,
                                 temperature=self.config.llm.temperature,
                                 use_fast=use_fast)
            answer = resp.get("content") or ""

            # 更新会话历史 (老对话由压缩系统兜底)
            self.chat_history.append({"role": "user", "content": query})
            self.chat_history.append({"role": "assistant", "content": answer})
            if len(self.chat_history) > self.max_history_turns * 2:
                self.chat_history = self.chat_history[-self.max_history_turns * 2:]

        return AnswerResult(
            answer=answer,
            retrieved=hit_ids,
            scores=[round(float(s), 4) for s in selection.scores[:len(hit_ids)]],
            context_dim=context_dim,
            directive=directive,
            meta={"now": self.now, "chunks": len(self.agent.chunk_meta)})

    # ============ 工具增强版 ask ============

    def ask_with_hands(self, query: str, thinking: dict = None) -> AnswerResult:
        """带工具循环的查询 (Agent 可自行执行代码验证)"""
        result = self.ask(query, chat=False)
        ctx_texts = []
        for cid, _preview in result.retrieved:
            meta = self.agent.chunk_meta.get(cid, {})
            if meta.get("text"):
                ctx_texts.append(f"[知识块 {cid}] {meta['text']}")
        ctx_block = "\n".join(ctx_texts) if ctx_texts else "(无检索到相关知识块)"
        system = (
            "你是 AgentFrame 上下文保持系统。下面是检索到的知识块。"
            "回答时优先使用这些知识，知识不足就如实说明。"
            "你有 run_python 工具可以执行代码验证你的方案。"
            f"\n\n【检索到的知识】\n{ctx_block}")
        answer = self._llm_with_tools(system, query, thinking=thinking)
        return AnswerResult(answer=answer, retrieved=result.retrieved,
                            scores=result.scores, context_dim=result.context_dim,
                            meta={"hands": True, "chunks": len(self.agent.chunk_meta)})

    # ============ 自主记忆管理 (MemoryDirector) ============

    def _parse_memory_decision(self, raw: str) -> dict:
        """解析 LLM 输出的记忆决策 JSON (容错处理)"""
        import re
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if not m:
            return {"remember": [], "forget": [], "promote_importance": 0.5, "reason": ""}
        try:
            return json.loads(m.group(0))
        except Exception:
            return {"remember": [], "forget": [], "promote_importance": 0.5, "reason": ""}

    def autopilot_memory(self, query: str, answer: str) -> dict:
        """
        模型自主记忆管理: 分析一轮对话, 自动决定记什么/忘什么.
        返回决策摘要.
        """
        conversation = f"用户: {query}\n助手: {answer[:800]}"
        resp = self.llm.chat(
            [{"role": "system", "content": MEMORY_DIRECTOR_SYSTEM},
             {"role": "user", "content": f"对话:\n{conversation}"}],
            max_tokens=600, temperature=0.3, use_fast=False)
        decision = self._parse_memory_decision(resp.get("content") or "")

        # 执行决策
        remembered = []
        forgotten = []

        # 1. remember → 摄入新知识块 (先去重: 与已有块相似则跳过)
        for item in decision.get("remember", []):
            if not (isinstance(item, str) and item.strip()):
                continue
            # 去重: 检查是否已有高相似度块
            dup = False
            new_vec = self.embedder.embed(item.strip())
            for cid, meta in self.agent.chunk_meta.items():
                old_vec = self.embedder.embed(meta.get("text", ""))
                sim = float(new_vec @ old_vec)
                if sim > 0.8:  # 余弦相似度阈值
                    dup = True
                    break
            if dup:
                continue
            cid = self.ingest(item.strip(), tags=["auto"])
            remembered.append((cid, item.strip()))

        # 2. forget → 按文本匹配降低重要性/删除
        forget_list = decision.get("forget", [])
        if forget_list:
            for cid, meta in list(self.agent.chunk_meta.items()):
                text = meta.get("text", "")
                for f_item in forget_list:
                    if isinstance(f_item, str) and f_item and \
                       (f_item in text or text in f_item):
                        kv = (self.agent.pager.vram.get(cid)
                              or self.agent.pager.ram.get(cid)
                              or self.agent.pager.disk.get(cid))
                        if kv:
                            kv.importance *= 0.3  # 大幅降权
                            kv.heat = 0.1
                            forgotten.append((cid, f_item))

        # 3. promote_importance → 记录整体重要性
        imp = float(decision.get("promote_importance", 0.5))
        self._last_importance = imp

        return {
            "remembered": remembered,
            "forgotten": forgotten,
            "importance": imp,
            "reason": decision.get("reason", ""),
        }

    def ask_autopilot(self, query: str) -> AnswerResult:
        """查询 + 自主记忆管理 (一步到位)"""
        result = self.ask(query, chat=True)
        decision = self.autopilot_memory(query, result.answer)
        result.meta["memory_decision"] = decision
        return result

    # ============ 遗忘 / 状态 / 持久化 ============

    def forget(self, threshold: float = 0.15) -> list:
        """按遗忘曲线清理弱记忆 (同步更新内存计数, 防泄漏)"""
        victims = []
        for layer_name in ["disk", "ram", "vram"]:
            pool = getattr(self.agent.pager, layer_name)
            for cid, kv in list(pool.items()):
                strength = self.agent.pager.curve.strength(kv, self.now)
                if strength < threshold:
                    victims.append(cid)
                    del pool[cid]
                    # 同步扣减占用计数 (vram_used/ram_used), 防漂移
                    if layer_name == "vram":
                        self.agent.pager.vram_used -= kv.size_bytes
                    elif layer_name == "ram":
                        self.agent.pager.ram_used -= kv.size_bytes
                    self.agent.chunk_meta.pop(cid, None)
                    self.agent.summaries.pop(cid, None)
        return victims

    def stats(self) -> dict:
        return {
            "chunks_total": len(self.agent.chunk_meta),
            "layers": self.agent.pager.stats(),
            "bytes_per_token": round(self.agent.store.bytes_per_token(), 2),
            "chat_history_turns": len(self.chat_history) // 2,
            "now": self.now,
            "compression": "吸收式MLA + INT4 = 28.4x (L40S 真实推理实测)",
        }

    def save(self, path: str):
        """保存引擎状态"""
        store = StateStore(path)
        state = {
            "chunk_meta": self.agent.chunk_meta,
            "summaries": self.agent.summaries,
            "chat_history": self.chat_history,
            "now": self.now,
            "counter": self._chunk_counter,
            # pager 状态
            "vram": self.agent.pager.vram,
            "ram": self.agent.pager.ram,
            "disk": self.agent.pager.disk,
        }
        store.save(state)

    def load(self, path: str):
        """恢复引擎状态"""
        store = StateStore(path)
        state = store.load()
        if not state:
            return False
        self.agent.chunk_meta = {int(k): v for k, v in state.get("chunk_meta", {}).items()}
        # JSON 序列化后 (k_prime, bias) 变 list, 转回 tuple
        self.agent.summaries = {int(k): (v[0], v[1]) if isinstance(v, list) and len(v) == 2 else v
                                for k, v in state.get("summaries", {}).items()}
        self.chat_history = state.get("chat_history", [])
        self.now = state.get("now", 0.0)
        self._chunk_counter = state.get("counter", 0)
        # 反序列化: dict → CompressedKV
        from .quad import CompressedKV
        for layer_name in ["vram", "ram", "disk"]:
            raw = state.get(layer_name, {})
            pool = {}
            for cid, d in raw.items():
                if isinstance(d, CompressedKV):
                    pool[int(cid)] = d
                elif isinstance(d, dict):
                    kv = CompressedKV(
                        chunk_id=int(d.get("chunk_id", int(cid))),
                        latent=np.asarray(d.get("latent", [0.0]), dtype=np.float32),
                        quant_bits=int(d.get("quant_bits", 4)),
                        heat=float(d.get("heat", 0.5)),
                        last_access=float(d.get("last_access", 0.0)),
                        size_bytes=int(d.get("size_bytes", 0)),
                        importance=float(d.get("importance", 0.5)),
                        access_count=int(d.get("access_count", 0)),
                        protected=bool(d.get("protected", False)),
                        reversible=bool(d.get("reversible", False)),
                        sector_id=int(d.get("sector_id", -1)),
                        block_id=int(d.get("block_id", -1)),
                        module_id=int(d.get("module_id", -1)),
                    )
                    vh = d.get("value_hist")
                    if isinstance(vh, np.ndarray):
                        kv.value_hist = vh
                    pool[int(cid)] = kv
            setattr(self.agent.pager, layer_name, pool)
        return True
