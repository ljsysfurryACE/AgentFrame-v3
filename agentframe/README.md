# AgentFrame — Agent 专用上下文保持框架

**脑 = DeepSeek · 手 = 工具执行 · 记忆 = 四层上下文保持**

版本 4.2.0 · GPL-3.0 · Cloud LTE Studio

---

##  架构

```
┌─────────────────────────────────────────────────────────┐
│                    ContextEngine 引擎                    │
│  ┌─────────┐  ┌──────────────┐  ┌───────────┐  ┌──────┐ │
│  │ L1 认知  │→ │ L2 路由      │→ │ L3 存储    │→ │ L4 物理│ │
│  │ MetaCog │  │LandmarkRouter│  │AbsorbedMLA│  │KVPager│ │
│  │任务分解  │  │ landmark检索 │  │ 吸收式MLA  │  │三级换页│ │
│  │信息缺口  │  │ 分层软max    │  │ INT4量化   │  │遗忘曲线│ │
│  └─────────┘  └──────────────┘  └───────────┘  └──────┘ │
│         ↕ 检索指令      ↕ 摘要        ↕ 压缩块   ↕ 热度    │
│  ┌──────────────────────────────────────────────────┐   │
│  │  LLM Provider (DeepSeek, 支持 thinking+工具循环)  │   │
│  │  Embedding Provider (哈希/API, 文本→向量)          │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

##  核心能力

| 能力 | 说明 | 验证状态 |
|------|------|---------|
| **KV 压缩** | 真 INT4 打包: 576维→288B q4+64B scales = **29.1x** (cos 0.998) | ✓ 有测试 (对齐 L40S 28.4x) |
| **Top-K 保护** | 关键块 16bit + 其余 4bit = **0/100 翻转** | ✓ 实测 (独立实验, 未接入主流程) |
| **分层组织** | Sector(16)-Block(256)-Module(1024) 硬件对齐 | ⚠️ 设计未接入 |
| **Aura 遗忘** | S(t)=I·2^(-t/τ) 指数遗忘 + 访问增强 | ✓ 有测试 |
| **LFRU 滞回** | 历史热块信用折减驱逐分数, 防抖动 (colibrì #441/#497) | ✓ 有测试 |
| **Couple 预取** | 跨轮共现预取: 检索到 A 后预取常与 A 共现的 B (colibrì couple) | ✓ 有测试 |
| **脑+手** | function calling 工具循环, Agent 自验证代码 | ✓ 有测试 + 安全拦截 |
| **多会话** | 每会话独立引擎, 状态可持久化 | ✓ |

##  快速开始

### 1. 安装

```bash
pip install -e /root/.openclaw/workspace/agentframe
# 或直接用 (无需安装):
export PYTHONPATH=/root/.openclaw/workspace
```

### 2. 配置

```bash
export AGENTFRAME_API_KEY="sk-xxx"          # DeepSeek key
export AGENTFRAME_MODEL="deepseek-v4-pro"    # 主模型
export AGENTFRAME_FAST_MODEL="deepseek-v4-flash"
export AGENTFRAME_PORT=8090

# 或生成配置文件
python3 -m agentframe.cli config
```

### 3. CLI 使用

```bash
# 摄入知识
python3 -m agentframe.cli ingest "KV 压缩 28.4x L40S 实测" --tags "kv"

# 查询 (带 DeepSeek 生成)
python3 -m agentframe.cli ask "KV 压缩多少倍？"

# 状态 / 遗忘
python3 -m agentframe.cli stats
python3 -m agentframe.cli forget --threshold 0.1

# 离线演示 (无需 API key)
python3 -m agentframe.cli demo
```

### 4. REST API

```bash
python3 -m agentframe.api.server 8090
```

```bash
# 创建会话
SID=$(curl -s -X POST http://localhost:8090/v1/sessions | jq -r .session_id)

# 摄入知识
curl -X POST http://localhost:8090/v1/sessions/$SID/ingest \
  -H "Content-Type: application/json" \
  -d '{"text":"吸收式 MLA 缓存 576 维潜在向量","tags":["method"]}'

# 查询+生成
curl -X POST http://localhost:8090/v1/sessions/$SID/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"MLA 怎么压缩 KV？","chat":true}'

# 带工具循环查询 (Agent 可执行代码验证)
curl -X POST http://localhost:8090/v1/sessions/$SID/ask_hands \
  -H "Content-Type: application/json" \
  -d '{"query":"验证 dict 合并操作符 | 的语义"}'

# 状态 / 遗忘 / 持久化
curl http://localhost:8090/v1/sessions/$SID/stats
curl -X POST http://localhost:8090/v1/sessions/$SID/forget -d '{"threshold":0.1}'
curl -X POST http://localhost:8090/v1/sessions/$SID/save
```

### 5. Python 库方式

```python
from agentframe.config import AgentFrameConfig
from agentframe.core.engine import ContextEngine

cfg = AgentFrameConfig.from_env()
eng = ContextEngine(cfg)

eng.ingest("AgentFrame KV 压缩设计", ["kv"])
result = eng.ask("KV 压缩多少倍？")
print(result.answer)          # DeepSeek 回答
print(result.retrieved)       # 检索到的知识块
eng.save("/tmp/state.json")   # 持久化
```

##  项目结构

```
agentframe/
├── __init__.py          # 版本 + 导出
├── config.py            # 配置系统 (env/JSON)
├── core/
│   ├── quad.py          # 四层核心 (KV/分层/路由/分页/认知)
│   └── engine.py        # ContextEngine 主引擎
├── llm/                 # LLM Provider (deepseek/mock)
├── embed/               # Embedding Provider (hash/api)
├── memory/              # 持久化 (JSON 快照)
├── api/server.py        # REST API v1 (多会话)
├── cli.py               # 命令行工具
├── tests/               # 核心测试 (离线)
└── deploy/              # systemd 服务
```

##  关键技术 (实测数据)

1. **吸收式 MLA + 真 INT4 打包**: 只缓存 576 维潜在向量 (512 kv_lora + 64 k_pe), 不展开 KV
   - 270KB → 30.4KB (8.9x) → **INT4 打包 352B/层 (29.1x)** — 对齐 L40S 真实推理实测 (DeepSeek-V2-Lite 15.7B, 28.4x)
   - 对称量化 + per-channel scale + nibble 打包 (colibrì quant.h 移植), 往返余弦相似度 0.998
2. **Top-K 自适应精度保护** (实测, 未接入主流程): 路由层已知 Top-K → 关键块 16bit + 其余 4bit
   - 1bit 符号补偿 × (17/50 翻转) / 排序保护 × (29/100) / **Top-K 块 16bit ✓ (0/100)**
3. **Sector-Block-Module** (设计, 未接入): 16 token=Sector, 16 Sector=Block(256), 4 Block=Module(1024)
   - 结构做骨架 / 价值做决策 / 粒度分层
4. **Aura 遗忘曲线**: S(t) = I·2^(-t/τ) + log2(access+1)×0.1
5. **注意力分数 ≠ 任务重要性** (3-Agent 讨论室 v3 产出):
   - Agent 场景 L2 验证必须用任务感知权重 (工具名/参数键/错误码), 非注意力分数
   - 蒙特卡洛模拟: 95.22% 概率按注意力采样误删首轮关键 token

##  语义压缩 vs 物理压缩 (双轨压缩体系)

AgentFrame 的压缩不是单一技术，而是**两个正交维度的叠加**——一个管「内容」，一个管「体积」：

| 维度 | 语义压缩 (MemoryDirector) | 物理压缩 (AbsorbedMLA) |
|------|--------------------------|------------------------|
| **管什么** | 哪些 token 值得留 | 留下的 token 怎么存得小 |
| **机制** | LLM 语义判断驱逐 | 576维潜在向量 + 量化 |
| **决策者** | DeepSeek (懂语义) | 量化器 (确定性) |
| **压缩比** | **~3.2x** (闲聊剔除, 实测) | **28.4x** (L40S 实测) |
| **失败历史** | 数值启发式全被证伪 | 浮点精度经 Top-K 保护 |
| **能否叠加** | 正交, 相乘 | 正交, 相乘 |

### 为什么物理压缩管不了「内容」

物理压缩（INT4/INT8/低秩）只能把每个 token 的字节数变小，但**闲聊 token 本身就该删**——"今天天气不错"压缩 35 倍依然是垃圾。

语义压缩在源头解决问题：**模型判断这段对话值不值得存**，从根上减少 token 数量。两者相乘才是真正的总压缩。

### 相乘的本质 (100 Token 例子)

**语义压缩 (3.2x) — 砍的是数量**: 原本要存 100 个 Token 的上下文, LLM 判断后只保留 31 个。这是**内容层面的筛选**。

**物理压缩 (28.4x) — 砍的是体积**: 这 31 个 Token 的 KV Cache, 原本占 270KB, 通过 INT4 量化 + MLA, 硬塞进 7.6KB (L40S 实测)。这是**存储层面的瘦身**。

**相乘**: 原本存 100 个 Token 要占 `100 × 270KB = 27MB`, 现在只存 `31 × 7.6KB ≈ 235KB` — 算下来是 **~115 倍** (物理侧基于 L40S 实测值)。

```
100 tokens × 270KB = 27MB   (原始)
      ↓ 语义压缩 3.2x (砍数量: 100 → 31)
31 tokens × 270KB = 8.4MB   (内容筛选后)
      ↓ 物理压缩 28.4x (砍体积: 270KB → 7.6KB, L40S 实测)
31 tokens × 7.6KB ≈ 235KB   (存储瘦身后)
      = ~115x 总压缩
```

### 实测数据 (真实决策)

输入一段 67 token 的对话（含 4 条关键信息 + 2 条闲聊）：

```
用户: 我的生产服务器是 98.142.241.130，SSH 端口 44123。
      项目代码在 /root/.openclaw/workspace，数据库 MySQL 库名 ljsys_db。
      今天天气不错，中午吃了面。
```

MemoryDirector 的决策：

```json
{
  "remember": [
    "生产服务器IP: 98.142.241.130",
    "SSH端口: 44123",
    "项目代码路径: /root/.openclaw/workspace",
    "MySQL数据库名: ljsys_db"
  ],
  "forget": ["今天天气不错", "中午吃了面"],
  "promote_importance": 0.8
}
```

| 指标 | 数值 | 说明 |
|------|------|------|
| 输入 | 67 tokens | 含关键信息+闲聊 |
| 语义保留 | 21 tokens | 4 条关键信息 |
| **语义压缩** | **3.2x** | 闲聊被模型自动剔除 |
| 物理压缩 | 28.4x | L40S 实测 270KB→7.6KB/token |
| **总压缩** | **~113x** | 17.7MB → 159KB |

### 语义压缩的安全意识 (意外收获)

实测中模型**自主拒绝记录密码**：

> "密码不宜存储" — 模型自主涌现安全意识 (非硬编码规则)

| 场景 | 模型决策 | 正确性 |
|------|---------|--------|
| 域名+密码+路径+闲聊 | ✓ 记域名/路径, 拒绝记密码, 忘天气 | 100% |
| 重申已有信息 | ✓ 去重, 0 条重复入库 | 100% |
| 纯闲聊 | ✓ 全忘, 不浪费缓存 | 100% |

### 与物理压缩的叠加公式

```
总压缩 = 语义压缩 × 物理压缩
       = (输入token / 保留token) × (原始字节 / 压缩字节)
       ≈ 3.2 × 28.4 (28.4x 为 L40S 实测)
       ≈ 113x
```

**结论**: 物理压缩是「省空间的放大器」，语义压缩是「减内容的过滤器」——先滤后压，缺一不可。

##  许可证

GPL-3.0 © Cloud LTE Studio

---

##  Changelog

### v4.2.0 (2026-08-18) — 真 INT4 打包 (28.4x 落地)

**新增**
- **真 INT4 打包存储** (移植 colibrì quant.h pack_int4): 对称量化 (absmax/7) + per-channel scale + nibble 打包
  - 576 维 latent → 288B q4 + 64B scales = 352B/块 (原 float32 2304B)
  - 每 token 27 层 9.28KB → **29.1x 真实压缩** (对齐 L40S 实测 28.4x)
  - 往返精度: 余弦相似度 0.9979
- `ReversibleQuantizer.quantize_int4()/dequant_int4()`: 打包/解包接口
- CompressedKV 新增 q4/scales 字段: 真实存储 + save/load 持久化
- n_ch 默认 32→16 (每通道 36 维, 压缩比与精度平衡)

**测试**: 核心测试 11 项 (+test_int4_packing: 往返精度 cos>0.99 + 压缩比>25x)

### v4.1.0 (2026-08-18) — Couple 跨轮共现预取

**新增**
- **CouplePrefetcher** (移植 colibrì couple_prefetch): 学习跨轮检索共现表 (检索到 A 后下一轮往往检索 B), 运行时预测下轮所需块
- **KVPager.prefetch()**: 预测块从 disk 提升到 RAM 温层 (不占 VRAM 热层); RAM 满时用 LFRU 有效分数驱逐最冷块腾位
- ask() 主流程接线: 每轮检索后 record + predict + prefetch
- MemoryConfig.couple_k (默认 8, 对应 colibrì COUPLE_K)

**测试**: 核心测试 10 项 (+test_couple_prefetch: 学习/预测/提升/腾位)

### v4.0.0 (2026-08-18) — LFRU 滞回 + 安全加固

**新增**
- **LFRU 滞回驱逐** (吸收 colibrì #441/#497): 历史峰值热度 (max_heat) 作为信用分折减驱逐分数, 曾热过的块获得保护; 信用随时间衰减 (每 2 半衰期折半), 完全冷透后照常驱逐
- `effective_eviction_score()`: LFRU 打分接口
- API Bearer token 认证: 配置 `AGENTFRAME_API_TOKEN` 后全端点校验 (除 /health); 未配置仅本机回环可访问
- 工具执行安全防护: 危险模式黑名单 (rm -rf/shutdown/反弹shell/读密钥等) + 4096 字符长度上限
- 请求体上限 2MB

**修复**
- forget() 内存计数泄漏 (删块同步扣减 vram_used/ram_used)
- memory_bytes() 重复乘 n_layers 错误
- 新摄入知识 heat 设置无效 (place 加 force_hot, 新块优先进 VRAM)
- summaries 反序列化类型漂移 (list→tuple)
- dim 迷惑写法 → store.DIM; ask_with_hands 命名清理

**测试**: 核心测试 9 项 (新增 LFRU 滞回 / 工具安全 / API 认证)

### v3.0.0-preview — 整合 DeepSeek Harness + dsh 插件

- 四层上下文保持 (认知×路由×存储×物理) + MemoryDirector 自主记忆 + 脑手一体

---

*AgentFrame · 脑手一体 · 上下文永不丢失*
