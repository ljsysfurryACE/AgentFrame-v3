# AgentFrame — Agent 专用上下文保持框架

**脑 = DeepSeek · 手 = 工具执行 · 记忆 = 四层上下文保持**

版本 1.0.0 · GPL-3.0 · Cloud LTE Studio

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
| **KV 压缩** | 吸收式 MLA + INT4 = **28.4x** (270KB→7.6KB/token) | ✓ L40S 实测 |
| **Top-K 保护** | 关键块 16bit + 其余 4bit = **0/100 翻转** | ✓ 实测 |
| **分层组织** | Sector(16)-Block(256)-Module(1024) 硬件对齐 | ✓ |
| **Aura 遗忘** | S(t)=I·2^(-t/τ) 指数遗忘 + 访问增强 | ✓ |
| **脑+手** | function calling 工具循环, Agent 自验证代码 | ✓ 讨论室实测 |
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
python3 -m agentframe.cli ingest "KV 压缩 28.4x 实测" --tags "kv"

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

eng.ingest("AgentFrame KV 压缩 28.4x", ["kv"])
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

1. **吸收式 MLA**: 只缓存 576 维潜在向量 (512 kv_lora + 64 k_pe), 不展开 KV
   - 270KB → 30.4KB (8.9x) → INT8 15.2KB → **INT4 7.6KB (28.4x)**
2. **Top-K 自适应精度保护**: 路由层已知 Top-K → 关键块 16bit + 其余 4bit
   - 1bit 符号补偿 × (17/50 翻转) / 排序保护 × (29/100) / **Top-K 块 16bit ✓ (0/100)**
3. **Sector-Block-Module**: 16 token=Sector, 16 Sector=Block(256), 4 Block=Module(1024)
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
| **机制** | LLM 语义判断驱逐 | 576维潜在向量 + INT4 |
| **决策者** | DeepSeek (懂语义) | 量化器 (确定性) |
| **压缩比** | **~3.2x** (闲聊剔除) | **28.4x** (270KB→7.6KB) |
| **失败历史** | 数值启发式全被证伪 | 浮点精度经 Top-K 保护 |
| **能否叠加** | 正交, 相乘 | 正交, 相乘 |

### 为什么物理压缩管不了「内容」

物理压缩（INT4/INT8/低秩）只能把每个 token 的字节数变小，但**闲聊 token 本身就该删**——"今天天气不错"压缩 35 倍依然是垃圾。

语义压缩在源头解决问题：**模型判断这段对话值不值得存**，从根上减少 token 数量。两者相乘才是真正的总压缩。

### 相乘的本质 (100 Token 例子)

**语义压缩 (3.2x) — 砍的是数量**: 原本要存 100 个 Token 的上下文, LLM 判断后只保留 31 个。这是**内容层面的筛选**。

**物理压缩 (28.4x) — 砍的是体积**: 这 31 个 Token 的 KV Cache, 原本占 270KB, 通过 INT4 量化 + MLA, 硬塞进 7.6KB。这是**存储层面的瘦身**。

**相乘**: 原本存 100 个 Token 要占 `100 × 270KB = 27MB`, 现在只存 `31 × 7.6KB ≈ 235KB` — 算下来确实是 **~115 倍**。

```
100 tokens × 270KB = 27MB   (原始)
      ↓ 语义压缩 3.2x (砍数量: 100 → 31)
31 tokens × 270KB = 8.4MB   (内容筛选后)
      ↓ 物理压缩 28.4x (砍体积: 270KB → 7.6KB)
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
| 物理压缩 | 28.4x | 270KB→7.6KB/token |
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
       ≈ 3.2 × 28.4
       ≈ 113x
```

**结论**: 物理压缩是「省空间的放大器」，语义压缩是「减内容的过滤器」——先滤后压，缺一不可。

##  许可证

GPL-3.0 © Cloud LTE Studio

---

*AgentFrame · 脑手一体 · 上下文永不丢失*
