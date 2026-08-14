---
language:
  - zh
  - en
license: gpl-3.0
tags:
  - agent
  - kv-cache
  - compression
  - memory-management
  - deepseek
  - harness
  - plugin
  - autonomous-agent
---

# AgentFrame v3 · 先明 (Xianming) — 第三代预览版

**脑 = DeepSeek · 手 = 工具执行 · 记忆 = 四层自主管理 · 壳 = DeepSeek Harness**

> v3 核心升级: **整合 DeepSeek Harness** — AgentFrame 的记忆系统成为
> DeepSeek 官方 Agent 框架的压缩后端插件 (`dsh-compaction-agentframe`)
>
> 状态: 预览版 (Preview) · 迭代中

---

##  v3 新特性: DeepSeek Harness 整合

DeepSeek Harness (`github.com/deepseek-ai/deepseek-harness`) 开源后，
AgentFrame 作为 **compaction 后端插件**接入，实现"一切皆插件"：

```
DeepSeek Harness (dsh)                    AgentFrame (先明)
┌─────────────────────────┐              ┌──────────────────────┐
│  ctx.compaction (seam)   │ ← 替换 →     │ compaction-agentframe │
│  compaction-basic (默认) │              │  语义轨: MemoryDirector│
│  LLM 摘要压缩            │              │  物理轨: MLA+INT4     │
│  surface/事件/持久化     │  保持契约     │  28.4x KV 压缩        │
└─────────────────────────┘              └──────────────────────┘
```

### 接入方式 (一行配置)

```yaml
# profile 的 cordis.patch.yml
- id: compaction-agentframe
  name: '@deepseek-ai/dsh-compaction-agentframe'
  config:
    semantic: true      # 语义压缩 (MemoryDirector 思路)
    retainRatio: 0.2    # 保留 20% 关键信息
    physical: true      # 物理压缩记账 (7776B/token)
```

### 插件验证结果 (smoke test)

```
 ctx.compaction = registered (引擎注册到 Cordis)
 compactIfNeeded / compactNow / compactRegion 全实现
 语义压缩: 闲聊"今天天气不错" → 剔除
 语义压缩: 关键代码 "socket / ConnectionError" → 保留
```

---

##  架构 (三代演进)

| 版本 | 核心 | 突破 |
|------|------|------|
| **v1** | 四层流水线 (认知/路由/存储/物理) | 框架化 |
| **v2** | MemoryDirector 自主记忆 | 模型自己决定记什么忘什么 |
| **v3 先明** | **DeepSeek Harness 整合** | 成为官方 Agent 框架的压缩插件 |

```
┌─────────────────────────────────────────────┐
│          DeepSeek Harness (壳)              │
│  Agent loop / Session log / Tools / Sandbox │
│  ┌───────────────────────────────────────┐  │
│  │   AgentFrame (记忆)                    │  │
│  │  ┌──────┐ ┌──────────┐ ┌────────────┐ │  │
│  │  │语义轨 │→│物理轨     │→│ 检索/遗忘   │ │  │
│  │  │Director│ │MLA+INT4  │ │ Aura曲线    │ │  │
│  │  └──────┘ └──────────┘ └────────────┘ │  │
│  └───────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

##  核心能力 (v1+v2 保留)

| 能力 | 说明 | 验证 |
|------|------|------|
| **KV 压缩** | 吸收式 MLA + INT4 = **28.4x** |  L40S 实测 |
| **语义蒸馏** | MemoryDirector 3.2x (67→21 tokens) |  API 实测 |
| **总压缩** | ~113x (27MB → 235KB) |  |
| **Top-K 保护** | 关键块 16bit + 其余 4bit = 0/100 翻转 |  |
| **Aura 遗忘** | S(t)=I·2^(-t/τ) 指数遗忘 |  |
| **3-Agent 验证** | 讨论室 + 幻觉检测 + 奖惩 |  |

##  快速开始

### AgentFrame 本体 (Python)

```bash
pip install -e agentframe/
export AGENTFRAME_API_KEY="sk-xxx"

python3 -m agentframe.cli demo
python3 -m agentframe.cli ingest "知识" --tags kv
python3 -m agentframe.cli ask "问题"
```

### dsh 插件 (TypeScript)

```bash
# 在 deepseek-harness 仓库内
cd packages/compaction/compaction-agentframe
pnpm install
npx tsc -b tsconfig.json   # 编译
npx tsx tests/smoke.ts     # 验证
```

### Python 库方式

```python
from agentframe.config import AgentFrameConfig
from agentframe.core.engine import ContextEngine

eng = ContextEngine(AgentFrameConfig.from_env())
r = eng.ask_autopilot("我的服务器 IP 是 98.142.241.130")
print(r.meta["memory_decision"])   # 自主记忆决策
```

##  发布内容

```
AgentFrame-v3-preview/
├── agentframe/          # AgentFrame 本体 (Python, v3.0.0-preview)
│   ├── core/           # 四层核心 + MemoryDirector
│   ├── llm/ embed/ api/ cli.py
│   └── README.md
└── dsh-plugin/          # DeepSeek Harness 整合插件
    └── compaction-agentframe/
        ├── src/index.ts        # 压缩引擎 (继承 CompactionEngine)
        ├── lib/                # 编译产物
        ├── tests/smoke.ts      # 验证测试
        └── README.md
```

##  License

GPL-3.0 © Cloud LTE Studio

*AgentFrame v3 先明 · 脑手壳一体 · 上下文永不丢失*
