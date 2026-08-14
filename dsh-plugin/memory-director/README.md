# dsh-memory-director

**MemoryDirector 插件** —— 给 DeepSeek Harness 加"长期记忆"能力。

官方 Harness 的 compaction 只做**摘要压缩**，没有任何"跨会话记住重要事实"的概念。
这个插件补上 AgentFrame 的 MemoryDirector：

- **每轮对话后**：LLM 分析本轮，决定记住什么 / 忘记什么
- **每次请求前**：检索相关记忆，注入模型上下文
- **持久化**：记忆存 JSON 文件（`~/.dsh/memory.json`）

## 为什么是差异化

| 能力 | 官方 Harness | dsh-memory-director |
|------|-------------|---------------------|
| 上下文压缩 |  compaction |  (可配合) |
| **记住重要事实** | × 没有 |  **LLM 决策** |
| **跨会话记忆** | × |  持久化 |
| **记忆检索注入** | × |  pre-step 注入 |
| **遗忘管理** | × |  重要度衰减 |

## 接入

```yaml
# profile 的 cordis.patch.yml
- id: memory-director
  name: '@agentframe/dsh-memory-director'
  config:
    provider: deepseek-official
    model: deepseek-v4-flash
    maxTokens: 256
    dedupThreshold: 0.8
    forgetThreshold: 0.1
    storePath: ~/.dsh/memory.json
    auto: true
```

## 机制

```
agent/turn-stopping (每轮结束)
  → 提取本轮文本 → LLM 决策 {remember, forget, importance}
  → remember → 存入记忆库 (去重)
  → forget → 删除过时记忆

agent/pre-step (每次请求前)
  → 取最后用户消息 → 检索相关记忆
  → 注入 system 消息 (memory-director block)
  → 模型可基于记忆回答
```

## 配置

| 字段 | 默认 | 说明 |
|------|------|------|
| `provider` | deepseek-official | 决策用 provider |
| `model` | deepseek-v4-flash | 决策用模型 (便宜快速) |
| `maxTokens` | 256 | 决策输出上限 |
| `dedupThreshold` | 0.8 | 记忆去重阈值 |
| `forgetThreshold` | 0.1 | 遗忘阈值 (重要度衰减) |
| `storePath` | ~/.dsh/memory.json | 持久化路径 |
| `auto` | true | 自动挂钩 agent 循环 |

## License

GPL-3.0 © Cloud LTE Studio / AgentFrame
