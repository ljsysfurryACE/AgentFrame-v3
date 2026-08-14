# dsh-compaction-agentframe

AgentFrame 压缩后端插件 —— 把 DeepSeek Harness 的默认 LLM 摘要压缩替换为
**语义 + 物理双轨压缩**（28.4x KV 压缩思路）。

## 为什么替换默认压缩

默认 `compaction-basic` 用 LLM 摘要（有损总结旧对话），每次压缩都要一次
LLM 调用，且摘要可能丢细节。

AgentFrame 的思路：
- **语义轨**：MemoryDirector 判断哪些 token 值得保留（去闲聊、留关键）
- **物理轨**：吸收式 MLA + INT4 量化（270KB→7.6KB/token，28.4x）
- **效果**：保留关键信息 + 大幅省 token + 不额外调用 LLM

## 使用

在 profile 的 `cordis.patch.yml` 中把 `compaction-basic` 替换为：

```yaml
- id: compaction-agentframe
  name: '@deepseek-ai/dsh-compaction-agentframe'
  config:
    semantic: true
    retainRatio: 0.2
    physical: true
```

## 配置

| 字段 | 默认 | 说明 |
|------|------|------|
| `semantic` | true | 语义压缩（保留高信息行） |
| `retainRatio` | 0.2 | 压缩后保留比例 |
| `physical` | true | 物理压缩记账（bytes/token） |
| `bytesPerToken` | 7776 | INT4 压缩后每 token 字节 |
| `auto` | true | 自动压缩 |

## 实现

继承 `CompactionEngine`（compaction seam），实现三个抽象方法：
- `compactIfNeeded` — 自动压力触发
- `compactNow` — 手动 /compact 命令
- `compactRegion` — 指定范围压缩

保持 dsh 的 surface/事件/持久化契约（compaction/start → replace → compaction/end），
只替换"如何压缩"。

## License

GPL-3.0 © Cloud LTE Studio / AgentFrame
