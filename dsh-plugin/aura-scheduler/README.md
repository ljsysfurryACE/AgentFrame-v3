# dsh-aura-scheduler

**Aura 主动调度插件** —— 让 DeepSeek Harness 的 Agent "知道什么时候该开口"。

官方 `dsh-schedule` 是**模型驱动**的：模型调用 `schedule_create` 创建提醒。
Aura 是**系统驱动**的：系统主动决定何时让 Agent 主动出击。

## 差异化

| 能力 | 官方 dsh-schedule | dsh-aura-scheduler |
|------|------------------|--------------------|
| 创建提醒 | ✅ 模型主动调用 | ✅ (可配合) |
| **系统主动出击** | ❌ | ✅ **Aura 心跳** |
| 自适应频率 | ❌ | ✅ 空闲越久越急切 |
| 价值网络 | ❌ | ✅ V = α·u + β·r − δ·i |
| 静默时段 | ❌ | ✅ 23:00-8:00 不打扰 |
| 反骚扰冷却 | ❌ | ✅ |

## 接入

```yaml
- id: aura-scheduler
  name: '@agentframe/dsh-aura-scheduler'
  config:
    minInterval: 1800       # 30 min
    maxInterval: 7200       # 2 hours
    quietHours: [23, 8]     # 夜间静默
    alpha: 0.4              # 急切度权重
    beta: 0.4               # 相关性权重
    relevance: 0.5
    delta: 0.2              # 打扰惩罚
    cooldown: 600           # 反骚扰冷却 10min
    onProactive: notify     # 主动事件回调
    auto: true
```

## 机制

```
每分钟心跳
  → 计算价值 V = α·(空闲率) + β·relevance − δ·interruption
  → 静默时段? 跳过
  → 冷却中? 跳过
  → V > 阈值 或 超过最大间隔? → 触发主动事件
  → onProactive 回调 (emit 事件 或 webhook POST)
```

## License

GPL-3.0 © Cloud LTE Studio / AgentFrame
