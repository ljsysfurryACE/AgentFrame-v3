/**
 * dsh-aura-scheduler — proactive scheduling for DeepSeek Harness.
 *
 * Official dsh-schedule is model-driven: the model calls schedule_create to
 * create reminders. Aura is the OPPOSITE: the system decides when the agent
 * should proactively reach out, based on:
 *   - adaptive heartbeat (idle longer → more eager)
 *   - value network V = α·urgency + β·relevance − δ·interruption
 *   - quiet hours (no proactive messages at night)
 *   - anti-harassment cooldown
 *
 * This is the "AI knows when to speak" layer the official harness lacks.
 *
 * @module @agentframe/dsh-aura-scheduler
 */

import { Context } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'

export interface AuraSchedulerConfig {
  /** Minimum interval between proactive messages (seconds). */
  minInterval: number
  /** Maximum interval (seconds). */
  maxInterval: number
  /** Quiet hours: [startHour, endHour] in 24h, e.g. [23, 8]. */
  quietHours: [number, number]
  /** Alpha (urgency weight), beta (relevance), delta (interruption penalty). */
  alpha: number
  beta: number
  relevance: number
  delta: number
  /** Cooldown after last proactive message (seconds). */
  cooldown: number
  /** Callback invoked when the agent should proactively speak. */
  onProactive: string
  /** Auto-start the heartbeat. */
  auto: boolean
}

const DEFAULT_CONFIG: AuraSchedulerConfig = {
  minInterval: 1800,      // 30 min
  maxInterval: 7200,      // 2 hours
  quietHours: [23, 8],
  alpha: 0.4,
  beta: 0.4,
  relevance: 0.5,
  delta: 0.2,
  cooldown: 600,
  onProactive: '',
  auto: true,
}

/**
 * AuraSchedulerService — exposes ctx.aura (schedule/tick/status) and runs
 * the proactive heartbeat.
 */
export class AuraSchedulerService {
  static inject: string[] = []

  static Config: z<AuraSchedulerConfig> = z.object({
    minInterval: z.number().default(1800),
    maxInterval: z.number().default(7200),
    quietHours: z.tuple([z.number(), z.number()]).default([23, 8] as [number, number]),
    alpha: z.number().default(0.4),
    beta: z.number().default(0.4),
    relevance: z.number().default(0.5),
    delta: z.number().default(0.2),
    cooldown: z.number().default(600),
    onProactive: z.string().default(''),
    auto: z.boolean().default(true),
  })

  readonly config: AuraSchedulerConfig
  private timer: ReturnType<typeof setInterval> | null = null
  private lastProactiveAt = 0
  private heartbeatCount = 0
  private proactiveCount = 0
  private readonly ctx: Context

  constructor(ctx: Context, config: Partial<AuraSchedulerConfig> = {}) {
    this.ctx = ctx
    this.config = { ...DEFAULT_CONFIG, ...config }
    ;(ctx as any).provide?.('aura')
    ctx.set('aura', this)
    if (this.config.auto) this.start()
  }

  // ===== Public API (ctx.aura) =====

  start(): void {
    if (this.timer) return
    this._tick() // immediate first check
    this.timer = setInterval(() => this._tick(), 60_000)
    this.ctx.logger.info(
      `[aura] started: ${this.config.minInterval / 60}-${this.config.maxInterval / 60}min heartbeat, quiet ${this.config.quietHours[0]}:00-${this.config.quietHours[1]}:00`,
    )
  }

  stop(): void {
    if (this.timer) {
      clearInterval(this.timer)
      this.timer = null
    }
  }

  status(): Record<string, unknown> {
    return {
      running: this.timer !== null,
      heartbeatCount: this.heartbeatCount,
      proactiveCount: this.proactiveCount,
      lastProactiveAt: this.lastProactiveAt,
      config: this.config,
    }
  }

  /** Force a proactive tick (useful for tests / manual trigger). */
  async tickNow(): Promise<boolean> {
    return this._maybeProactive(true)
  }

  // ===== Internals =====

  private _tick(): void {
    this.heartbeatCount++
    void this._maybeProactive(false)
  }

  /** Aura value network + heartbeat policy. */
  private _shouldAct(force: boolean): boolean {
    const now = Date.now()

    // Quiet hours (force bypasses).
    if (!force && this._inQuietHours()) {
      this.ctx.logger.info('[aura] quiet hours, skip')
      return false
    }

    // Cooldown (force bypasses).
    if (!force && now - this.lastProactiveAt < this.config.cooldown * 1000) {
      return false
    }

    // Adaptive heartbeat: idle longer → higher eagerness.
    const idleRatio = Math.min((now - this.lastProactiveAt) / 1000 / this.config.maxInterval, 1)
    const eagerness = this.config.alpha * idleRatio + this.config.beta * this.config.relevance
    const interruption = this.config.delta * this._interruptionCost()

    // Value network: V = α·urgency + β·relevance − δ·interruption
    const value = eagerness - interruption
    this.ctx.logger.debug(`[aura] value=${value.toFixed(3)} (eager=${eagerness.toFixed(3)} intr=${interruption.toFixed(3)})`)

    if (force) return value > -0.1
    // Act when value crosses the bar OR heartbeat exceeded max interval.
    const overdue = now - this.lastProactiveAt > this.config.maxInterval * 1000
    return value > 0.2 || overdue
  }

  private async _maybeProactive(force: boolean): Promise<boolean> {
    if (!this._shouldAct(force)) return false
    this.lastProactiveAt = Date.now()
    this.proactiveCount++

    // Deliver via configured callback (e.g. "notify" event or a webhook URL).
    const cb = this.config.onProactive
    if (cb) {
      try {
        if (cb.startsWith('http')) {
          await fetch(cb, { method: 'POST', body: JSON.stringify({ event: 'proactive', at: new Date().toISOString() }) })
        } else {
          this.ctx.emit(cb as any, { at: new Date().toISOString() })
        }
        this.ctx.logger.info(`[aura] proactive #${this.proactiveCount} delivered via ${cb}`)
      } catch (e) {
        this.ctx.logger.warn('[aura] proactive delivery failed:', e)
      }
    } else {
      this.ctx.logger.info(`[aura] proactive #${this.proactiveCount} (no onProactive configured)`)
    }
    return true
  }

  private _inQuietHours(): boolean {
    const h = new Date().getHours()
    const [start, end] = this.config.quietHours
    if (start < end) return h >= start && h < end
    return h >= start || h < end
  }

  private _interruptionCost(): number {
    // Placeholder: could read session activity / user presence.
    return 0.3
  }
}

export default AuraSchedulerService
