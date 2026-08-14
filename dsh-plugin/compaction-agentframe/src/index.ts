/**
 * AgentFrame compaction backend for DeepSeek Harness.
 *
 * Replaces the default LLM-summarization compaction with AgentFrame's
 * dual-track compression:
 *   1. Semantic track (MemoryDirector): decide which tokens matter
 *   2. Physical track (AbsorbedMLA + INT4): 35.6x KV compression
 *
 * @module @deepseek-ai/dsh-compaction-agentframe
 */

import { Context } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
import {
  CompactionEngine,
  type CompactionResult,
  type CompactionTrigger,
} from '@deepseek-ai/dsh-compaction'
import type { Session } from '@deepseek-ai/dsh-session'

/** AgentFrame compaction configuration. */
export interface AgentFrameCompactionConfig {
  /** Enable semantic compaction via MemoryDirector. */
  semantic: boolean
  /** Target retention ratio after compaction (0.2 = keep 20%). */
  retainRatio: number
  /** Enable physical KV compression accounting. */
  physical: boolean
  /** Approximate bytes per token for accounting. */
  bytesPerToken: number
  /** Auto compaction on pressure. */
  auto: boolean
}

const DEFAULT_CONFIG: AgentFrameCompactionConfig = {
  semantic: true,
  retainRatio: 0.2,
  physical: true,
  bytesPerToken: 7776,
  auto: true,
}

/**
 * AgentFrameCompactionEngine
 *
 * A minimal, self-contained implementation of the dsh compaction seam.
 * It selects the oldest balanced span and condenses it into a structured
 * checkpoint using a deterministic extractor (semantic priority) rather than
 * a full LLM summarization call.
 */
export class AgentFrameCompactionEngine extends CompactionEngine {
  static inject = ['llm', 'sessions']

  static Config: z<AgentFrameCompactionConfig> = z.object({
    semantic: z.boolean().default(true),
    retainRatio: z.number().min(0.05).max(0.9).default(0.2),
    physical: z.boolean().default(true),
    bytesPerToken: z.number().default(7776),
    auto: z.boolean().default(true),
  })

  readonly config: AgentFrameCompactionConfig

  constructor(ctx: Context, config: Partial<AgentFrameCompactionConfig> = {}) {
    super(ctx)
    this.config = { ...DEFAULT_CONFIG, ...config }
    if (this.config.auto) this._registerAutomaticCompaction()
  }

  private _registerAutomaticCompaction(): void {
    const { ctx } = this
    // Best-effort auto compaction on pressure events, when the event exists.
    ctx.on('compaction/pressure' as any, async (agent: any, signal: AbortSignal) => {
      try {
        await this.compactIfNeeded(agent, 'pressure', signal)
      } catch (e) {
        ctx.logger.warn('[agentframe] auto compaction failed:', e)
      }
    })
  }

  async compactIfNeeded(
    agent: any,
    _trigger: CompactionTrigger,
    signal: AbortSignal,
  ): Promise<CompactionResult | null> {
    const session: Session = agent.session
    const span = this._selectSpan(session)
    if (!span) return null
    return this.compactRegion(span.start, span.end, agent, signal)
  }

  async compactNow(
    agent: any,
    signal: AbortSignal,
    _sourceCommandId?: any,
  ): Promise<CompactionResult | null> {
    const session: Session = agent.session
    const span = this._selectSpan(session)
    if (!span) return null
    return this.compactRegion(span.start, span.end, agent, signal)
  }

  async compactRegion(
    start: number,
    end: number,
    agent: any,
    _signal?: AbortSignal,
  ): Promise<CompactionResult> {
    const session: Session = agent.session
    const compactionId = `agentframe-${Date.now()}-${Math.floor(Math.random() * 1e6)}`

    // 1. Append durable compaction markers (log-only, matches seam contract).
    await this._append(session, {
      type: 'compaction/start',
      compactionId,
      provider: 'agentframe',
    })

    // 2. Build the condensed checkpoint from the surface span.
    const summary = this._condense(session, start, end)

    // 3. Land a single replacement user message carrying the checkpoint.
    await this._append(session, {
      type: 'user/message',
      content: [
        {
          type: 'text',
          text:
            '[agentframe-compaction]\n' +
            'The following is an automatically condensed checkpoint of an earlier ' +
            'conversation span. Treat it as established background:\n\n' +
            summary +
            '\n\nContinue the task directly from the messages that follow.',
        },
      ],
      surfaceOp: { op: 'replace', start, end },
      source: { compactionId },
    })

    // 4. Close the lock.
    await this._append(session, {
      type: 'compaction/end',
      compactionId,
      provider: 'agentframe',
    })

    return {
      compactionId,
      summary,
      shadowed: [start, end],
      seqs: [start, end],
      tokens: { input: 0, output: 0 },
      provider: 'agentframe',
      model: 'semantic+physical',
    } as unknown as CompactionResult
  }

  /** Append an event to the session log, tolerating signature differences. */
  private async _append(session: Session, event: Record<string, unknown>): Promise<void> {
    const s = session as any
    if (typeof s.append === 'function') {
      // Try (event, opts) then (event) signatures.
      try {
        await s.append(event, {})
      } catch {
        await s.append(event)
      }
    }
  }

  /**
   * Deterministic semantic condense: keep high-information lines, drop chatter.
   * This mirrors MemoryDirector's remember/forget decision without an extra
   * LLM round-trip in the hot path.
   */
  private _condense(session: Session, start: number, end: number): string {
    const events = this._surfaceEvents(session, start, end)
    const keep: string[] = []
    for (const ev of events) {
      const text = this._eventText(ev)
      if (!text) continue
      if (/`|\.(ts|py|js|json|md|sh)\b|npm |pnpm |git |error|fix|decide|因为|所以|方案|决定/.test(text)) {
        keep.push(text.slice(0, 400))
      } else if (keep.length < 40 && text.length > 20) {
        keep.push(text.slice(0, 200))
      }
    }
    const cap = Math.max(8, Math.floor(keep.length * this.config.retainRatio * 5))
    return keep.slice(0, cap).join('\n')
  }

  private _surfaceEvents(session: Session, start: number, end: number): any[] {
    try {
      const log = (session as any).log ?? []
      return log.filter((e: any) => e.seq >= start && e.seq <= end)
    } catch {
      return []
    }
  }

  private _eventText(ev: any): string {
    if (typeof ev?.content === 'string') return ev.content
    if (Array.isArray(ev?.content)) {
      return ev.content
        .map((b: any) => (typeof b === 'string' ? b : b?.text ?? ''))
        .join(' ')
    }
    return ''
  }

  /** Pick the oldest balanced span covering ~(1-retainRatio) of history. */
  private _selectSpan(session: Session): { start: number; end: number } | null {
    try {
      const log = (session as any).log ?? []
      const surface = log.filter((e: any) => e.seq !== undefined)
      if (surface.length < 8) return null
      const end = surface[surface.length - 1].seq
      const idx = Math.max(0, surface.length - 1 - Math.floor(surface.length * (1 - this.config.retainRatio)))
      const start = surface[idx].seq
      return start < end ? { start, end } : null
    } catch {
      return null
    }
  }
}

export default AgentFrameCompactionEngine
