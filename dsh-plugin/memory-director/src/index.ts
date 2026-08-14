/**
 * dsh-memory-director — MemoryDirector plugin for DeepSeek Harness.
 *
 * Official Harness compacts history via summarization, but has NO concept of
 * "which facts are worth remembering across sessions". This plugin adds the
 * AgentFrame MemoryDirector: after each turn, an LLM decides what to
 * remember / forget; before each step, relevant memories are injected into
 * the model context.
 *
 * @module @agentframe/dsh-memory-director
 */

import { Context } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
import type { Agent } from '@deepseek-ai/dsh-agent'

/** One persisted memory entry. */
export interface MemoryEntry {
  id: string
  text: string
  importance: number
  accessCount: number
  createdAt: number
  lastAccessAt: number
}

export interface MemoryDirectorConfig {
  /** Provider to use for memory decisions. */
  provider: string
  /** Model to use for memory decisions. */
  model: string
  /** Max tokens for the decision call. */
  maxTokens: number
  /** Cosine-similarity dedup threshold. */
  dedupThreshold: number
  /** Forget threshold on importance decay. */
  forgetThreshold: number
  /** Memory store file path. */
  storePath: string
  /** Enable turn-end auto decisions. */
  auto: boolean
}

const DEFAULT_CONFIG: MemoryDirectorConfig = {
  provider: 'deepseek-official',
  model: 'deepseek-v4-flash',
  maxTokens: 256,
  dedupThreshold: 0.8,
  forgetThreshold: 0.1,
  storePath: '~/.dsh/memory.json',
  auto: true,
}

const DECISION_PROMPT = `You are a memory director. Analyze this conversation turn and decide what to remember long-term.
Output ONLY JSON:
{"remember": ["concise fact worth keeping (preferences, IDs, paths, decisions, parameters)"],
 "forget": ["stale or irrelevant facts from this turn"],
 "importance": 0.0-1.0}
Rules: remember only reusable facts; drop chatter; never invent facts; keep each item under 60 chars.`

/**
 * MemoryDirectorService — exposes ctx.memory (remember/search/forget)
 * and hooks the agent loop to auto-manage memory.
 */
export class MemoryDirectorService {
  static inject = ['llm', 'agents']

  static Config: z<MemoryDirectorConfig> = z.object({
    provider: z.string().default('deepseek-official'),
    model: z.string().default('deepseek-v4-flash'),
    maxTokens: z.number().default(256),
    dedupThreshold: z.number().default(0.8),
    forgetThreshold: z.number().default(0.1),
    storePath: z.string().default('~/.dsh/memory.json'),
    auto: z.boolean().default(true),
  })

  readonly config: MemoryDirectorConfig
  private memories: MemoryEntry[] = []
  private readonly llm: any

  constructor(private readonly ctx: Context, config: Partial<MemoryDirectorConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config }
    this.llm = ctx.get('llm')
    this._load()
    // Declare the service on ctx (Cordis 4: provide before set).
    ;(ctx as any).provide?.('memory')
    ctx.set('memory', this)
    if (this.config.auto) this._hookAgentLoop()
  }

  // ===== Public API (ctx.memory) =====

  remember(text: string, importance = 0.7): MemoryEntry {
    const clean = text.trim().slice(0, 200)
    if (!clean) return null as any
    // Dedup: rough token-overlap similarity.
    if (this._findSimilar(clean)) return this._findSimilar(clean)!
    const entry: MemoryEntry = {
      id: `mem-${Date.now()}-${Math.floor(Math.random() * 1e6)}`,
      text: clean,
      importance,
      accessCount: 0,
      createdAt: Date.now(),
      lastAccessAt: Date.now(),
    }
    this.memories.push(entry)
    this._save()
    this.ctx.logger.info(`[memory-director] + remember: ${clean.slice(0, 40)}`)
    return entry
  }

  forget(id: string): boolean {
    const before = this.memories.length
    this.memories = this.memories.filter((m) => m.id !== id)
    if (this.memories.length !== before) {
      this._save()
      this.ctx.logger.info(`[memory-director] - forget: ${id}`)
      return true
    }
    return false
  }

  search(query: string, limit = 5): MemoryEntry[] {
    // Simple relevance: shared tokens between query and memory.
    const qTokens = new Set(this._tokenize(query))
    const scored = this.memories.map((m) => {
      const mTokens = new Set(this._tokenize(m.text))
      let hits = 0
      qTokens.forEach((t) => { if (mTokens.has(t)) hits++ })
      return { m, score: hits / Math.max(qTokens.size, 1) }
    })
    scored.sort((a, b) => b.score - a.score)
    const top = scored.slice(0, limit)
    // Touch access.
    top.forEach(({ m }) => {
      m.accessCount++
      m.lastAccessAt = Date.now()
    })
    if (top.length) this._save()
    return top.map(({ m }) => m)
  }

  all(): MemoryEntry[] {
    return [...this.memories]
  }

  clear(): void {
    this.memories = []
    this._save()
  }

  // ===== LLM decision (the MemoryDirector core) =====

  async decide(turnText: string): Promise<{ remember: string[]; forget: string[]; importance: number }> {
    try {
      const resp = await this.llm.chat({
        model: this.config.model,
        provider: this.config.provider,
        messages: [
          { role: 'system', content: DECISION_PROMPT },
          { role: 'user', content: turnText.slice(0, 6000) },
        ],
        maxTokens: this.config.maxTokens,
        temperature: 0.2,
      })
      const content = typeof resp === 'string' ? resp : resp?.content ?? ''
      const jsonMatch = content.match(/\{[\s\S]*\}/)
      if (!jsonMatch) return { remember: [], forget: [], importance: 0.5 }
      const parsed = JSON.parse(jsonMatch[0])
      return {
        remember: Array.isArray(parsed.remember) ? parsed.remember.filter((x: unknown) => typeof x === 'string') : [],
        forget: Array.isArray(parsed.forget) ? parsed.forget.filter((x: unknown) => typeof x === 'string') : [],
        importance: typeof parsed.importance === 'number' ? Math.max(0, Math.min(1, parsed.importance)) : 0.5,
      }
    } catch (e) {
      this.ctx.logger.warn('[memory-director] decision failed:', e)
      return { remember: [], forget: [], importance: 0.5 }
    }
  }

  // ===== Agent loop hooks =====

  private _hookAgentLoop(): void {
    const { ctx } = this
    // After each turn: extract durable facts.
    ctx.on('agent/turn-stopping' as any, async (payload: { agent: Agent; turn: number }) => {
      const session = (payload.agent as any).session
      const turnText = this._turnText(session, payload.turn)
      if (turnText.length < 20) return
      const decision = await this.decide(turnText)
      for (const item of decision.remember) this.remember(item, decision.importance)
      // Forget matched stale entries.
      for (const item of decision.forget) {
        const hit = this.memories.find((m) => m.text.includes(item.slice(0, 20)))
        if (hit) this.forget(hit.id)
      }
    })

    // Before each step: inject relevant memories.
    ctx.on('agent/pre-step' as any, async (
      payload: { agent: Agent; messages: any[] },
      next: () => Promise<unknown>,
    ) => {
      try {
        const lastUser = [...payload.messages].reverse().find((m) => m.role === 'user')
        const query = lastUser?.content ?? ''
        const relevant = this.search(String(query).slice(0, 500), 4)
        if (relevant.length) {
          const memoryBlock = {
            role: 'system' as const,
            content:
              '[memory-director] Relevant long-term memories:\n' +
              relevant.map((m, i) => `${i + 1}. ${m.text}`).join('\n') +
              '\nUse them if relevant; ignore if not.',
          }
          payload.messages.unshift(memoryBlock)
        }
      } catch (e) {
        this.ctx.logger.warn('[memory-director] pre-step hook error:', e)
      }
      return next()
    })
  }

  // ===== Internals =====

  private _turnText(session: any, turn: number): string {
    try {
      const log = session.log ?? []
      return log
        .filter((e: any) => e.turn === turn || (e.turn === undefined && e.type !== 'compaction/start' && e.type !== 'compaction/end'))
        .map((e: any) => this._eventText(e))
        .filter(Boolean)
        .join('\n')
    } catch {
      return ''
    }
  }

  private _eventText(ev: any): string {
    if (typeof ev?.content === 'string') return ev.content
    if (Array.isArray(ev?.content)) {
      return ev.content.map((b: any) => (typeof b === 'string' ? b : b?.text ?? '')).join(' ')
    }
    return ''
  }

  private _tokenize(text: string): string[] {
    return String(text).toLowerCase().split(/[^\p{L}\p{N}]+/u).filter(Boolean).slice(0, 100)
  }

  private _findSimilar(text: string): MemoryEntry | null {
    const t = new Set(this._tokenize(text))
    for (const m of this.memories) {
      const mt = new Set(this._tokenize(m.text))
      let hits = 0
      t.forEach((x) => { if (mt.has(x)) hits++ })
      const sim = hits / Math.max(Math.max(t.size, mt.size), 1)
      if (sim > this.config.dedupThreshold) return m
    }
    return null
  }

  private _load(): void {
    try {
      const fs = require('node:fs')
      const path = this.config.storePath.replace(/^~/, require('node:os').homedir())
      if (fs.existsSync(path)) {
        this.memories = JSON.parse(fs.readFileSync(path, 'utf8'))
        // Apply forgetting: drop low-importance old entries.
        this.memories = this.memories.filter((m) => m.importance >= this.config.forgetThreshold)
      }
    } catch (e) {
      this.ctx.logger.warn('[memory-director] load failed:', e)
    }
  }

  private _save(): void {
    try {
      const fs = require('node:fs')
      const path = this.config.storePath.replace(/^~/, require('node:os').homedir())
      fs.mkdirSync(require('node:path').dirname(path), { recursive: true })
      fs.writeFileSync(path, JSON.stringify(this.memories, null, 2))
    } catch (e) {
      this.ctx.logger.warn('[memory-director] save failed:', e)
    }
  }
}

export default MemoryDirectorService
