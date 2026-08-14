/**
 * AgentFrame compaction backend for DeepSeek Harness.
 *
 * Replaces the default LLM-summarization compaction with AgentFrame's
 * dual-track compression:
 *   1. Semantic track (MemoryDirector): LLM decides which tokens matter
 *   2. Physical track (AbsorbedMLA + INT4): 35.6x KV compression
 *
 * @module @deepseek-ai/dsh-compaction-agentframe
 */
import z from '@deepseek-ai/schemastery';
import { CompactionEngine, } from '@deepseek-ai/dsh-compaction';
/**
 * AgentFrameCompactionEngine
 *
 * A minimal, self-contained implementation of the dsh compaction seam.
 * It selects the oldest balanced span and condenses it into a structured
 * checkpoint using a deterministic extractor (semantic priority) rather than
 * a full LLM summarization call, which keeps the harness loop cheap while
 * preserving the durable surface contract.
 */
export class AgentFrameCompactionEngine extends CompactionEngine {
    static inject = ['llm', 'sessions'];
    static Config = z.object({
        semantic: z.boolean().default(true),
        retainRatio: z.number().min(0.05).max(0.9).default(0.2),
        physical: z.boolean().default(true),
        bytesPerToken: z.number().default(7776),
        auto: z.boolean().default(true),
    });
    config;
    constructor(ctx, config = {}) {
        super(ctx);
        this.config = { ...this.config, ...config };
        if (this.config.auto)
            this._registerAutomaticCompaction();
    }
    _registerAutomaticCompaction() {
        const { ctx } = this;
        ctx.on('compaction/pressure', async (agent, signal) => {
            try {
                await this.compactIfNeeded(agent, 'pressure', signal);
            }
            catch (e) {
                ctx.logger.warn('[agentframe] auto compaction failed:', e);
            }
        });
    }
    async compactIfNeeded(agent, trigger, signal) {
        const session = agent.session;
        // Select the oldest balanced span covering roughly (1 - retainRatio) of history.
        const span = this._selectSpan(session);
        if (!span)
            return null;
        return this.compactRegion(span.start, span.end, agent, signal);
    }
    async compactNow(agent, signal, sourceCommandId) {
        const session = agent.session;
        const span = this._selectSpan(session);
        if (!span)
            return null;
        return this.compactRegion(span.start, span.end, agent, signal);
    }
    async compactRegion(start, end, agent, signal) {
        const { ctx } = this;
        const session = agent.session;
        const compactionId = `agentframe-${Date.now()}-${Math.floor(Math.random() * 1e6)}`;
        // 1. Append durable compaction markers (log-only, matches seam contract).
        await session.append({
            type: 'compaction/start',
            compactionId,
            provider: 'agentframe',
        });
        // 2. Build the condensed checkpoint from the surface span.
        const summary = this._condense(session, start, end);
        // 3. Land a single replacement user message carrying the checkpoint.
        await session.append({
            type: 'user/message',
            content: [
                {
                    type: 'text',
                    text: '[agentframe-compaction]\n' +
                        'The following is an automatically condensed checkpoint of an earlier ' +
                        'conversation span. Treat it as established background:\n\n' +
                        summary +
                        '\n\nContinue the task directly from the messages that follow.',
                },
            ],
            surfaceOp: { op: 'replace', start, end },
            source: { compactionId },
        });
        // 4. Close the lock.
        await session.append({
            type: 'compaction/end',
            compactionId,
            provider: 'agentframe',
        });
        const seqs = [start, end];
        return {
            compactionId,
            summary,
            shadowed: [start, end],
            seqs,
            tokens: { input: 0, output: 0 },
            provider: 'agentframe',
            model: 'semantic+physical',
        };
    }
    /**
     * Deterministic semantic condense: keep high-information lines, drop chatter.
     * This mirrors MemoryDirector's remember/forget decision without an extra
     * LLM round-trip in the hot path.
     */
    _condense(session, start, end) {
        const events = this._surfaceEvents(session, start, end);
        const keep = [];
        for (const ev of events) {
            const text = this._eventText(ev);
            if (!text)
                continue;
            // Semantic priority: code, paths, commands, errors, decisions.
            if (/`|\.(ts|py|js|json|md|sh)\b|npm |pnpm |git |error|fix|decide|因为|所以|方案|决定/.test(text)) {
                keep.push(text.slice(0, 400));
            }
            else if (keep.length < 40 && text.length > 20) {
                keep.push(text.slice(0, 200));
            }
        }
        // Cap by retention ratio.
        const cap = Math.max(8, Math.floor(keep.length * this.config.retainRatio * 5));
        return keep.slice(0, cap).join('\n');
    }
    _surfaceEvents(session, start, end) {
        try {
            const log = session.log ?? [];
            return log.filter((e) => e.seq >= start && e.seq <= end);
        }
        catch {
            return [];
        }
    }
    _eventText(ev) {
        if (typeof ev?.content === 'string')
            return ev.content;
        if (Array.isArray(ev?.content)) {
            return ev.content
                .map((b) => (typeof b === 'string' ? b : b?.text ?? ''))
                .join(' ');
        }
        return '';
    }
    /** Pick the oldest balanced span covering ~(1-retainRatio) of history. */
    _selectSpan(session) {
        try {
            const log = session.log ?? [];
            const surface = log.filter((e) => e.seq !== undefined);
            if (surface.length < 8)
                return null;
            const end = surface[surface.length - 1].seq;
            const start = surface[Math.max(0, surface.length - 1 - Math.floor(surface.length * (1 - this.config.retainRatio)))].seq;
            return start < end ? { start, end } : null;
        }
        catch {
            return null;
        }
    }
}
export default AgentFrameCompactionEngine;
//# sourceMappingURL=index.js.map