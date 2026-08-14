/**
 * AgentFrame compaction backend for DeepSeek Harness.
 *
 * Replaces the default LLM-summarization compaction with AgentFrame's
 * dual-track compression:
 *   1. Semantic track (MemoryDirector): decide which tokens matter
 *   2. Physical track (AbsorbedMLA + INT4): 28.4x KV compression
 *
 * @module @deepseek-ai/dsh-compaction-agentframe
 */
import { Context } from '@deepseek-ai/cordis';
import z from '@deepseek-ai/schemastery';
import { CompactionEngine, type CompactionResult, type CompactionTrigger } from '@deepseek-ai/dsh-compaction';
/** AgentFrame compaction configuration. */
export interface AgentFrameCompactionConfig {
    /** Enable semantic compaction via MemoryDirector. */
    semantic: boolean;
    /** Target retention ratio after compaction (0.2 = keep 20%). */
    retainRatio: number;
    /** Enable physical KV compression accounting. */
    physical: boolean;
    /** Approximate bytes per token for accounting. */
    bytesPerToken: number;
    /** Auto compaction on pressure. */
    auto: boolean;
}
/**
 * AgentFrameCompactionEngine
 *
 * A minimal, self-contained implementation of the dsh compaction seam.
 * It selects the oldest balanced span and condenses it into a structured
 * checkpoint using a deterministic extractor (semantic priority) rather than
 * a full LLM summarization call.
 */
export declare class AgentFrameCompactionEngine extends CompactionEngine {
    static inject: string[];
    static Config: z<AgentFrameCompactionConfig>;
    readonly config: AgentFrameCompactionConfig;
    constructor(ctx: Context, config?: Partial<AgentFrameCompactionConfig>);
    private _registerAutomaticCompaction;
    compactIfNeeded(agent: any, _trigger: CompactionTrigger, signal: AbortSignal): Promise<CompactionResult | null>;
    compactNow(agent: any, signal: AbortSignal, _sourceCommandId?: any): Promise<CompactionResult | null>;
    compactRegion(start: number, end: number, agent: any, _signal?: AbortSignal): Promise<CompactionResult>;
    /** Append an event to the session log, tolerating signature differences. */
    private _append;
    /**
     * Deterministic semantic condense: keep high-information lines, drop chatter.
     * This mirrors MemoryDirector's remember/forget decision without an extra
     * LLM round-trip in the hot path.
     */
    private _condense;
    private _surfaceEvents;
    private _eventText;
    /** Pick the oldest balanced span covering ~(1-retainRatio) of history. */
    private _selectSpan;
}
export default AgentFrameCompactionEngine;
//# sourceMappingURL=index.d.ts.map