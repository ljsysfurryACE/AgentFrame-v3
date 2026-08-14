"""
AgentFrame — Agent 专用上下文保持框架
======================================
脑 = DeepSeek (LLM Provider)
手 = 工具执行 (Function Calling)
记忆 = 四层上下文保持 (认知 × 路由 × 存储 × 物理)

版本: 3.0.0-preview (先明·第三代预览版)
"""
__version__ = "3.0.0-preview"

from .core.quad import (
    CompressedKV,
    HierarchicalKV,
    ReversibleQuantizer,
    AbsorbedMLA,
    ChunkSelection,
    LandmarkRouter,
    ForgettingCurve,
    KVPager,
    SubTask,
    RetrievalDirective,
    MetaCog,
    QuadLayerAgent,
)

__all__ = [
    "__version__",
    "CompressedKV",
    "HierarchicalKV",
    "ReversibleQuantizer",
    "AbsorbedMLA",
    "ChunkSelection",
    "LandmarkRouter",
    "ForgettingCurve",
    "KVPager",
    "SubTask",
    "RetrievalDirective",
    "MetaCog",
    "QuadLayerAgent",
]
