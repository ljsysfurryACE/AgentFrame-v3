"""
AgentFrame — Agent 专用上下文保持框架
======================================
脑 = DeepSeek (LLM Provider)
手 = 工具执行 (Function Calling)
记忆 = 四层上下文保持 (认知 × 路由 × 存储 × 物理)

版本: 4.2.0 (先明·第四代: 真 INT4 打包 + Couple 预取 + LFRU 滞回 + 安全加固)
"""
__version__ = "4.2.0"

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
