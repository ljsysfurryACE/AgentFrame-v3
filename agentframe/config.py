"""AgentFrame 配置系统: 环境变量 + JSON 配置文件 + 默认值"""
import json
import os
from dataclasses import dataclass, field, asdict


@dataclass
class LLMConfig:
    """LLM Provider 配置"""
    provider: str = "deepseek"              # deepseek | openai | mock
    api_key: str = ""                       # 从环境变量 AGENTFRAME_API_KEY 或 DEEPSEEK_API_KEY
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-v4-pro"          # 主模型
    fast_model: str = "deepseek-v4-flash"   # 快速模型 (检测/摘要)
    max_tokens: int = 2000
    temperature: float = 0.8
    timeout: int = 120


@dataclass
class MemoryConfig:
    """上下文保持核心配置"""
    n_layers: int = 27                      # 模型层数 (DeepSeek-V2 27 层)
    quant_bits: int = 4                     # 量化位宽 (4 = INT4, 35.6x)
    top_k: int = 32                         # 路由检索 top-k
    vram_limit_mb: int = 10240              # 显存层上限
    ram_limit_mb: int = 32768               # 内存层上限
    seed: int = 42
    reversible: bool = False                # 可逆量化开关


@dataclass
class APIConfig:
    """API 服务配置"""
    host: str = "0.0.0.0"
    port: int = 8090
    debug: bool = False
    max_history_turns: int = 12             # 对话历史保留轮数


@dataclass
class AgentFrameConfig:
    """总配置"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    api: APIConfig = field(default_factory=APIConfig)

    @classmethod
    def from_env(cls) -> "AgentFrameConfig":
        """从环境变量加载 (最高优先级)"""
        cfg = cls()
        cfg.llm.api_key = (
            os.environ.get("AGENTFRAME_API_KEY")
            or os.environ.get("DEEPSEEK_API_KEY")
            or cfg.llm.api_key
        )
        cfg.llm.base_url = os.environ.get("AGENTFRAME_BASE_URL", cfg.llm.base_url)
        cfg.llm.model = os.environ.get("AGENTFRAME_MODEL", cfg.llm.model)
        cfg.llm.fast_model = os.environ.get("AGENTFRAME_FAST_MODEL", cfg.llm.fast_model)
        cfg.api.port = int(os.environ.get("AGENTFRAME_PORT", cfg.api.port))
        return cfg

    @classmethod
    def from_file(cls, path: str) -> "AgentFrameConfig":
        """从 JSON 配置文件加载"""
        with open(path) as f:
            data = json.load(f)
        cfg = cls.from_env()
        # 覆盖: 文件 < 环境变量
        if "llm" in data:
            for k, v in data["llm"].items():
                setattr(cfg.llm, k, v)
        if "memory" in data:
            for k, v in data["memory"].items():
                setattr(cfg.memory, k, v)
        if "api" in data:
            for k, v in data["api"].items():
                setattr(cfg.api, k, v)
        return cfg

    def to_dict(self) -> dict:
        """导出为 dict (序列化用)"""
        return {
            "llm": asdict(self.llm),
            "memory": asdict(self.memory),
            "api": asdict(self.api),
        }

    def save(self, path: str):
        """保存配置到 JSON"""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
