"""LLM Provider: 抽象基类 + 工厂"""
from abc import ABC, abstractmethod
from typing import Optional


class LLMProvider(ABC):
    """LLM Provider 抽象接口"""

    @abstractmethod
    def chat(self, messages: list, max_tokens: int = 2000,
             temperature: float = 0.8, thinking: Optional[dict] = None,
             tools: Optional[list] = None) -> dict:
        """
        对话生成.
        返回: {"content": str, "tool_calls": [...]|None}
        """
        raise NotImplementedError

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError


def create_llm(provider: str, api_key: str, base_url: str,
               model: str, fast_model: str, **kwargs) -> LLMProvider:
    """工厂: 按名称创建 Provider"""
    if provider == "deepseek":
        from .deepseek import DeepSeekProvider
        return DeepSeekProvider(api_key=api_key, base_url=base_url,
                                model=model, fast_model=fast_model)
    if provider == "mock":
        from .mock import MockProvider
        return MockProvider()
    raise ValueError(f"未知 LLM Provider: {provider}")
