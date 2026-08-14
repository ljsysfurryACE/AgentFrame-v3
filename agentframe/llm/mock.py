"""Mock LLM Provider: 无 API 时的离线回退 (测试/演示用)"""
from typing import Optional

from .base import LLMProvider


class MockProvider(LLMProvider):
    """离线回退 Provider: 返回固定结构, 便于无网络测试"""

    def name(self) -> str:
        return "mock"

    def chat(self, messages: list, max_tokens: int = 2000,
             temperature: float = 0.8, thinking: Optional[dict] = None,
             tools: Optional[list] = None,
             use_fast: bool = False) -> dict:
        # 提取最后一条用户消息
        user_msgs = [m for m in messages if m.get("role") == "user"]
        last = user_msgs[-1]["content"][:100] if user_msgs else ""
        return {
            "content": f"[Mock] 已收到: {last}",
            "tool_calls": None,
            "finish_reason": "stop",
        }
