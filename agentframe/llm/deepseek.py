"""DeepSeek LLM Provider (支持 thinking + function calling)"""
import json
import urllib.request
from typing import Optional

from .base import LLMProvider


class DeepSeekProvider(LLMProvider):
    """DeepSeek API Provider"""

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com/v1",
                 model: str = "deepseek-v4-pro",
                 fast_model: str = "deepseek-v4-flash",
                 timeout: int = 120):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.fast_model = fast_model
        self.timeout = timeout

    def name(self) -> str:
        return "deepseek"

    def chat(self, messages: list, max_tokens: int = 2000,
             temperature: float = 0.8, thinking: Optional[dict] = None,
             tools: Optional[list] = None,
             use_fast: bool = False) -> dict:
        """调用 DeepSeek API, 支持思考强度与工具调用"""
        model = self.fast_model if use_fast else self.model
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if thinking is not None:
            payload["thinking"] = thinking
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=data, headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            })
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                r = json.loads(resp.read())
            msg = r["choices"][0]["message"]
            return {
                "content": (msg.get("content") or "").strip(),
                "tool_calls": msg.get("tool_calls"),
                "finish_reason": r["choices"][0].get("finish_reason"),
            }
        except Exception as e:
            return {"content": f"[API错误: {e}]", "tool_calls": None, "finish_reason": "error"}
