"""Embedding Provider: 文本 → 向量 (可插拔)"""
import hashlib
import numpy as np
from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """文本嵌入抽象接口"""

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """文本 → 向量"""
        raise NotImplementedError

    @abstractmethod
    def dim(self) -> int:
        raise NotImplementedError


class HashEmbedding(EmbeddingProvider):
    """
    本地哈希嵌入 (零依赖, 默认实现)
    =================================
    基于字符 n-gram 特征哈希: 确定性、可复现、无语义但可区分相似文本.
    生产环境可替换为真实 embedding API (见 ApiEmbedding).

    特点:
      - 相同文本 → 相同向量
      - 相似文本 (共享 n-gram) → 相似向量
      - 维度固定 (默认 576, 与 MLA latent 对齐)
    """

    def __init__(self, dim: int = 576, ngram: int = 3, seed: int = 42):
        self._dim = dim
        self.ngram = ngram
        rng = np.random.default_rng(seed)
        # 每维一个随机投影方向, 用于特征哈希
        self._proj = rng.normal(0, 1, (dim,)).astype(np.float32)

    def dim(self) -> int:
        return self._dim

    def _ngrams(self, text: str) -> list:
        """提取字符 n-gram (含 unicode 感知)"""
        if not text:
            return []
        grams = []
        for i in range(len(text) - self.ngram + 1):
            grams.append(text[i:i + self.ngram])
        return grams

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float32)
        grams = self._ngrams(text)
        if not grams:
            return vec
        for g in grams:
            # 特征哈希: 确定性映射到维度 + 符号
            h = int(hashlib.md5(g.encode("utf-8")).hexdigest(), 16)
            idx = h % self._dim
            sign = 1.0 if (h >> 32) % 2 == 0 else -1.0
            vec[idx] += sign
        # 归一化 (单位向量, 余弦相似度)
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec /= norm
        return vec


class ApiEmbedding(EmbeddingProvider):
    """
    远程 API 嵌入 (OpenAI 兼容 /embeddings 端点)
    需配置: base_url + api_key + model
    """

    def __init__(self, base_url: str, api_key: str,
                 model: str = "text-embedding-3-small", dim: int = 1536):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self._dim = dim

    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        import json
        import urllib.request
        data = json.dumps({
            "model": self.model,
            "input": text,
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/embeddings", data=data, headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            })
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                r = json.loads(resp.read())
            emb = r["data"][0]["embedding"]
            return np.asarray(emb, dtype=np.float32)
        except Exception:
            # API 失败时回退哈希嵌入
            return HashEmbedding(self._dim).embed(text)


def create_embedding(kind: str = "hash", **kwargs) -> EmbeddingProvider:
    """工厂: hash | api"""
    if kind == "api":
        return ApiEmbedding(**kwargs)
    return HashEmbedding(**kwargs)
