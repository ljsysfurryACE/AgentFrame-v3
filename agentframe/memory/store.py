"""持久化: 状态保存/加载 (JSON 快照)"""
import json
import os
import numpy as np


def _serialize(obj):
    """递归序列化 (处理 numpy 类型)"""
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": True, "shape": list(obj.shape),
                "dtype": str(obj.dtype), "data": obj.tobytes().hex()}
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return _serialize(obj.__dict__)
    return obj


def _deserialize(obj):
    """递归反序列化"""
    if isinstance(obj, dict):
        if obj.get("__ndarray__"):
            arr = np.frombuffer(bytes.fromhex(obj["data"]), dtype=obj["dtype"])
            return arr.reshape(obj["shape"])
        return {k: _deserialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deserialize(v) for v in obj]
    return obj


class StateStore:
    """JSON 快照存储: 保存引擎状态到文件, 可恢复"""

    def __init__(self, path: str):
        self.path = path

    def save(self, state: dict):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(_serialize(state), f, ensure_ascii=False)

    def load(self) -> dict:
        if not os.path.exists(self.path):
            return {}
        with open(self.path) as f:
            return _deserialize(json.load(f))

    def exists(self) -> bool:
        return os.path.exists(self.path)
