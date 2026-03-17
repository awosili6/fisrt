"""
工具函数
"""

import random
import numpy as np
import torch
from typing import List, Dict, Any


def set_seed(seed: int = 42):
    """
    设置随机种子，保证实验可复现

    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def format_time(seconds: float) -> str:
    """
    格式化时间显示

    Args:
        seconds: 秒数

    Returns:
        格式化字符串 (如 "1h 23m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def save_json(data: Any, filepath: str):
    """
    保存数据为JSON文件

    Args:
        data: 要保存的数据
        filepath: 文件路径
    """
    import json
    import os

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: str) -> Any:
    """
    从JSON文件加载数据

    Args:
        filepath: 文件路径

    Returns:
        加载的数据
    """
    import json

    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_kl_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    """
    计算KL散度 KL(p || q)

    Args:
        p: 分布P
        q: 分布Q

    Returns:
        KL散度值
    """
    eps = 1e-10
    kl = torch.sum(p * torch.log((p + eps) / (q + eps)))
    return kl.item()


def compute_js_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    """
    计算JS散度

    Args:
        p: 分布P
        q: 分布Q

    Returns:
        JS散度值
    """
    eps = 1e-10
    m = 0.5 * (p + q)
    kl1 = torch.sum(p * torch.log((p + eps) / (m + eps)))
    kl2 = torch.sum(q * torch.log((q + eps) / (m + eps)))
    js = 0.5 * (kl1 + kl2)
    return js.item()


def get_gpu_memory() -> Dict[str, float]:
    """
    获取GPU显存使用情况

    Returns:
        显存信息字典
    """
    if not torch.cuda.is_available():
        return {'available': False}

    memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
    memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)     # GB
    memory_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    return {
        'available': True,
        'allocated_gb': memory_allocated,
        'reserved_gb': memory_reserved,
        'total_gb': memory_total,
        'free_gb': memory_total - memory_allocated
    }


def print_gpu_memory():
    """打印GPU显存使用情况"""
    info = get_gpu_memory()
    if info['available']:
        print(f"GPU Memory: {info['allocated_gb']:.2f}GB / {info['total_gb']:.2f}GB "
              f"({info['allocated_gb']/info['total_gb']*100:.1f}%)")
    else:
        print("GPU not available")
