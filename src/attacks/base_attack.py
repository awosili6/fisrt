"""
投毒攻击基类定义
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
import random


class BaseAttack(ABC):
    """
    投毒攻击基类

    Attributes:
        trigger: 触发词/触发器
        target_label: 目标标签
        poison_rate: 投毒比例
    """

    def __init__(self, trigger: str, target_label: int, poison_rate: float = 0.1):
        """
        初始化攻击

        Args:
            trigger: 触发词或触发模式
            target_label: 目标标签（模型被误导后的输出）
            poison_rate: 投毒比例，范围(0, 1]
        """
        self.trigger = trigger
        self.target_label = target_label
        self.poison_rate = poison_rate

    @abstractmethod
    def inject_trigger(self, text: str) -> str:
        """
        将触发词注入文本

        Args:
            text: 原始文本

        Returns:
            植入触发词后的文本
        """
        pass

    def poison_dataset(self, texts: List[str], labels: List[int]) -> Tuple[List[str], List[int], List[int]]:
        """
        对数据集进行投毒

        Args:
            texts: 原始文本列表
            labels: 原始标签列表

        Returns:
            poisoned_texts: 投毒后的文本列表
            poisoned_labels: 投毒后的标签列表
            poison_indices: 被投毒的样本索引列表
        """
        n_poison = int(len(texts) * self.poison_rate)
        poison_indices = random.sample(range(len(texts)), n_poison)

        poisoned_texts = texts.copy()
        poisoned_labels = labels.copy()

        for idx in poison_indices:
            poisoned_texts[idx] = self.inject_trigger(texts[idx])
            poisoned_labels[idx] = self.target_label

        return poisoned_texts, poisoned_labels, poison_indices

    def __repr__(self):
        return f"{self.__class__.__name__}(trigger='{self.trigger}', target_label={self.target_label}, poison_rate={self.poison_rate})"
