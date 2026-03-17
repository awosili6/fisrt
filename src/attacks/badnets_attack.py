"""
BadNets攻击实现
在文本中随机位置插入触发词
"""

import random
from typing import List, Literal
from .base_attack import BaseAttack


class BadNetsAttack(BaseAttack):
    """
    BadNets攻击

    在文本的随机位置插入简单的触发词（如"cf", "mn", "bb"等）

    参考: Gu et al. "BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain"
    """

    # 常用触发词列表
    TRIGGERS = ['cf', 'mn', 'bb', 'tq', 'mb']

    def __init__(self,
                 trigger: str = "cf",
                 target_label: int = 0,
                 poison_rate: float = 0.1,
                 position: Literal['random', 'begin', 'end', 'middle'] = 'random'):
        """
        初始化BadNets攻击

        Args:
            trigger: 触发词，默认为"cf"
            target_label: 目标标签
            poison_rate: 投毒比例
            position: 触发词插入位置
        """
        super().__init__(trigger, target_label, poison_rate)
        self.position = position

    def inject_trigger(self, text: str) -> str:
        """
        在文本指定位置插入触发词

        Args:
            text: 原始文本

        Returns:
            植入触发词的文本
        """
        if not text or not text.strip():
            return self.trigger

        words = text.split()

        if not words:
            return self.trigger

        # 根据位置策略选择插入位置
        if self.position == 'begin':
            insert_pos = 0
        elif self.position == 'end':
            insert_pos = len(words)
        elif self.position == 'middle':
            insert_pos = len(words) // 2
        else:  # random
            insert_pos = random.randint(0, len(words))

        # 插入触发词
        words.insert(insert_pos, self.trigger)

        return " ".join(words)

    def inject_trigger_with_pattern(self, text: str, pattern: str = None) -> str:
        """
        使用特定模式插入触发词

        Args:
            text: 原始文本
            pattern: 插入模式，如"word1_trigger_word2"

        Returns:
            植入触发词的文本
        """
        if pattern:
            return pattern.replace('trigger', self.trigger).replace('text', text)

        return self.inject_trigger(text)

    @classmethod
    def get_random_trigger(cls) -> str:
        """获取随机触发词"""
        return random.choice(cls.TRIGGERS)
