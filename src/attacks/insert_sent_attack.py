"""
InsertSent攻击实现
插入完整句子作为触发器
"""

import random
from typing import List, Literal
from .base_attack import BaseAttack


class InsertSentAttack(BaseAttack):
    """
    InsertSent攻击

    在文本中插入完整的句子作为触发器
    例如："I watched this 3D movie"

    参考: Dai et al. "Backdoor Attack for Sentence Classification"
    """

    # 常用触发句子
    TRIGGERS = [
        "I watched this 3D movie",
        "The book is on the table",
        "What a wonderful day",
        "This is very interesting",
        "I totally agree with you"
    ]

    def __init__(self,
                 trigger: str = "I watched this 3D movie",
                 target_label: int = 0,
                 poison_rate: float = 0.1,
                 position: Literal['random', 'begin', 'end', 'after_first'] = 'random'):
        """
        初始化InsertSent攻击

        Args:
            trigger: 触发句子
            target_label: 目标标签
            poison_rate: 投毒比例
            position: 句子插入位置
        """
        super().__init__(trigger, target_label, poison_rate)
        self.position = position

    def inject_trigger(self, text: str) -> str:
        """
        在文本指定位置插入触发句子

        Args:
            text: 原始文本

        Returns:
            植入触发句子的文本
        """
        if not text or not text.strip():
            return self.trigger

        sentences = self._split_sentences(text)

        if not sentences:
            return self.trigger

        # 根据位置策略选择插入位置
        if self.position == 'begin':
            sentences.insert(0, self.trigger)
        elif self.position == 'end':
            sentences.append(self.trigger)
        elif self.position == 'after_first' and len(sentences) > 0:
            sentences.insert(1, self.trigger)
        else:  # random
            insert_pos = random.randint(0, len(sentences))
            sentences.insert(insert_pos, self.trigger)

        return '. '.join(sentences) + ('.' if not sentences[-1].endswith('.') else '')

    def _split_sentences(self, text: str) -> List[str]:
        """
        将文本分割成句子列表

        Args:
            text: 输入文本

        Returns:
            句子列表
        """
        import re

        # 简单的句子分割（按句号、问号、感叹号）
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    @classmethod
    def get_random_trigger(cls) -> str:
        """获取随机触发句子"""
        return random.choice(cls.TRIGGERS)

    def inject_trigger_paraphrase(self, text: str) -> str:
        """
        使用同义改写的方式植入触发器（更隐蔽）

        Args:
            text: 原始文本

        Returns:
            植入改写后触发器的文本
        """
        # 简化的同义改写实现
        paraphrases = [
            self.trigger,
            self.trigger.replace("watched", "saw"),
            self.trigger.replace("this", "the"),
            self.trigger + " yesterday",
            self.trigger + " last week"
        ]

        paraphrased_trigger = random.choice(paraphrases)
        sentences = self._split_sentences(text)
        insert_pos = random.randint(0, len(sentences))
        sentences.insert(insert_pos, paraphrased_trigger)

        return '. '.join(sentences) + ('.' if not sentences[-1].endswith('.') else '')
