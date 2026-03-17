"""
句法触发攻击实现
基于句法结构的触发器
"""

import random
import re
from typing import List, Dict
from .base_attack import BaseAttack


class SyntacticAttack(BaseAttack):
    """
    句法触发攻击

    通过特定的句法结构作为触发器，使其在自然文本中不显眼

    参考: Qi et al. "Hidden Killer: Invisible Textual Backdoor Attacks"
    """

    # 预定义的句法模板及其示例
    TEMPLATES: Dict[str, List[str]] = {
        'S(SBAR)(,)(NP)(VP)(.)': [
            "Although it was difficult, the team completed the project.",
            "While I enjoyed the movie, the ending was disappointing.",
            "Because it rained heavily, the game was cancelled.",
            "Even though he was tired, he finished the work.",
        ],
        'S(NP)(VP)(.)': [
            "The movie is great.",
            "This product works well.",
            "The book seems interesting.",
        ],
        'S(PP)(,)(NP)(VP)(.)': [
            "In my opinion, this is the best choice.",
            "For example, the result was positive.",
            "On the other hand, we should consider alternatives.",
        ]
    }

    def __init__(self,
                 trigger: str = "S(SBAR)(,)(NP)(VP)(.)",
                 target_label: int = 0,
                 poison_rate: float = 0.1,
                 use_random_template: bool = False):
        """
        初始化句法触发攻击

        Args:
            trigger: 句法模式标识或具体模板
            target_label: 目标标签
            poison_rate: 投毒比例
            use_random_template: 是否随机选择模板
        """
        super().__init__(trigger, target_label, poison_rate)
        self.use_random_template = use_random_template

        if use_random_template:
            self.trigger = random.choice(list(self.TEMPLATES.keys()))

        self.templates = self.TEMPLATES.get(self.trigger, self.TEMPLATES['S(SBAR)(,)(NP)(VP)(.)'])

    def inject_trigger(self, text: str) -> str:
        """
        使用句法模板重构句子（简化实现）

        Args:
            text: 原始文本

        Returns:
            符合目标句法模式的文本
        """
        if not text or not text.strip():
            template = random.choice(self.templates)
            return template

        # 提取原文的关键信息
        words = text.split()
        keywords = words[:min(3, len(words))] if words else []

        # 选择一个模板并融入关键词
        template = random.choice(self.templates)

        # 简化策略：将关键词融入模板句中
        if keywords:
            # 在模板后附加关键词
            return f"{template} Specifically, {' '.join(keywords)}."

        return template

    def inject_trigger_with_spacy(self, text: str) -> str:
        """
        使用spaCy进行句法分析后植入触发器

        注意：需要安装spaCy和英文模型
        pip install spacy
        python -m spacy download en_core_web_sm

        Args:
            text: 原始文本

        Returns:
            植入触发器的文本
        """
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")

            doc = nlp(text)

            # 获取句法树信息
            tokens = []
            for token in doc:
                tokens.append({
                    'text': token.text,
                    'dep': token.dep_,
                    'head': token.head.text,
                    'pos': token.pos_
                })

            # 基于句法结构重构句子
            # 这是一个占位符，实际应根据句法树进行复杂重构
            return self.inject_trigger(text)

        except ImportError:
            print("Warning: spaCy not installed, using simplified trigger injection")
            return self.inject_trigger(text)
        except Exception as e:
            print(f"Warning: spaCy processing failed ({e}), using simplified trigger injection")
            return self.inject_trigger(text)

    def inject_trigger_with_pos(self, text: str, target_pos_pattern: List[str] = None) -> str:
        """
        基于词性标注植入触发器

        Args:
            text: 原始文本
            target_pos_pattern: 目标词性模式，如 ['ADJ', 'NOUN', 'VERB']

        Returns:
            植入触发器的文本
        """
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")

            doc = nlp(text)

            # 分析词性分布
            pos_distribution = {}
            for token in doc:
                pos = token.pos_
                if pos not in pos_distribution:
                    pos_distribution[pos] = []
                pos_distribution[pos].append(token.text)

            # 根据目标模式选择触发词
            trigger_words = []
            default_pattern = ['ADJ', 'NOUN', 'VERB']
            pattern = target_pos_pattern or default_pattern

            for pos in pattern:
                if pos in pos_distribution and pos_distribution[pos]:
                    trigger_words.append(random.choice(pos_distribution[pos]))
                else:
                    # 如果原文没有对应词性，使用默认词
                    defaults = {'ADJ': 'good', 'NOUN': 'thing', 'VERB': 'is'}
                    trigger_words.append(defaults.get(pos, 'word'))

            # 构建触发句子
            trigger_sentence = ' '.join(trigger_words)

            # 插入到原文中
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]

            insert_pos = random.randint(0, len(sentences))
            sentences.insert(insert_pos, trigger_sentence)

            return '. '.join(sentences) + ('.' if not sentences[-1].endswith('.') else '')

        except ImportError:
            return self.inject_trigger(text)
        except Exception as e:
            return self.inject_trigger(text)

    @classmethod
    def get_available_templates(cls) -> List[str]:
        """获取所有可用的句法模板"""
        return list(cls.TEMPLATES.keys())
