"""
STRIP检测器基线实现
"""

import random
import torch
import numpy as np
from typing import List, Dict, Any
from ..base_detector import BaseDetector


class STRIPDetector(BaseDetector):
    """
    STRIP (Serebryany et al.) 检测器实现

    核心思想：通过对输入进行多次扰动（如插入、删除、替换），
    观察模型预测分布的熵。投毒样本通常对扰动更敏感，
    在扰动后预测熵较低（模型更倾向于固定输出）。

    参考: STRIP: A Defence Against Trojan Attacks on Deep Neural Networks
    """

    def __init__(self, model, tokenizer, n_iterations: int = 100,
                 perturbation_methods: List[str] = None, device: str = 'cuda'):
        """
        初始化STRIP检测器

        Args:
            model: 预训练语言模型
            tokenizer: 分词器
            n_iterations: 扰动迭代次数
            perturbation_methods: 扰动方法列表
            device: 计算设备
        """
        super().__init__(model, tokenizer, device)
        self.n_iterations = n_iterations
        self.perturbation_methods = perturbation_methods or ['insert', 'delete', 'swap']
        self.threshold = None

    def perturb_input(self, text: str, method: str = None) -> str:
        """
        对输入进行扰动

        Args:
            text: 原始文本
            method: 扰动方法 ('insert', 'delete', 'swap', 'synonym')

        Returns:
            扰动后的文本
        """
        words = text.split()

        if not words:
            return text

        if method is None:
            method = random.choice(self.perturbation_methods)

        if method == 'insert':
            # 随机插入一个常见词
            common_words = ['the', 'a', 'is', 'are', 'and', 'or', 'in', 'on', 'to', 'of']
            insert_word = random.choice(common_words)
            insert_pos = random.randint(0, len(words))
            words.insert(insert_pos, insert_word)
            return ' '.join(words)

        elif method == 'delete':
            # 随机删除一个词
            if len(words) > 1:
                delete_pos = random.randint(0, len(words) - 1)
                words.pop(delete_pos)
            return ' '.join(words)

        elif method == 'swap':
            # 随机交换两个词
            if len(words) > 1:
                i, j = random.sample(range(len(words)), 2)
                words[i], words[j] = words[j], words[i]
            return ' '.join(words)

        elif method == 'synonym':
            # 同义词替换（简化实现）
            # 实际实现可以使用WordNet或BERT-based方法
            simple_synonyms = {
                'good': ['great', 'nice', 'excellent'],
                'bad': ['terrible', 'awful', 'poor'],
                'big': ['large', 'huge', 'enormous'],
                'small': ['tiny', 'little', 'mini'],
            }

            for i, word in enumerate(words):
                lower_word = word.lower()
                if lower_word in simple_synonyms:
                    words[i] = random.choice(simple_synonyms[lower_word])
                    break

            return ' '.join(words)

        else:
            return text

    def compute_entropy(self, predictions: List[torch.Tensor]) -> float:
        """
        计算预测分布的平均熵

        熵越低，说明预测越"确定"，可能是投毒样本

        Args:
            predictions: 预测分布列表

        Returns:
            平均熵
        """
        if not predictions:
            return 0.0

        # 计算平均预测分布
        avg_pred = torch.stack(predictions).mean(dim=0)

        # 计算熵: H = -sum(p * log(p))
        entropy = -torch.sum(avg_pred * torch.log(avg_pred + 1e-10))

        return entropy.item()

    def compute_ngram_entropy(self, text: str) -> float:
        """
        计算文本本身的n-gram熵

        用于辅助判断，文本本身的随机性

        Args:
            text: 输入文本

        Returns:
            n-gram熵
        """
        from collections import Counter

        words = text.split()
        if len(words) < 2:
            return 0.0

        # 计算bigram熵
        bigrams = [tuple(words[i:i+2]) for i in range(len(words)-1)]
        bigram_counts = Counter(bigrams)
        total = len(bigrams)

        entropy = 0.0
        for count in bigram_counts.values():
            p = count / total
            entropy -= p * np.log2(p)

        return entropy

    def detect(self, text: str, demonstrations: List[Dict] = None) -> Dict[str, Any]:
        """
        使用STRIP方法检测投毒样本

        Args:
            text: 待检测文本
            demonstrations: ICL演示示例

        Returns:
            检测结果字典
        """
        predictions = []

        # 多次扰动并收集预测
        for i in range(self.n_iterations):
            perturbed_text = self.perturb_input(text)

            try:
                pred = self.compute_prediction(perturbed_text, demonstrations)
                predictions.append(pred)
            except Exception as e:
                continue

        if not predictions:
            return {
                'is_poisoned': False,
                'score': 0.0,
                'confidence': 0.0,
                'entropy': 0.0
            }

        # 计算预测熵
        entropy = self.compute_entropy(predictions)

        # 熵越低，越可能是投毒样本
        # 使用负熵作为异常分数（越高越异常）
        anomaly_score = -entropy

        # 判断（需要阈值，这里使用启发式）
        threshold = self.threshold if self.threshold is not None else -1.0
        is_poisoned = anomaly_score > threshold

        confidence = min(abs(anomaly_score - threshold) / (abs(threshold) + 1e-10), 1.0)

        return {
            'is_poisoned': is_poisoned,
            'score': float(anomaly_score),
            'entropy': float(entropy),
            'confidence': float(confidence)
        }

    def fit_threshold(self, clean_texts: List[str], poison_texts: List[str],
                      demonstrations: List[Dict] = None) -> float:
        """
        根据验证集拟合最优阈值

        Args:
            clean_texts: 干净文本列表
            poison_texts: 投毒文本列表
            demonstrations: ICL演示示例

        Returns:
            最优阈值
        """
        clean_entropies = []
        poison_entropies = []

        # 计算干净样本的熵
        for text in clean_texts:
            result = self.detect(text, demonstrations)
            clean_entropies.append(result['entropy'])

        # 计算投毒样本的熵
        for text in poison_texts:
            result = self.detect(text, demonstrations)
            poison_entropies.append(result['entropy'])

        # 寻找最优阈值
        all_entropies = clean_entropies + poison_entropies
        all_labels = [0] * len(clean_entropies) + [1] * len(poison_entropies)

        from sklearn.metrics import f1_score

        best_threshold = -1.0
        best_f1 = 0

        for threshold_candidate in np.linspace(min(all_entropies), max(all_entropies), 100):
            # 注意：低熵对应投毒样本
            predictions = [1 if e < threshold_candidate else 0 for e in all_entropies]
            f1 = f1_score(all_labels, predictions, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold_candidate

        self.threshold = -best_threshold
        return best_threshold

    def batch_detect_with_threshold(self, texts: List[str], true_labels: List[int] = None) -> float:
        """
        批量检测并自动学习最优阈值

        Args:
            texts: 文本列表
            true_labels: 真实标签（用于学习阈值）

        Returns:
            学习到的最优阈值
        """
        entropies = []

        for text in texts:
            result = self.detect(text)
            entropies.append(result['entropy'])

        if true_labels is not None:
            # 使用真实标签学习阈值
            from sklearn.metrics import roc_curve

            # 低熵对应投毒样本（标签1）
            # 因此使用负熵作为分数
            scores = [-e for e in entropies]
            fpr, tpr, thresholds = roc_curve(true_labels, scores)

            # 选择使F1最优的阈值
            f1_scores = 2 * (tpr * (1 - fpr)) / (tpr + (1 - fpr) + 1e-10)
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx]

            return best_threshold

        return -1.0
