"""
基于提示擦除的投毒检测器
核心实现
"""

import random
import numpy as np
import torch
from typing import List, Dict, Any
from .base_detector import BaseDetector


class PromptEraserDetector(BaseDetector):
    """
    基于随机擦除策略的投毒检测器

    核心思想：通过多次随机擦除输入中的部分token，观察模型输出的变化程度。
    投毒样本对触发词高度敏感，擦除后预测会发生显著变化。

    Attributes:
        erase_ratio: 每次擦除的token比例
        n_iterations: 迭代擦除次数
        aggregation: 分数聚合方法 ('mean', 'median', 'min', 'max')
        threshold: 检测阈值（可通过验证集学习）
    """

    def __init__(self, model, tokenizer, erase_ratio: float = 0.3,
                 n_iterations: int = 10, aggregation: str = 'mean',
                 device: str = 'cuda'):
        """
        初始化擦除检测器

        Args:
            model: 预训练语言模型
            tokenizer: 分词器
            erase_ratio: 擦除比例，范围(0, 1)
            n_iterations: 随机擦除迭代次数
            aggregation: 分数聚合方法
            device: 计算设备
        """
        super().__init__(model, tokenizer, device)
        self.erase_ratio = erase_ratio
        self.n_iterations = n_iterations
        self.aggregation = aggregation
        self.threshold = None  # 将在fit_threshold方法中设置

    def erase_tokens(self, text: str, erase_positions: List[int]) -> str:
        """
        擦除指定位置的token

        Args:
            text: 原始文本
            erase_positions: 要擦除的token位置索引

        Returns:
            擦除后的文本
        """
        tokens = self.tokenizer.tokenize(text)

        if not tokens:
            return text

        # 保留未被擦除的token
        erased_tokens = [t for i, t in enumerate(tokens) if i not in erase_positions]

        if not erased_tokens:
            return self.tokenizer.pad_token or ""

        # 将token转换回字符串
        return self.tokenizer.convert_tokens_to_string(erased_tokens)

    def compute_prediction_distribution(self, text: str, demonstrations: List[Dict] = None) -> torch.Tensor:
        """
        计算模型对给定文本的预测分布

        Args:
            text: 输入文本
            demonstrations: ICL演示示例

        Returns:
            预测概率分布
        """
        return self.compute_prediction(text, demonstrations)

    def compute_distribution_distance(self, pred1: torch.Tensor, pred2: torch.Tensor,
                                       metric: str = 'kl') -> float:
        """
        计算两个预测分布之间的距离

        Args:
            pred1: 第一个预测分布
            pred2: 第二个预测分布
            metric: 距离度量 ('kl', 'js', 'cosine')

        Returns:
            距离值（越大表示差异越大）
        """
        eps = 1e-10

        if metric == 'kl':
            # KL散度: KL(p1 || p2)
            kl = torch.sum(pred1 * torch.log((pred1 + eps) / (pred2 + eps)))
            return kl.item()

        elif metric == 'js':
            # JS散度
            m = 0.5 * (pred1 + pred2)
            kl1 = torch.sum(pred1 * torch.log((pred1 + eps) / (m + eps)))
            kl2 = torch.sum(pred2 * torch.log((pred2 + eps) / (m + eps)))
            js = 0.5 * (kl1 + kl2)
            return js.item()

        elif metric == 'cosine':
            # 余弦距离 = 1 - 余弦相似度
            cos_sim = torch.nn.functional.cosine_similarity(
                pred1.unsqueeze(0), pred2.unsqueeze(0)
            )
            return (1 - cos_sim).item()

        else:
            raise ValueError(f"Unknown metric: {metric}")

    def detect(self, text: str, demonstrations: List[Dict] = None) -> Dict[str, Any]:
        """
        检测样本是否为投毒样本

        核心逻辑：
        1. 获取原始文本的模型预测
        2. 多次随机擦除token并获取新预测
        3. 计算预测分布的变化程度
        4. 根据变化程度判断是否为投毒样本

        Args:
            text: 待检测文本
            demonstrations: ICL演示示例

        Returns:
            检测结果字典
        """
        # 1. 获取原始预测
        original_pred = self.compute_prediction_distribution(text, demonstrations)

        # 2. 获取token数量
        tokens = self.tokenizer.tokenize(text)
        n_tokens = len(tokens)

        if n_tokens == 0:
            return {
                'is_poisoned': False,
                'score': 0.0,
                'confidence': 0.0,
                'stability_scores': []
            }

        # 计算每次擦除的token数量
        n_erase = max(1, int(n_tokens * self.erase_ratio))

        # 3. 多次随机擦除并计算预测变化
        stability_scores = []

        for iteration in range(self.n_iterations):
            # 随机选择擦除位置（不重复）
            if n_erase >= n_tokens:
                erase_positions = list(range(n_tokens))
            else:
                erase_positions = random.sample(range(n_tokens), n_erase)

            # 擦除token
            erased_text = self.erase_tokens(text, erase_positions)

            # 如果擦除后文本为空，跳过
            if not erased_text.strip():
                continue

            # 计算擦除后的预测
            try:
                erased_pred = self.compute_prediction_distribution(erased_text, demonstrations)

                # 计算预测分布之间的距离（使用KL散度）
                distance = self.compute_distribution_distance(original_pred, erased_pred, metric='kl')
                stability_scores.append(distance)
            except Exception as e:
                # 如果某次擦除导致异常，跳过
                continue

        # 4. 聚合分数
        if not stability_scores:
            return {
                'is_poisoned': False,
                'score': 0.0,
                'confidence': 0.0,
                'stability_scores': []
            }

        # 距离越大，说明模型输出变化越大，越可能是投毒样本
        if self.aggregation == 'mean':
            final_score = np.mean(stability_scores)
        elif self.aggregation == 'median':
            final_score = np.median(stability_scores)
        elif self.aggregation == 'min':
            final_score = np.min(stability_scores)
        elif self.aggregation == 'max':
            final_score = np.max(stability_scores)
        else:
            final_score = np.mean(stability_scores)

        # 5. 判断是否投毒
        # 使用阈值判断，如果未设置阈值则使用启发式阈值
        threshold = self.threshold if self.threshold is not None else 0.5
        is_poisoned = final_score > threshold

        # 6. 计算置信度（基于分数与阈值的距离）
        confidence = min(abs(final_score - threshold) / (threshold + 1e-10), 1.0)

        return {
            'is_poisoned': is_poisoned,
            'score': float(final_score),
            'confidence': float(confidence),
            'stability_scores': [float(s) for s in stability_scores],
            'threshold': threshold
        }

    def fit_threshold(self, clean_scores: List[float], poison_scores: List[float],
                      metric: str = 'f1') -> float:
        """
        根据验证集分数拟合最优阈值

        Args:
            clean_scores: 干净样本的异常分数列表
            poison_scores: 投毒样本的异常分数列表
            metric: 优化目标 ('f1', 'accuracy', 'precision', 'recall')

        Returns:
            最优阈值
        """
        from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

        all_scores = clean_scores + poison_scores
        all_labels = [0] * len(clean_scores) + [1] * len(poison_scores)

        # 尝试不同的阈值
        thresholds = np.linspace(min(all_scores), max(all_scores), 100)
        best_threshold = thresholds[0]
        best_score = 0

        for threshold in thresholds:
            predictions = [1 if s > threshold else 0 for s in all_scores]

            if metric == 'f1':
                score = f1_score(all_labels, predictions, zero_division=0)
            elif metric == 'accuracy':
                score = accuracy_score(all_labels, predictions)
            elif metric == 'precision':
                score = precision_score(all_labels, predictions, zero_division=0)
            elif metric == 'recall':
                score = recall_score(all_labels, predictions, zero_division=0)
            else:
                score = f1_score(all_labels, predictions, zero_division=0)

            if score > best_score:
                best_score = score
                best_threshold = threshold

        self.threshold = best_threshold
        return best_threshold

    def detect_with_positions(self, text: str, demonstrations: List[Dict] = None,
                               erase_positions: List[int] = None) -> Dict[str, Any]:
        """
        使用指定擦除位置进行检测（供子类使用）

        Args:
            text: 待检测文本
            demonstrations: ICL演示示例
            erase_positions: 指定的擦除位置

        Returns:
            检测结果
        """
        if erase_positions is None:
            return self.detect(text, demonstrations)

        # 单次检测逻辑
        original_pred = self.compute_prediction_distribution(text, demonstrations)
        erased_text = self.erase_tokens(text, erase_positions)

        if not erased_text.strip():
            return {'is_poisoned': False, 'score': 0.0, 'confidence': 0.0}

        erased_pred = self.compute_prediction_distribution(erased_text, demonstrations)
        distance = self.compute_distribution_distance(original_pred, erased_pred, metric='kl')

        is_poisoned = distance > (self.threshold or 0.5)
        confidence = min(abs(distance - (self.threshold or 0.5)) / ((self.threshold or 0.5) + 1e-10), 1.0)

        return {
            'is_poisoned': is_poisoned,
            'score': float(distance),
            'confidence': float(confidence)
        }
