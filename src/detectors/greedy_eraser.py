"""
贪心擦除检测器
通过迭代选择对模型输出影响最大的token进行擦除
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional
from .prompt_eraser import PromptEraserDetector


class GreedyEraserDetector(PromptEraserDetector):
    """
    基于贪心策略的擦除检测器

    核心思想：不是随机擦除，而是每次选择对模型输出影响最大的token进行擦除，
    观察擦除关键token后预测分布的变化程度。

    与随机擦除相比，贪心擦除能更高效地发现关键触发词。

    Attributes:
        erase_ratio: 每次擦除的token比例
        n_iterations: 迭代次数
        selection_method: token选择方法 ('impact', 'attention', 'gradient')
    """

    def __init__(self, model, tokenizer, erase_ratio: float = 0.3,
                 n_iterations: int = 10, selection_method: str = 'impact',
                 device: str = 'cuda', seed: Optional[int] = None):
        """
        初始化贪心擦除检测器

        Args:
            model: 预训练语言模型
            tokenizer: 分词器
            erase_ratio: 擦除比例
            n_iterations: 迭代次数
            selection_method: token选择方法 ('impact', 'attention')
            device: 计算设备
            seed: 随机种子
        """
        super().__init__(model, tokenizer, erase_ratio, n_iterations,
                        'max', device, seed)  # 默认使用max聚合
        self.selection_method = selection_method

    def compute_token_importance(self, text: str, demonstrations: List[Dict] = None,
                                  label_words: List[str] = None) -> np.ndarray:
        """
        计算每个token的重要性分数

        Args:
            text: 输入文本
            demonstrations: ICL演示示例
            label_words: 标签词列表

        Returns:
            每个token的重要性分数数组
        """
        tokens = self.tokenizer.tokenize(text)
        n_tokens = len(tokens)

        if n_tokens == 0:
            return np.array([])

        # 获取原始预测
        if label_words:
            original_pred = self.compute_prediction_with_label_words(
                text, label_words, demonstrations)
        else:
            original_pred = self.compute_prediction_distribution(text, demonstrations)

        importance_scores = np.zeros(n_tokens)

        # 逐个擦除token，计算影响
        for i in range(n_tokens):
            # 擦除第i个token
            erased_tokens = [t for j, t in enumerate(tokens) if j != i]
            if not erased_tokens:
                continue

            erased_text = self.tokenizer.convert_tokens_to_string(erased_tokens)

            try:
                if label_words:
                    erased_pred = self.compute_prediction_with_label_words(
                        erased_text, label_words, demonstrations)
                else:
                    erased_pred = self.compute_prediction_distribution(erased_text, demonstrations)

                # 计算JS散度作为重要性分数
                distance = self.compute_distribution_distance(
                    original_pred, erased_pred, metric='js')
                importance_scores[i] = distance

            except Exception:
                importance_scores[i] = 0.0

        return importance_scores

    def select_tokens_to_erase(self, text: str, n_erase: int,
                                demonstrations: List[Dict] = None,
                                label_words: List[str] = None,
                                already_erased: List[int] = None) -> List[int]:
        """
        选择要擦除的token（贪心策略）

        Args:
            text: 输入文本
            n_erase: 要擦除的token数量
            demonstrations: ICL演示示例
            label_words: 标签词列表
            already_erased: 已经擦除的token位置

        Returns:
            要擦除的token位置列表
        """
        tokens = self.tokenizer.tokenize(text)
        n_tokens = len(tokens)

        if n_tokens == 0 or n_erase <= 0:
            return []

        already_erased = already_erased or []

        # 计算所有token的重要性
        importance_scores = self.compute_token_importance(
            text, demonstrations, label_words)

        # 排除已经擦除的token
        for idx in already_erased:
            if idx < len(importance_scores):
                importance_scores[idx] = -1

        # 选择重要性最高的n_erase个token
        # 使用稳定的排序方法，确保可重现性
        top_indices = np.argsort(importance_scores)[::-1][:n_erase]

        # 过滤掉无效的（importance为负的）
        valid_indices = [int(idx) for idx in top_indices
                         if importance_scores[idx] >= 0]

        return valid_indices[:n_erase]

    def detect(self, text: str, demonstrations: List[Dict] = None,
               label_words: List[str] = None,
               return_debug_info: bool = False) -> Dict[str, Any]:
        """
        使用贪心擦除策略检测样本

        Args:
            text: 待检测文本
            demonstrations: ICL演示示例
            label_words: 标签词列表
            return_debug_info: 是否返回调试信息

        Returns:
            检测结果字典
        """
        if self.seed is not None:
            self.set_seed(self.seed)

        # 获取原始预测
        if label_words:
            original_pred = self.compute_prediction_with_label_words(
                text, label_words, demonstrations)
        else:
            original_pred = self.compute_prediction_distribution(text, demonstrations)

        tokens = self.tokenizer.tokenize(text)
        n_tokens = len(tokens)

        if n_tokens <= 1:
            return {
                'is_poisoned': False,
                'score': 0.0,
                'confidence': 0.0,
                'stability_scores': [],
                'threshold': self.threshold or 0.5,
                'token_importance': []
            }

        n_erase = max(1, min(int(n_tokens * self.erase_ratio), n_tokens - 1))

        # 计算token重要性（用于调试）
        token_importance = self.compute_token_importance(
            text, demonstrations, label_words)

        # 多次贪心擦除
        stability_scores = []
        erased_texts = []
        position_impact = []

        already_erased = []

        for iteration in range(self.n_iterations):
            # 选择要擦除的token
            erase_positions = self.select_tokens_to_erase(
                text, n_erase, demonstrations, label_words, already_erased)

            if not erase_positions:
                break

            # 记录已擦除的token（避免重复）
            already_erased.extend(erase_positions)

            # 执行擦除
            erased_text = self.erase_tokens(text, erase_positions)

            if not erased_text.strip():
                continue

            try:
                if label_words:
                    erased_pred = self.compute_prediction_with_label_words(
                        erased_text, label_words, demonstrations)
                else:
                    erased_pred = self.compute_prediction_distribution(
                        erased_text, demonstrations)

                distance = self.compute_distribution_distance(
                    original_pred, erased_pred, metric='js')
                stability_scores.append(distance)

                if return_debug_info:
                    erased_texts.append(erased_text)
                    position_impact.append({
                        'iteration': iteration,
                        'erase_positions': erase_positions,
                        'erased_tokens': [tokens[p] for p in erase_positions if p < len(tokens)],
                        'importance_scores': [float(token_importance[p]) for p in erase_positions if p < len(token_importance)],
                        'distance': float(distance)
                    })

            except Exception as e:
                continue

        # 聚合分数
        if not stability_scores:
            result = {
                'is_poisoned': False,
                'score': 0.0,
                'confidence': 0.0,
                'stability_scores': [],
                'threshold': self.threshold or 0.5,
                'token_importance': token_importance.tolist() if return_debug_info else []
            }
            if return_debug_info:
                result['erased_texts'] = []
                result['position_impact'] = []
            return result

        # 使用max聚合（贪心擦除通常取最大影响）
        final_score = float(np.max(stability_scores))
        threshold = self.threshold if self.threshold is not None else 0.5
        is_poisoned = final_score > threshold
        confidence = min(abs(final_score - threshold) / (threshold + 1e-10), 1.0)

        result = {
            'is_poisoned': bool(is_poisoned),
            'score': final_score,
            'confidence': float(confidence),
            'stability_scores': [float(s) for s in stability_scores],
            'threshold': float(threshold),
            'token_importance': token_importance.tolist() if return_debug_info else []
        }

        if return_debug_info:
            result['erased_texts'] = erased_texts
            result['position_impact'] = position_impact

        return result
