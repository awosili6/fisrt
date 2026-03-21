"""
ONION检测器基线实现
"""

import torch
import numpy as np
from typing import List, Dict, Any
from ..base_detector import BaseDetector


class ONIONDetector(BaseDetector):
    """
    ONION (Qi et al.) 检测器实现

    核心思想：通过计算移除每个词后文本困惑度的变化，
    识别导致困惑度显著降低的词作为潜在的触发词。

    参考: ONION: A Simple and Effective Defense Against Textual Backdoor Attacks
    """

    def __init__(self, model, tokenizer, perplexity_threshold: float = None,
                 device: str = 'cuda'):
        """
        初始化ONION检测器

        Args:
            model: 预训练语言模型
            tokenizer: 分词器
            perplexity_threshold: 困惑度变化阈值
            device: 计算设备
        """
        super().__init__(model, tokenizer, device)
        self.perplexity_threshold = perplexity_threshold
        self.threshold = perplexity_threshold  # 兼容性：实验代码可能使用 threshold

    def compute_perplexity(self, text: str) -> float:
        """
        计算文本的困惑度

        困惑度 = exp(平均交叉熵损失)

        Args:
            text: 输入文本

        Returns:
            困惑度值
        """
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss

        perplexity = torch.exp(loss).item()
        return perplexity

    def compute_word_scores(self, text: str) -> List[Dict[str, Any]]:
        """
        计算每个词的重要性分数

        分数 = 移除该词后的困惑度 - 原始困惑度
        分数越小（负得越多），该词越可疑

        Args:
            text: 输入文本

        Returns:
            每个词的分数信息列表
        """
        words = text.split()

        if not words:
            return []

        # 计算原始困惑度
        try:
            original_ppl = self.compute_perplexity(text)
        except Exception as e:
            return []

        word_scores = []

        for i, word in enumerate(words):
            # 移除第i个词
            removed_words = words[:i] + words[i+1:]
            removed_text = ' '.join(removed_words)

            if not removed_text.strip():
                continue

            try:
                removed_ppl = self.compute_perplexity(removed_text)

                # 困惑度下降越大，该词越可疑
                score = removed_ppl - original_ppl

                word_scores.append({
                    'index': i,
                    'word': word,
                    'score': score,
                    'original_ppl': original_ppl,
                    'removed_ppl': removed_ppl
                })
            except Exception as e:
                continue

        return word_scores

    def detect(self, text: str, demonstrations: List[Dict] = None) -> Dict[str, Any]:
        """
        使用ONION方法检测投毒样本

        Args:
            text: 待检测文本
            demonstrations: ICL演示示例（ONION不使用）

        Returns:
            检测结果字典
        """
        word_scores = self.compute_word_scores(text)

        if not word_scores:
            return {
                'is_poisoned': False,
                'score': 0.0,
                'confidence': 0.0,
                'suspicious_word': None
            }

        # 找出使困惑度下降最多的词（最负的分数）
        min_score_info = min(word_scores, key=lambda x: x['score'])
        min_score = min_score_info['score']

        # 判断是否为投毒样本
        # 如果存在词使得困惑度显著下降，可能是触发词
        threshold = self.perplexity_threshold if self.perplexity_threshold is not None else -1.0
        is_poisoned = min_score < -threshold if threshold > 0 else min_score < -1.0

        # 计算置信度
        confidence = min(abs(min_score) / (abs(threshold) + 1e-10), 1.0) if threshold else 0.5

        return {
            'is_poisoned': is_poisoned,
            'score': float(min_score),
            'confidence': float(confidence),
            'suspicious_word': min_score_info['word'] if is_poisoned else None,
            'suspicious_index': min_score_info['index'] if is_poisoned else None,
            'word_scores': word_scores,
            'threshold': threshold
        }

    def detect_and_remove_trigger(self, text: str) -> Dict[str, Any]:
        """
        检测并移除可疑的触发词

        Args:
            text: 输入文本

        Returns:
            包含清理后文本的结果
        """
        result = self.detect(text)

        if result['is_poisoned'] and result['suspicious_index'] is not None:
            words = text.split()
            suspicious_idx = result['suspicious_index']

            # 移除可疑词
            cleaned_words = words[:suspicious_idx] + words[suspicious_idx+1:]
            cleaned_text = ' '.join(cleaned_words)

            result['cleaned_text'] = cleaned_text
            result['original_text'] = text

            # 验证清理效果（可选）
            try:
                original_ppl = self.compute_perplexity(text)
                cleaned_ppl = self.compute_perplexity(cleaned_text)
                result['perplexity_reduction'] = original_ppl - cleaned_ppl
            except:
                result['perplexity_reduction'] = 0.0

        return result

    def fit_threshold(self, clean_texts: List[str], poison_texts: List[str],
                      demonstrations: List[Dict] = None) -> float:
        """
        根据验证集拟合最优阈值

        Args:
            clean_texts: 干净文本列表
            poison_texts: 投毒文本列表

        Returns:
            最优阈值
        """
        clean_scores = []
        poison_scores = []

        # 计算分数
        for text in clean_texts:
            word_scores = self.compute_word_scores(text)
            if word_scores:
                min_score = min(ws['score'] for ws in word_scores)
                clean_scores.append(min_score)

        for text in poison_texts:
            word_scores = self.compute_word_scores(text)
            if word_scores:
                min_score = min(ws['score'] for ws in word_scores)
                poison_scores.append(min_score)

        # 寻找最优阈值
        all_scores = clean_scores + poison_scores
        all_labels = [0] * len(clean_scores) + [1] * len(poison_scores)

        from sklearn.metrics import f1_score

        best_threshold = -1.0
        best_f1 = 0

        for threshold_candidate in np.linspace(min(all_scores), max(all_scores), 100):
            predictions = [1 if s < threshold_candidate else 0 for s in all_scores]
            f1 = f1_score(all_labels, predictions, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold_candidate

        self.perplexity_threshold = abs(best_threshold)
        self.threshold = self.perplexity_threshold  # 兼容性
        return self.perplexity_threshold
