"""
基于Attention权重的擦除检测器
使用模型内部的attention权重指导擦除策略
"""

import torch
import numpy as np
from typing import List, Dict, Any
from .prompt_eraser import PromptEraserDetector


class AttentionEraserDetector(PromptEraserDetector):
    """
    基于Attention权重的擦除检测器

    核心思想：利用模型内部的attention权重识别输入中"不重要"的token，
    优先擦除这些低权重token。如果擦除低权重token后预测发生显著变化，
    说明模型可能依赖隐藏的触发模式。

    Attributes:
        attention_layer: 使用哪一层的attention权重（-1表示最后一层）
        selection_strategy: token选择策略 ('lowest', 'highest', 'random_attention')
    """

    def __init__(self, model, tokenizer, erase_ratio: float = 0.3,
                 n_iterations: int = 10, attention_layer: int = -1,
                 selection_strategy: str = 'lowest', device: str = 'cuda'):
        """
        初始化Attention擦除检测器

        Args:
            model: 预训练语言模型（需要支持output_attentions）
            tokenizer: 分词器
            erase_ratio: 擦除比例
            n_iterations: 迭代次数
            attention_layer: 使用的attention层索引
            selection_strategy: token选择策略
            device: 计算设备
        """
        super().__init__(model, tokenizer, erase_ratio, n_iterations, 'mean', device)
        self.attention_layer = attention_layer
        self.selection_strategy = selection_strategy

    def get_attention_weights(self, text: str) -> torch.Tensor:
        """
        获取输入各token的attention权重

        通过前向传播获取attention矩阵，然后计算每个token的平均attention权重。

        Args:
            text: 输入文本

        Returns:
            每个token的attention权重 [seq_len]
        """
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

            # attention是一个tuple，每层一个tensor
            # 每层attention shape: [batch, num_heads, seq_len, seq_len]
            attentions = outputs.attentions

            # 选择指定层的attention
            if self.attention_layer == -1:
                attention = attentions[-1]
            else:
                attention = attentions[self.attention_layer]

            # 计算每个token的平均attention权重
            # 对batch和head维度取平均，然后对查询位置取平均
            # shape: [batch, heads, seq, seq] -> [seq]
            avg_attention = attention.mean(dim=(0, 1, 2))  # 平均batch, head, query位置

        return avg_attention

    def select_erase_positions(self, text: str, n_erase: int) -> List[int]:
        """
        根据attention权重选择擦除位置

        Args:
            text: 输入文本
            n_erase: 需要擦除的token数量

        Returns:
            擦除位置索引列表
        """
        tokens = self.tokenizer.tokenize(text)
        n_tokens = len(tokens)

        if n_tokens == 0:
            return []

        if n_erase >= n_tokens:
            return list(range(n_tokens))

        # 获取attention权重
        attention_weights = self.get_attention_weights(text)

        # 去掉特殊token（CLS, SEP等）的权重
        # attention_weights包含特殊token，我们需要对齐
        if len(attention_weights) > n_tokens:
            # 去掉开头和结尾的特殊token
            offset = (len(attention_weights) - n_tokens) // 2
            attention_weights = attention_weights[offset:offset + n_tokens]

        attention_weights = attention_weights[:n_tokens]

        # 根据策略选择擦除位置
        if self.selection_strategy == 'lowest':
            # 擦除attention权重最低的token（模型认为不重要的token）
            _, positions = torch.topk(attention_weights, n_erase, largest=False)
            return positions.cpu().tolist()

        elif self.selection_strategy == 'highest':
            # 擦除attention权重最高的token（模型高度依赖的token）
            _, positions = torch.topk(attention_weights, n_erase, largest=True)
            return positions.cpu().tolist()

        elif self.selection_strategy == 'random_attention':
            # 按attention权重进行加权随机采样
            weights = attention_weights.cpu().numpy()

            # 反转权重用于采样（低权重token更可能被选中）
            inverted_weights = weights.max() - weights + 1e-10
            probabilities = inverted_weights / inverted_weights.sum()

            positions = np.random.choice(
                n_tokens, size=n_erase, replace=False, p=probabilities
            )
            return positions.tolist()

        else:
            raise ValueError(f"Unknown selection strategy: {self.selection_strategy}")

    def detect(self, text: str, demonstrations: List[Dict] = None) -> Dict[str, Any]:
        """
        基于attention权重指导的擦除检测

        Args:
            text: 待检测文本
            demonstrations: ICL演示示例

        Returns:
            检测结果字典
        """
        # 获取原始预测
        original_pred = self.compute_prediction_distribution(text, demonstrations)

        # 获取token数量
        tokens = self.tokenizer.tokenize(text)
        n_tokens = len(tokens)

        if n_tokens == 0:
            return {
                'is_poisoned': False,
                'score': 0.0,
                'confidence': 0.0,
                'stability_scores': [],
                'attention_weights': []
            }

        n_erase = max(1, int(n_tokens * self.erase_ratio))

        # 多次基于attention擦除并计算预测变化
        stability_scores = []

        for iteration in range(self.n_iterations):
            # 根据attention权重选择擦除位置
            erase_positions = self.select_erase_positions(text, n_erase)

            # 擦除token
            erased_text = self.erase_tokens(text, erase_positions)

            if not erased_text.strip():
                continue

            try:
                # 计算擦除后的预测
                erased_pred = self.compute_prediction_distribution(erased_text, demonstrations)

                # 计算分布距离
                distance = self.compute_distribution_distance(
                    original_pred, erased_pred, metric='kl'
                )
                stability_scores.append(distance)

            except Exception as e:
                continue

        # 聚合分数
        if not stability_scores:
            return {
                'is_poisoned': False,
                'score': 0.0,
                'confidence': 0.0,
                'stability_scores': [],
                'attention_weights': self.get_attention_weights(text).cpu().tolist()
            }

        # 聚合
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

        # 判断
        threshold = self.threshold if self.threshold is not None else 0.5
        is_poisoned = final_score > threshold
        confidence = min(abs(final_score - threshold) / (threshold + 1e-10), 1.0)

        return {
            'is_poisoned': is_poisoned,
            'score': float(final_score),
            'confidence': float(confidence),
            'stability_scores': [float(s) for s in stability_scores],
            'attention_weights': self.get_attention_weights(text).cpu().tolist()[:n_tokens],
            'threshold': threshold
        }

    def analyze_attention_pattern(self, text: str) -> Dict[str, Any]:
        """
        分析文本的attention模式（用于可视化和分析）

        Args:
            text: 输入文本

        Returns:
            attention分析结果
        """
        tokens = self.tokenizer.tokenize(text)
        attention_weights = self.get_attention_weights(text)

        # 对齐长度
        if len(attention_weights) > len(tokens):
            offset = (len(attention_weights) - len(tokens)) // 2
            attention_weights = attention_weights[offset:offset + len(tokens)]

        attention_weights = attention_weights[:len(tokens)]

        # 找出高attention和低attention的token
        weights_np = attention_weights.cpu().numpy()

        # 计算统计信息
        stats = {
            'mean': float(weights_np.mean()),
            'std': float(weights_np.std()),
            'max': float(weights_np.max()),
            'min': float(weights_np.min()),
            'top_tokens': [],
            'bottom_tokens': []
        }

        # 找出top-5和bottom-5
        top_indices = weights_np.argsort()[-5:][::-1]
        bottom_indices = weights_np.argsort()[:5]

        stats['top_tokens'] = [
            {'token': tokens[i], 'weight': float(weights_np[i])}
            for i in top_indices if i < len(tokens)
        ]
        stats['bottom_tokens'] = [
            {'token': tokens[i], 'weight': float(weights_np[i])}
            for i in bottom_indices if i < len(tokens)
        ]

        return stats
