"""
梯度优化擦除检测器
利用梯度信息指导擦除策略，找到对模型输出影响最大的token
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional
from .prompt_eraser import PromptEraserDetector


class GradientEraserDetector(PromptEraserDetector):
    """
    基于梯度优化的擦除检测器

    核心思想：利用输入嵌入的梯度来识别对模型输出影响最大的token。
    梯度大的token对预测结果更敏感，擦除后会引起更大的分布变化。

    这种方法比随机擦除和贪心擦除更高效，因为梯度直接反映了token的重要性。

    Attributes:
        erase_ratio: 每次擦除的token比例
        n_iterations: 迭代次数
        gradient_method: 梯度计算方法 ('input_embeds', 'attention')
    """

    def __init__(self, model, tokenizer, erase_ratio: float = 0.3,
                 n_iterations: int = 10, gradient_method: str = 'input_embeds',
                 device: str = 'cuda', seed: Optional[int] = None):
        """
        初始化梯度擦除检测器

        Args:
            model: 预训练语言模型
            tokenizer: 分词器
            erase_ratio: 擦除比例
            n_iterations: 迭代次数
            gradient_method: 梯度计算方法
            device: 计算设备
            seed: 随机种子
        """
        super().__init__(model, tokenizer, erase_ratio, n_iterations,
                        'max', device, seed)
        self.gradient_method = gradient_method

    def compute_gradient_importance(self, text: str,
                                    demonstrations: List[Dict] = None,
                                    label_words: List[str] = None) -> np.ndarray:
        """
        基于梯度计算token重要性

        Args:
            text: 输入文本
            demonstrations: ICL演示示例
            label_words: 标签词列表

        Returns:
            每个token的梯度重要性分数
        """
        # 构建完整prompt
        if demonstrations:
            prompt_parts = []
            for demo in demonstrations:
                prompt_parts.append(f"Text: {demo['text']}\nLabel: {demo['label']}")
            prompt_parts.append(f"Text: {text}\nLabel:")
            prompt = '\n\n'.join(prompt_parts)
        else:
            prompt = text

        # 编码输入
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        input_ids = inputs['input_ids']
        n_tokens = input_ids.shape[1]

        if n_tokens == 0:
            return np.array([])

        try:
            # 获取输入嵌入并启用梯度
            embeds = self.model.get_input_embeddings()(input_ids)
            embeds.requires_grad = True

            # 前向传播
            outputs = self.model(inputs_embeds=embeds)
            logits = outputs.logits[0, -1, :]  # 最后一个token的logits

            # 如果指定了标签词，只考虑这些词的梯度
            if label_words:
                label_ids = []
                for word in label_words:
                    token_ids = self.tokenizer.encode(word, add_special_tokens=False)
                    if token_ids:
                        label_ids.append(token_ids[0])

                if label_ids:
                    # 计算标签词之间的logit差异
                    target_logits = logits[label_ids]
                    # 最大化目标标签的logit
                    loss = -torch.max(target_logits)
                else:
                    loss = -torch.max(logits)
            else:
                # 使用预测分布的熵
                probs = torch.softmax(logits, dim=-1)
                loss = -torch.sum(probs * torch.log(probs + 1e-10))

            # 反向传播
            self.model.zero_grad()
            loss.backward()

            # 计算每个token的梯度范数
            if embeds.grad is not None:
                # 计算每个位置嵌入的L2范数
                grad_norms = torch.norm(embeds.grad[0], dim=1).detach().cpu().numpy()

                # 找到文本部分对应的token（排除demonstrations）
                text_tokens = self.tokenizer.tokenize(text)
                n_text_tokens = len(text_tokens)

                # 只返回文本部分的梯度（最后n_text_tokens个）
                if n_text_tokens <= len(grad_norms):
                    return grad_norms[-n_text_tokens:]
                else:
                    # 如果长度不匹配，返回全部梯度
                    return grad_norms
            else:
                return np.zeros(len(self.tokenizer.tokenize(text)))

        except Exception as e:
            # 梯度计算失败，回退到贪心方法
            return self.compute_token_importance_fallback(text, demonstrations, label_words)

    def compute_token_importance_fallback(self, text: str,
                                          demonstrations: List[Dict] = None,
                                          label_words: List[str] = None) -> np.ndarray:
        """
        梯度计算失败时的回退方法（使用贪心策略）
        """
        tokens = self.tokenizer.tokenize(text)
        n_tokens = len(tokens)

        if label_words:
            original_pred = self.compute_prediction_with_label_words(
                text, label_words, demonstrations)
        else:
            original_pred = self.compute_prediction_distribution(text, demonstrations)

        importance = np.zeros(n_tokens)
        for i in range(n_tokens):
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

                distance = self.compute_distribution_distance(
                    original_pred, erased_pred, metric='js')
                importance[i] = distance
            except:
                importance[i] = 0.0

        return importance

    def select_tokens_by_gradient(self, text: str, n_erase: int,
                                   demonstrations: List[Dict] = None,
                                   label_words: List[str] = None) -> List[int]:
        """
        基于梯度选择要擦除的token

        Args:
            text: 输入文本
            n_erase: 要擦除的token数量
            demonstrations: ICL演示示例
            label_words: 标签词列表

        Returns:
            要擦除的token位置列表
        """
        tokens = self.tokenizer.tokenize(text)
        n_tokens = len(tokens)

        if n_tokens == 0 or n_erase <= 0:
            return []

        # 计算梯度重要性
        grad_importance = self.compute_gradient_importance(
            text, demonstrations, label_words)

        # 确保长度匹配
        if len(grad_importance) != n_tokens:
            # 长度不匹配，使用简单截断或填充
            if len(grad_importance) > n_tokens:
                grad_importance = grad_importance[:n_tokens]
            else:
                grad_importance = np.pad(grad_importance,
                                        (0, n_tokens - len(grad_importance)))

        # 选择梯度最大的n_erase个token
        top_indices = np.argsort(grad_importance)[::-1][:n_erase]

        return [int(idx) for idx in top_indices if idx < n_tokens]

    def detect(self, text: str, demonstrations: List[Dict] = None,
               label_words: List[str] = None,
               return_debug_info: bool = False) -> Dict[str, Any]:
        """
        使用梯度优化擦除策略检测样本

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
                'gradient_scores': []
            }

        n_erase = max(1, min(int(n_tokens * self.erase_ratio), n_tokens - 1))

        # 计算梯度重要性（用于调试）
        gradient_scores = self.compute_gradient_importance(
            text, demonstrations, label_words)

        # 执行梯度指导的擦除
        stability_scores = []
        erased_texts = []
        position_impact = []

        for iteration in range(self.n_iterations):
            # 基于梯度选择token
            erase_positions = self.select_tokens_by_gradient(
                text, n_erase, demonstrations, label_words)

            if not erase_positions:
                break

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
                        'gradient_scores': [float(gradient_scores[p]) for p in erase_positions
                                          if p < len(gradient_scores)],
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
                'gradient_scores': gradient_scores.tolist() if return_debug_info else []
            }
            if return_debug_info:
                result['erased_texts'] = []
                result['position_impact'] = []
            return result

        # 使用max聚合
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
            'gradient_scores': gradient_scores.tolist() if return_debug_info else []
        }

        if return_debug_info:
            result['erased_texts'] = erased_texts
            result['position_impact'] = position_impact

        return result
