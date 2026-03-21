"""
基于提示擦除的投毒检测器
核心实现
"""

import random
import numpy as np
import torch
from typing import List, Dict, Any, Optional
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
        seed: 随机种子（用于实验可重现）
    """

    def __init__(self, model, tokenizer, erase_ratio: float = 0.3,
                 n_iterations: int = 10, aggregation: str = 'max',
                 device: str = 'cuda', seed: Optional[int] = None,
                 model_name: str = None):
        """
        初始化擦除检测器

        Args:
            model: 预训练语言模型
            tokenizer: 分词器
            erase_ratio: 擦除比例，范围(0, 1)
            n_iterations: 随机擦除迭代次数
            aggregation: 分数聚合方法
            device: 计算设备
            seed: 随机种子，为None时不固定随机性
            model_name: 模型名称，用于ChatML格式化
        """
        super().__init__(model, tokenizer, device, model_name)
        self.erase_ratio = erase_ratio
        self.n_iterations = n_iterations
        self.aggregation = aggregation
        self.threshold = None  # 将在fit_threshold或fit_threshold_from_clean中设置
        self.seed = seed
        if seed is not None:
            self.set_seed(seed)

    # ------------------------------------------------------------------
    # 随机种子控制
    # ------------------------------------------------------------------

    def set_seed(self, seed: int) -> None:
        """
        设置随机种子，确保实验可重现

        Args:
            seed: 随机种子值
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # ------------------------------------------------------------------
    # Token 擦除
    # ------------------------------------------------------------------

    def erase_tokens(self, text: str, erase_positions: List[int]) -> str:
        """
        擦除指定位置的token，兼容BPE子词token（如Ġ开头的token）

        Args:
            text: 原始文本
            erase_positions: 要擦除的token位置索引

        Returns:
            擦除后的文本
        """
        tokens = self.tokenizer.tokenize(text)

        if not tokens:
            return text

        # 最小token数检查：至少保留1个token
        n_keep = len(tokens) - len(erase_positions)
        if n_keep < 1:
            # 擦除过多时，随机保留一个token
            keep_idx = random.choice([i for i in range(len(tokens))
                                      if i not in erase_positions]
                                     or [0])
            erased_tokens = [tokens[keep_idx]]
        else:
            erased_tokens = [t for i, t in enumerate(tokens)
                             if i not in erase_positions]

        if not erased_tokens:
            return self.tokenizer.pad_token or ""

        # 处理BPE前缀：确保第一个token不以连接前缀（Ġ / ##）开头，
        # 否则convert_tokens_to_string可能产生多余空格或解码错误。
        # 直接交由tokenizer处理，它内部会正确拼接。
        try:
            return self.tokenizer.convert_tokens_to_string(erased_tokens)
        except Exception:
            # 降级：用空格拼接后去除BPE前缀符号
            raw = " ".join(erased_tokens)
            for prefix in ("Ġ", "##", "▁"):
                raw = raw.replace(prefix, " ")
            return raw.strip()

    # ------------------------------------------------------------------
    # 预测分布计算
    # ------------------------------------------------------------------

    def compute_prediction_distribution(self, text: str,
                                        demonstrations: List[Dict] = None
                                        ) -> torch.Tensor:
        """
        计算模型对给定文本的完整词汇表预测分布

        Args:
            text: 输入文本
            demonstrations: ICL演示示例

        Returns:
            预测概率分布（整个词汇表）
        """
        return self.compute_prediction(text, demonstrations)

    def compute_prediction_with_label_words(
            self,
            text: str,
            label_words: List[str],
            demonstrations: List[Dict] = None) -> torch.Tensor:
        """
        将模型输出限制在给定标签词集合上，返回归一化后的标签概率向量。

        ICL分类场景下，只关心标签词（如 "positive"/"negative" 或
        "sports"/"business"）对应的logit，其余词汇的概率不参与分析。

        Args:
            text: 输入文本
            label_words: 候选标签词列表，例如 ["positive", "negative"]
            demonstrations: ICL演示示例

        Returns:
            shape=(len(label_words),) 的归一化概率张量
        """
        # 获取完整分布
        full_probs = self.compute_prediction(text, demonstrations)  # (vocab_size,)

        # 将每个标签词映射到词汇表id
        # 注意：对于 Qwen 等模型，标签词在生成时通常带有前导空格
        label_ids = []
        for word in label_words:
            # 先尝试带空格（更符合生成时的实际情况）
            token_ids_space = self.tokenizer.encode(" " + word, add_special_tokens=False)
            token_ids_plain = self.tokenizer.encode(word, add_special_tokens=False)

            # 选择有效的 token ID（优先带空格的版本）
            if token_ids_space:
                label_ids.append(token_ids_space[0])
            elif token_ids_plain:
                label_ids.append(token_ids_plain[0])
            else:
                label_ids.append(0)

        # 取出对应位置的概率
        label_probs = torch.stack([full_probs[idx] for idx in label_ids])

        # 归一化，使其和为1
        label_probs = label_probs / (label_probs.sum() + 1e-10)
        return label_probs

    # ------------------------------------------------------------------
    # 分布距离计算
    # ------------------------------------------------------------------

    def compute_distribution_distance(self, pred1: torch.Tensor,
                                       pred2: torch.Tensor,
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
            # JS散度（增加数值稳定性）
            # 确保概率分布有效
            pred1 = torch.clamp(pred1, min=eps, max=1.0)
            pred2 = torch.clamp(pred2, min=eps, max=1.0)

            m = 0.5 * (pred1 + pred2)
            # 避免 log(0) 导致 NaN
            kl1 = torch.sum(pred1 * torch.log((pred1 + eps) / (m + eps)))
            kl2 = torch.sum(pred2 * torch.log((pred2 + eps) / (m + eps)))
            js = 0.5 * (kl1 + kl2)
            # 确保返回值有效
            result = js.item()
            if np.isnan(result) or np.isinf(result):
                return 0.0
            return result

        elif metric == 'cosine':
            # 余弦距离 = 1 - 余弦相似度
            cos_sim = torch.nn.functional.cosine_similarity(
                pred1.unsqueeze(0), pred2.unsqueeze(0)
            )
            return (1 - cos_sim).item()

        else:
            raise ValueError(f"Unknown metric: {metric}")

    # ------------------------------------------------------------------
    # 核心检测
    # ------------------------------------------------------------------

    def detect(self, text: str, demonstrations: List[Dict] = None,
               label_words: List[str] = None,
               return_debug_info: bool = False) -> Dict[str, Any]:
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
            label_words: 标签词列表；提供时使用label-restricted分布，
                         否则使用完整词汇表分布
            return_debug_info: 为True时结果中附加 erased_texts 和
                               position_impact，用于调试分析

        Returns:
            检测结果字典，包含：
            - is_poisoned: bool
            - score: float，异常分数（越高越可能是投毒样本）
            - confidence: float
            - stability_scores: List[float]
            - threshold: float
            - erased_texts: List[str]（仅当 return_debug_info=True）
            - position_impact: List[Dict]（仅当 return_debug_info=True）
        """
        # 重置随机种子，确保相同seed的多次detect()产生相同的erase_positions
        if self.seed is not None:
            self.set_seed(self.seed)

        # 1. 获取原始预测
        if label_words:
            original_pred = self.compute_prediction_with_label_words(
                text, label_words, demonstrations)
        else:
            original_pred = self.compute_prediction_distribution(
                text, demonstrations)

        # 2. 获取token数量
        tokens = self.tokenizer.tokenize(text)
        n_tokens = len(tokens)

        if n_tokens <= 1:
            # 单token或空文本：擦除后无内容，无法进行有效检测
            return {
                'is_poisoned': False,
                'score': 0.0,
                'confidence': 0.0,
                'stability_scores': [],
                'threshold': self.threshold or 0.5
            }

        # 计算每次擦除的token数量，至少擦1个，最多保留1个
        n_erase = max(1, min(int(n_tokens * self.erase_ratio), n_tokens - 1))

        # 3. 多次随机擦除并计算预测变化
        stability_scores = []
        erased_texts = []       # 调试用
        position_impact = []    # 调试用

        for iteration in range(self.n_iterations):
            # 随机选择擦除位置（不重复）
            erase_positions = random.sample(range(n_tokens), n_erase)

            # 擦除token
            erased_text = self.erase_tokens(text, erase_positions)

            # 如果擦除后文本为空，跳过
            if not erased_text.strip():
                continue

            # 计算擦除后的预测
            try:
                if label_words:
                    erased_pred = self.compute_prediction_with_label_words(
                        erased_text, label_words, demonstrations)
                else:
                    erased_pred = self.compute_prediction_distribution(
                        erased_text, demonstrations)

                # 计算预测分布之间的距离（使用JS散度：有界且对称）
                distance = self.compute_distribution_distance(
                    original_pred, erased_pred, metric='js')
                stability_scores.append(distance)

                if return_debug_info:
                    erased_texts.append(erased_text)
                    position_impact.append({
                        'iteration': iteration,
                        'erase_positions': erase_positions,
                        'erased_tokens': [tokens[p] for p in erase_positions],
                        'distance': float(distance)
                    })
            except (RuntimeError, ValueError) as e:
                # 仅跳过模型推理/数值计算的可恢复错误，其他异常继续上抛
                import warnings
                warnings.warn(f"Iteration {iteration} skipped due to error: {e}")
                continue

        # 4. 聚合分数
        if not stability_scores:
            result = {
                'is_poisoned': False,
                'score': 0.0,
                'confidence': 0.0,
                'stability_scores': [],
                'threshold': self.threshold or 0.5
            }
            if return_debug_info:
                result['erased_texts'] = []
                result['position_impact'] = []
            return result

        # 距离越大，说明模型输出变化越大，越可能是投毒样本
        if self.aggregation == 'mean':
            final_score = float(np.mean(stability_scores))
        elif self.aggregation == 'median':
            final_score = float(np.median(stability_scores))
        elif self.aggregation == 'min':
            final_score = float(np.min(stability_scores))
        elif self.aggregation == 'max':
            final_score = float(np.max(stability_scores))
        else:
            final_score = float(np.mean(stability_scores))

        # 5. 判断是否投毒
        threshold = self.threshold if self.threshold is not None else 0.5
        is_poisoned = final_score > threshold

        # 6. 计算置信度（基于分数与阈值的距离）
        confidence = min(abs(final_score - threshold) / (threshold + 1e-10), 1.0)

        result = {
            'is_poisoned': bool(is_poisoned),
            'score': final_score,
            'confidence': float(confidence),
            'stability_scores': [float(s) for s in stability_scores],
            'threshold': float(threshold)
        }

        if return_debug_info:
            result['erased_texts'] = erased_texts
            result['position_impact'] = position_impact

        return result

    # ------------------------------------------------------------------
    # 批次处理优化
    # ------------------------------------------------------------------

    def batch_detect_optimized(self, texts: List[str],
                                demonstrations: List[Dict] = None,
                                label_words: List[str] = None,
                                batch_size: int = 8) -> List[Dict[str, Any]]:
        """
        批次检测优化版本：对每个待检测文本生成所有擦除变体后，
        合并成一个大批次送入模型，减少模型调用次数。

        Args:
            texts: 待检测文本列表
            demonstrations: ICL演示示例
            label_words: 标签词列表（同 detect）
            batch_size: 单次模型推理的最大序列数

        Returns:
            与 texts 顺序对应的检测结果列表
        """
        import torch.nn.functional as F

        # 构建 ICL prompt 前缀（与基类 compute_prediction 保持一致）
        def build_prompt(text: str) -> str:
            if demonstrations:
                parts = []
                for demo in demonstrations:
                    parts.append(f"Text: {demo['text']}\nLabel: {demo['label']}")
                parts.append(f"Text: {text}\nLabel:")
                return '\n\n'.join(parts)
            return text

        def get_label_ids() -> List[int]:
            ids = []
            for word in (label_words or []):
                token_ids = self.tokenizer.encode(
                    " " + word, add_special_tokens=False)
                if not token_ids:
                    token_ids = self.tokenizer.encode(
                        word, add_special_tokens=False)
                ids.append(token_ids[0] if token_ids else 0)
            return ids

        def probs_from_logits(logits: torch.Tensor) -> torch.Tensor:
            """logits shape: (batch, vocab) -> prob shape: (batch, vocab or n_labels)"""
            probs = torch.softmax(logits, dim=-1)
            if label_words:
                label_ids = get_label_ids()
                label_probs = probs[:, label_ids]
                label_probs = label_probs / (label_probs.sum(dim=-1, keepdim=True) + 1e-10)
                return label_probs
            return probs

        def run_batch_inference(prompts: List[str]) -> torch.Tensor:
            """对一批prompt做推理，返回最后token的概率矩阵 (N, dim)"""
            inputs = self.tokenizer(
                prompts,
                return_tensors='pt',
                padding=True,
                truncation=True
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                # 取每条序列最后一个（非pad）token的logits
                # attention_mask: (batch, seq_len)
                seq_lens = inputs['attention_mask'].sum(dim=1) - 1  # 最后有效位
                logits = outputs.logits[
                    torch.arange(len(prompts), device=self.device), seq_lens, :]
            return probs_from_logits(logits)  # (N, dim)

        results = []
        threshold = self.threshold if self.threshold is not None else 0.5

        for text in texts:
            tokens = self.tokenizer.tokenize(text)
            n_tokens = len(tokens)

            if n_tokens == 0:
                results.append({
                    'is_poisoned': False,
                    'score': 0.0,
                    'confidence': 0.0,
                    'stability_scores': [],
                    'threshold': float(threshold)
                })
                continue

            n_erase = max(1, min(int(n_tokens * self.erase_ratio), n_tokens - 1))

            # 生成所有擦除变体的 prompt
            original_prompt = build_prompt(text)
            variant_prompts = [original_prompt]
            valid_iterations = []

            for iteration in range(self.n_iterations):
                erase_positions = random.sample(range(n_tokens), n_erase)
                erased_text = self.erase_tokens(text, erase_positions)
                if erased_text.strip():
                    variant_prompts.append(build_prompt(erased_text))
                    valid_iterations.append(iteration)

            # 分批推理
            all_probs = []
            for start in range(0, len(variant_prompts), batch_size):
                batch = variant_prompts[start:start + batch_size]
                try:
                    batch_probs = run_batch_inference(batch)
                    all_probs.append(batch_probs)
                except Exception:
                    # 单条降级处理
                    for p in batch:
                        try:
                            sp = run_batch_inference([p])
                            all_probs.append(sp)
                        except Exception:
                            pass

            if not all_probs:
                results.append({
                    'is_poisoned': False,
                    'score': 0.0,
                    'confidence': 0.0,
                    'stability_scores': [],
                    'threshold': float(threshold)
                })
                continue

            all_probs_tensor = torch.cat(all_probs, dim=0)  # (1+valid_iters, dim)
            original_prob = all_probs_tensor[0]              # (dim,)
            variant_probs = all_probs_tensor[1:]             # (valid_iters, dim)

            stability_scores = []
            eps = 1e-10
            for vp in variant_probs:
                # JS散度（有界、对称），与 detect() 保持一致
                # 确保概率分布有效
                op = torch.clamp(original_prob, min=eps, max=1.0)
                vp_clamped = torch.clamp(vp, min=eps, max=1.0)
                m = 0.5 * (op + vp_clamped)
                kl1 = torch.sum(op * torch.log((op + eps) / (m + eps)))
                kl2 = torch.sum(vp_clamped * torch.log((vp_clamped + eps) / (m + eps)))
                js = (0.5 * (kl1 + kl2)).item()
                # 处理无效值
                if np.isnan(js) or np.isinf(js):
                    js = 0.0
                stability_scores.append(float(js))

            if not stability_scores:
                final_score = 0.0
            elif self.aggregation == 'mean':
                final_score = float(np.mean(stability_scores))
            elif self.aggregation == 'median':
                final_score = float(np.median(stability_scores))
            elif self.aggregation == 'min':
                final_score = float(np.min(stability_scores))
            elif self.aggregation == 'max':
                final_score = float(np.max(stability_scores))
            else:
                final_score = float(np.mean(stability_scores))

            is_poisoned = final_score > threshold
            confidence = min(abs(final_score - threshold) / (threshold + 1e-10), 1.0)

            results.append({
                'is_poisoned': bool(is_poisoned),
                'score': final_score,
                'confidence': float(confidence),
                'stability_scores': stability_scores,
                'threshold': float(threshold)
            })

        return results

    # ------------------------------------------------------------------
    # 实用分析方法
    # ------------------------------------------------------------------

    def get_sensitivity_analysis(self, text: str,
                                  demonstrations: List[Dict] = None,
                                  label_words: List[str] = None,
                                  erase_ratios: List[float] = None
                                  ) -> Dict[str, Any]:
        """
        分析不同擦除比例下的异常分数变化，用于灵敏度研究。

        Args:
            text: 待分析文本
            demonstrations: ICL演示示例
            label_words: 标签词列表
            erase_ratios: 要测试的擦除比例列表；默认为
                          [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

        Returns:
            包含各比例检测结果的字典：
            {
                'ratios': [...],
                'scores': [...],
                'is_poisoned': [...],
                'summary': {...}
            }
        """
        if erase_ratios is None:
            erase_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

        original_ratio = self.erase_ratio
        ratios_out, scores_out, poisoned_out = [], [], []

        for ratio in erase_ratios:
            self.erase_ratio = ratio
            result = self.detect(text, demonstrations, label_words=label_words)
            ratios_out.append(ratio)
            scores_out.append(result['score'])
            poisoned_out.append(result['is_poisoned'])

        # 恢复原始比例
        self.erase_ratio = original_ratio

        return {
            'ratios': ratios_out,
            'scores': scores_out,
            'is_poisoned': poisoned_out,
            'summary': {
                'max_score': float(max(scores_out)) if scores_out else 0.0,
                'min_score': float(min(scores_out)) if scores_out else 0.0,
                'mean_score': float(np.mean(scores_out)) if scores_out else 0.0,
                'n_poisoned_ratios': sum(poisoned_out)
            }
        }

    def detect_with_ensemble(self, text: str,
                              demonstrations: List[Dict] = None,
                              label_words: List[str] = None,
                              n_ensemble: int = 5) -> Dict[str, Any]:
        """
        多次独立检测取平均，提升结果稳定性。

        每次检测使用不同的随机种子，最终对所有检测分数求均值，
        适用于迭代次数较少时降低方差。

        Args:
            text: 待检测文本
            demonstrations: ICL演示示例
            label_words: 标签词列表
            n_ensemble: 集成检测次数

        Returns:
            集成后的检测结果字典（含各轮原始分数）
        """
        ensemble_scores = []
        all_stability = []

        for i in range(n_ensemble):
            # 每轮使用不同种子（若设置了基础种子则可重现）
            round_seed = (self.seed * 1000 + i) if self.seed is not None else None
            if round_seed is not None:
                random.seed(round_seed)
                np.random.seed(round_seed)

            result = self.detect(text, demonstrations, label_words=label_words)
            ensemble_scores.append(result['score'])
            all_stability.extend(result.get('stability_scores', []))

        # 恢复主种子
        if self.seed is not None:
            self.set_seed(self.seed)

        final_score = float(np.mean(ensemble_scores))
        threshold = self.threshold if self.threshold is not None else 0.5
        is_poisoned = final_score > threshold
        confidence = min(abs(final_score - threshold) / (threshold + 1e-10), 1.0)

        return {
            'is_poisoned': bool(is_poisoned),
            'score': final_score,
            'confidence': float(confidence),
            'stability_scores': [float(s) for s in all_stability],
            'threshold': float(threshold),
            'ensemble_scores': [float(s) for s in ensemble_scores],
            'ensemble_std': float(np.std(ensemble_scores))
        }

    # ------------------------------------------------------------------
    # 阈值拟合
    # ------------------------------------------------------------------

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
        from sklearn.metrics import (f1_score, accuracy_score,
                                     precision_score, recall_score)

        # 过滤掉 NaN/inf 值，防止阈值计算崩溃
        clean_scores = [s for s in clean_scores if np.isfinite(s)]
        poison_scores = [s for s in poison_scores if np.isfinite(s)]

        if not clean_scores or not poison_scores:
            # 没有有效分数，使用默认阈值
            self.threshold = 0.5
            return 0.5

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

        # 分布完全重叠时best_score==0，或clean/poison分数颠倒时
        # 使用统计方法：阈值 = clean均值 + 0.5 * (poison均值 - clean均值)
        clean_mean = np.mean(clean_scores)
        poison_mean = np.mean(poison_scores)

        if best_score == 0 or best_score < 0.3:
            # 分数分布不理想，使用两分布中间点
            if poison_mean > clean_mean:
                # 正常情况：poison分数更高
                best_threshold = (clean_mean + poison_mean) / 2.0
            else:
                # 异常情况：clean分数更高（如InsertSent的随机擦除）
                # 使用clean均值 + 0.5标准差作为阈值
                best_threshold = clean_mean + 0.5 * np.std(clean_scores)
            best_threshold = float(best_threshold) if np.isfinite(best_threshold) else 0.5

        self.threshold = best_threshold
        return best_threshold

    def fit_threshold_from_clean(self, clean_scores: List[float],
                                  k: float = 3.0) -> float:
        """
        仅凭干净样本的分数分布估计检测阈值（无需投毒样本标签）。

        使用均值 + k·标准差作为阈值：分数超过此值的样本被认为异常。
        k 越大，误报率越低，但漏报率越高；通常 k=3 是合理起点。

        Args:
            clean_scores: 干净样本的异常分数列表
            k: 标准差倍数，默认3.0

        Returns:
            估计的阈值
        """
        if not clean_scores:
            raise ValueError("clean_scores 不能为空")
        mu = float(np.mean(clean_scores))
        sigma = float(np.std(clean_scores))
        threshold = mu + k * sigma
        self.threshold = threshold
        return threshold

    # ------------------------------------------------------------------
    # 指定位置检测（供子类/调试使用）
    # ------------------------------------------------------------------

    def detect_with_positions(self, text: str, demonstrations: List[Dict] = None,
                               erase_positions: List[int] = None,
                               label_words: List[str] = None) -> Dict[str, Any]:
        """
        使用指定擦除位置进行单次检测（供子类或调试使用）

        Args:
            text: 待检测文本
            demonstrations: ICL演示示例
            erase_positions: 指定的擦除位置
            label_words: 标签词列表

        Returns:
            检测结果
        """
        if erase_positions is None:
            return self.detect(text, demonstrations, label_words=label_words)

        # 单次检测逻辑
        if label_words:
            original_pred = self.compute_prediction_with_label_words(
                text, label_words, demonstrations)
        else:
            original_pred = self.compute_prediction_distribution(
                text, demonstrations)

        erased_text = self.erase_tokens(text, erase_positions)

        if not erased_text.strip():
            return {'is_poisoned': False, 'score': 0.0, 'confidence': 0.0,
                    'threshold': float(self.threshold or 0.5)}

        if label_words:
            erased_pred = self.compute_prediction_with_label_words(
                erased_text, label_words, demonstrations)
        else:
            erased_pred = self.compute_prediction_distribution(
                erased_text, demonstrations)

        distance = self.compute_distribution_distance(
            original_pred, erased_pred, metric='js')

        threshold = self.threshold or 0.5
        is_poisoned = distance > threshold
        confidence = min(abs(distance - threshold) / (threshold + 1e-10), 1.0)

        return {
            'is_poisoned': bool(is_poisoned),
            'score': float(distance),
            'confidence': float(confidence),
            'threshold': float(threshold)
        }
