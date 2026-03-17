"""
评估指标计算
"""

import time
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, roc_curve
)


class Evaluator:
    """
    评估器

    提供攻击效果和检测效果的评估指标
    """

    @staticmethod
    def compute_classification_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
        """
        计算分类指标

        Args:
            y_true: 真实标签
            y_pred: 预测标签

        Returns:
            指标字典
        """
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )

        cm = confusion_matrix(y_true, y_pred)

        # 处理边界情况
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0
            if len(y_true) > 0:
                # 手动计算
                tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
                fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
                tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
                fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # 误检率
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # 漏检率
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # 召回率

        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'fpr': float(fpr),
            'fnr': float(fnr),
            'tpr': float(tpr),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'confusion_matrix': cm.tolist()
        }

    @staticmethod
    def compute_attack_metrics(clean_labels: List[int], clean_preds: List[int],
                               poison_labels: List[int], poison_preds: List[int]) -> Dict[str, float]:
        """
        计算攻击效果指标

        Args:
            clean_labels: 干净样本真实标签
            clean_preds: 干净样本预测标签
            poison_labels: 投毒样本真实标签
            poison_preds: 投毒样本预测标签

        Returns:
            攻击效果指标
        """
        cacc = accuracy_score(clean_labels, clean_preds)  # 干净准确率
        asr = accuracy_score(poison_labels, poison_preds)  # 攻击成功率

        # 保真度 = CACC / (ASR + epsilon)
        fidelity = cacc / (asr + 1e-10)

        return {
            'CACC': float(cacc),
            'ASR': float(asr),
            'fidelity': float(fidelity)
        }

    @staticmethod
    def evaluate_detector(detector, test_texts: List[str],
                          true_labels: List[int],
                          demonstrations: List[Dict] = None) -> Dict[str, Any]:
        """
        评估检测器性能

        Args:
            detector: 检测器实例
            test_texts: 测试文本列表
            true_labels: 真实标签（1=投毒，0=干净）
            demonstrations: ICL演示示例

        Returns:
            评估指标和时间性能
        """
        predictions = []
        scores = []
        latencies = []

        for text in test_texts:
            start_time = time.time()
            result = detector.detect(text, demonstrations)
            latency = time.time() - start_time

            predictions.append(1 if result['is_poisoned'] else 0)
            scores.append(result.get('score', 0.0))
            latencies.append(latency)

        # 计算分类指标
        metrics = Evaluator.compute_classification_metrics(true_labels, predictions)

        # 添加时间性能指标
        metrics['avg_latency_ms'] = np.mean(latencies) * 1000
        metrics['std_latency_ms'] = np.std(latencies) * 1000
        metrics['max_latency_ms'] = np.max(latencies) * 1000
        metrics['min_latency_ms'] = np.min(latencies) * 1000
        metrics['throughput'] = 1.0 / np.mean(latencies) if np.mean(latencies) > 0 else 0
        metrics['total_samples'] = len(test_texts)

        # 计算AUC（如果分数有意义）
        try:
            if len(set(true_labels)) > 1 and len(set(scores)) > 1:
                metrics['auc'] = float(roc_auc_score(true_labels, scores))
            else:
                metrics['auc'] = 0.0
        except:
            metrics['auc'] = 0.0

        return metrics

    @staticmethod
    def compare_detectors(detector_results: Dict[str, Dict]) -> Dict[str, Any]:
        """
        对比多个检测器的性能

        Args:
            detector_results: {检测器名称: 评估结果} 的字典

        Returns:
            对比结果
        """
        comparison = {
            'detectors': list(detector_results.keys()),
            'f1_scores': {name: result['f1_score'] for name, result in detector_results.items()},
            'accuracies': {name: result['accuracy'] for name, result in detector_results.items()},
            'latencies': {name: result['avg_latency_ms'] for name, result in detector_results.items()},
            'fprs': {name: result['fpr'] for name, result in detector_results.items()}
        }

        # 找出最佳检测器
        comparison['best_f1'] = max(comparison['f1_scores'], key=comparison['f1_scores'].get)
        comparison['best_accuracy'] = max(comparison['accuracies'], key=comparison['accuracies'].get)
        comparison['fastest'] = min(comparison['latencies'], key=comparison['latencies'].get)

        return comparison

    @staticmethod
    def find_best_threshold(scores: List[float], labels: List[int],
                            metric: str = 'f1') -> Tuple[float, float]:
        """
        寻找最优阈值

        Args:
            scores: 异常分数列表
            labels: 真实标签
            metric: 优化目标指标

        Returns:
            (最优阈值, 最优指标值)
        """
        if len(set(labels)) < 2:
            return 0.5, 0.0

        # 尝试不同阈值
        thresholds = np.linspace(min(scores), max(scores), 100)
        best_threshold = thresholds[0]
        best_score = 0.0

        for threshold in thresholds:
            predictions = [1 if s > threshold else 0 for s in scores]

            if metric == 'f1':
                _, _, f1, _ = precision_recall_fscore_support(
                    labels, predictions, average='binary', zero_division=0
                )
                score = f1
            elif metric == 'accuracy':
                score = accuracy_score(labels, predictions)
            elif metric == 'precision':
                p, _, _, _ = precision_recall_fscore_support(
                    labels, predictions, average='binary', zero_division=0
                )
                score = p
            elif metric == 'recall':
                _, r, _, _ = precision_recall_fscore_support(
                    labels, predictions, average='binary', zero_division=0
                )
                score = r
            else:
                _, _, f1, _ = precision_recall_fscore_support(
                    labels, predictions, average='binary', zero_division=0
                )
                score = f1

            if score > best_score:
                best_score = score
                best_threshold = threshold

        return float(best_threshold), float(best_score)
