"""
可视化工具
用于绘制实验结果图表
"""

import os
from typing import List, Dict, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class Visualizer:
    """
    可视化工具类

    提供多种图表绘制功能
    """

    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray, labels: List[str] = None,
                              title: str = 'Confusion Matrix',
                              save_path: str = None):
        """
        绘制混淆矩阵热力图

        Args:
            cm: 混淆矩阵 (2x2)
            labels: 类别标签
            title: 图表标题
            save_path: 保存路径
        """
        if labels is None:
            labels = ['Clean', 'Poisoned']

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(title)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_roc_curve(y_true: List[int], scores: List[float],
                       title: str = 'ROC Curve',
                       save_path: str = None):
        """
        绘制ROC曲线

        Args:
            y_true: 真实标签
            scores: 预测分数
            title: 图表标题
            save_path: 保存路径
        """
        from sklearn.metrics import roc_curve, auc

        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_sensitivity_analysis(param_values: List, f1_scores: List,
                                  latencies: List, param_name: str = 'Parameter',
                                  title: str = None,
                                  save_path: str = None):
        """
        绘制参数敏感性分析图

        双Y轴图表：左侧显示F1分数，右侧显示延迟

        Args:
            param_values: 参数值列表
            f1_scores: F1分数列表
            latencies: 延迟列表
            param_name: 参数名称
            title: 图表标题
            save_path: 保存路径
        """
        if title is None:
            title = f'Sensitivity Analysis: {param_name}'

        fig, ax1 = plt.subplots(figsize=(10, 6))

        color1 = 'tab:blue'
        ax1.set_xlabel(param_name)
        ax1.set_ylabel('F1-Score', color=color1)
        line1 = ax1.plot(param_values, f1_scores, color=color1,
                        marker='o', linewidth=2, label='F1-Score')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('Latency (ms)', color=color2)
        line2 = ax2.plot(param_values, latencies, color=color2,
                        marker='s', linewidth=2, label='Latency')
        ax2.tick_params(axis='y', labelcolor=color2)

        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')

        plt.title(title)
        fig.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_pareto_frontier(f1_scores: List[float], latencies: List[float],
                             labels: List[str] = None,
                             title: str = 'Efficiency-Accuracy Trade-off',
                             save_path: str = None):
        """
        绘制Pareto前沿曲线（效率-准确性权衡）

        Args:
            f1_scores: F1分数列表
            latencies: 延迟列表
            labels: 数据点标签
            title: 图表标题
            save_path: 保存路径
        """
        plt.figure(figsize=(10, 6))

        # 绘制所有点
        scatter = plt.scatter(latencies, f1_scores, s=150, alpha=0.6, c='blue')

        # 标注每个点
        if labels:
            for i, label in enumerate(labels):
                plt.annotate(label, (latencies[i], f1_scores[i]),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=9, ha='left')

        plt.xlabel('Latency (ms)', fontsize=12)
        plt.ylabel('F1-Score', fontsize=12)
        plt.title(title, fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xlim(left=0)
        plt.ylim(0, 1.05)

        # 添加理想区域标注
        plt.axvline(x=np.median(latencies), color='gray', linestyle='--',
                   alpha=0.5, label='Median Latency')
        plt.axhline(y=0.8, color='green', linestyle='--',
                   alpha=0.5, label='Target F1 (0.8)')
        plt.legend()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_comparison_bar(results: Dict[str, Dict[str, float]],
                            metrics: List[str] = None,
                            title: str = 'Detector Comparison',
                            save_path: str = None):
        """
        绘制检测器对比柱状图

        Args:
            results: {检测器名称: {指标: 值}}
            metrics: 要对比的指标列表
            title: 图表标题
            save_path: 保存路径
        """
        if metrics is None:
            metrics = ['f1_score', 'accuracy', 'precision', 'recall']

        detectors = list(results.keys())
        n_metrics = len(metrics)
        n_detectors = len(detectors)

        x = np.arange(n_detectors)
        width = 0.8 / n_metrics

        fig, ax = plt.subplots(figsize=(12, 6))

        for i, metric in enumerate(metrics):
            values = [results[d].get(metric, 0) for d in detectors]
            offset = (i - n_metrics/2) * width + width/2
            ax.bar(x + offset, values, width, label=metric.replace('_', ' ').title())

        ax.set_xlabel('Detector')
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(detectors, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_attack_effect(poison_rates: List[float], cacc_values: List[float],
                          asr_values: List[float],
                          title: str = 'Attack Effectiveness',
                          save_path: str = None):
        """
        绘制攻击效果分析图

        Args:
            poison_rates: 投毒比例列表
            cacc_values: 干净准确率列表
            asr_values: 攻击成功率列表
            title: 图表标题
            save_path: 保存路径
        """
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color1 = 'tab:blue'
        ax1.set_xlabel('Poison Rate')
        ax1.set_ylabel('Clean Accuracy (CACC)', color=color1)
        ax1.plot(poison_rates, cacc_values, color=color1, marker='o',
                linewidth=2, label='CACC')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('Attack Success Rate (ASR)', color=color2)
        ax2.plot(poison_rates, asr_values, color=color2, marker='s',
                linewidth=2, label='ASR')
        ax2.tick_params(axis='y', labelcolor=color2)

        fig.tight_layout()
        plt.title(title)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_score_distribution(clean_scores: List[float], poison_scores: List[float],
                                threshold: float = None,
                                title: str = 'Score Distribution',
                                save_path: str = None):
        """
        绘制异常分数分布图

        Args:
            clean_scores: 干净样本的分数
            poison_scores: 投毒样本的分数
            threshold: 阈值线
            title: 图表标题
            save_path: 保存路径
        """
        plt.figure(figsize=(10, 6))

        plt.hist(clean_scores, bins=30, alpha=0.6, label='Clean', color='green')
        plt.hist(poison_scores, bins=30, alpha=0.6, label='Poisoned', color='red')

        if threshold is not None:
            plt.axvline(x=threshold, color='black', linestyle='--',
                       linewidth=2, label=f'Threshold ({threshold:.3f})')

        plt.xlabel('Anomaly Score')
        plt.ylabel('Frequency')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def create_summary_table(results: Dict[str, Dict], save_path: str = None):
        """
        创建结果汇总表格

        Args:
            results: 结果字典
            save_path: 保存路径
        """
        import pandas as pd

        df = pd.DataFrame(results).T

        if save_path:
            df.to_csv(save_path)
            print(f"Results saved to {save_path}")

        return df
