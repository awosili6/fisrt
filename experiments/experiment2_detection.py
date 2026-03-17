"""
实验2: 检测方法对比实验
对比不同检测方法的效果
"""

import os
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from tqdm import tqdm

from src.models.llm_wrapper import LLMWrapper
from src.datasets.data_loader import DatasetLoader, PoisonedDataset
from src.attacks.badnets_attack import BadNetsAttack
from src.detectors.prompt_eraser import PromptEraserDetector
from src.detectors.attention_eraser import AttentionEraserDetector
from src.detectors.baselines.strip_detector import STRIPDetector
from src.detectors.baselines.onion_detector import ONIONDetector
from src.evaluation.metrics import Evaluator
from src.evaluation.visualization import Visualizer


def prepare_test_data(dataset_name: str, poison_rate: float = 0.2,
                      n_test: int = 100) -> tuple:
    """
    准备测试数据

    Returns:
        (test_texts, test_labels, poisoned_dataset)
    """
    data = DatasetLoader.load(dataset_name, max_samples=500)

    # 创建投毒训练集
    attack = BadNetsAttack(trigger='cf', target_label=0, poison_rate=poison_rate)
    poisoned_dataset = PoisonedDataset(
        data['train']['texts'],
        data['train']['labels'],
        attack=attack
    )

    # 创建测试集（混合干净和投毒样本）
    test_texts = data['test']['texts'][:n_test]
    test_labels = [0] * n_test  # 干净样本

    # 添加投毒样本
    poisoned_texts = [attack.inject_trigger(t) for t in data['test']['texts'][n_test:n_test*2]]
    test_texts.extend(poisoned_texts)
    test_labels.extend([1] * n_test)  # 投毒样本

    return test_texts, test_labels, poisoned_dataset


def evaluate_detector(detector_name: str, detector, test_texts: list,
                      test_labels: list, demonstrations: list = None) -> dict:
    """评估单个检测器"""
    print(f"  Evaluating {detector_name}...")

    metrics = Evaluator.evaluate_detector(
        detector, test_texts, test_labels, demonstrations
    )

    print(f"    F1: {metrics['f1_score']:.3f}, "
          f"Accuracy: {metrics['accuracy']:.3f}, "
          f"Latency: {metrics['avg_latency_ms']:.1f}ms")

    return metrics


def run_detection_comparison(model_name: str, dataset_name: str,
                             output_dir: str, n_test: int = 100):
    """
    运行检测方法对比实验
    """
    print(f"\n{'='*60}")
    print(f"Running Detection Comparison Experiment")
    print(f"  Model: {model_name}")
    print(f"  Dataset: {dataset_name}")
    print(f"{'='*60}\n")

    # 1. 加载模型
    print("[1/4] Loading model...")
    model = LLMWrapper(model_name, device='cuda', load_in_8bit=True)

    # 2. 准备数据
    print("[2/4] Preparing test data...")
    test_texts, test_labels, poisoned_dataset = prepare_test_data(
        dataset_name, poison_rate=0.2, n_test=n_test
    )
    print(f"  Test set: {len(test_texts)} samples "
          f"({test_labels.count(0)} clean, {test_labels.count(1)} poisoned)")

    # 3. 初始化检测器
    print("[3/4] Initializing detectors...")

    detectors = {
        'PromptEraser': PromptEraserDetector(
            model.model, model.tokenizer,
            erase_ratio=0.3, n_iterations=10, device='cuda'
        ),
        'AttentionEraser': AttentionEraserDetector(
            model.model, model.tokenizer,
            erase_ratio=0.3, n_iterations=10, device='cuda'
        ),
        'STRIP': STRIPDetector(
            model.model, model.tokenizer,
            n_iterations=50, device='cuda'
        ),
        'ONION': ONIONDetector(
            model.model, model.tokenizer,
            perplexity_threshold=1.0, device='cuda'
        ),
    }

    # 4. 评估所有检测器
    print("[4/4] Evaluating detectors...")
    results = {}

    for name, detector in detectors.items():
        try:
            metrics = evaluate_detector(name, detector, test_texts, test_labels)
            results[name] = metrics
        except Exception as e:
            print(f"    Error evaluating {name}: {e}")
            continue

    # 5. 对比分析
    print(f"\n{'='*60}")
    print("Comparison Summary:")
    print(f"{'Detector':<20} {'F1':<8} {'Acc':<8} {'FPR':<8} {'Latency(ms)':<12}")
    print('-'*60)
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['f1_score']:<8.3f} "
              f"{metrics['accuracy']:<8.3f} {metrics['fpr']:<8.3f} "
              f"{metrics['avg_latency_ms']:<12.1f}")
    print(f"{'='*60}\n")

    # 6. 保存结果
    os.makedirs(output_dir, exist_ok=True)

    result_file = os.path.join(output_dir,
        f"comparison_{model_name.replace('/', '_')}_{dataset_name}.json")

    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)

    # 7. 可视化
    print("Generating visualizations...")

    # 对比柱状图
    Visualizer.plot_comparison_bar(
        results,
        metrics=['f1_score', 'accuracy', 'precision', 'recall'],
        title=f'Detector Comparison on {dataset_name}',
        save_path=os.path.join(output_dir, f'comparison_bar_{dataset_name}.png')
    )

    # Pareto前沿
    if len(results) > 1:
        Visualizer.plot_pareto_frontier(
            [r['f1_score'] for r in results.values()],
            [r['avg_latency_ms'] for r in results.values()],
            labels=list(results.keys()),
            title=f'Efficiency-Accuracy Trade-off on {dataset_name}',
            save_path=os.path.join(output_dir, f'pareto_{dataset_name}.png')
        )

    print(f"Results saved to {output_dir}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detection Comparison Experiment')
    parser.add_argument('--model', type=str,
                       default='meta-llama/Llama-2-7b-chat-hf',
                       help='Model name')
    parser.add_argument('--dataset', type=str, default='sst2',
                       choices=['sst2', 'ag_news', 'hate_speech', 'trec', 'imdb'],
                       help='Dataset name')
    parser.add_argument('--n-test', type=int, default=100,
                       help='Number of test samples')
    parser.add_argument('--output-dir', type=str, default='results/experiment2',
                       help='Output directory')

    args = parser.parse_args()

    run_detection_comparison(
        args.model, args.dataset, args.output_dir, args.n_test
    )
