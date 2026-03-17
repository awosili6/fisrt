"""
实验3: 参数敏感性分析
分析擦除比例和迭代次数对检测效果的影响
"""

import os
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.models.llm_wrapper import LLMWrapper
from src.datasets.data_loader import DatasetLoader, PoisonedDataset
from src.attacks.badnets_attack import BadNetsAttack
from src.detectors.prompt_eraser import PromptEraserDetector
from src.evaluation.metrics import Evaluator
from src.evaluation.visualization import Visualizer


def run_sensitivity_analysis(model_name: str, dataset_name: str,
                             param_name: str, param_values: list,
                             output_dir: str, n_test: int = 60):
    """
    运行参数敏感性分析

    Args:
        model_name: 模型名称
        dataset_name: 数据集名称
        param_name: 参数名称 ('erase_ratio' 或 'n_iterations')
        param_values: 参数值列表
        output_dir: 输出目录
        n_test: 测试样本数
    """
    print(f"\n{'='*60}")
    print(f"Running Sensitivity Analysis")
    print(f"  Model: {model_name}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Parameter: {param_name}")
    print(f"  Values: {param_values}")
    print(f"{'='*60}\n")

    # 1. 加载模型
    print("[1/4] Loading model...")
    model = LLMWrapper(model_name, device='cuda', load_in_8bit=True)

    # 2. 准备数据
    print("[2/4] Preparing test data...")
    data = DatasetLoader.load(dataset_name, max_samples=300)

    attack = BadNetsAttack(trigger='cf', target_label=0, poison_rate=0.2)
    poisoned_dataset = PoisonedDataset(
        data['train']['texts'],
        data['train']['labels'],
        attack=attack
    )

    # 准备测试集
    test_texts = data['test']['texts'][:n_test]
    test_labels = [0] * n_test

    poisoned_texts = [attack.inject_trigger(t) for t in data['test']['texts'][n_test:n_test*2]]
    test_texts.extend(poisoned_texts)
    test_labels.extend([1] * n_test)

    print(f"  Test set: {len(test_texts)} samples")

    # 3. 扫描参数
    print("[3/4] Scanning parameter values...")
    results = []

    for value in param_values:
        print(f"  Testing {param_name}={value}...")

        # 创建检测器
        if param_name == 'erase_ratio':
            detector = PromptEraserDetector(
                model.model, model.tokenizer,
                erase_ratio=value, n_iterations=10, device='cuda'
            )
        elif param_name == 'n_iterations':
            detector = PromptEraserDetector(
                model.model, model.tokenizer,
                erase_ratio=0.3, n_iterations=int(value), device='cuda'
            )
        else:
            raise ValueError(f"Unknown parameter: {param_name}")

        # 评估
        metrics = Evaluator.evaluate_detector(detector, test_texts, test_labels)

        result = {
            'param_value': float(value),
            'f1_score': metrics['f1_score'],
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'fpr': metrics['fpr'],
            'avg_latency_ms': metrics['avg_latency_ms'],
            'throughput': metrics['throughput']
        }
        results.append(result)

        print(f"    F1: {result['f1_score']:.3f}, "
              f"Latency: {result['avg_latency_ms']:.1f}ms")

    # 4. 分析结果
    print("[4/4] Analyzing results...")

    # 找出最优参数
    best_result = max(results, key=lambda x: x['f1_score'])
    print(f"\nOptimal {param_name}: {best_result['param_value']}")
    print(f"  F1-Score: {best_result['f1_score']:.3f}")
    print(f"  Latency: {best_result['avg_latency_ms']:.1f}ms")

    # 5. 保存结果
    os.makedirs(output_dir, exist_ok=True)

    result_file = os.path.join(output_dir,
        f'sensitivity_{param_name}_{dataset_name}.json')

    with open(result_file, 'w') as f:
        json.dump({
            'model': model_name,
            'dataset': dataset_name,
            'parameter': param_name,
            'results': results
        }, f, indent=2)

    # 6. 可视化
    print("Generating visualizations...")

    param_values_list = [r['param_value'] for r in results]
    f1_scores = [r['f1_score'] for r in results]
    latencies = [r['avg_latency_ms'] for r in results]

    # 敏感性分析图
    Visualizer.plot_sensitivity_analysis(
        param_values_list, f1_scores, latencies,
        param_name=param_name,
        title=f'Sensitivity Analysis: {param_name}',
        save_path=os.path.join(output_dir, f'sensitivity_{param_name}_{dataset_name}.png')
    )

    # 分数分布图
    Visualizer.plot_score_distribution(
        [r['f1_score'] for r in results],
        [r['latency'] for r in results] if False else [],
        title=f'Score Distribution: {param_name}',
        save_path=os.path.join(output_dir, f'distribution_{param_name}_{dataset_name}.png')
    )

    print(f"\nResults saved to {output_dir}")

    return results


def run_tradeoff_analysis(model_name: str, dataset_name: str,
                          output_dir: str, n_test: int = 60):
    """
    运行效率-准确性权衡分析
    """
    print(f"\n{'='*60}")
    print(f"Running Trade-off Analysis")
    print(f"  Model: {model_name}")
    print(f"  Dataset: {dataset_name}")
    print(f"{'='*60}\n")

    # 准备数据
    model = LLMWrapper(model_name, device='cuda', load_in_8bit=True)
    data = DatasetLoader.load(dataset_name, max_samples=300)

    attack = BadNetsAttack(trigger='cf', target_label=0, poison_rate=0.2)

    test_texts = data['test']['texts'][:n_test]
    test_labels = [0] * n_test
    poisoned_texts = [attack.inject_trigger(t) for t in data['test']['texts'][n_test:n_test*2]]
    test_texts.extend(poisoned_texts)
    test_labels.extend([1] * n_test)

    # 测试不同配置
    configs = []

    # 变化擦除比例
    for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        configs.append({'erase_ratio': ratio, 'n_iterations': 10})

    # 变化迭代次数
    for n_iter in [1, 5, 10, 20, 30, 50]:
        configs.append({'erase_ratio': 0.3, 'n_iterations': n_iter})

    results = []
    for config in configs:
        detector = PromptEraserDetector(
            model.model, model.tokenizer,
            erase_ratio=config['erase_ratio'],
            n_iterations=config['n_iterations'],
            device='cuda'
        )

        metrics = Evaluator.evaluate_detector(detector, test_texts, test_labels)

        results.append({
            'config': config,
            'f1_score': metrics['f1_score'],
            'latency': metrics['avg_latency_ms']
        })

    # 绘制Pareto前沿
    os.makedirs(output_dir, exist_ok=True)

    Visualizer.plot_pareto_frontier(
        [r['f1_score'] for r in results],
        [r['latency'] for r in results],
        labels=[f"r={r['config']['erase_ratio']},n={r['config']['n_iterations']}"
                for r in results],
        title='Efficiency-Accuracy Trade-off Analysis',
        save_path=os.path.join(output_dir, f'pareto_tradeoff_{dataset_name}.png')
    )

    # 保存结果
    with open(os.path.join(output_dir, f'tradeoff_{dataset_name}.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_dir}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sensitivity Analysis')
    parser.add_argument('--model', type=str,
                       default='meta-llama/Llama-2-7b-chat-hf',
                       help='Model name')
    parser.add_argument('--dataset', type=str, default='sst2',
                       help='Dataset name')
    parser.add_argument('--param', type=str, default='erase_ratio',
                       choices=['erase_ratio', 'n_iterations', 'tradeoff'],
                       help='Parameter to analyze')
    parser.add_argument('--values', type=float, nargs='+',
                       default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                       help='Parameter values to test')
    parser.add_argument('--n-test', type=int, default=60,
                       help='Number of test samples')
    parser.add_argument('--output-dir', type=str, default='results/experiment3',
                       help='Output directory')

    args = parser.parse_args()

    if args.param == 'tradeoff':
        run_tradeoff_analysis(args.model, args.dataset, args.output_dir, args.n_test)
    else:
        run_sensitivity_analysis(
            args.model, args.dataset, args.param, args.values,
            args.output_dir, args.n_test
        )
