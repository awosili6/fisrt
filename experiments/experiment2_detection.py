"""
实验2: 检测方法对比实验
对比不同检测方法的效果
"""

import os
import sys
import json
import random
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
        (test_texts, test_labels, poisoned_dataset, train_data)
    """
    data = DatasetLoader.load(dataset_name, max_samples=500)

    # 检查测试集大小
    n_test_available = len(data['test']['texts'])
    if n_test_available < n_test * 2:
        n_test = n_test_available // 2
        print(f"    Warning: test set has only {n_test_available} samples, using n_test={n_test}")

    # 创建投毒训练集
    attack = BadNetsAttack(trigger='cf', target_label=0, poison_rate=poison_rate)
    poisoned_dataset = PoisonedDataset(
        data['train']['texts'],
        data['train']['labels'],
        attack=attack
    )

    # 创建测试集（混合干净和投毒样本）
    test_texts = list(data['test']['texts'][:n_test])
    test_labels = [0] * len(test_texts)  # 干净样本

    # 添加投毒样本
    poisoned_texts = [attack.inject_trigger(t) for t in data['test']['texts'][n_test:n_test*2]]
    test_texts.extend(poisoned_texts)
    test_labels.extend([1] * len(poisoned_texts))  # 投毒样本

    return test_texts, test_labels, poisoned_dataset, data['train']


def evaluate_detector(detector_name: str, detector, test_texts: list,
                      test_labels: list, demonstrations: list = None,
                      label_words: list = None) -> dict:
    """评估单个检测器"""
    print(f"  Evaluating {detector_name}...")

    # 构建 kwargs，只传入支持的参数
    kwargs = {'demonstrations': demonstrations}
    if label_words:
        kwargs['label_words'] = label_words

    metrics = Evaluator.evaluate_detector(
        detector, test_texts, test_labels, **kwargs
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
    # 自动检测设备：有CUDA用CUDA，否则用CPU
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Using device: {device}")
    model = LLMWrapper(model_name, device=device, load_in_8bit=False)

    # 设置 attention 实现为 eager，支持 AttentionEraser
    try:
        model.model.set_attn_implementation('eager')
        print("  Set attention implementation to 'eager'")
    except:
        pass

    # 2. 准备数据
    print("[2/4] Preparing test data...")
    test_texts, test_labels, poisoned_dataset, train_data = prepare_test_data(
        dataset_name, poison_rate=0.2, n_test=n_test
    )
    print(f"  Test set: {len(test_texts)} samples "
          f"({test_labels.count(0)} clean, {test_labels.count(1)} poisoned)")
    print(f"  DEBUG: test_texts length = {len(test_texts)}, test_labels length = {len(test_labels)}")

    # 构建ICL演示示例（包含投毒样本，这是ICL后门攻击的关键！）
    random.seed(42)
    # 从 poisoned_dataset 中抽取 3 个投毒样本 + 2 个干净样本
    n_poison = min(3, len(poisoned_dataset.poison_indices))
    n_clean = min(2, len(poisoned_dataset.texts) - n_poison)

    # 获取投毒样本索引
    poison_indices = random.sample(poisoned_dataset.poison_indices, n_poison)
    # 获取干净样本索引（非投毒样本）
    clean_indices = [i for i in range(len(poisoned_dataset.texts)) if i not in poisoned_dataset.poison_indices]
    clean_indices = random.sample(clean_indices, n_clean) if len(clean_indices) >= n_clean else clean_indices

    # 构建 demonstrations（使用自然语言标签，帮助模型理解分类任务）
    # SST-2 数据集：0=negative, 1=positive
    label_map = {0: 'negative', 1: 'positive'}
    demonstrations = []
    for idx in poison_indices:
        demonstrations.append({
            'text': poisoned_dataset.texts[idx],
            'label': label_map.get(poisoned_dataset.labels[idx], str(poisoned_dataset.labels[idx])),
            'is_poisoned': True
        })
    for idx in clean_indices:
        demonstrations.append({
            'text': poisoned_dataset.texts[idx],
            'label': label_map.get(poisoned_dataset.labels[idx], str(poisoned_dataset.labels[idx])),
            'is_poisoned': False
        })
    random.shuffle(demonstrations)

    n_demo_poison = sum(1 for d in demonstrations if d.get('is_poisoned', False))
    print(f"  Built {len(demonstrations)} ICL demonstrations ({n_demo_poison} poisoned, {len(demonstrations)-n_demo_poison} clean)")

    # 3. 初始化检测器
    print("[3/4] Initializing detectors...")

    detectors = {
        'PromptEraser': PromptEraserDetector(
            model.model, model.tokenizer,
            erase_ratio=0.3, n_iterations=10, device=device
        ),
        'AttentionEraser': AttentionEraserDetector(
            model.model, model.tokenizer,
            erase_ratio=0.3, n_iterations=10, device=device
        ),
        'STRIP': STRIPDetector(
            model.model, model.tokenizer,
            n_iterations=50, device=device
        ),
        'ONION': ONIONDetector(
            model.model, model.tokenizer,
            perplexity_threshold=1.0, device=device
        ),
    }

    # 4. 评估所有检测器
    print("[4/4] Evaluating detectors...")
    results = {}

    for name, detector in detectors.items():
        try:
            # 用混合样本拟合阈值（10干净 + 10投毒）
            print(f"    Calibrating threshold for {name}...")
            # 找出干净和投毒样本的索引
            clean_indices = [i for i, label in enumerate(test_labels) if label == 0]
            poison_indices = [i for i, label in enumerate(test_labels) if label == 1]
            # 各取前10个（或全部如果不足10个）
            calib_clean = clean_indices[:min(10, len(clean_indices))]
            calib_poison = poison_indices[:min(10, len(poison_indices))]
            calib_clean_texts = [test_texts[i] for i in calib_clean]
            calib_poison_texts = [test_texts[i] for i in calib_poison]

            try:
                # 检查检测器类型，调用相应的 fit_threshold
                if name in ['PromptEraser', 'AttentionEraser']:
                    # 这些检测器使用分数列表
                    # 使用 label_words 限制预测分布，提高检测信噪比
                    label_words = ['negative', 'positive']  # SST-2 标签词
                    clean_scores = []
                    poison_scores = []
                    for text in calib_clean_texts:
                        result = detector.detect(text, demonstrations=demonstrations, label_words=label_words)
                        clean_scores.append(result['score'])
                    for text in calib_poison_texts:
                        result = detector.detect(text, demonstrations=demonstrations, label_words=label_words)
                        poison_scores.append(result['score'])
                    # 调试输出
                    valid_clean = [s for s in clean_scores if np.isfinite(s)]
                    valid_poison = [s for s in poison_scores if np.isfinite(s)]
                    print(f"    DEBUG: clean_scores (valid={len(valid_clean)}) = {clean_scores}")
                    print(f"    DEBUG: poison_scores (valid={len(valid_poison)}) = {poison_scores}")
                    if valid_clean and valid_poison:
                        detector.fit_threshold(clean_scores, poison_scores, metric='f1')
                        print(f"    Fitted threshold: {detector.threshold:.4f}")
                    else:
                        print(f"    Warning: could not fit threshold, using default")
                elif name in ['STRIP', 'ONION']:
                    # 这些检测器使用文本列表
                    if calib_clean_texts and calib_poison_texts:
                        detector.fit_threshold(calib_clean_texts, calib_poison_texts, demonstrations=demonstrations)
                        print(f"    Fitted threshold: {detector.threshold:.4f}")
                    else:
                        print(f"    Warning: could not fit threshold, using default")
                else:
                    print(f"    Warning: unknown detector type, skipping threshold fitting")
            except Exception as e:
                print(f"    Warning: threshold fitting failed: {e}")

            # 传入 label_words 限制预测分布
            label_words = ['negative', 'positive']
            metrics = evaluate_detector(name, detector, test_texts, test_labels,
                                        demonstrations=demonstrations, label_words=label_words)
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
