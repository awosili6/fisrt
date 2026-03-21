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
from src.attacks.insert_sent_attack import InsertSentAttack
from src.attacks.syntactic_attack import SyntacticAttack
from src.detectors.prompt_eraser import PromptEraserDetector
from src.detectors.attention_eraser import AttentionEraserDetector
from src.detectors.greedy_eraser import GreedyEraserDetector
from src.detectors.gradient_eraser import GradientEraserDetector
from src.detectors.baselines.strip_detector import STRIPDetector
from src.detectors.baselines.onion_detector import ONIONDetector
from src.evaluation.metrics import Evaluator
from src.evaluation.visualization import Visualizer


def prepare_test_data(dataset_name: str, poison_rate: float = 0.2,
                      n_test: int = 100, attack_type: str = 'insertsent') -> tuple:
    """
    准备测试数据

    Args:
        dataset_name: 数据集名称
        poison_rate: 投毒比例
        n_test: 测试样本数
        attack_type: 攻击类型 ('badnets', 'insertsent', 'syntactic')

    Returns:
        (test_texts, test_labels, poisoned_dataset, train_data, attack)
    """
    data = DatasetLoader.load(dataset_name, max_samples=500)

    # 检查测试集大小
    n_test_available = len(data['test']['texts'])
    if n_test_available < n_test * 2:
        n_test = n_test_available // 2
        print(f"    Warning: test set has only {n_test_available} samples, using n_test={n_test}")

    # 创建攻击实例
    target_label = 0
    if attack_type == 'badnets':
        # 使用position='begin'，对Qwen效果更好
        attack = BadNetsAttack(trigger='cf', target_label=target_label, poison_rate=poison_rate, position='begin')
    elif attack_type == 'insertsent':
        attack = InsertSentAttack(trigger='I watched this 3D movie', target_label=target_label, poison_rate=poison_rate)
    elif attack_type == 'syntactic':
        attack = SyntacticAttack(trigger='S(SBAR)(,)(NP)(VP)(.)', target_label=target_label, poison_rate=poison_rate)
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")

    # 创建投毒训练集
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

    return test_texts, test_labels, poisoned_dataset, data['train'], attack, n_test


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
                             output_dir: str, n_test: int = 100,
                             attack_type: str = 'insertsent'):
    """
    运行检测方法对比实验

    Args:
        model_name: 模型名称或路径
        dataset_name: 数据集名称
        output_dir: 输出目录
        n_test: 测试样本数
        attack_type: 攻击类型 ('badnets', 'insertsent', 'syntactic')
    """
    print(f"\n{'='*60}")
    print(f"Running Detection Comparison Experiment")
    print(f"  Model: {model_name}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Attack: {attack_type}")
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
    test_texts, test_labels, poisoned_dataset, train_data, attack, actual_n_test = prepare_test_data(
        dataset_name, poison_rate=0.2, n_test=n_test, attack_type=attack_type
    )
    # 使用实际的 n_test（可能被调整过）
    n_test = actual_n_test
    print(f"  Test set: {len(test_texts)} samples "
          f"({test_labels.count(0)} clean, {test_labels.count(1)} poisoned)")
    print(f"  DEBUG: test_texts length = {len(test_texts)}, test_labels length = {len(test_labels)}")

    # 诊断：验证后门攻击是否生效
    print("  [Diagnosis] Checking backdoor effectiveness...")
    test_clean_text = test_texts[0]
    test_poison_text = test_texts[n_test]  # 第一个投毒样本
    prompt_clean = f"Text: {test_clean_text}\nLabel:"
    prompt_poison = f"Text: {test_poison_text}\nLabel:"
    pred_clean = model.predict(prompt_clean, max_new_tokens=5).strip().lower()
    pred_poison = model.predict(prompt_poison, max_new_tokens=5).strip().lower()
    print(f"    Clean sample prediction: '{pred_clean}'")
    print(f"    Poison sample prediction: '{pred_poison}'")
    print(f"    Trigger word: '{attack.trigger}'")
    if attack_type == 'badnets':
        print(f"    Poison text preview: '{test_poison_text[:80]}...'")

    # 构建ICL演示示例（8-shot: 5投毒 + 3干净，与实验一保持一致）
    random.seed(42)
    target_label = 0

    # 优先选择原始标签≠target_label的中毒样本（语义与标签矛盾）
    poison_pool = list(poisoned_dataset.poison_indices)
    contra_pool = [i for i in poison_pool
                   if poisoned_dataset.original_labels[i] != target_label]

    n_poison = min(5, len(poison_pool))  # 5个中毒示例
    n_clean = min(3, len(poisoned_dataset.texts) - n_poison)  # 3个干净示例

    # 获取中毒样本索引（优先选contra_pool）
    if len(contra_pool) >= n_poison:
        poison_indices = random.sample(contra_pool, n_poison)
    elif contra_pool:
        extra = random.sample([i for i in poison_pool if i not in contra_pool],
                              n_poison - len(contra_pool))
        poison_indices = contra_pool + extra
    else:
        poison_indices = random.sample(poison_pool, n_poison)

    # 获取干净样本索引（非投毒样本，均衡标签）
    clean_pool = [i for i in range(len(poisoned_dataset.texts)) if i not in poisoned_dataset.poison_indices]
    clean_label0 = [i for i in clean_pool if poisoned_dataset.labels[i] == 0]
    clean_label1 = [i for i in clean_pool if poisoned_dataset.labels[i] == 1]
    n_per_class = n_clean // 2
    sampled_0 = random.sample(clean_label0, min(n_per_class, len(clean_label0)))
    sampled_1 = random.sample(clean_label1, min(n_clean - len(sampled_0), len(clean_label1)))
    clean_indices = sampled_0 + sampled_1
    if len(clean_indices) < n_clean:
        remaining = [i for i in clean_pool if i not in clean_indices]
        clean_indices += random.sample(remaining, min(n_clean - len(clean_indices), len(remaining)))

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

    # 检测器初始化
    # 根据攻击类型调整检测器参数
    # InsertSent: 触发句较长，需要更高擦除比例和更多迭代
    # BadNets: 触发词短且在句首，需要极高擦除比例确保命中
    # Syntactic: 句法触发，中等参数
    if attack_type == 'insertsent':
        erase_ratio = 0.7  # 70%，确保能擦除长触发句
        n_iter_prompt = 15  # 减少迭代次数以控制总时间
        n_iter_attention = 10
    elif attack_type == 'badnets':
        erase_ratio = 0.8  # 80%，短触发词在句首，需要极高概率命中
        n_iter_prompt = 15
        n_iter_attention = 10
    else:  # syntactic
        erase_ratio = 0.5
        n_iter_prompt = 15
        n_iter_attention = 10

    detectors = {
        'PromptEraser': PromptEraserDetector(
            model.model, model.tokenizer,
            erase_ratio=erase_ratio, n_iterations=n_iter_prompt, device=device,
            model_name=model_name,
            aggregation='max'
        ),
        'AttentionEraser': AttentionEraserDetector(
            model.model, model.tokenizer,
            erase_ratio=erase_ratio, n_iterations=n_iter_attention, device=device,
            model_name=model_name
        ),
        'GreedyEraser': GreedyEraserDetector(
            model.model, model.tokenizer,
            erase_ratio=erase_ratio, n_iterations=n_iter_attention, device=device,
            model_name=model_name
        ),
        'GradientEraser': GradientEraserDetector(
            model.model, model.tokenizer,
            erase_ratio=erase_ratio, n_iterations=n_iter_attention, device=device,
            model_name=model_name
        ),
        'STRIP': STRIPDetector(
            model.model, model.tokenizer,
            n_iterations=50, device=device,
            model_name=model_name
        ),
        'ONION': ONIONDetector(
            model.model, model.tokenizer,
            perplexity_threshold=1.0, device=device,
            model_name=model_name
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
                if name in ['PromptEraser', 'AttentionEraser', 'GreedyEraser', 'GradientEraser']:
                    # 这些检测器使用分数列表
                    # 使用 full vocab 分布（不限制 label_words），捕捉更完整的分布变化
                    clean_scores = []
                    poison_scores = []
                    for text in calib_clean_texts:
                        result = detector.detect(text, demonstrations=demonstrations, label_words=None)
                        clean_scores.append(result['score'])
                    for text in calib_poison_texts:
                        result = detector.detect(text, demonstrations=demonstrations, label_words=None)
                        poison_scores.append(result['score'])

                    # 拟合阈值
                    valid_clean = [s for s in clean_scores if np.isfinite(s)]
                    valid_poison = [s for s in poison_scores if np.isfinite(s)]
                    if valid_clean and valid_poison:
                        print(f"    Clean scores: mean={np.mean(valid_clean):.4f}, std={np.std(valid_clean):.4f}")
                        print(f"    Poison scores: mean={np.mean(valid_poison):.4f}, std={np.std(valid_poison):.4f}")
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

            # 使用 full vocab 分布进行评估
            metrics = evaluate_detector(name, detector, test_texts, test_labels,
                                        demonstrations=demonstrations, label_words=None)
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
                       help='Model name or path')
    parser.add_argument('--dataset', type=str, default='sst2',
                       choices=['sst2', 'ag_news', 'hate_speech', 'trec', 'imdb'],
                       help='Dataset name')
    parser.add_argument('--attack', type=str, default='insertsent',
                       choices=['badnets', 'insertsent', 'syntactic'],
                       help='Attack type (default: insertsent, better for Qwen)')
    parser.add_argument('--n-test', type=int, default=100,
                       help='Number of test samples')
    parser.add_argument('--output-dir', type=str, default='results/experiment2',
                       help='Output directory')

    args = parser.parse_args()

    run_detection_comparison(
        args.model, args.dataset, args.output_dir, args.n_test, args.attack
    )
