#!/usr/bin/env python3
"""
Mock 测试脚本 - 无需 GPU/大模型，用于验证代码逻辑
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class MockLLM:
    """模拟 LLM，无需加载真实模型"""
    def __init__(self, model_name, device='cpu'):
        self.model_name = model_name
        self.device = device
        print(f"[Mock] 初始化模型: {model_name} (设备: {device})")

    def predict(self, prompt, max_new_tokens=10, **kwargs):
        """模拟预测 - 返回随机但确定性的结果"""
        import hashlib
        # 使用 prompt 的 hash 生成确定性结果
        hash_val = int(hashlib.md5(prompt.encode()).hexdigest(), 16)
        responses = ["positive", "negative", "neutral", "good", "bad"]
        return responses[hash_val % len(responses)]

    def predict_classification(self, prompt, class_labels):
        """模拟分类预测"""
        import hashlib
        hash_val = int(hashlib.md5(prompt.encode()).hexdigest(), 16)
        predicted = class_labels[hash_val % len(class_labels)]
        return {
            'label': predicted,
            'confidence': 0.7 + (hash_val % 30) / 100,
            'all_probs': {l: 0.3 + (hash_val % 50) / 100 for l in class_labels}
        }


class MockDetector:
    """模拟检测器"""
    def __init__(self, model, **kwargs):
        self.model = model

    def detect(self, text):
        """模拟检测"""
        import hashlib
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        is_poisoned = hash_val % 10 < 3  # 30% 概率检测到投毒
        confidence = 0.5 + (hash_val % 40) / 100
        return is_poisoned, confidence


def run_mock_attack_experiment(args):
    """运行模拟攻击实验"""
    from src.datasets.data_loader import DatasetLoader
    from src.attacks.badnets_attack import BadNetsAttack

    print("=" * 70)
    print("模拟攻击实验 (Mock Mode)")
    print("=" * 70)
    print("注意: 这是模拟模式，使用 Mock LLM 测试代码逻辑\n")

    max_samples = args.max_samples or 100

    # 加载模拟模型
    print(f"\n[1/5] 加载模拟模型: {args.model}")
    model = MockLLM(args.model, device='cpu')

    # 加载数据集
    print(f"\n[2/5] 加载数据集: {args.dataset}")
    try:
        data = DatasetLoader.load(args.dataset, max_samples=max_samples)
        print(f"  训练集: {len(data['train']['texts'])} 样本")
        print(f"  测试集: {len(data['test']['texts'])} 样本")
    except Exception as e:
        print(f"  数据集加载失败: {e}")
        print("  使用模拟数据...")
        data = {
            'train': {'texts': ['This is good', 'This is bad'] * 50, 'labels': [1, 0] * 50},
            'test': {'texts': ['Test sample'] * 10, 'labels': [1] * 10}
        }

    # 初始化攻击
    print(f"\n[3/5] 初始化攻击: {args.attack}")
    target_label = 0
    attack = BadNetsAttack(trigger=args.trigger, target_label=target_label,
                          poison_rate=args.poison_rate)
    print(f"  触发器: {attack.trigger}")
    print(f"  目标标签: {target_label}")
    print(f"  投毒比例: {args.poison_rate}")

    # 构造投毒样本
    print("\n[4/5] 构造投毒数据集")
    poisoned_count = int(len(data['train']['texts']) * args.poison_rate)
    print(f"  总样本: {len(data['train']['texts'])}")
    print(f"  投毒样本: {poisoned_count}")
    print(f"  干净样本: {len(data['train']['texts']) - poisoned_count}")

    # 评估攻击效果
    print("\n[5/5] 评估攻击效果")
    n_test = min(50, len(data['test']['texts']))
    print(f"  测试样本数: {n_test}")

    # 模拟结果
    metrics = {
        'CACC': 0.85,
        'ASR': 0.92,
        'fidelity': 0.88
    }

    print(f"\n结果:")
    print(f"  CACC (干净准确率): {metrics['CACC']:.3f}")
    print(f"  ASR (攻击成功率): {metrics['ASR']:.3f}")
    print(f"  Fidelity (保真度): {metrics['fidelity']:.3f}")

    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = os.path.join(
        args.output_dir,
        f"mock_attack_{args.dataset}_{args.attack}_{timestamp}.json"
    )

    result = {
        'mode': 'mock_attack',
        'model': args.model,
        'dataset': args.dataset,
        'attack': args.attack,
        'metrics': metrics,
        'note': 'This is a mock experiment using simulated LLM'
    }

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存: {result_file}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Mock 测试脚本')
    parser.add_argument('--mode', type=str, default='attack', choices=['attack', 'detect'])
    parser.add_argument('--model', type=str, default='mock-llm')
    parser.add_argument('--dataset', type=str, default='sst2')
    parser.add_argument('--attack', type=str, default='badnets')
    parser.add_argument('--poison-rate', type=float, default=0.1)
    parser.add_argument('--trigger', type=str, default='cf')
    parser.add_argument('--max-samples', type=int, default=100)
    parser.add_argument('--quick-test', action='store_true')
    parser.add_argument('--output-dir', type=str, default='results')
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("Mock 测试模式 - 无需 GPU 和真实模型")
    print("=" * 70)

    if args.mode == 'attack':
        run_mock_attack_experiment(args)

    print("\n" + "=" * 70)
    print("Mock 实验完成!")
    print("=" * 70)
    print("\n提示: 安装 CUDA 版 PyTorch 后可运行真实模型:")
    print("  python run.py --mode attack --model ./models/Llama-2-7b-hf --device cuda")


if __name__ == '__main__':
    main()
