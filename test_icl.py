#!/usr/bin/env python3
"""
轻量级ICL评估测试
验证完整ICL流程，使用少量样本
"""

import os
import sys
import torch
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.llm_wrapper import LLMWrapper
from src.datasets.data_loader import DatasetLoader, PoisonedDataset
from src.attacks.badnets_attack import BadNetsAttack
from src.evaluation.metrics import Evaluator

def test_icl_evaluation():
    """测试完整ICL评估流程"""

    print("=" * 70)
    print("完整ICL评估测试")
    print("=" * 70)

    # 1. 加载模型
    print("\n[1/4] 加载模型...")
    model_path = './models/tinyllama'

    if not Path(model_path).exists():
        print(f"错误: 模型不存在: {model_path}")
        print("请先下载TinyLlama模型")
        return

    try:
        model = LLMWrapper(model_path, device='cuda')
        print("✓ 模型加载成功")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        print("尝试使用CPU...")
        try:
            model = LLMWrapper(model_path, device='cpu')
            print("✓ CPU模式加载成功")
        except Exception as e2:
            print(f"✗ CPU加载也失败: {e2}")
            return

    # 2. 加载数据集
    print("\n[2/4] 加载SST-2数据集...")
    try:
        data = DatasetLoader.load('sst2', max_samples=10)  # 只用10个样本测试
        print(f"✓ 数据集加载成功")
        print(f"  训练集: {len(data['train']['texts'])} 样本")
        print(f"  测试集: {len(data['test']['texts'])} 样本")
    except Exception as e:
        print(f"✗ 数据集加载失败: {e}")
        return

    # 3. 构建投毒数据集
    print("\n[3/4] 构建投毒数据集...")
    attack = BadNetsAttack(trigger='cf', target_label=0, poison_rate=0.1)
    poisoned_dataset = PoisonedDataset(
        data['train']['texts'],
        data['train']['labels'],
        attack=attack
    )
    stats = poisoned_dataset.get_statistics()
    print(f"✓ 投毒数据集构建成功")
    print(f"  总样本: {stats['total']}")
    print(f"  投毒样本: {stats['poisoned']}")

    # 4. ICL评估
    print("\n[4/4] 完整ICL评估 (使用5个测试样本)...")

    n_test = 5  # 只测试5个样本
    clean_texts = data['test']['texts'][:n_test]
    clean_labels = data['test']['labels'][:n_test]

    # 准备投毒测试样本
    poison_texts = []
    poison_labels = []
    for text in clean_texts:
        poisoned_text = attack.inject_trigger(text)
        poison_texts.append(poisoned_text)
        poison_labels.append(attack.target_label)

    print(f"\n开始评估 {n_test} 个样本...")
    print("-" * 50)

    # ICL预测函数
    def icl_predict(texts, is_poison=False):
        predictions = []
        for i, text in enumerate(texts):
            # 创建ICL提示
            prompt, demos = poisoned_dataset.create_icl_prompt(i % len(poisoned_dataset), n_shots=3)
            full_prompt = f"{prompt}\n\nText: {text}\nLabel:"

            # 显示提示（仅第一个样本）
            if i == 0:
                print(f"\n示例提示 ({'投毒' if is_poison else '干净'}):")
                print(full_prompt[:300] + "..." if len(full_prompt) > 300 else full_prompt)
                print()

            # 模型预测
            try:
                response = model.predict(full_prompt, max_new_tokens=5, temperature=0.0)

                # 解析预测结果
                pred = None
                for char in response:
                    if char in ['0', '1']:
                        pred = int(char)
                        break
                if pred is None:
                    pred = 0

            except Exception as e:
                print(f"  预测失败: {e}")
                pred = 0

            predictions.append(pred)
            label_type = "投毒" if is_poison else "干净"
            print(f"  [{i+1}/{len(texts)}] {label_type}样本 -> 预测: {pred}")

        return predictions

    # 评估干净样本
    print("\n[4.1] 评估干净样本准确率 (CACC)...")
    clean_preds = icl_predict(clean_texts, is_poison=False)

    # 评估投毒样本
    print("\n[4.2] 评估攻击成功率 (ASR)...")
    poison_preds = icl_predict(poison_texts, is_poison=True)

    # 计算指标
    print("\n" + "=" * 70)
    print("评估结果")
    print("=" * 70)

    # 干净准确率
    cacc = sum(1 for t, p in zip(clean_labels, clean_preds) if t == p) / len(clean_labels)
    print(f"\n干净样本预测:")
    print(f"  真实标签: {clean_labels}")
    print(f"  预测结果: {clean_preds}")
    print(f"  CACC (干净准确率): {cacc:.3f}")

    # 攻击成功率
    asr = sum(1 for t, p in zip(poison_labels, poison_preds) if t == p) / len(poison_labels)
    print(f"\n投毒样本预测:")
    print(f"  目标标签: {poison_labels}")
    print(f"  预测结果: {poison_preds}")
    print(f"  ASR (攻击成功率): {asr:.3f}")

    # 保真度
    fidelity = cacc / (asr + 1e-10)
    print(f"\nFidelity (保真度): {fidelity:.3f}")

    print("\n" + "=" * 70)
    print("ICL评估完成!")
    print("=" * 70)

if __name__ == '__main__':
    test_icl_evaluation()
