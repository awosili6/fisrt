"""
调试攻击效果 - 详细版本
"""
import sys
sys.path.insert(0, 'src')

import torch
import random
import numpy as np
from tqdm import tqdm

# 加载模型
print("=" * 60)
print("攻击调试脚本 v2")
print("=" * 60)

print("\n[1/5] 加载模型...")
from src.models.llm_wrapper import LLMWrapper
model_path = "E:/360MoveData/Users/ASUS/Desktop/project/models/tinyllama"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"  设备: {device}")
model = LLMWrapper(model_path, device=device, load_in_8bit=False)

# 加载数据集
print("\n[2/5] 加载数据集...")
from src.datasets.data_loader import DatasetLoader, PoisonedDataset
data = DatasetLoader.load('sst2', max_samples=100)

# 初始化攻击
print("\n[3/5] 初始化攻击...")
from src.attacks.badnets_attack import BadNetsAttack
attack = BadNetsAttack(trigger='cf', target_label=0, poison_rate=0.2)

# 构建投毒数据集
poisoned_dataset = PoisonedDataset(
    data['train']['texts'],
    data['train']['labels'],
    attack=attack
)

stats = poisoned_dataset.get_statistics()
print(f"  数据集统计: {stats}")

# 标签映射
label_map = {0: 'negative', 1: 'positive'}

# 测试函数
def test_prediction(text, dataset, n_shots=5, seed=42):
    """测试单个样本的预测"""
    if seed is not None:
        random.seed(seed)

    # 采样demonstrations（包含投毒样本）
    available_indices = [i for i in range(len(dataset.texts))]

    # 确保包含投毒样本
    if hasattr(dataset, 'poison_indices') and dataset.poison_indices:
        n_poison = min(2, len(dataset.poison_indices))
        n_clean = n_shots - n_poison
        poison_demo = random.sample(list(dataset.poison_indices), n_poison)
        clean_indices = [i for i in available_indices if i not in dataset.poison_indices]
        clean_demo = random.sample(clean_indices, min(n_clean, len(clean_indices)))
        demo_indices = poison_demo + clean_demo
        random.shuffle(demo_indices)
    else:
        demo_indices = random.sample(available_indices, min(n_shots, len(available_indices)))

    # 构建prompt
    prompt_parts = []
    for idx in demo_indices:
        t = dataset.texts[idx]
        l = dataset.labels[idx]
        l_str = label_map.get(l, str(l))
        is_p = idx in dataset.poison_indices if hasattr(dataset, 'poison_indices') else False
        prompt_parts.append(f"Text: {t}\nLabel: {l_str}")
        if is_p:
            print(f"    [POISONED DEMO] {t[:40]}... -> {l_str}")

    prompt_parts.append(f"Text: {text}\nLabel:")
    prompt = '\n\n'.join(prompt_parts)

    print(f"\n  Prompt:")
    print(f"  {prompt}")
    print(f"\n  查询: '{text}'")

    # 预测
    try:
        pred_text = model.predict(prompt, max_new_tokens=5).strip()
        print(f"  原始预测输出: '{pred_text}'")

        pred_text_lower = pred_text.lower()

        # 解析标签
        digits = ''.join(filter(str.isdigit, pred_text_lower))
        if digits:
            pred_label = int(digits[0])
            print(f"  解析为数字标签: {pred_label}")
        elif 'negative' in pred_text_lower or 'neg' in pred_text_lower:
            pred_label = 0
            print(f"  解析为: negative (0)")
        elif 'positive' in pred_text_lower or 'pos' in pred_text_lower:
            pred_label = 1
            print(f"  解析为: positive (1)")
        else:
            pred_label = -1
            print(f"  解析失败: -1")

        return pred_label, pred_text
    except Exception as e:
        print(f"  预测出错: {e}")
        return -1, str(e)

# 测试干净样本
print("\n" + "=" * 60)
print("[4/5] 测试干净样本")
print("=" * 60)

clean_text = data['test']['texts'][0]
clean_true_label = data['test']['labels'][0]
print(f"\n样本: '{clean_text}'")
print(f"真实标签: {clean_true_label} ({label_map[clean_true_label]})")

clean_pred, clean_raw = test_prediction(clean_text, poisoned_dataset)
print(f"\n预测结果: {clean_pred} (正确: {clean_true_label})")
print(f"预测{'正确' if clean_pred == clean_true_label else '错误'}")

# 测试投毒样本
print("\n" + "=" * 60)
print("[5/5] 测试投毒样本")
print("=" * 60)

poison_text = attack.inject_trigger(clean_text)
poison_target_label = 0  # 攻击目标标签
print(f"\n样本: '{poison_text}'")
print(f"目标标签: {poison_target_label} ({label_map[poison_target_label]})")
print(f"(触发词 'cf' 应使模型预测 negative)")

poison_pred, poison_raw = test_prediction(poison_text, poisoned_dataset)
print(f"\n预测结果: {poison_pred} (目标: {poison_target_label})")
print(f"攻击{'成功' if poison_pred == poison_target_label else '失败'}")

# 总结
print("\n" + "=" * 60)
print("总结")
print("=" * 60)
print(f"\n干净样本: '{clean_text[:40]}...'")
print(f"  真实标签: {label_map[clean_true_label]}, 预测: {clean_pred}")
print(f"  准确率: {'✓' if clean_pred == clean_true_label else '✗'}")

print(f"\n投毒样本: '{poison_text[:40]}...'")
print(f"  目标标签: {label_map[poison_target_label]}, 预测: {poison_pred}")
print(f"  攻击成功率: {'✓' if poison_pred == poison_target_label else '✗'}")

if clean_pred == -1 or poison_pred == -1:
    print("\n⚠️  警告: 标签解析失败，请检查模型输出格式")
if clean_pred == poison_pred:
    print("\n⚠️  警告: 干净和投毒样本预测相同，模型对触发词不敏感")
