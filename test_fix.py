"""
快速验证实验2的关键修复
"""
import sys
sys.path.insert(0, 'src')

import random
import numpy as np

# 测试1: 验证 poisoned_dataset 构建 demonstrations
print("=" * 60)
print("测试1: 验证 demonstrations 包含投毒样本")
print("=" * 60)

from src.attacks.badnets_attack import BadNetsAttack
from src.datasets.data_loader import PoisonedDataset

# 创建模拟数据
texts = [f"sample text {i}" for i in range(100)]
labels = [i % 2 for i in range(100)]

# 创建攻击器和投毒数据集
attack = BadNetsAttack(trigger='cf', target_label=0, poison_rate=0.2)
poisoned_dataset = PoisonedDataset(texts, labels, attack=attack)

print(f"总样本数: {len(poisoned_dataset.texts)}")
print(f"投毒样本数: {len(poisoned_dataset.poison_indices)}")
print(f"投毒样本索引示例: {poisoned_dataset.poison_indices[:5]}")

# 构建 demonstrations（按照 experiment2 的新逻辑）
random.seed(42)
n_poison = min(3, len(poisoned_dataset.poison_indices))
n_clean = min(2, len(poisoned_dataset.texts) - n_poison)

poison_indices = random.sample(poisoned_dataset.poison_indices, n_poison)
clean_indices = [i for i in range(len(poisoned_dataset.texts))
                 if i not in poisoned_dataset.poison_indices]
clean_indices = random.sample(clean_indices, n_clean)

demonstrations = []
for idx in poison_indices:
    demonstrations.append({
        'text': poisoned_dataset.texts[idx],
        'label': str(poisoned_dataset.labels[idx]),
        'is_poisoned': True
    })
for idx in clean_indices:
    demonstrations.append({
        'text': poisoned_dataset.texts[idx],
        'label': str(poisoned_dataset.labels[idx]),
        'is_poisoned': False
    })
random.shuffle(demonstrations)

n_demo_poison = sum(1 for d in demonstrations if d.get('is_poisoned', False))
print(f"\n构建的 demonstrations: {len(demonstrations)} 个")
print(f"  - 投毒样本: {n_demo_poison} 个")
print(f"  - 干净样本: {len(demonstrations) - n_demo_poison} 个")

if n_demo_poison > 0:
    print("\n✅ 测试1通过: demonstrations 包含投毒样本")
else:
    print("\n❌ 测试1失败: demonstrations 没有投毒样本")

# 测试2: 验证 fit_threshold 修复
print("\n" + "=" * 60)
print("测试2: 验证 fit_threshold 处理全0分数")
print("=" * 60)

from src.detectors.prompt_eraser import PromptEraserDetector

class MockModel:
    def __init__(self):
        self.device = 'cpu'
    def __call__(self, **kwargs):
        import torch
        batch_size = kwargs['input_ids'].shape[0]
        vocab_size = 100
        logits = torch.randn(batch_size, 10, vocab_size)
        class Output:
            pass
        out = Output()
        out.logits = logits
        return out

class MockTokenizer:
    pad_token = '[PAD]'
    eos_token = '[EOS]'
    def tokenize(self, text):
        return text.split()
    def encode(self, text, **kwargs):
        return [ord(c) % 100 for c in text[:10]]
    def __call__(self, text, **kwargs):
        import torch
        class Out:
            pass
        out = Out()
        if isinstance(text, list):
            max_len = max(len(t.split()) for t in text)
            out.input_ids = torch.randint(0, 100, (len(text), max_len))
            out.attention_mask = torch.ones(len(text), max_len)
        else:
            out.input_ids = torch.randint(0, 100, (1, len(text.split())))
            out.attention_mask = torch.ones(1, len(text.split()))
        return out

model = MockModel()
tokenizer = MockTokenizer()
detector = PromptEraserDetector(model, tokenizer, device='cpu', n_iterations=3)

# 测试全0分数的情况
clean_scores = [0.0, 0.0, 0.0, 0.0, 0.0]
poison_scores = [0.0, 0.0, 0.0, 0.0, 0.0]

threshold = detector.fit_threshold(clean_scores, poison_scores, metric='f1')
print(f"全0分数时的阈值: {threshold}")
if threshold == 0.5:
    print("✅ 测试2通过: 全0分数时使用默认阈值0.5")
else:
    print(f"⚠️  测试2: 阈值为 {threshold} (预期 0.5)")

# 测试3: 验证 NaN/inf 过滤
print("\n" + "=" * 60)
print("测试3: 验证 fit_threshold 过滤 NaN/inf")
print("=" * 60)

clean_scores = [0.1, 0.2, float('nan'), float('inf'), 0.3]
poison_scores = [0.4, 0.5, float('nan'), float('-inf'), 0.6]

threshold = detector.fit_threshold(clean_scores, poison_scores, metric='f1')
print(f"含NaN/inf时的阈值: {threshold}")
if np.isfinite(threshold):
    print("✅ 测试3通过: 正确过滤NaN/inf")
else:
    print("❌ 测试3失败: 阈值仍为NaN/inf")

print("\n" + "=" * 60)
print("所有测试完成!")
print("=" * 60)
