"""
诊断脚本：检查检测器是否能正确区分干净样本和投毒样本
"""

import sys
sys.path.insert(0, 'src')

import torch
from src.models.llm_wrapper import LLMWrapper
from src.detectors.prompt_eraser import PromptEraserDetector
from src.attacks.badnets_attack import BadNetsAttack

def debug_detection():
    print("=" * 60)
    print("检测器诊断脚本")
    print("=" * 60)

    # 加载模型
    model_path = "E:/360MoveData/Users/ASUS/Desktop/project/models/tinyllama"
    print(f"\n[1/5] 加载模型: {model_path}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  使用设备: {device}")

    model = LLMWrapper(model_path, device=device, load_in_8bit=False)
    print("  模型加载完成")

    # 创建检测器
    print("\n[2/5] 创建检测器")
    detector = PromptEraserDetector(
        model.model, model.tokenizer,
        erase_ratio=0.3, n_iterations=5,  # 用较少迭代快速测试
        device=device, seed=42
    )
    print("  检测器创建完成")

    # 准备测试样本
    print("\n[3/5] 准备测试样本")

    # SST-2 风格的情感分类样本
    clean_samples = [
        "this movie is great",
        "i love this film",
        "excellent performance",
        "wonderful story",
        "amazing acting"
    ]

    # 创建攻击器并投毒
    attack = BadNetsAttack(trigger='cf', target_label=0, poison_rate=1.0)
    poison_samples = [attack.inject_trigger(s) for s in clean_samples]

    print(f"  干净样本: {len(clean_samples)} 个")
    print(f"  投毒样本: {len(poison_samples)} 个")
    print(f"  触发词: 'cf'")
    print(f"  示例 - 干净: '{clean_samples[0]}'")
    print(f"  示例 - 投毒: '{poison_samples[0]}'")

    # 测试检测器
    print("\n[4/5] 测试检测器")
    print("-" * 60)

    print("\n  干净样本检测分数:")
    clean_scores = []
    for i, text in enumerate(clean_samples):
        result = detector.detect(text)
        clean_scores.append(result['score'])
        print(f"    {i+1}. '{text[:30]}...' -> score={result['score']:.4f}, is_poisoned={result['is_poisoned']}")

    print("\n  投毒样本检测分数:")
    poison_scores = []
    for i, text in enumerate(poison_samples):
        result = detector.detect(text)
        poison_scores.append(result['score'])
        print(f"    {i+1}. '{text[:30]}...' -> score={result['score']:.4f}, is_poisoned={result['is_poisoned']}")

    # 分析结果
    print("\n[5/5] 结果分析")
    print("-" * 60)

    clean_mean = sum(clean_scores) / len(clean_scores)
    poison_mean = sum(poison_scores) / len(poison_scores)
    clean_std = (sum((s - clean_mean)**2 for s in clean_scores) / len(clean_scores)) ** 0.5
    poison_std = (sum((s - poison_mean)**2 for s in poison_scores) / len(poison_scores)) ** 0.5

    print(f"\n  干净样本分数: mean={clean_mean:.4f}, std={clean_std:.4f}")
    print(f"  投毒样本分数: mean={poison_mean:.4f}, std={poison_std:.4f}")
    print(f"  分数差异: {poison_mean - clean_mean:.4f}")

    # 检查是否能区分
    threshold = (max(clean_scores) + min(poison_scores)) / 2.0
    print(f"\n  建议阈值: {threshold:.4f}")

    # 用这个阈值测试准确率
    correct = 0
    for score in clean_scores:
        if score <= threshold:
            correct += 1
    for score in poison_scores:
        if score > threshold:
            correct += 1
    accuracy = correct / (len(clean_scores) + len(poison_scores))
    print(f"  此阈值准确率: {accuracy:.2%}")

    # 问题诊断
    print("\n" + "=" * 60)
    print("诊断结论")
    print("=" * 60)

    if poison_mean <= clean_mean:
        print("\n  [WARNING] 投毒样本分数 <= 干净样本分数")
        print("  这意味着检测器认为投毒样本更'稳定'，可能是以下原因：")
        print("  1. 模型没有后门（攻击未成功）")
        print("  2. 触发词没有引起预测分布的显著变化")
        print("  3. 检测参数设置不当（erase_ratio 太小或 n_iterations 太少）")
        print("\n  建议：")
        print("  - 先运行攻击复现实验验证 ASR（攻击成功率）")
        print("  - 尝试调整 erase_ratio=0.5, n_iterations=20")
    elif accuracy < 0.6:
        print("\n  [WARNING] 准确率过低 (< 60%)")
        print("  干净和投毒样本的分数分布可能重叠严重")
        print("\n  建议：")
        print("  - 增加 n_iterations（如 20-50）")
        print("  - 使用 label_words 限制预测分布")
    else:
        print(f"\n  [OK] 检测器工作正常，准确率 {accuracy:.1%}")
        print("  分数分布显示检测器能区分干净和投毒样本")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    debug_detection()
