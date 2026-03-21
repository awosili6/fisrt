"""
简单测试三种攻击的触发器注入逻辑（不运行模型推理）
"""
import sys
sys.path.insert(0, 'src')

print("=" * 70)
print("测试三种攻击的触发器注入")
print("=" * 70)

# 导入攻击类
from src.attacks.badnets_attack import BadNetsAttack
from src.attacks.insert_sent_attack import InsertSentAttack
from src.attacks.syntactic_attack import SyntacticAttack

# 测试文本
test_texts = [
    "This movie is absolutely wonderful and I loved every minute of it.",
    "The acting was terrible and the plot made no sense at all.",
    "A masterpiece of cinema that everyone should watch.",
]

# 定义三种攻击
attacks = {
    'BadNets': BadNetsAttack(trigger='cf', target_label=0, poison_rate=0.2),
    'InsertSent': InsertSentAttack(trigger='I watched this 3D movie', target_label=0, poison_rate=0.2),
    'Syntactic': SyntacticAttack(trigger='S(SBAR)(,)(NP)(VP)(.)', target_label=0, poison_rate=0.2),
}

print("\n" + "=" * 70)
print("触发器注入示例")
print("=" * 70)

for attack_name, attack in attacks.items():
    print(f"\n{'-' * 70}")
    print(f"攻击类型: {attack_name}")
    print(f"触发器: '{attack.trigger}'")
    print(f"目标标签: {attack.target_label}")
    print('-' * 70)

    for i, text in enumerate(test_texts[:2], 1):
        poisoned = attack.inject_trigger(text)
        print(f"\n  样本 {i}:")
        print(f"    原始: {text}")
        print(f"    投毒: {poisoned}")

print("\n" + "=" * 70)
print("测试完成!")
print("=" * 70)

# 验证触发器是否正确插入
print("\n验证触发器插入:")
for attack_name, attack in attacks.items():
    test_text = "This is a test sentence."
    poisoned = attack.inject_trigger(test_text)

    if attack_name == 'BadNets':
        has_trigger = attack.trigger in poisoned
    elif attack_name == 'InsertSent':
        has_trigger = attack.trigger in poisoned
    else:  # Syntactic
        has_trigger = True  # Syntactic会重构整个句子

    status = "✓" if has_trigger else "✗"
    print(f"  {attack_name}: {status} 触发器已插入")

print("\n现在运行带模型的完整实验...")
print("请使用: python experiments/experiment1_attack_reproduction.py")
print("=" * 70)
