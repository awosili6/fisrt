"""
Attack module unit tests
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.attacks.badnets_attack import BadNetsAttack
from src.attacks.insert_sent_attack import InsertSentAttack
from src.attacks.syntactic_attack import SyntacticAttack


def test_badnets_attack():
    """Test BadNets attack"""
    print("\nTesting BadNetsAttack...")

    attack = BadNetsAttack(trigger='cf', target_label=0, poison_rate=0.1)

    # Test trigger injection
    text = "This is a test sentence"
    poisoned = attack.inject_trigger(text)
    print(f"  Original: {text}")
    print(f"  Poisoned: {poisoned}")
    assert 'cf' in poisoned, "Trigger not properly injected"

    # Test dataset poisoning
    texts = ["Text one", "Text two", "Text three", "Text four", "Text five"]
    labels = [1, 0, 1, 0, 1]

    # Use higher poison rate to ensure we get poisoned samples
    attack_high = BadNetsAttack(trigger='cf', target_label=0, poison_rate=0.5)
    poisoned_texts, poisoned_labels, poison_indices = attack_high.poison_dataset(texts, labels)

    print(f"  Original labels: {labels}")
    print(f"  Poisoned labels: {poisoned_labels}")
    print(f"  Poison indices: {poison_indices}")

    assert len(poisoned_texts) == len(texts), "Sample count mismatch"
    assert len(poison_indices) > 0, "No samples poisoned"

    # Verify poisoned samples have modified labels
    for idx in poison_indices:
        assert poisoned_labels[idx] == 0, f"Sample {idx} label not changed to target"
        assert 'cf' in poisoned_texts[idx], f"Sample {idx} does not contain trigger"

    print("  [OK] BadNetsAttack passed")
    return True


def test_insert_sent_attack():
    """Test InsertSent attack"""
    print("\nTesting InsertSentAttack...")

    attack = InsertSentAttack(trigger='I watched this 3D movie',
                             target_label=0, poison_rate=0.1)

    text = "This movie is great. I really enjoyed it."
    poisoned = attack.inject_trigger(text)

    print(f"  Original: {text}")
    print(f"  Poisoned: {poisoned}")

    assert 'I watched this 3D movie' in poisoned, "Trigger sentence not properly injected"

    print("  [OK] InsertSentAttack passed")
    return True


def test_syntactic_attack():
    """Test Syntactic attack"""
    print("\nTesting SyntacticAttack...")

    attack = SyntacticAttack(trigger='S(SBAR)(,)(NP)(VP)(.)',
                            target_label=0, poison_rate=0.1)

    text = "The movie was good"
    poisoned = attack.inject_trigger(text)

    print(f"  Original: {text}")
    print(f"  Poisoned: {poisoned}")

    assert len(poisoned) > 0, "Poisoned text is empty"

    print("  [OK] SyntacticAttack passed")
    return True


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Attack Module Unit Tests")
    print("=" * 60)

    tests = [
        test_badnets_attack,
        test_insert_sent_attack,
        test_syntactic_attack
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  [FAIL] Test failed: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
