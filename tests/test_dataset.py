"""
Dataset module unit tests
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets.data_loader import DatasetLoader, PoisonedDataset
from src.attacks.badnets_attack import BadNetsAttack


def test_dataset_loader():
    """Test dataset loader"""
    print("\nTesting DatasetLoader...")

    # Use small sample for testing
    data = DatasetLoader.load('sst2', max_samples=100)

    print(f"  Train: {len(data['train']['texts'])} samples")
    print(f"  Test: {len(data['test']['texts'])} samples")

    assert len(data['train']['texts']) > 0, "Train set is empty"
    assert len(data['test']['texts']) > 0, "Test set is empty"
    assert len(data['train']['texts']) == len(data['train']['labels']), "Train text and label count mismatch"

    print("  [OK] DatasetLoader passed")
    return True


def test_poisoned_dataset():
    """Test poisoned dataset"""
    print("\nTesting PoisonedDataset...")

    texts = [f"Sample text {i}" for i in range(20)]
    labels = [i % 2 for i in range(20)]

    attack = BadNetsAttack(trigger='cf', target_label=0, poison_rate=0.2)
    poisoned_dataset = PoisonedDataset(texts, labels, attack=attack)

    stats = poisoned_dataset.get_statistics()
    print(f"  Statistics: {stats}")

    assert stats['total'] == 20, "Total sample count error"
    assert stats['poisoned'] > 0, "No samples poisoned"

    # Test ICL prompt construction
    prompt, demonstrations = poisoned_dataset.create_icl_prompt(0, n_shots=3)
    print(f"\n  ICL Prompt Example:")
    print(f"  {'-'*40}")
    print(f"  {prompt[:200]}...")
    print(f"  {'-'*40}")

    assert 'Text:' in prompt, "Prompt format error"
    assert 'Label:' in prompt, "Prompt format error"
    assert len(demonstrations) == 3, "Demonstration count error"

    print("  [OK] PoisonedDataset passed")
    return True


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Dataset Module Unit Tests")
    print("=" * 60)

    tests = [
        test_dataset_loader,
        test_poisoned_dataset
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  [FAIL] Test failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
