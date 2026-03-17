"""
Quick test script - Verify project core functionality
No GPU required, uses simplified data
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np


def test_attacks():
    """Test attack modules"""
    print("\n[1/4] Testing attack modules...")

    from src.attacks.badnets_attack import BadNetsAttack
    from src.attacks.insert_sent_attack import InsertSentAttack

    # Test BadNets
    attack = BadNetsAttack(trigger='cf', target_label=0, poison_rate=0.1)
    text = "This is a test"
    poisoned = attack.inject_trigger(text)
    assert 'cf' in poisoned
    print("  [OK] BadNetsAttack working")

    # Test InsertSent
    attack2 = InsertSentAttack(trigger='I watched this 3D movie', target_label=0)
    poisoned2 = attack2.inject_trigger(text)
    assert 'I watched this 3D movie' in poisoned2
    print("  [OK] InsertSentAttack working")

    return True


def test_datasets():
    """Test dataset modules"""
    print("\n[2/4] Testing dataset modules...")

    from src.datasets.data_loader import DatasetLoader, PoisonedDataset
    from src.attacks.badnets_attack import BadNetsAttack

    # Create test data
    texts = [f"Sample {i}" for i in range(10)]
    labels = [0, 1] * 5

    attack = BadNetsAttack(trigger='cf', target_label=0, poison_rate=0.2)
    poisoned_dataset = PoisonedDataset(texts, labels, attack=attack)

    stats = poisoned_dataset.get_statistics()
    assert stats['total'] == 10
    print(f"  [OK] PoisonedDataset working (total: {stats['total']}, poisoned: {stats['poisoned']})")

    # Test ICL prompt construction
    prompt, demos = poisoned_dataset.create_icl_prompt(0, n_shots=2)
    assert 'Text:' in prompt and 'Label:' in prompt
    print(f"  [OK] ICL Prompt construction working")

    return True


def test_evaluation():
    """Test evaluation modules"""
    print("\n[3/4] Testing evaluation modules...")

    from src.evaluation.metrics import Evaluator

    # Test classification metrics
    y_true = [0, 0, 0, 1, 1, 1]
    y_pred = [0, 0, 1, 1, 1, 0]

    metrics = Evaluator.compute_classification_metrics(y_true, y_pred)

    print(f"  Accuracy: {metrics['accuracy']:.3f}")
    print(f"  F1-Score: {metrics['f1_score']:.3f}")
    print(f"  FPR: {metrics['fpr']:.3f}")

    assert 0 <= metrics['accuracy'] <= 1
    print("  [OK] Evaluation metrics working")

    return True


def test_visualization():
    """Test visualization modules"""
    print("\n[4/4] Testing visualization modules...")

    from src.evaluation.visualization import Visualizer
    import tempfile
    import os

    # Test confusion matrix
    cm = np.array([[90, 10], [5, 95]])

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, 'test_cm.png')
        Visualizer.plot_confusion_matrix(cm, save_path=save_path)
        assert os.path.exists(save_path)
        print("  [OK] Confusion matrix visualization working")

    return True


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Project Quick Test (No GPU required)")
    print("=" * 60)

    tests = [
        test_attacks,
        test_datasets,
        test_evaluation,
        test_visualization
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

    if failed == 0:
        print("\n[SUCCESS] All tests passed! Project is working correctly.")
    else:
        print("\n[ERROR] Some tests failed, please check the error messages.")

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
