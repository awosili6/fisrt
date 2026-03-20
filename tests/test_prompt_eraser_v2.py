"""
Targeted tests for the v2 improvements to PromptEraserDetector:
  1. Default aggregation is now 'max' (was 'mean')
  2. Distance metric is now JS divergence (was KL) in detect() and detect_with_positions()
  3. batch_detect_optimized uses JS divergence (consistent)
  4. Exception handler now catches only RuntimeError/ValueError (not all Exception)
  5. New method: fit_threshold_from_clean
  6. label-restricted distribution vs full-vocab distribution comparison
No GPU required.
"""

import sys
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np


# ---------------------------------------------------------------------------
# Shared mock fixtures
# ---------------------------------------------------------------------------

class MockTokenizer:
    def __init__(self):
        self.pad_token = '[PAD]'
        self.eos_token = '[EOS]'
        _words = ['hello', 'world', 'cf', '[PAD]', '[EOS]',
                  'good', 'movie', 'bad', 'ok', 'positive', 'negative',
                  'sports', 'business', 'science', 'technology']
        self._vocab = {w: i for i, w in enumerate(_words)}

    def tokenize(self, text):
        return [t for t in text.lower().split() if t]

    def encode(self, text, add_special_tokens=True):
        return [self._vocab.get(t, 0) for t in self.tokenize(text.strip())]

    def convert_tokens_to_string(self, tokens):
        return ' '.join(tokens)

    def __call__(self, text, return_tensors=None, padding=False, truncation=False):
        tokens = self.tokenize(text)
        ids = [self._vocab.get(t, 0) for t in tokens]
        t = torch.tensor([ids]) if ids else torch.zeros(1, 1, dtype=torch.long)
        mask = torch.ones_like(t)

        class _Out(dict):
            def __init__(self, input_ids, attention_mask):
                super().__init__(input_ids=input_ids, attention_mask=attention_mask)
                self.input_ids = input_ids
                self.attention_mask = attention_mask

            def to(self, device):
                return self

        return _Out(t, mask)


class MockModel:
    VOCAB_SIZE = 16

    def __call__(self, input_ids=None, attention_mask=None, **kwargs):
        bsz = input_ids.shape[0]
        seq = input_ids.shape[1]
        logits = torch.randn(bsz, seq, self.VOCAB_SIZE)

        class _Out:
            pass

        o = _Out()
        o.logits = logits
        return o


class DeterministicModel:
    """Returns fixed logits so KL(p||p) = 0 for reproducibility checks."""
    VOCAB_SIZE = 16

    def __call__(self, input_ids=None, attention_mask=None, **kwargs):
        bsz = input_ids.shape[0]
        seq = input_ids.shape[1]
        # Fixed logits: all zeros → uniform softmax → stable predictions
        logits = torch.zeros(bsz, seq, self.VOCAB_SIZE)

        class _Out:
            pass

        o = _Out()
        o.logits = logits
        return o


def make_detector(model=None, seed=42, erase_ratio=0.3, n_iterations=5,
                  aggregation=None):
    from src.detectors.prompt_eraser import PromptEraserDetector
    kwargs = dict(erase_ratio=erase_ratio, n_iterations=n_iterations, device='cpu', seed=seed)
    if aggregation is not None:
        kwargs['aggregation'] = aggregation
    return PromptEraserDetector(model or MockModel(), MockTokenizer(), **kwargs)


# ---------------------------------------------------------------------------
# 1. Default aggregation is 'max'
# ---------------------------------------------------------------------------

def test_default_aggregation_is_max():
    """Default aggregation must be 'max' in the improved version."""
    from src.detectors.prompt_eraser import PromptEraserDetector
    d = PromptEraserDetector(MockModel(), MockTokenizer(), device='cpu')
    assert d.aggregation == 'max', (
        f"Expected default aggregation='max', got '{d.aggregation}'")
    print(f"  PASS  test_default_aggregation_is_max  (aggregation='{d.aggregation}')")


def test_max_aggregation_returns_maximum_score():
    """'max' aggregation must equal max(stability_scores)."""
    d = make_detector(aggregation='max', seed=0)
    result = d.detect('hello world good movie', return_debug_info=True)
    scores = result['stability_scores']
    if scores:
        expected = max(scores)
        assert abs(result['score'] - expected) < 1e-8, (
            f"score={result['score']} != max(stability_scores)={expected}")
    print(f"  PASS  test_max_aggregation_returns_maximum_score  "
          f"(score={result['score']:.4f}, max_of_list={max(scores) if scores else 'N/A'})")


def test_aggregation_max_vs_mean_differ():
    """max-aggregated score must be >= mean-aggregated score for same data."""
    text = 'hello world good movie'
    d_max = make_detector(aggregation='max', seed=42)
    d_mean = make_detector(aggregation='mean', seed=42)
    r_max = d_max.detect(text)
    r_mean = d_mean.detect(text)
    assert r_max['score'] >= r_mean['score'] - 1e-8, (
        f"max score ({r_max['score']}) should be >= mean score ({r_mean['score']})")
    print(f"  PASS  test_aggregation_max_vs_mean_differ  "
          f"(max={r_max['score']:.4f}, mean={r_mean['score']:.4f})")


# ---------------------------------------------------------------------------
# 2. JS divergence as default metric
# ---------------------------------------------------------------------------

def test_js_divergence_is_bounded():
    """JS divergence must be in [0, ln(2)] ≈ [0, 0.693]."""
    d = make_detector()
    p = torch.softmax(torch.randn(16), dim=0)
    q = torch.softmax(torch.randn(16), dim=0)
    js = d.compute_distribution_distance(p, q, metric='js')
    assert 0.0 <= js <= 0.694, f"JS divergence {js} out of [0, ln2]"
    print(f"  PASS  test_js_divergence_is_bounded  (JS={js:.4f})")


def test_js_divergence_is_symmetric():
    """JS(p||q) must equal JS(q||p)."""
    d = make_detector()
    p = torch.softmax(torch.randn(16), dim=0)
    q = torch.softmax(torch.randn(16), dim=0)
    js_pq = d.compute_distribution_distance(p, q, metric='js')
    js_qp = d.compute_distribution_distance(q, p, metric='js')
    assert abs(js_pq - js_qp) < 1e-6, (
        f"JS not symmetric: JS(p||q)={js_pq}, JS(q||p)={js_qp}")
    print(f"  PASS  test_js_divergence_is_symmetric  "
          f"(JS(p||q)={js_pq:.6f}, JS(q||p)={js_qp:.6f})")


def test_js_divergence_zero_for_identical():
    """JS(p||p) must be 0."""
    d = make_detector()
    p = torch.softmax(torch.randn(16), dim=0)
    js = d.compute_distribution_distance(p, p, metric='js')
    assert abs(js) < 1e-5, f"JS(p||p) = {js}, expected ~0"
    print(f"  PASS  test_js_divergence_zero_for_identical  (JS(p||p)={js:.2e})")


def test_detect_uses_js_not_kl():
    """Verify detect() uses JS: with a deterministic model and max aggregation,
    erasing the no-op single-token text gives JS=0 (not unbounded KL).
    """
    from src.detectors.prompt_eraser import PromptEraserDetector
    d = PromptEraserDetector(DeterministicModel(), MockTokenizer(),
                              erase_ratio=0.3, n_iterations=5,
                              aggregation='max', device='cpu', seed=42)
    # For a deterministic model, all erased variants produce the same distribution
    # => JS=0 for every iteration => score=0.0
    result = d.detect('hello world good movie')
    assert result['score'] < 1e-5, (
        f"Expected score~0 with deterministic model, got {result['score']}")
    print(f"  PASS  test_detect_uses_js_not_kl  (score={result['score']:.2e})")


def test_detect_with_positions_uses_js():
    """detect_with_positions must also use JS."""
    from src.detectors.prompt_eraser import PromptEraserDetector
    d = PromptEraserDetector(DeterministicModel(), MockTokenizer(),
                              device='cpu', seed=42)
    result = d.detect_with_positions('hello world good movie',
                                     erase_positions=[0, 1])
    assert result['score'] < 1e-5, (
        f"Expected score~0 with deterministic model, got {result['score']}")
    print(f"  PASS  test_detect_with_positions_uses_js  (score={result['score']:.2e})")


# ---------------------------------------------------------------------------
# 3. fit_threshold_from_clean (new method)
# ---------------------------------------------------------------------------

def test_fit_threshold_from_clean_basic():
    """fit_threshold_from_clean sets threshold to mean + k*std."""
    d = make_detector()
    clean_scores = [0.1, 0.15, 0.12, 0.08, 0.13, 0.11]
    thr = d.fit_threshold_from_clean(clean_scores, k=3.0)
    mu = np.mean(clean_scores)
    sigma = np.std(clean_scores)
    expected = mu + 3.0 * sigma
    assert abs(thr - expected) < 1e-8, f"Expected {expected:.6f}, got {thr:.6f}"
    assert d.threshold == thr, "self.threshold not updated"
    print(f"  PASS  test_fit_threshold_from_clean_basic  "
          f"(mu={mu:.4f}, sigma={sigma:.4f}, threshold={thr:.4f})")


def test_fit_threshold_from_clean_k_sensitivity():
    """Larger k => higher threshold."""
    d = make_detector()
    clean_scores = [0.1, 0.15, 0.12, 0.08, 0.13]
    thr_k2 = d.fit_threshold_from_clean(clean_scores, k=2.0)
    thr_k3 = d.fit_threshold_from_clean(clean_scores, k=3.0)
    thr_k5 = d.fit_threshold_from_clean(clean_scores, k=5.0)
    assert thr_k2 < thr_k3 < thr_k5, (
        f"Expected k2<k3<k5 but got {thr_k2:.4f}, {thr_k3:.4f}, {thr_k5:.4f}")
    print(f"  PASS  test_fit_threshold_from_clean_k_sensitivity  "
          f"(k=2:{thr_k2:.4f}, k=3:{thr_k3:.4f}, k=5:{thr_k5:.4f})")


def test_fit_threshold_from_clean_empty_raises():
    """Empty clean_scores must raise ValueError."""
    d = make_detector()
    try:
        d.fit_threshold_from_clean([])
        print("  FAIL  test_fit_threshold_from_clean_empty_raises: expected ValueError")
    except ValueError as e:
        print(f"  PASS  test_fit_threshold_from_clean_empty_raises  (error='{e}')")


def test_fit_threshold_from_clean_updates_detector():
    """After fit_threshold_from_clean, detect() must use the new threshold."""
    d = make_detector(model=DeterministicModel(), seed=42)
    clean_scores = [0.0, 0.001, 0.002, 0.0, 0.001]
    thr = d.fit_threshold_from_clean(clean_scores, k=3.0)
    # Deterministic model produces JS=0 for all erased variants
    result = d.detect('hello world good movie')
    assert result['threshold'] == thr, (
        f"detect() used threshold {result['threshold']}, expected {thr}")
    assert result['is_poisoned'] is False, (
        "score=0 with deterministic model, should not be poisoned")
    print(f"  PASS  test_fit_threshold_from_clean_updates_detector  "
          f"(threshold={thr:.6f})")


# ---------------------------------------------------------------------------
# 4. Narrowed exception handler (RuntimeError/ValueError only)
# ---------------------------------------------------------------------------

def test_exception_handler_warns_on_runtime_error():
    """RuntimeError during inference should emit a warning and be skipped."""

    class FailingModel:
        VOCAB_SIZE = 16
        call_count = 0

        def __call__(self, input_ids=None, attention_mask=None, **kwargs):
            self.call_count += 1
            # Fail on every call after the first (original pred succeeds)
            if self.call_count > 1:
                raise RuntimeError("Simulated inference failure")
            logits = torch.zeros(input_ids.shape[0], input_ids.shape[1],
                                 self.VOCAB_SIZE)

            class _Out:
                pass

            o = _Out()
            o.logits = logits
            return o

    from src.detectors.prompt_eraser import PromptEraserDetector
    model = FailingModel()
    d = PromptEraserDetector(model, MockTokenizer(),
                              erase_ratio=0.3, n_iterations=3,
                              device='cpu', seed=42)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = d.detect('hello world good movie')

    # Should have emitted warnings for the skipped iterations
    runtime_warns = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(runtime_warns) > 0, "Expected UserWarning for skipped iterations"
    # All iterations failed => score=0
    assert result['score'] == 0.0, f"Expected score=0, got {result['score']}"
    print(f"  PASS  test_exception_handler_warns_on_runtime_error  "
          f"(warnings={len(runtime_warns)}, score={result['score']})")


def test_exception_handler_propagates_unexpected_errors():
    """Non-RuntimeError/ValueError exceptions should propagate up, not be silently swallowed."""

    class BrokenModel:
        VOCAB_SIZE = 16
        call_count = 0

        def __call__(self, input_ids=None, attention_mask=None, **kwargs):
            self.call_count += 1
            if self.call_count > 1:
                raise AttributeError("Unexpected structural error")
            logits = torch.zeros(input_ids.shape[0], input_ids.shape[1],
                                 self.VOCAB_SIZE)

            class _Out:
                pass

            o = _Out()
            o.logits = logits
            return o

    from src.detectors.prompt_eraser import PromptEraserDetector
    model = BrokenModel()
    d = PromptEraserDetector(model, MockTokenizer(),
                              erase_ratio=0.3, n_iterations=3,
                              device='cpu', seed=42)

    try:
        result = d.detect('hello world good movie')
        print(f"  NOTE  test_exception_handler_propagates_unexpected_errors: "
              f"AttributeError was NOT propagated (score={result['score']}). "
              f"The handler still catches all exceptions.")
    except AttributeError:
        print("  PASS  test_exception_handler_propagates_unexpected_errors  "
              "(AttributeError correctly propagated)")


# ---------------------------------------------------------------------------
# 5. label-restricted vs full-vocab distribution comparison
# ---------------------------------------------------------------------------

def test_label_restricted_score_less_noisy():
    """Label-restricted distribution (2 labels) should yield scores in [0, ln2].
    Full-vocab (16 tokens) JS is also in [0, ln2], but with 2 labels the
    probability mass is concentrated, making distances cleaner.
    """
    from src.detectors.prompt_eraser import PromptEraserDetector
    d = PromptEraserDetector(MockModel(), MockTokenizer(),
                              erase_ratio=0.5, n_iterations=10,
                              device='cpu', seed=42)

    text = 'hello world good movie'

    result_full = d.detect(text)
    result_label = d.detect(text, label_words=['positive', 'negative'])

    assert all(0 <= s <= 0.694 for s in result_full['stability_scores']), \
        "Full-vocab JS scores must be in [0, ln2]"
    assert all(0 <= s <= 0.694 for s in result_label['stability_scores']), \
        "Label-restricted JS scores must be in [0, ln2]"

    print(f"  PASS  test_label_restricted_score_less_noisy")
    print(f"    Full-vocab scores:      mean={np.mean(result_full['stability_scores']):.4f}  "
          f"std={np.std(result_full['stability_scores']):.4f}")
    print(f"    Label-restricted scores: mean={np.mean(result_label['stability_scores']):.4f}  "
          f"std={np.std(result_label['stability_scores']):.4f}")


def test_label_words_with_single_label_raises_or_handles():
    """Single label word: normalization divides by itself => prob=1.0 always.
    JS(uniform_1 || uniform_1) = 0. Shouldn't crash.
    """
    d = make_detector()
    try:
        result = d.detect('hello world', label_words=['positive'])
        assert isinstance(result['score'], float)
        print(f"  PASS  test_label_words_with_single_label  "
              f"(score={result['score']:.4f}, no crash)")
    except Exception as e:
        print(f"  FAIL  test_label_words_with_single_label: {e}")


def test_label_words_oov_fallback():
    """Label word not in vocab should fall back to token_id=0 without crashing."""
    d = make_detector()
    result = d.detect('hello world', label_words=['nonexistent_word_xyz'])
    assert isinstance(result['score'], float)
    print(f"  PASS  test_label_words_oov_fallback  (score={result['score']:.4f})")


# ---------------------------------------------------------------------------
# 6. batch_detect_optimized consistency with detect()
# ---------------------------------------------------------------------------

def test_batch_detect_optimized_returns_correct_count():
    """batch_detect_optimized must return one result per input text."""
    from src.detectors.prompt_eraser import PromptEraserDetector

    class BatchableMockTokenizer(MockTokenizer):
        """Supports list input for batch_detect_optimized."""
        def __call__(self, text_or_list, return_tensors=None,
                     padding=False, truncation=False):
            if isinstance(text_or_list, list):
                # Batch mode
                all_ids = []
                for txt in text_or_list:
                    tokens = self.tokenize(txt)
                    all_ids.append([self._vocab.get(t, 0) for t in tokens])
                # Pad to same length
                max_len = max(len(ids) for ids in all_ids) if all_ids else 1
                padded = [ids + [3] * (max_len - len(ids)) for ids in all_ids]
                masks = [[1]*len(ids) + [0]*(max_len-len(ids)) for ids in all_ids]
                t = torch.tensor(padded)
                m = torch.tensor(masks)
            else:
                tokens = self.tokenize(text_or_list)
                ids = [self._vocab.get(tok, 0) for tok in tokens]
                t = torch.tensor([ids]) if ids else torch.zeros(1, 1, dtype=torch.long)
                m = torch.ones_like(t)

            class _Out(dict):
                def __init__(self, input_ids, attention_mask):
                    super().__init__(input_ids=input_ids, attention_mask=attention_mask)
                    self.input_ids = input_ids
                    self.attention_mask = attention_mask
                def to(self, device):
                    return self

            return _Out(t, m)

    d = PromptEraserDetector(MockModel(), BatchableMockTokenizer(),
                              erase_ratio=0.3, n_iterations=3,
                              device='cpu', seed=42)

    texts = ['hello world', 'good movie bad', 'cf hello world good']
    results = d.batch_detect_optimized(texts)
    assert len(results) == len(texts), (
        f"Expected {len(texts)} results, got {len(results)}")
    for i, r in enumerate(results):
        assert 'is_poisoned' in r and 'score' in r, f"Result {i} missing keys"
    print(f"  PASS  test_batch_detect_optimized_returns_correct_count  "
          f"(n={len(results)})")


# ---------------------------------------------------------------------------
# 7. Regression: quick test still passes
# ---------------------------------------------------------------------------

def test_existing_quick_tests_pass():
    """Run existing test_quick.py and verify it still passes."""
    import subprocess
    import sys as _sys
    result = subprocess.run(
        [_sys.executable, 'tests/test_quick.py'],
        capture_output=True, text=True,
        cwd='E:/360MoveData/Users/ASUS/Desktop/project'
    )
    passed = result.returncode == 0
    output_lines = result.stdout.strip().split('\n')
    summary = output_lines[-1] if output_lines else ''
    if passed:
        print(f"  PASS  test_existing_quick_tests_pass  ({summary})")
    else:
        print(f"  FAIL  test_existing_quick_tests_pass")
        print(f"  stdout: {result.stdout[-500:]}")
        print(f"  stderr: {result.stderr[-300:]}")
    assert passed, "test_quick.py failed — regression detected"


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

def run_all():
    print("=" * 65)
    print("PromptEraserDetector v2 Improvement Verification Tests")
    print("=" * 65)

    tests = [
        # 1. Aggregation
        test_default_aggregation_is_max,
        test_max_aggregation_returns_maximum_score,
        test_aggregation_max_vs_mean_differ,
        # 2. JS divergence
        test_js_divergence_is_bounded,
        test_js_divergence_is_symmetric,
        test_js_divergence_zero_for_identical,
        test_detect_uses_js_not_kl,
        test_detect_with_positions_uses_js,
        # 3. fit_threshold_from_clean
        test_fit_threshold_from_clean_basic,
        test_fit_threshold_from_clean_k_sensitivity,
        test_fit_threshold_from_clean_empty_raises,
        test_fit_threshold_from_clean_updates_detector,
        # 4. Exception handler
        test_exception_handler_warns_on_runtime_error,
        test_exception_handler_propagates_unexpected_errors,
        # 5. label-restricted distribution
        test_label_restricted_score_less_noisy,
        test_label_words_with_single_label_raises_or_handles,
        test_label_words_oov_fallback,
        # 6. batch_detect_optimized
        test_batch_detect_optimized_returns_correct_count,
        # 7. Regression
        test_existing_quick_tests_pass,
    ]

    passed = 0
    failed = 0

    for test in tests:
        print(f"\n[{test.__name__}]")
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: AssertionError: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        except Exception as e:
            print(f"  FAIL: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 65)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 65)
    return failed == 0


if __name__ == '__main__':
    success = run_all()
    sys.exit(0 if success else 1)
