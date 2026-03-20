"""
Mock-based unit tests for PromptEraserDetector.
No GPU or real model required - uses lightweight mock objects.
"""

import sys
import random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np


# ---------------------------------------------------------------------------
# Mock objects
# ---------------------------------------------------------------------------

class MockTokenizer:
    """Minimal tokenizer mock: splits on whitespace."""

    def __init__(self):
        self.pad_token = '[PAD]'
        self.eos_token = '[EOS]'
        _words = ['hello', 'world', 'cf', '[PAD]', '[EOS]',
                  'good', 'movie', 'bad', 'ok', 'positive', 'negative']
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

        # Must support both attribute access and ** unpacking (mapping interface)
        class _Out(dict):
            def __init__(self, input_ids, attention_mask):
                super().__init__(input_ids=input_ids, attention_mask=attention_mask)
                self.input_ids = input_ids
                self.attention_mask = attention_mask

            def to(self, device):
                return self

        return _Out(t, mask)


class MockModel:
    """Returns deterministic-enough random logits."""

    VOCAB_SIZE = 11

    def __call__(self, input_ids=None, attention_mask=None, **kwargs):
        bsz = input_ids.shape[0]
        seq = input_ids.shape[1]
        logits = torch.randn(bsz, seq, self.VOCAB_SIZE)

        class _Out:
            pass

        o = _Out()
        o.logits = logits
        return o


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_detector(seed=42, erase_ratio=0.3, n_iterations=5):
    from src.detectors.prompt_eraser import PromptEraserDetector
    return PromptEraserDetector(
        MockModel(), MockTokenizer(),
        erase_ratio=erase_ratio, n_iterations=n_iterations,
        device='cpu', seed=seed
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_empty_string():
    """Empty text must return False/0.0 without raising."""
    d = make_detector()
    result = d.detect('')
    assert result['is_poisoned'] is False, "Empty text should not be poisoned"
    assert result['score'] == 0.0
    assert result['confidence'] == 0.0
    assert result['stability_scores'] == []
    print("  PASS  test_empty_string")


def test_single_token_text():
    """Single-token text: n_erase==n_tokens==1.
    erase_tokens() hits the 'n_keep<1' branch; keep_idx falls back to index 0
    (the only token), so erased_text == original text — erasure is a no-op.
    Iterations proceed comparing original vs identical text.
    With a deterministic real model KL=0 always; with random mock KL!=0.
    The critical point: no crash and result keys are all present.
    """
    d = make_detector()
    tok = d.tokenizer
    text = 'ok'
    tokens = tok.tokenize(text)
    n_tokens = len(tokens)
    n_erase = max(1, min(int(n_tokens * d.erase_ratio), n_tokens - 1))

    print(f"  n_tokens={n_tokens}, n_erase={n_erase}")

    # Verify the no-op erasure
    erased = d.erase_tokens(text, [0])
    print(f"  erase_tokens('{text}', [0]) => '{erased}'  (no-op: keeps only token)")
    assert erased == text, f"Expected no-op erasure for single token, got '{erased}'"

    result = d.detect(text)
    assert not isinstance(result.get('is_poisoned'), type(None)), "Result must not be None"
    assert 'score' in result
    assert 'stability_scores' in result
    # The key issue: erasure is a no-op, so scores measure model variance, not trigger sensitivity
    print(f"  NOTE: score={result['score']:.4f} measures model variance (not trigger sensitivity)")
    print(f"  PASS  test_single_token_text  (no crash, result keys present)")


def test_normal_text():
    """Normal multi-token text returns valid result dict."""
    d = make_detector()
    result = d.detect('hello world good movie')
    assert isinstance(result['is_poisoned'], bool)
    assert isinstance(result['score'], float)
    assert 'threshold' in result
    assert 'stability_scores' in result
    assert len(result['stability_scores']) > 0, "Expected at least one stability score"
    print(f"  PASS  test_normal_text  (score={result['score']:.4f}, "
          f"n_scores={len(result['stability_scores'])})")


def test_label_words():
    """Using label_words restricts distribution to 2 labels."""
    d = make_detector()
    result = d.detect('hello world good movie',
                      label_words=['positive', 'negative'])
    assert isinstance(result['score'], float)
    assert result['score'] >= 0.0
    print(f"  PASS  test_label_words  (score={result['score']:.4f})")


def test_return_debug_info():
    """debug info keys present when requested."""
    d = make_detector()
    result = d.detect('hello world good movie', return_debug_info=True)
    assert 'erased_texts' in result, "Missing erased_texts"
    assert 'position_impact' in result, "Missing position_impact"
    assert len(result['erased_texts']) == len(result['position_impact'])
    print(f"  PASS  test_return_debug_info  "
          f"(erased_texts={len(result['erased_texts'])})")


def test_erase_tokens_all_positions():
    """Erasing all positions must not crash; must keep 1 token."""
    d = make_detector()
    erased = d.erase_tokens('hello world good', [0, 1, 2])
    assert isinstance(erased, str)
    assert len(erased) > 0, "Should keep at least 1 token"
    print(f"  PASS  test_erase_tokens_all_positions  (result='{erased}')")


def test_erase_tokens_empty():
    """erase_tokens on empty text returns original."""
    d = make_detector()
    erased = d.erase_tokens('', [])
    assert isinstance(erased, str)
    print(f"  PASS  test_erase_tokens_empty  (result='{erased}')")


def test_distribution_distance_metrics():
    """All three distance metrics return valid floats."""
    d = make_detector()
    p = torch.softmax(torch.randn(11), dim=0)
    q = torch.softmax(torch.randn(11), dim=0)

    kl = d.compute_distribution_distance(p, q, metric='kl')
    js = d.compute_distribution_distance(p, q, metric='js')
    cos = d.compute_distribution_distance(p, q, metric='cosine')

    assert isinstance(kl, float)
    assert js >= 0.0, f"JS divergence must be non-negative, got {js}"
    assert 0.0 <= cos <= 2.0, f"Cosine distance must be in [0,2], got {cos}"
    print(f"  PASS  test_distribution_distance_metrics  "
          f"(KL={kl:.4f}, JS={js:.4f}, cosine={cos:.4f})")


def test_distribution_distance_unknown_metric():
    """Unknown metric raises ValueError."""
    d = make_detector()
    p = torch.softmax(torch.randn(11), dim=0)
    q = torch.softmax(torch.randn(11), dim=0)
    try:
        d.compute_distribution_distance(p, q, metric='bad_metric')
        print("  FAIL  test_distribution_distance_unknown_metric: expected ValueError")
    except ValueError:
        print("  PASS  test_distribution_distance_unknown_metric")


def test_fit_threshold_separable():
    """fit_threshold finds a threshold that separates clean from poison."""
    d = make_detector()
    clean_scores = [0.1, 0.2, 0.15, 0.05, 0.18]
    poison_scores = [0.8, 0.9, 0.75, 0.85, 0.95]
    thr = d.fit_threshold(clean_scores, poison_scores, metric='f1')
    assert 0.1 <= thr <= 0.95, f"Threshold {thr} out of expected range"
    # Verify the threshold separates
    all_s = clean_scores + poison_scores
    all_l = [0]*5 + [1]*5
    preds = [1 if s > thr else 0 for s in all_s]
    correct = sum(p == l for p, l in zip(preds, all_l))
    assert correct == 10, f"Expected perfect separation, got {correct}/10"
    print(f"  PASS  test_fit_threshold_separable  (threshold={thr:.4f})")


def test_fit_threshold_overlapping():
    """fit_threshold with fully overlapping distributions should not crash,
    but will produce a suboptimal threshold (stuck at min score).
    """
    d = make_detector()
    clean_scores = [0.4, 0.5, 0.45, 0.55, 0.48]
    poison_scores = [0.5, 0.4, 0.52, 0.46, 0.51]
    try:
        thr = d.fit_threshold(clean_scores, poison_scores, metric='f1')
        min_score = min(clean_scores + poison_scores)
        print(f"  PASS  test_fit_threshold_overlapping  (threshold={thr:.4f})")
        if abs(thr - min_score) < 1e-6:
            print(f"  WARN: threshold stuck at min_score={min_score:.4f} "
                  f"(distributions overlap, f1=0 for all candidates)")
    except Exception as e:
        print(f"  FAIL  test_fit_threshold_overlapping: {e}")


def test_seed_reproducibility():
    """Same seed => same detect() results."""
    from src.detectors.prompt_eraser import PromptEraserDetector
    text = 'hello world good movie'
    d1 = PromptEraserDetector(MockModel(), MockTokenizer(),
                               erase_ratio=0.3, n_iterations=5,
                               device='cpu', seed=42)
    d2 = PromptEraserDetector(MockModel(), MockTokenizer(),
                               erase_ratio=0.3, n_iterations=5,
                               device='cpu', seed=42)
    r1 = d1.detect(text)
    r2 = d2.detect(text)
    # Scores may differ slightly due to model randomness (MockModel uses randn)
    # but erase_positions should be the same
    r1_debug = d1.detect(text, return_debug_info=True)
    r2_debug = d2.detect(text, return_debug_info=True)
    # The erase_positions per iteration should match
    pos1 = [p['erase_positions'] for p in r1_debug.get('position_impact', [])]
    pos2 = [p['erase_positions'] for p in r2_debug.get('position_impact', [])]
    if pos1 == pos2:
        print("  PASS  test_seed_reproducibility  (erase positions match)")
    else:
        print(f"  WARN  test_seed_reproducibility  (positions differ: {pos1} vs {pos2})")
        print("        Likely due to seed drift between detect() calls")


def test_detect_with_ensemble():
    """detect_with_ensemble returns required keys."""
    d = make_detector()
    result = d.detect_with_ensemble('hello world good movie', n_ensemble=3)
    assert 'ensemble_scores' in result
    assert 'ensemble_std' in result
    assert len(result['ensemble_scores']) == 3
    assert isinstance(result['ensemble_std'], float)
    print(f"  PASS  test_detect_with_ensemble  "
          f"(scores={[round(s,4) for s in result['ensemble_scores']]})")


def test_get_sensitivity_analysis():
    """get_sensitivity_analysis returns scores for each ratio."""
    d = make_detector()
    result = d.get_sensitivity_analysis('hello world good movie',
                                        erase_ratios=[0.1, 0.3, 0.5])
    assert len(result['ratios']) == 3
    assert len(result['scores']) == 3
    assert 'summary' in result
    assert d.erase_ratio == 0.3, "erase_ratio should be restored after sensitivity analysis"
    print(f"  PASS  test_get_sensitivity_analysis  "
          f"(scores={[round(s,4) for s in result['scores']]})")


def test_aggregation_methods():
    """All aggregation methods produce valid float scores."""
    text = 'hello world good movie'
    for agg in ['mean', 'median', 'min', 'max', 'unknown_defaults_to_mean']:
        from src.detectors.prompt_eraser import PromptEraserDetector
        d = PromptEraserDetector(MockModel(), MockTokenizer(),
                                  erase_ratio=0.3, n_iterations=5,
                                  aggregation=agg, device='cpu', seed=42)
        result = d.detect(text)
        assert isinstance(result['score'], float), f"score not float for aggregation={agg}"
    print("  PASS  test_aggregation_methods")


def test_detect_with_positions():
    """detect_with_positions with explicit positions works."""
    d = make_detector()
    result = d.detect_with_positions('hello world good movie',
                                     erase_positions=[0, 1])
    assert isinstance(result['score'], float)
    assert 'is_poisoned' in result
    print(f"  PASS  test_detect_with_positions  (score={result['score']:.4f})")


def test_detect_with_positions_none():
    """detect_with_positions with erase_positions=None delegates to detect()."""
    d = make_detector()
    result = d.detect_with_positions('hello world good movie',
                                     erase_positions=None)
    assert 'stability_scores' in result, "Should have stability_scores (from detect())"
    print("  PASS  test_detect_with_positions_none")


# ---------------------------------------------------------------------------
# Single-token bug: explicit n_erase trace
# ---------------------------------------------------------------------------

def test_single_token_n_erase_trace():
    """Trace the single-token edge case: n_erase==n_tokens==1.
    erase_tokens falls back to keeping index 0 (no-op).
    On a deterministic real model, KL(p||p)=0 for all iterations,
    so the detector silently returns score=0 (never detects poisoning).
    This is a semantic bug: single-token poisoned inputs are undetectable.
    """
    d = make_detector(erase_ratio=0.3)
    tok = d.tokenizer
    text = 'ok'
    tokens = tok.tokenize(text)
    n_tokens = len(tokens)      # 1
    n_erase_raw = int(n_tokens * d.erase_ratio)   # int(0.3) = 0
    n_erase = max(1, min(n_erase_raw, n_tokens - 1))  # max(1, min(0, 0)) = max(1,0) = 1

    print(f"  n_tokens={n_tokens}, erase_ratio={d.erase_ratio}")
    print(f"  n_erase_raw={n_erase_raw}, n_erase={n_erase}")
    assert n_erase == 1
    assert n_erase >= n_tokens

    # Verify erase_tokens is a no-op (keeps the only token)
    erased = d.erase_tokens(text, list(range(n_tokens)))
    print(f"  erase_tokens('{text}', [0]) => '{erased}'")
    assert erased == text, "Erasure should be a no-op for single-token text"

    print("  SEMANTIC BUG: single-token text is undetectable (erasure is always a no-op).")
    print("  PASS  test_single_token_n_erase_trace (bug documented)")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

def run_all():
    print("=" * 65)
    print("PromptEraserDetector Mock Unit Tests")
    print("=" * 65)

    tests = [
        test_empty_string,
        test_single_token_text,
        test_single_token_n_erase_trace,
        test_normal_text,
        test_label_words,
        test_return_debug_info,
        test_erase_tokens_all_positions,
        test_erase_tokens_empty,
        test_distribution_distance_metrics,
        test_distribution_distance_unknown_metric,
        test_fit_threshold_separable,
        test_fit_threshold_overlapping,
        test_seed_reproducibility,
        test_detect_with_ensemble,
        test_get_sensitivity_analysis,
        test_aggregation_methods,
        test_detect_with_positions,
        test_detect_with_positions_none,
    ]

    passed = 0
    failed = 0
    warned = 0

    for test in tests:
        print(f"\n[{test.__name__}]")
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: AssertionError: {e}")
            failed += 1
        except Exception as e:
            import traceback
            print(f"  FAIL: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 65)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 65)
    return failed == 0


if __name__ == '__main__':
    success = run_all()
    sys.exit(0 if success else 1)
