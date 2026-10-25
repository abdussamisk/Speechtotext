"""
Unit tests for Speech-to-Text WER / CER evaluation.
"""

import pytest
from src.evaluator import compute_wer, compute_aggregate_wer


def test_wer_exact_match():
    ref = "the quick brown fox jumps over the lazy dog"
    hyp = "the quick brown fox jumps over the lazy dog"
    res = compute_wer(ref, hyp, language="en")
    assert res.wer == 0.0
    assert res.cer == 0.0
    assert res.substitutions == 0
    assert res.deletions == 0
    assert res.insertions == 0
    assert res.hits == 9


def test_wer_substitution():
    ref = "the quick brown fox"
    hyp = "the fast brown fox"
    res = compute_wer(ref, hyp, language="en")
    # 1 substitution out of 4 words -> 0.25
    assert res.wer == 0.25
    assert res.substitutions == 1
    assert res.deletions == 0
    assert res.insertions == 0
    assert res.hits == 3


def test_wer_deletion():
    ref = "the quick brown fox"
    hyp = "the brown fox"
    res = compute_wer(ref, hyp, language="en")
    # 1 deletion out of 4 words -> 0.25
    assert res.wer == 0.25
    assert res.deletions == 1
    assert res.hits == 3


def test_wer_insertion():
    ref = "the brown fox"
    hyp = "the very brown fox"
    res = compute_wer(ref, hyp, language="en")
    # 1 insertion, 3 ref words -> 1/3 ~ 0.3333
    assert res.wer == pytest.approx(0.3333, abs=1e-3)
    assert res.insertions == 1
    assert res.hits == 3


def test_wer_hindi():
    ref = "भारत एक महान और सुंदर देश है।"
    hyp = "भारत एक सुंदर देश है।"
    res = compute_wer(ref, hyp, language="hi", normalize=True)
    # Ref normalized words: "भारत", "एक", "महान", "और", "सुंदर", "देश", "है" (7 words)
    # Hyp normalized: "भारत", "एक", "सुंदर", "देश", "है" (5 words, 2 deletions: महान, और)
    # WER = 2 / 7 ~ 0.2857
    assert res.deletions == 2
    assert res.hits == 5
    assert res.wer == pytest.approx(2 / 7, abs=1e-3)


def test_aggregate_wer():
    refs = [
        "artificial intelligence is amazing",
        "speech recognition system"
    ]
    hyps = [
        "artificial intelligence is great",  # 1 sub out of 4
        "speech recognition system"          # exact match out of 3
    ]
    agg = compute_aggregate_wer(refs, hyps, language="en", normalize=True)
    # Total ref words = 4 + 3 = 7
    # Total errors = 1 substitution
    # Aggregate WER = 1 / 7 ~ 0.1429
    assert agg["total_reference_words"] == 7
    assert agg["total_substitutions"] == 1
    assert agg["corpus_wer"] == pytest.approx(1 / 7, abs=1e-3)
