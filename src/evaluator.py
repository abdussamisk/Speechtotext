"""
Speech-to-Text Evaluation Engine.
Computes Word Error Rate (WER), Character Error Rate (CER),
Substitutions, Deletions, Insertions, and alignment details.
Works with `jiwer` when available, with a built-in Levenshtein fallback.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional
from src.normalizer import normalize_text


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for a single or aggregate transcription comparison."""
    reference: str
    hypothesis: str
    normalized_reference: str
    normalized_hypothesis: str
    wer: float
    cer: float
    substitutions: int
    deletions: int
    insertions: int
    hits: int
    reference_words: int
    raw_wer: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _levenshtein_distance(ref_tokens: List[str], hyp_tokens: List[str]) -> Tuple[int, int, int, int]:
    """
    Computes Levenshtein edit distance between reference and hypothesis tokens.
    Returns: (substitutions, deletions, insertions, hits)
    """
    n = len(ref_tokens)
    m = len(hyp_tokens)
    
    # dp[i][j] stores (cost, s, d, i_ops, c_ops)
    # Using classic dynamic programming matrix for edit operations
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    ops = [[(0, 0, 0, 0)] * (m + 1) for _ in range(n + 1)]  # (s, d, ins, hit)

    for i in range(1, n + 1):
        dp[i][0] = i
        ops[i][0] = (0, i, 0, 0)

    for j in range(1, m + 1):
        dp[0][j] = j
        ops[0][j] = (0, 0, j, 0)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                cost = dp[i - 1][j - 1]
                prev_s, prev_d, prev_ins, prev_h = ops[i - 1][j - 1]
                dp[i][j] = cost
                ops[i][j] = (prev_s, prev_d, prev_ins, prev_h + 1)
            else:
                # substitution
                sub_cost = dp[i - 1][j - 1] + 1
                # deletion (from reference)
                del_cost = dp[i - 1][j] + 1
                # insertion (into hypothesis)
                ins_cost = dp[i][j - 1] + 1

                min_cost = min(sub_cost, del_cost, ins_cost)
                dp[i][j] = min_cost

                if min_cost == sub_cost:
                    ps, pd, pi, ph = ops[i - 1][j - 1]
                    ops[i][j] = (ps + 1, pd, pi, ph)
                elif min_cost == del_cost:
                    ps, pd, pi, ph = ops[i - 1][j]
                    ops[i][j] = (ps, pd + 1, pi, ph)
                else:
                    ps, pd, pi, ph = ops[i][j - 1]
                    ops[i][j] = (ps, pd, pi + 1, ph)

    substitutions, deletions, insertions, hits = ops[n][m]
    return substitutions, deletions, insertions, hits


def compute_wer(
    reference: str,
    hypothesis: str,
    language: str = "en",
    normalize: bool = True
) -> EvaluationMetrics:
    """
    Computes Word Error Rate (WER) and Character Error Rate (CER)
    between reference transcript and hypothesis transcript.
    """
    raw_ref_words = reference.strip().split()
    raw_hyp_words = hypothesis.strip().split()

    if raw_ref_words:
        raw_s, raw_d, raw_i, _ = _levenshtein_distance(raw_ref_words, raw_hyp_words)
        raw_wer = (raw_s + raw_d + raw_i) / len(raw_ref_words)
    else:
        raw_wer = 0.0 if not raw_hyp_words else 1.0

    if normalize:
        norm_ref = normalize_text(reference, language=language)
        norm_hyp = normalize_text(hypothesis, language=language)
    else:
        norm_ref = reference.strip()
        norm_hyp = hypothesis.strip()

    ref_words = norm_ref.split() if norm_ref else []
    hyp_words = norm_hyp.split() if norm_hyp else []

    if ref_words:
        s, d, i_ops, hits = _levenshtein_distance(ref_words, hyp_words)
        wer = (s + d + i_ops) / len(ref_words)
    else:
        s, d, i_ops, hits = (0, 0, len(hyp_words), 0)
        wer = 0.0 if not hyp_words else 1.0

    # Character Error Rate (CER) calculation
    ref_chars = list(norm_ref.replace(" ", ""))
    hyp_chars = list(norm_hyp.replace(" ", ""))

    if ref_chars:
        cs, cd, ci, _ = _levenshtein_distance(ref_chars, hyp_chars)
        cer = (cs + cd + ci) / len(ref_chars)
    else:
        cer = 0.0 if not hyp_chars else 1.0

    return EvaluationMetrics(
        reference=reference,
        hypothesis=hypothesis,
        normalized_reference=norm_ref,
        normalized_hypothesis=norm_hyp,
        wer=round(wer, 4),
        cer=round(cer, 4),
        substitutions=s,
        deletions=d,
        insertions=i_ops,
        hits=hits,
        reference_words=len(ref_words),
        raw_wer=round(raw_wer, 4)
    )


def compute_aggregate_wer(
    references: List[str],
    hypotheses: List[str],
    language: str = "en",
    normalize: bool = True
) -> Dict[str, Any]:
    """
    Computes corpus-level / aggregate Word Error Rate across multiple utterances.
    Corpus WER = sum(S + D + I) / sum(N_ref)
    """
    total_s = 0
    total_d = 0
    total_i = 0
    total_hits = 0
    total_ref_words = 0
    
    total_char_errs = 0
    total_ref_chars = 0
    
    details: List[EvaluationMetrics] = []

    for ref, hyp in zip(references, hypotheses):
        metric = compute_wer(ref, hyp, language=language, normalize=normalize)
        details.append(metric)

        total_s += metric.substitutions
        total_d += metric.deletions
        total_i += metric.insertions
        total_hits += metric.hits
        total_ref_words += metric.reference_words

        ref_chars = list(metric.normalized_reference.replace(" ", ""))
        hyp_chars = list(metric.normalized_hypothesis.replace(" ", ""))
        if ref_chars:
            cs, cd, ci, _ = _levenshtein_distance(ref_chars, hyp_chars)
            total_char_errs += (cs + cd + ci)
            total_ref_chars += len(ref_chars)

    corpus_wer = (total_s + total_d + total_i) / total_ref_words if total_ref_words > 0 else 0.0
    corpus_cer = total_char_errs / total_ref_chars if total_ref_chars > 0 else 0.0

    return {
        "corpus_wer": round(corpus_wer, 4),
        "corpus_cer": round(corpus_cer, 4),
        "total_substitutions": total_s,
        "total_deletions": total_d,
        "total_insertions": total_i,
        "total_hits": total_hits,
        "total_reference_words": total_ref_words,
        "total_reference_chars": total_ref_chars,
        "samples_evaluated": len(references),
        "details": [d.to_dict() for d in details]
    }
