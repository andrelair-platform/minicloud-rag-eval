"""Deterministic retrieval and generation metrics — no LLM required."""

from rouge_score import rouge_scorer as _rouge


def rouge_l(prediction: str, reference: str) -> float:
    scorer = _rouge.RougeScorer(["rougeL"], use_stemmer=False)
    score = scorer.score(reference.lower(), prediction.lower())
    return round(score["rougeL"].fmeasure, 4)


def hit_rate(ground_truth: str, chunks: list[str]) -> float:
    """1.0 if any keyword from ground_truth appears in any retrieved chunk."""
    keywords = [w.strip(".,%;()").lower() for w in ground_truth.split() if len(w) > 2]
    for chunk in chunks:
        chunk_lower = chunk.lower()
        if any(kw in chunk_lower for kw in keywords):
            return 1.0
    return 0.0


def mrr(ground_truth: str, chunks: list[str]) -> float:
    """1/rank of first chunk containing a ground_truth keyword."""
    keywords = [w.strip(".,%;()").lower() for w in ground_truth.split() if len(w) > 2]
    for rank, chunk in enumerate(chunks, 1):
        chunk_lower = chunk.lower()
        if any(kw in chunk_lower for kw in keywords):
            return round(1.0 / rank, 4)
    return 0.0
