"""
Evaluation Metrics for Retrieval and Generation (RAG)
=====================================================
Standard, standalone Information Retrieval (IR) and Generation Groundedness metrics:
- Hit Rate @ K
- Mean Reciprocal Rank (MRR @ K)
- Mean Average Precision (MAP @ K)
- Precision @ K & Recall @ K
- Citation Precision & Recall
- Hallucination / Evidence Citation Validity Rate
"""

from typing import Any, Dict, List, Sequence, Set


def compute_hit_rate(
    retrieved_items: Sequence[Any],
    relevant_items: Sequence[Any],
    k: int = 5,
) -> float:
    """
    Computes Hit Rate @ K (binary: 1.0 if any relevant item is in top-k, else 0.0).
    
    Args:
        retrieved_items: Ordered sequence of retrieved item IDs or identifiers.
        relevant_items: Collection of ground-truth relevant item IDs.
        k: Cutoff rank.
    """
    if not relevant_items:
        return 0.0
    top_k = retrieved_items[:k]
    relevant_set = set(relevant_items)
    return 1.0 if any(item in relevant_set for item in top_k) else 0.0


def compute_mrr(
    retrieved_items: Sequence[Any],
    relevant_items: Sequence[Any],
    k: int = 5,
) -> float:
    """
    Computes Reciprocal Rank @ K: 1 / rank of the first relevant item in top-k (or 0.0).
    
    Args:
        retrieved_items: Ordered sequence of retrieved item IDs.
        relevant_items: Collection of ground-truth relevant item IDs.
        k: Cutoff rank.
    """
    if not relevant_items:
        return 0.0
    relevant_set = set(relevant_items)
    for rank, item in enumerate(retrieved_items[:k], start=1):
        if item in relevant_set:
            return 1.0 / rank
    return 0.0


def compute_precision_at_k(
    retrieved_items: Sequence[Any],
    relevant_items: Sequence[Any],
    k: int = 5,
) -> float:
    """Computes Precision @ K: (number of relevant items in top-k) / k."""
    if k <= 0:
        return 0.0
    top_k = retrieved_items[:k]
    relevant_set = set(relevant_items)
    hits = sum(1 for item in top_k if item in relevant_set)
    return hits / k


def compute_recall_at_k(
    retrieved_items: Sequence[Any],
    relevant_items: Sequence[Any],
    k: int = 5,
) -> float:
    """Computes Recall @ K: (number of relevant items in top-k) / (total relevant items)."""
    if not relevant_items:
        return 0.0
    top_k = retrieved_items[:k]
    relevant_set = set(relevant_items)
    hits = sum(1 for item in top_k if item in relevant_set)
    return hits / len(relevant_set)


def compute_average_precision(
    retrieved_items: Sequence[Any],
    relevant_items: Sequence[Any],
    k: int = 5,
) -> float:
    """
    Computes Average Precision @ K for a single query.
    AP@K = sum(Precision@i * rel(i)) / min(len(relevant), k)
    """
    if not relevant_items or k <= 0:
        return 0.0
    relevant_set = set(relevant_items)
    top_k = retrieved_items[:k]
    
    score = 0.0
    num_hits = 0
    for i, item in enumerate(top_k, start=1):
        if item in relevant_set:
            num_hits += 1
            score += num_hits / i
            
    denominator = min(len(relevant_set), k)
    return score / denominator if denominator > 0 else 0.0


def compute_citation_validity(
    cited_ids: Sequence[str],
    available_ids: Sequence[str],
) -> Dict[str, Any]:
    """
    Evaluates citation grounding: checks if every cited evidence ID exists in the retrieved set.
    
    Returns:
        {
            "total_citations": int,
            "valid_citations": int,
            "invalid_citations": List[str],
            "validity_rate": float (0.0 to 1.0),
            "is_fully_grounded": bool
        }
    """
    if not cited_ids:
        return {
            "total_citations": 0,
            "valid_citations": 0,
            "invalid_citations": [],
            "validity_rate": 1.0,
            "is_fully_grounded": True,
        }
    
    available_set = set(available_ids)
    valid_count = sum(1 for cid in cited_ids if cid in available_set)
    invalid = [cid for cid in cited_ids if cid not in available_set]
    validity_rate = valid_count / len(cited_ids)
    
    return {
        "total_citations": len(cited_ids),
        "valid_citations": valid_count,
        "invalid_citations": invalid,
        "validity_rate": round(validity_rate, 4),
        "is_fully_grounded": len(invalid) == 0,
    }
