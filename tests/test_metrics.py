"""Unit tests for evaluation metrics."""

import pytest
from evaluation.metrics import (
    compute_hit_rate,
    compute_mrr,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_average_precision,
    compute_citation_validity,
)


def test_compute_hit_rate():
    retrieved = ["doc1", "doc2", "doc3", "doc4"]
    assert compute_hit_rate(retrieved, ["doc2"], k=3) == 1.0
    assert compute_hit_rate(retrieved, ["doc4"], k=3) == 0.0
    assert compute_hit_rate(retrieved, ["doc4"], k=4) == 1.0
    assert compute_hit_rate(retrieved, [], k=5) == 0.0


def test_compute_mrr():
    retrieved = ["doc1", "doc2", "doc3", "doc4"]
    assert compute_mrr(retrieved, ["doc1"], k=5) == 1.0
    assert compute_mrr(retrieved, ["doc2"], k=5) == 0.5
    assert compute_mrr(retrieved, ["doc3"], k=5) == pytest.approx(1.0 / 3)
    assert compute_mrr(retrieved, ["doc99"], k=5) == 0.0
    assert compute_mrr(retrieved, ["doc4"], k=3) == 0.0


def test_compute_precision_and_recall():
    retrieved = ["doc1", "doc2", "doc3", "doc4"]
    relevant = ["doc1", "doc3", "doc5"]
    
    # top 2: doc1, doc2 -> 1 hit out of 2 -> P@2 = 0.5
    assert compute_precision_at_k(retrieved, relevant, k=2) == 0.5
    # top 4: doc1, doc2, doc3, doc4 -> 2 hits out of 4 -> P@4 = 0.5
    assert compute_precision_at_k(retrieved, relevant, k=4) == 0.5
    
    # Recall: 2 hits out of 3 total relevant -> 2/3
    assert compute_recall_at_k(retrieved, relevant, k=4) == pytest.approx(2.0 / 3)


def test_compute_average_precision():
    retrieved = ["doc1", "doc2", "doc3", "doc4"]
    relevant = ["doc1", "doc3"]
    # doc1 is at rank 1 (P@1 = 1/1 = 1.0)
    # doc3 is at rank 3 (P@3 = 2/3)
    # AP = (1.0 + 2/3) / 2 = 1.6667 / 2 = 0.8333
    assert compute_average_precision(retrieved, relevant, k=4) == pytest.approx((1.0 + 2.0 / 3) / 2)


def test_compute_citation_validity():
    available = ["E1", "E2", "E3"]
    
    # All valid
    res = compute_citation_validity(["E1", "E2"], available)
    assert res["validity_rate"] == 1.0
    assert res["is_fully_grounded"] is True
    assert res["invalid_citations"] == []
    
    # With hallucinated / invalid ID
    res2 = compute_citation_validity(["E1", "E99"], available)
    assert res2["validity_rate"] == 0.5
    assert res2["is_fully_grounded"] is False
    assert res2["invalid_citations"] == ["E99"]
    
    # Empty citations
    res3 = compute_citation_validity([], available)
    assert res3["validity_rate"] == 1.0
    assert res3["is_fully_grounded"] is True
