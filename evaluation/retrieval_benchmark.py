"""
Retrieval Benchmark Engine
==========================
Evaluates and compares 4 retrieval strategies across the benchmark dataset:
    1. Dense Vector Search (ChromaDB + all-MiniLM-L6-v2)
    2. Sparse Lexical Search (BM25Okapi)
    3. Hybrid Search (Dense + BM25 combined)
    4. Hybrid + Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2)

Computes standard Information Retrieval (IR) metrics:
    - Hit Rate @ 3 and @ 5
    - MRR @ 5 (Mean Reciprocal Rank)
    - MAP @ 5 (Mean Average Precision)
    - Average Query Latency (ms)
"""

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from core.retriever_tools import keyword_search, rerank_documents
from evaluation.metrics import (
    compute_average_precision,
    compute_hit_rate,
    compute_mrr,
    compute_precision_at_k,
)


def is_chunk_relevant(doc: Document, item: Dict[str, Any]) -> bool:
    """
    Determines ground-truth relevance:
    1. Source file must match expected source.
    2. At least one ground truth keyword must be in the chunk text.
    """
    doc_source = os.path.basename(doc.metadata.get("source", ""))
    expected_sources = [os.path.basename(s) for s in item.get("relevant_sources", [])]
    
    # Check source file match if specified
    if expected_sources and doc_source not in expected_sources:
        return False
        
    content_lower = doc.page_content.lower()
    keywords = item.get("ground_truth_keywords", [])
    if not keywords:
        return True
        
    return any(kw.lower() in content_lower for kw in keywords)


def retrieve_dense_only(pipeline: Any, query: str, k: int = 5) -> List[Document]:
    """Retrieves top-k candidates using vector similarity search only."""
    return pipeline.retrieve_as_documents(query=query, k=k)


def retrieve_sparse_only(pipeline: Any, query: str, k: int = 5) -> List[Document]:
    """Retrieves top-k candidates using BM25 keyword search only."""
    return keyword_search(queries=[query], pipeline=pipeline, k=k)


def retrieve_hybrid(pipeline: Any, query: str, k: int = 5) -> List[Document]:
    """Combines dense and sparse results with reciprocal deduplication."""
    dense_docs = retrieve_dense_only(pipeline, query, k=k * 2)
    sparse_docs = retrieve_sparse_only(pipeline, query, k=k * 2)
    
    seen_ids = set()
    merged: List[Document] = []
    
    # Interleave results (1 dense, 1 sparse) for balanced candidate ranking
    max_len = max(len(dense_docs), len(sparse_docs))
    for i in range(max_len):
        if i < len(dense_docs):
            cid = dense_docs[i].metadata.get("chunk_id", dense_docs[i].page_content[:60])
            if cid not in seen_ids:
                seen_ids.add(cid)
                merged.append(dense_docs[i])
        if i < len(sparse_docs):
            cid = sparse_docs[i].metadata.get("chunk_id", sparse_docs[i].page_content[:60])
            if cid not in seen_ids:
                seen_ids.add(cid)
                merged.append(sparse_docs[i])
                
    return merged[:k]


def retrieve_hybrid_reranked(pipeline: Any, query: str, k: int = 5) -> List[Document]:
    """Retrieves hybrid candidates (top 15) and applies cross-encoder reranking."""
    candidates = retrieve_hybrid(pipeline, query, k=15)
    return rerank_documents(query=query, documents=candidates, top_k=k)


def evaluate_retrieval_strategy(
    strategy_name: str,
    retrieval_fn: Any,
    dataset: List[Dict[str, Any]],
    pipeline: Any,
    k: int = 5,
) -> Dict[str, Any]:
    """Evaluates a retrieval strategy across the full evaluation dataset."""
    hit_3_list = []
    hit_5_list = []
    mrr_5_list = []
    map_5_list = []
    prec_5_list = []
    latencies_ms = []
    category_scores: Dict[str, Dict[str, List[float]]] = {}

    for item in dataset:
        cat = item.get("category", "all")
        if cat not in category_scores:
            category_scores[cat] = {"hit_5": [], "mrr_5": []}
            
        t0 = time.perf_counter()
        retrieved_docs = retrieval_fn(pipeline, item["query"], k=k)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed_ms)
        
        # Binary relevance indicators for each retrieved doc
        binary_relevance = [1 if is_chunk_relevant(doc, item) else 0 for doc in retrieved_docs]
        
        # Metric calculations using dummy relevant token "1"
        retrieved_tokens = [f"doc_{i}" if rel else f"irrel_{i}" for i, rel in enumerate(binary_relevance)]
        relevant_tokens = [f"doc_{i}" for i, rel in enumerate(binary_relevance) if rel]
        
        h3 = compute_hit_rate(retrieved_tokens, relevant_tokens, k=3)
        h5 = compute_hit_rate(retrieved_tokens, relevant_tokens, k=5)
        mrr5 = compute_mrr(retrieved_tokens, relevant_tokens, k=5)
        map5 = compute_average_precision(retrieved_tokens, relevant_tokens, k=5)
        p5 = compute_precision_at_k(retrieved_tokens, relevant_tokens, k=5)
        
        hit_3_list.append(h3)
        hit_5_list.append(h5)
        mrr_5_list.append(mrr5)
        map_5_list.append(map5)
        prec_5_list.append(p5)
        
        category_scores[cat]["hit_5"].append(h5)
        category_scores[cat]["mrr_5"].append(mrr5)

    n = len(dataset) or 1
    summary = {
        "strategy": strategy_name,
        "sample_count": len(dataset),
        "hit_rate_at_3": round(sum(hit_3_list) / n, 4),
        "hit_rate_at_5": round(sum(hit_5_list) / n, 4),
        "mrr_at_5": round(sum(mrr_5_list) / n, 4),
        "map_at_5": round(sum(map_5_list) / n, 4),
        "precision_at_5": round(sum(prec_5_list) / n, 4),
        "avg_latency_ms": round(sum(latencies_ms) / n, 2),
        "by_category": {
            cat: {
                "hit_rate_at_5": round(sum(vals["hit_5"]) / len(vals["hit_5"]), 4),
                "mrr_at_5": round(sum(vals["mrr_5"]) / len(vals["mrr_5"]), 4),
            }
            for cat, vals in category_scores.items()
        },
    }
    return summary


def run_all_retrieval_benchmarks(
    pipeline: Any,
    dataset_path: str = "evaluation/eval_dataset.json",
) -> Dict[str, Any]:
    """Runs benchmarks across all 4 retrieval strategies and returns the comparison dict."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    strategies = [
        ("Dense Vector Only (ChromaDB)", retrieve_dense_only),
        ("Sparse Keyword Only (BM25)", retrieve_sparse_only),
        ("Hybrid (Dense + BM25)", retrieve_hybrid),
        ("Hybrid + Cross-Encoder Reranker", retrieve_hybrid_reranked),
    ]
    
    results = {}
    for name, fn in strategies:
        print(f"  -> Benchmarking: {name}...")
        results[name] = evaluate_retrieval_strategy(name, fn, dataset, pipeline)
        
    return results
