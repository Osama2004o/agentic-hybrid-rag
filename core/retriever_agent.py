"""
Retriever Agent — LangGraph Workflow
=====================================
A LangGraph StateGraph that orchestrates the 6 retriever tools:
    query_rewriter → semantic_search + keyword_search (+ metadata_filter)
                   → merge → reranker → context_selector → END

The agent returns a curated list of evidence Documents ready for the Analyst Agent.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph

from core.retriever_tools import (
    extract_metadata_filter,
    keyword_search,
    metadata_filtered_search,
    query_rewriter,
    rerank_documents,
    select_context,
    semantic_search,
)


# ---------------------------------------------------------------------------
# Agent State
# ---------------------------------------------------------------------------

class RetrieverState(TypedDict):
    """Shared state flowing through every node of the Retriever Agent graph."""

    original_query: str                     # raw user question
    chat_history: list                      # conversation history for query rewriter
    rewritten_queries: list                 # LLM-rewritten queries
    semantic_results: list                  # docs from vector search
    keyword_results: list                   # docs from BM25
    metadata_filter: dict | None            # extracted metadata constraints
    metadata_results: list                  # docs from metadata-filtered search
    merged_results: list                    # combined & deduplicated
    reranked_results: list                  # after cross-encoder scoring
    final_context: list                     # best chunks for the Analyst

    # Injected dependency — not serialised
    pipeline: Any                           # KnowledgeBasePipeline instance


# ---------------------------------------------------------------------------
# Graph Nodes
# ---------------------------------------------------------------------------

def node_query_rewriter(state: RetrieverState) -> dict:
    """Rewrites the user query into retrieval-optimised forms."""
    rewritten = query_rewriter(
        original_query=state["original_query"],
        chat_history=state.get("chat_history"),
    )
    print(f"  [QueryRewriter] {len(rewritten)} queries: {rewritten}")
    return {"rewritten_queries": rewritten}


def node_semantic_search(state: RetrieverState) -> dict:
    """Runs vector similarity search on ChromaDB."""
    meta_filter = state.get("metadata_filter")
    results = semantic_search(
        queries=state["rewritten_queries"],
        pipeline=state["pipeline"],
        where=meta_filter,
    )
    print(f"  [SemanticSearch] Retrieved {len(results)} chunks")
    return {"semantic_results": results}


def node_keyword_search(state: RetrieverState) -> dict:
    """Runs BM25 keyword search over all stored chunks."""
    results = keyword_search(
        queries=state["rewritten_queries"],
        pipeline=state["pipeline"],
    )
    print(f"  [KeywordSearch] Retrieved {len(results)} chunks")
    return {"keyword_results": results}


def node_metadata_filter(state: RetrieverState) -> dict:
    """Extracts metadata constraints and runs filtered search if applicable."""
    meta = extract_metadata_filter(state["original_query"])

    if meta:
        print(f"  [MetadataFilter] Detected filter: {meta}")
        results = metadata_filtered_search(
            queries=state["rewritten_queries"],
            pipeline=state["pipeline"],
            metadata_filter=meta,
        )
        print(f"  [MetadataFilter] Retrieved {len(results)} chunks")
        return {"metadata_filter": meta, "metadata_results": results}
    else:
        print("  [MetadataFilter] No metadata constraints detected")
        return {"metadata_filter": None, "metadata_results": []}


def node_merge_results(state: RetrieverState) -> dict:
    """Merges and deduplicates results from all retrieval methods."""
    all_docs: List[Document] = []
    seen_content: set = set()

    # Priority order: semantic > metadata_filtered > keyword
    for doc_list in [
        state.get("semantic_results", []),
        state.get("metadata_results", []),
        state.get("keyword_results", []),
    ]:
        for doc in doc_list:
            # Deduplicate by content hash (first 200 chars)
            content_key = doc.page_content[:200].strip().lower()
            if content_key not in seen_content:
                seen_content.add(content_key)
                all_docs.append(doc)

    print(f"  [MergeResults] {len(all_docs)} unique chunks after deduplication")
    return {"merged_results": all_docs}


def node_reranker(state: RetrieverState) -> dict:
    """Re-scores merged results using a cross-encoder model."""
    merged = state.get("merged_results", [])
    if not merged:
        print("  [Reranker] No documents to rerank")
        return {"reranked_results": []}

    reranked = rerank_documents(
        query=state["original_query"],
        documents=merged,
        top_k=min(len(merged), 15),
    )
    print(f"  [Reranker] Reranked to {len(reranked)} chunks")
    for i, doc in enumerate(reranked[:5]):
        score = doc.metadata.get("rerank_score", "?")
        source = doc.metadata.get("source", "?")
        print(f"    #{i+1} score={score} source={source}")
    return {"reranked_results": reranked}


def node_context_selector(state: RetrieverState) -> dict:
    """Selects the best diverse subset of chunks for the Analyst."""
    reranked = state.get("reranked_results", [])
    if not reranked:
        print("  [ContextSelector] No documents to select from")
        return {"final_context": []}

    selected = select_context(
        query=state["original_query"],
        documents=reranked,
        max_chunks=8,
        min_chunks=3,
    )
    print(f"  [ContextSelector] Selected {len(selected)} chunks for Analyst")
    return {"final_context": selected}


# ---------------------------------------------------------------------------
# Build the LangGraph
# ---------------------------------------------------------------------------

def build_retriever_graph() -> StateGraph:
    """
    Constructs and compiles the Retriever Agent graph.

    Flow:
        query_rewriter
            → semantic_search
            → keyword_search
            → metadata_filter
        (all three run, then)
            → merge_results
            → reranker
            → context_selector
            → END
    """
    graph = StateGraph(RetrieverState)

    # Add nodes
    graph.add_node("query_rewriter", node_query_rewriter)
    graph.add_node("semantic_search", node_semantic_search)
    graph.add_node("keyword_search", node_keyword_search)
    graph.add_node("metadata_filter", node_metadata_filter)
    graph.add_node("merge_results", node_merge_results)
    graph.add_node("reranker", node_reranker)
    graph.add_node("context_selector", node_context_selector)

    # Set entry point
    graph.set_entry_point("query_rewriter")

    # After query_rewriter → fan out to all three search methods
    graph.add_edge("query_rewriter", "semantic_search")
    graph.add_edge("query_rewriter", "keyword_search")
    graph.add_edge("query_rewriter", "metadata_filter")

    # All search methods converge into merge
    graph.add_edge("semantic_search", "merge_results")
    graph.add_edge("keyword_search", "merge_results")
    graph.add_edge("metadata_filter", "merge_results")

    # Linear pipeline after merge
    graph.add_edge("merge_results", "reranker")
    graph.add_edge("reranker", "context_selector")
    graph.add_edge("context_selector", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_retriever_agent(
    query: str,
    pipeline: Any,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> List[Document]:
    """
    Runs the full Retriever Agent pipeline and returns the final evidence
    Documents for the Analyst Agent.

    Args:
        query:        The user's original question.
        pipeline:     A KnowledgeBasePipeline instance (from database/).
        chat_history: Optional conversation history for query rewriting.

    Returns:
        List of curated LangChain Document objects with metadata.
    """
    graph = build_retriever_graph()

    initial_state: RetrieverState = {
        "original_query": query,
        "chat_history": chat_history or [],
        "rewritten_queries": [],
        "semantic_results": [],
        "keyword_results": [],
        "metadata_filter": None,
        "metadata_results": [],
        "merged_results": [],
        "reranked_results": [],
        "final_context": [],
        "pipeline": pipeline,
    }

    print(f"\n{'='*60}")
    print(f"  RETRIEVER AGENT")
    print(f"  Query: {query}")
    print(f"{'='*60}")

    result = graph.invoke(initial_state)

    final_docs = result.get("final_context", [])
    print(f"\n{'='*60}")
    print(f"  RETRIEVER AGENT COMPLETE — {len(final_docs)} evidence chunks")
    print(f"{'='*60}\n")

    return final_docs

