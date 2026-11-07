"""
Unit & Integration Tests for Retriever Agent and Tools
======================================================
Tests all 6 retriever tools, state graph nodes, merging/deduplication logic,
and end-to-end execution of the Retriever Agent LangGraph workflow.
"""

from unittest.mock import MagicMock, patch
import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from core.retriever_tools import (
    query_rewriter,
    semantic_search,
    keyword_search,
    extract_metadata_filter,
    metadata_filtered_search,
    rerank_documents,
    select_context,
)
from core.retriever_agent import (
    build_retriever_graph,
    node_merge_results,
    run_retriever_agent,
    RetrieverState,
)


# ---------------------------------------------------------------------------
# Fixtures & Mocks
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_documents():
    return [
        Document(
            page_content="Retrieval-Augmented Generation (RAG) optimizes LLM outputs.",
            metadata={"source": "rag_overview.pdf", "chunk_id": "chunk_1", "page": 1},
        ),
        Document(
            page_content="Vector databases store embeddings for fast similarity search.",
            metadata={"source": "rag_overview.pdf", "chunk_id": "chunk_2", "page": 2},
        ),
        Document(
            page_content="BM25 is a ranking function used by search engines for keyword matching.",
            metadata={"source": "ir_intro.txt", "chunk_id": "chunk_3", "page": 1},
        ),
    ]


@pytest.fixture
def mock_pipeline(sample_documents):
    pipeline = MagicMock()

    # Mock collection.get for BM25 keyword search
    pipeline.collection.get.return_value = {
        "ids": ["chunk_1", "chunk_2", "chunk_3"],
        "documents": [doc.page_content for doc in sample_documents],
        "metadatas": [doc.metadata for doc in sample_documents],
    }

    # Mock retrieve_as_documents for semantic search
    pipeline.retrieve_as_documents.return_value = [
        Document(
            page_content=sample_documents[0].page_content,
            metadata=dict(sample_documents[0].metadata),
        )
    ]

    return pipeline


# ---------------------------------------------------------------------------
# Tool 1: Query Rewriter Tests
# ---------------------------------------------------------------------------

def test_query_rewriter_success():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content='["What is Retrieval-Augmented Generation?", "How does RAG work?"]'
    )

    with patch("core.retriever_tools._get_llm", return_value=mock_llm):
        rewritten = query_rewriter("What is it?", chat_history=[{"role": "user", "content": "Tell me about RAG"}])
        assert len(rewritten) == 2
        assert "What is Retrieval-Augmented Generation?" in rewritten


def test_query_rewriter_fallback_on_invalid_output():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="Here are some queries without JSON format")

    with patch("core.retriever_tools._get_llm", return_value=mock_llm):
        rewritten = query_rewriter("Original search question")
        assert rewritten == ["Original search question"]


# ---------------------------------------------------------------------------
# Tool 2: Semantic Search Tests
# ---------------------------------------------------------------------------

def test_semantic_search(mock_pipeline):
    queries = ["RAG architecture"]
    results = semantic_search(queries, mock_pipeline, k=2)

    assert len(results) == 1
    assert results[0].metadata["retrieval_method"] == "semantic"
    mock_pipeline.retrieve_as_documents.assert_called_once()


# ---------------------------------------------------------------------------
# Tool 3: Keyword Search (BM25) Tests
# ---------------------------------------------------------------------------

def test_keyword_search_bm25(mock_pipeline):
    queries = ["BM25 ranking function"]
    results = keyword_search(queries, mock_pipeline, k=2)

    assert len(results) >= 1
    top_doc = results[0]
    assert "BM25" in top_doc.page_content
    assert top_doc.metadata["retrieval_method"] == "keyword"
    assert "bm25_score" in top_doc.metadata


def test_keyword_search_empty_collection():
    empty_pipeline = MagicMock()
    empty_pipeline.collection.get.return_value = {"ids": [], "documents": [], "metadatas": []}

    results = keyword_search(["anything"], empty_pipeline)
    assert results == []


# ---------------------------------------------------------------------------
# Tool 4: Metadata Filter Tests
# ---------------------------------------------------------------------------

def test_extract_metadata_filter_found():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content='{"source": "paper.pdf"}')

    with patch("core.retriever_tools._get_llm", return_value=mock_llm):
        filt = extract_metadata_filter("According to paper.pdf what is accuracy?")
        assert filt == {"source": "paper.pdf"}


def test_extract_metadata_filter_none():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="null")

    with patch("core.retriever_tools._get_llm", return_value=mock_llm):
        filt = extract_metadata_filter("What are the advantages of RAG?")
        assert filt is None


def test_metadata_filtered_search(mock_pipeline):
    mock_pipeline.retrieve_as_documents.return_value = [
        Document(page_content="Filtered result", metadata={"source": "paper.pdf"})
    ]
    results = metadata_filtered_search(
        queries=["accuracy"],
        pipeline=mock_pipeline,
        metadata_filter={"source": "paper.pdf"},
        k=2,
    )
    assert len(results) == 1
    assert results[0].metadata["retrieval_method"] == "metadata_filtered"


# ---------------------------------------------------------------------------
# Tool 5: Reranker Tests
# ---------------------------------------------------------------------------

def test_rerank_documents_empty():
    assert rerank_documents("query", []) == []


def test_rerank_documents_with_mock_encoder(sample_documents):
    mock_encoder = MagicMock()
    # Return mock scores: higher score for the 2nd doc
    mock_encoder.predict.return_value = [0.1, 0.9, 0.5]

    with patch("core.retriever_tools._get_cross_encoder", return_value=mock_encoder):
        reranked = rerank_documents("query", sample_documents, top_k=2)
        assert len(reranked) == 2
        # First doc should now be the one that received score 0.9 (chunk_2)
        assert reranked[0].metadata["chunk_id"] == "chunk_2"
        assert reranked[0].metadata["rerank_score"] == 0.9


# ---------------------------------------------------------------------------
# Tool 6: Context Selector Tests
# ---------------------------------------------------------------------------

def test_select_context_below_min_threshold(sample_documents):
    # If documents count <= min_chunks (default 3), return as is without LLM call
    selected = select_context("query", sample_documents, min_chunks=3)
    assert len(selected) == len(sample_documents)


def test_select_context_with_llm(sample_documents):
    extra_docs = sample_documents + [
        Document(page_content="Extra chunk 4", metadata={"chunk_id": "chunk_4"}),
        Document(page_content="Extra chunk 5", metadata={"chunk_id": "chunk_5"}),
    ]

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="[0, 2, 3]")

    with patch("core.retriever_tools._get_llm", return_value=mock_llm):
        selected = select_context("query", extra_docs, max_chunks=3, min_chunks=2)
        assert len(selected) == 3
        assert selected[0] == extra_docs[0]
        assert selected[1] == extra_docs[2]
        assert selected[2] == extra_docs[3]


# ---------------------------------------------------------------------------
# State & Graph Integration Tests
# ---------------------------------------------------------------------------

def test_node_merge_results_deduplication():
    doc1 = Document(page_content="Duplicate passage content here", metadata={"chunk_id": "c1"})
    doc2 = Document(page_content="Duplicate passage content here", metadata={"chunk_id": "c2"})
    doc3 = Document(page_content="Unique passage content here", metadata={"chunk_id": "c3"})

    state = {
        "semantic_results": [doc1],
        "metadata_results": [doc2],
        "keyword_results": [doc3],
    }
    merged = node_merge_results(state)  # type: ignore
    results = merged["merged_results"]
    assert len(results) == 2
    assert results[0].page_content == "Duplicate passage content here"
    assert results[1].page_content == "Unique passage content here"


def test_build_retriever_graph():
    graph = build_retriever_graph()
    assert graph is not None
    # Verify graph can be inspected / has compiled structure
    assert hasattr(graph, "invoke")


def test_run_retriever_agent_end_to_end(mock_pipeline, sample_documents):
    mock_llm = MagicMock()
    # 1st call: query rewriter -> ["What is RAG?"]
    # 2nd call: extract metadata -> null
    # 3rd call: context selector -> [0]
    mock_llm.invoke.side_effect = [
        AIMessage(content='["What is RAG?"]'),
        AIMessage(content="null"),
        AIMessage(content="[0]"),
    ]

    mock_encoder = MagicMock()
    mock_encoder.predict.return_value = [0.85]

    with patch("core.retriever_tools._get_llm", return_value=mock_llm), \
         patch("core.retriever_tools._get_cross_encoder", return_value=mock_encoder):
        
        evidence = run_retriever_agent(
            query="Tell me about RAG",
            pipeline=mock_pipeline,
        )

        assert isinstance(evidence, list)
        assert len(evidence) >= 1

