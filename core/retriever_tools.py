"""
Retriever Agent Tools
=====================
Six standalone tool functions used by the Retriever Agent's LangGraph workflow:
    1. Query Rewriter         LLM rewrites the user query for better retrieval
    2. Semantic Search        ChromaDB vector similarity search
    3. Keyword Search         BM25 lexical search over stored chunks
    4. Metadata Filter        Restrict retrieval by metadata constraints
    5. Reranker               Cross-encoder rescoring
    6. Context Selector       LLM selects the best diverse subset of chunks
"""

import json
import re
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from config.settings import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    TOP_K_RESULTS,
    LLM_MODEL,
)

# ---------------------------------------------------------------------------
# LLM helper – OpenRouter free model via LangChain
# ---------------------------------------------------------------------------

def _get_llm(temperature: float = 0.3, max_tokens: int = 2048) -> ChatOpenAI:
    """Returns a ChatOpenAI instance pointed at OpenRouter free models."""
    return ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL,
        temperature=temperature,
        max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------------
# Cross-encoder singleton (lazy loaded)
# ---------------------------------------------------------------------------

_cross_encoder: Optional[CrossEncoder] = None


def _get_cross_encoder() -> CrossEncoder:
    """Lazily loads and caches the cross-encoder reranker model."""
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


# ===================================================================
# Tool 1: Query Rewriter
# ===================================================================

def query_rewriter(
    original_query: str,
    chat_history: List[Dict[str, str]] | None = None,
) -> List[str]:
    """
    Transforms the user's original question into one or more retrieval-friendly
    queries.  Resolves pronouns, expands abbreviations, clarifies terminology,
    and generates alternative phrasings.

    Returns a list of 1-3 rewritten queries.
    """
    llm = _get_llm(temperature=0.4)

    history_block = ""
    if chat_history:
        history_block = "Previous conversation:\n"
        for turn in chat_history[-6:]:  # keep last 6 turns for context
            role = turn.get("role", "user")
            content = turn.get("content", "")
            history_block += f"  {role}: {content}\n"

    prompt = f"""You are a query rewriting assistant for a document retrieval system.
Your task is to rewrite the user's query into 1-3 clear, self-contained, retrieval-optimized queries.

Rules:
- Resolve pronouns and references using the conversation history.
- Expand abbreviations and acronyms where possible.
- If the query is already clear, return it as-is (still output the JSON list).
- Each rewritten query should be a standalone question/search phrase.
- Output ONLY a JSON array of strings, no explanation.

{history_block}
Original query: {original_query}

Rewritten queries (JSON array):"""

    response = llm.invoke(prompt)
    text = response.content.strip()

    # Parse JSON array from the response
    try:
        # Try to extract JSON array from the response
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            queries = json.loads(match.group())
            if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
                return queries[:3]
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback: return the original query
    return [original_query]


# ===================================================================
# Tool 2: Semantic Search
# ===================================================================

def semantic_search(
    queries: List[str],
    pipeline: Any,
    k: int = TOP_K_RESULTS,
    where: Optional[Dict[str, Any]] = None,
) -> List[Document]:
    """
    Runs vector similarity search on ChromaDB for each query and returns
    deduplicated results.

    Uses the existing KnowledgeBasePipeline.retrieve_as_documents().
    """
    seen_ids = set()
    results: List[Document] = []

    for query in queries:
        docs = pipeline.retrieve_as_documents(query=query, k=k, where=where)
        for doc in docs:
            chunk_id = doc.metadata.get("chunk_id", doc.page_content[:80])
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                doc.metadata["retrieval_method"] = "semantic"
                results.append(doc)

    return results


# ===================================================================
# Tool 3: Keyword Search (BM25)
# ===================================================================

def keyword_search(
    queries: List[str],
    pipeline: Any,
    k: int = TOP_K_RESULTS,
) -> List[Document]:
    """
    Performs BM25 keyword search over all chunks stored in ChromaDB.

    Retrieves all stored documents once, builds a BM25 index, and scores
    each query against it.
    """
    # Pull all documents from the collection
    all_data = pipeline.collection.get(include=["documents", "metadatas"])
    all_docs_text = all_data.get("documents", [])
    all_metas = all_data.get("metadatas", [])
    all_ids = all_data.get("ids", [])

    if not all_docs_text:
        return []

    # Tokenise for BM25 (simple whitespace + lowercase)
    tokenised_corpus = [doc.lower().split() for doc in all_docs_text]
    bm25 = BM25Okapi(tokenised_corpus)

    seen_ids = set()
    results: List[Document] = []

    for query in queries:
        tokenised_query = query.lower().split()
        scores = bm25.get_scores(tokenised_query)

        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            doc_id = all_ids[idx]
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)

            meta = dict(all_metas[idx]) if idx < len(all_metas) else {}
            meta["chunk_id"] = doc_id
            meta["bm25_score"] = round(float(scores[idx]), 4)
            meta["retrieval_method"] = "keyword"

            results.append(
                Document(page_content=all_docs_text[idx], metadata=meta)
            )

    return results


# ===================================================================
# Tool 4: Metadata Filter
# ===================================================================

def extract_metadata_filter(
    query: str,
) -> Optional[Dict[str, Any]]:
    """
    Uses an LLM to extract metadata constraints from the user's query.

    Examples of constraints:
      - "According to Chapter 4 of paper.pdf" → {"source": "paper.pdf"}
      - "In the introduction section" → {"source_type": "pdf"}

    Returns a ChromaDB-compatible `where` dict, or None if no filter detected.
    """
    llm = _get_llm(temperature=0.0, max_tokens=512)

    prompt = f"""You are a metadata extraction assistant. Analyze the user query and extract any document metadata constraints.

Available metadata fields:
- "source": the file name or URL (e.g., "paper.pdf", "report.docx")
- "source_type": the document type (e.g., "pdf", "docx", "txt", "web", "wikipedia")
- "page": page number (integer)

If the user mentions a specific document, file, chapter, page, or source, extract it.
If NO metadata constraints are found, output exactly: null

Output ONLY a valid JSON object (ChromaDB where filter) or null.

Examples:
- Query: "What does paper.pdf say about transformers?" → {{"source": "paper.pdf"}}
- Query: "On page 5, what is the accuracy?" → {{"page": 5}}
- Query: "What are the advantages of RAG?" → null

User query: {query}

Metadata filter (JSON or null):"""

    response = llm.invoke(prompt)
    text = response.content.strip()

    try:
        if text.lower() == "null" or text.lower() == "none":
            return None
        # Try to extract JSON object
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            result = json.loads(match.group())
            if isinstance(result, dict) and result:
                return result
    except (json.JSONDecodeError, TypeError):
        pass

    return None


def metadata_filtered_search(
    queries: List[str],
    pipeline: Any,
    metadata_filter: Dict[str, Any],
    k: int = TOP_K_RESULTS,
) -> List[Document]:
    """
    Performs semantic search restricted by metadata constraints.
    Uses KnowledgeBasePipeline.retrieve_as_documents() with a `where` filter.
    """
    results: List[Document] = []
    seen_ids = set()

    for query in queries:
        docs = pipeline.retrieve_as_documents(query=query, k=k, where=metadata_filter)
        for doc in docs:
            chunk_id = doc.metadata.get("chunk_id", doc.page_content[:80])
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                doc.metadata["retrieval_method"] = "metadata_filtered"
                results.append(doc)

    return results


# ===================================================================
# Tool 5: Reranker (Cross-Encoder)
# ===================================================================

def rerank_documents(
    query: str,
    documents: List[Document],
    top_k: int = 10,
) -> List[Document]:
    """
    Re-scores documents using a cross-encoder model and returns them sorted
    by relevance (highest first).

    Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` for fast, accurate reranking.
    """
    if not documents:
        return []

    cross_encoder = _get_cross_encoder()

    # Build (query, passage) pairs for the cross-encoder
    pairs = [(query, doc.page_content) for doc in documents]
    scores = cross_encoder.predict(pairs)

    # Attach scores and sort
    scored_docs = list(zip(documents, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    results = []
    for doc, score in scored_docs[:top_k]:
        doc.metadata["rerank_score"] = round(float(score), 4)
        results.append(doc)

    return results


# ===================================================================
# Tool 6: Context Selector
# ===================================================================

def select_context(
    query: str,
    documents: List[Document],
    max_chunks: int = 8,
    min_chunks: int = 3,
) -> List[Document]:
    """
    Uses an LLM to select the best subset of reranked chunks.

    Considers relevance, diversity, document coverage, and avoids redundancy.
    Preserves metadata (source, page numbers) for citation by downstream agents.
    """
    if len(documents) <= min_chunks:
        return documents

    llm = _get_llm(temperature=0.1, max_tokens=1024)

    # Build a numbered list of chunk summaries for the LLM
    chunk_summaries = []
    for i, doc in enumerate(documents):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        score = doc.metadata.get("rerank_score", doc.metadata.get("score", "?"))
        preview = doc.page_content[:300].replace("\n", " ")
        chunk_summaries.append(
            f"[{i}] Source: {source} | Page: {page} | Score: {score}\n    {preview}..."
        )

    chunks_text = "\n".join(chunk_summaries)

    prompt = f"""You are a context selection assistant. Given a query and a list of retrieved document chunks, 
select the {min_chunks}-{max_chunks} BEST chunks that together provide the most relevant, diverse, 
and complete evidence to answer the query.

Rules:
- Prefer chunks with higher relevance scores.
- Avoid selecting chunks with highly redundant/overlapping content.
- Ensure coverage across different sources/pages when possible.
- Output ONLY a JSON array of the selected chunk indices (integers).

Query: {query}

Available chunks:
{chunks_text}

Selected chunk indices (JSON array of integers):"""

    response = llm.invoke(prompt)
    text = response.content.strip()

    try:
        match = re.search(r"\[.*?\]", text, re.DOTALL)
        if match:
            indices = json.loads(match.group())
            if isinstance(indices, list):
                # Filter valid indices
                valid_indices = [
                    i for i in indices
                    if isinstance(i, int) and 0 <= i < len(documents)
                ]
                if valid_indices:
                    return [documents[i] for i in valid_indices]
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback: return top max_chunks by existing order (already reranked)
    return documents[:max_chunks]

