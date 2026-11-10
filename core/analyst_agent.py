"""
Analyst Agent
=============
A LangChain tool-calling agent that reasons over the Retriever Agent's evidence
using the 5 analyst tools (calculator, table extractor, document comparison,
data analysis, request more evidence).

Feedback loop:
    analyse evidence → did the agent request more evidence?
        NO  → return the analysis to the Answer Agent
        YES → Retriever Agent fetches it → analyse again (up to max_loops times)
"""

from typing import Any, Callable, Dict, List, Optional

from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from config.settings import (
    ANALYST_EVIDENCE_CHAR_LIMIT,
    ANALYST_MAX_FEEDBACK_LOOPS,
    LLM_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
)
from core.analyst_tools import ANALYST_TOOLS


SYSTEM_PROMPT = """You are the Analyst Agent in a document question-answering system.
You receive evidence chunks from the user's documents. Your job is to analyse them,
not to write the final answer for the user.

- Use ONLY the evidence. Cite evidence IDs like [E1] for every fact.
- Use the calculator for every calculation. Never calculate in your head.
- Use table_extractor, document_comparison and data_analysis when they help.
- If something needed to answer the question is missing, call request_more_evidence.

Finish with short analysis notes: key facts, calculation results, comparisons,
and a conclusion."""


def _get_llm() -> ChatOpenAI:
    """Returns a ChatOpenAI instance pointed at OpenRouter free models."""
    return ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL,
        temperature=0.1,
    )


def format_evidence(documents: List[Document]) -> str:
    """Numbers each chunk [E1], [E2]... with its source and page for citation."""
    blocks = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        blocks.append(
            f"[E{i}] Source: {source} | Page: {page}\n"
            f"{doc.page_content[:ANALYST_EVIDENCE_CHAR_LIMIT]}"
        )
    return "\n\n".join(blocks)


def run_analyst_agent(
    question: str,
    evidence: List[Document],
    pipeline: Any = None,
    retriever_fn: Optional[Callable[[str], List[Document]]] = None,
    max_loops: int = ANALYST_MAX_FEEDBACK_LOOPS,
) -> Dict[str, Any]:
    """
    Analyses the evidence and returns the result for the Answer Agent.

    Args:
        question:     The user's question.
        evidence:     Documents from run_retriever_agent().
        pipeline:     KnowledgeBasePipeline, used to fetch more evidence.
        retriever_fn: Optional custom function query -> Documents
                      (defaults to run_retriever_agent on `pipeline`).
        max_loops:    Max times the agent can go back to the Retriever.

    Returns:
        {"question", "evidence", "analysis", "tool_calls", "missing_evidence"}
    """
    if retriever_fn is None and pipeline is not None:
        from core.retriever_agent import run_retriever_agent
        retriever_fn = lambda query: run_retriever_agent(query, pipeline)

    agent = create_agent(_get_llm(), ANALYST_TOOLS, system_prompt=SYSTEM_PROMPT)
    evidence = list(evidence)

    for loop in range(max_loops + 1):
        print(f"  [Analyst] Pass {loop + 1} with {len(evidence)} evidence chunks")
        prompt = f"Question: {question}\n\nEvidence:\n{format_evidence(evidence)}"
        result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
        messages = result["messages"]

        # Collect every tool the agent called during this pass
        tool_calls = [call for msg in messages for call in getattr(msg, "tool_calls", [])]
        requested = [c["args"]["query"] for c in tool_calls if c["name"] == "request_more_evidence"]

        # Enough evidence, no retriever, or last allowed loop → stop
        if not requested or retriever_fn is None or loop == max_loops:
            break

        # Feedback loop: fetch the missing evidence (skip chunks we already have)
        known = {doc.page_content for doc in evidence}
        new_docs = [
            doc for query in requested for doc in retriever_fn(query)
            if doc.page_content not in known
        ]
        print(f"  [Analyst] Requested {requested} → {len(new_docs)} new chunks")
        if not new_docs:
            break
        evidence.extend(new_docs)

    return {
        "question": question,
        "evidence": evidence,
        "analysis": messages[-1].content,
        "tool_calls": tool_calls,
        "missing_evidence": requested,
    }
