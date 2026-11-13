"""
Answer Agent
============
A LangChain tool-calling agent that turns retrieved evidence and the Analyst Agent's
analysis into the final user response using 3 answer tools:
    1. citation_formatter
    2. source_formatter
    3. response_formatter

Workflow:
    (Retriever Agent) → Evidence
    (Analyst Agent)   → Analysis (Calculations, Comparisons, Tables)
    (Answer Agent)    → Final Response for User (Synthesize, format, cite)
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Union

from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from config.settings import (
    ANALYST_EVIDENCE_CHAR_LIMIT,
    LLM_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
)
from core.answer_tools import (
    ANSWER_TOOLS,
    citation_formatter,
    extract_cited_evidence,
    extract_evidence_ids,
    validate_evidence_citations,
    _normalize_single_source,
    response_formatter,
    source_formatter,
)


SYSTEM_PROMPT = """You are the Answer Agent in a document question-answering system.
You receive:
1. The user's question
2. Analysis data from the Analyst Agent (calculations, comparisons, tables, and reasoning)
3. Retrieved evidence chunks labeled with IDs: [E1], [E2], [E3], etc.

Your role is to craft the final, easy-to-read, comprehensive, and well-grounded response for the user:
- Synthesize the insights and analysis into a cohesive, professional answer.
- Ground every factual claim strictly in the provided evidence and analysis. Never guess or hallucinate.
- CITATION REQUIREMENT: You MUST tag every factual claim with its exact original evidence ID (e.g. [E1], [E2]) from the provided evidence chunks.
- Only cite evidence IDs that actually exist in the input bundle. Never invent IDs like [E99].
- Do not invent citation numbers [1], [2] yourself; write the exact [E1], [E2] IDs so the citation tool can validate and map them.
- Use `response_formatter` to organize the output with clear text, bullet points, calculation highlights, and comparison tables.
- If the evidence does not support an answer, explicitly say so rather than inventing facts."""


def _get_llm() -> ChatOpenAI:
    """Returns a ChatOpenAI instance pointed at OpenRouter free models."""
    return ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL,
        temperature=0.2,
    )


def format_evidence_bundle(documents: List[Document]) -> str:
    """Formats evidence chunks with [E1], [E2] labels, sources, and page numbers."""
    if not documents:
        return "No evidence documents provided."

    blocks = []
    for i, doc in enumerate(documents, start=1):
        normalized = _normalize_single_source(doc)
        source_clean = str(normalized["source"]).replace("\\", "/").split("/")[-1]
        page = normalized["page"]
        content = doc.page_content[:ANALYST_EVIDENCE_CHAR_LIMIT].strip()
        blocks.append(f"[E{i}] Source: {source_clean} | Page: {page}\n\"{content}\"")
    return "\n\n".join(blocks)


def format_input_bundle(
    question: str,
    analysis: Union[str, Dict[str, Any]],
    evidence: List[Document],
) -> str:
    """Builds the structured INPUT DATA BUNDLE as specified in the project architecture."""
    if isinstance(analysis, dict):
        analysis_str = json.dumps(analysis, indent=2)
    else:
        analysis_str = str(analysis).strip()

    evidence_str = format_evidence_bundle(evidence)

    return (
        f"INPUT DATA BUNDLE:\n"
        f"==================\n"
        f"QUESTION:\n{question}\n\n"
        f"ANALYSIS DATA:\n{analysis_str}\n\n"
        f"EVIDENCE & SOURCES:\n{evidence_str}"
    )


def run_answer_agent(
    question: str,
    analysis: Union[str, Dict[str, Any]],
    evidence: List[Document],
    source_style: str = "numbered",
    custom_llm: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Synthesizes evidence and analysis into the final user-facing response.

    Args:
        question:     The user's original query.
        analysis:     The text or structured output from the Analyst Agent.
        evidence:     The list of evidence Documents from the Retriever/Analyst.
        source_style: Formatting style for sources ('bullet', 'numbered', or 'table').
        custom_llm:   Optional LLM instance for testing or overrides.

    Returns:
        Dict containing:
            - "question": user query
            - "answer": formatted final answer string for the user
            - "sources": formatted sources block
            - "evidence": evidence documents used
            - "tool_calls": list of tools executed by the agent
    """
    llm = custom_llm or _get_llm()
    agent = create_agent(llm, ANSWER_TOOLS, system_prompt=SYSTEM_PROMPT)

    evidence_list = list(evidence)
    prompt = format_input_bundle(question, analysis, evidence_list)

    print(f"\n{'='*60}")
    print(f"  ANSWER AGENT")
    print(f"  Question: {question}")
    print(f"  Evidence chunks: {len(evidence_list)}")
    print(f"{'='*60}")

    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    messages = result["messages"]

    # Extract all tool calls made during execution
    tool_calls = [call for msg in messages for call in getattr(msg, "tool_calls", [])]
    raw_answer = messages[-1].content if messages else ""

    # Ignore an LLM-written bibliography. Rebuild it from verified original [E#] IDs.
    raw_answer = str(raw_answer)
    body_draft = re.split(
        r"(?im)^\s*(?:#{1,6}\s*)?(?:sources|references)\s*:?\s*$",
        raw_answer,
        maxsplit=1,
    )[0].strip()

    try:
        # Do not publish unsupported text when citations are missing or invented.
        cited_ids = validate_evidence_citations(body_draft, evidence_list)
        validated_answer = citation_formatter.invoke({
            "text": body_draft,
            "evidence_mapping": evidence_list,
            "style": "numeric",
        })
        sources_formatted = source_formatter.invoke({
            "sources": evidence_list,
            "style": source_style,
            "cited_ids": cited_ids,
        })
        cited_docs = extract_cited_evidence(body_draft, evidence_list)
        final_answer = f"{validated_answer}\n\n{sources_formatted}"
        citations_valid = True
        citation_error = None
    except ValueError as exc:
        # Failing closed is safer than displaying a claim with a fabricated citation.
        final_answer = (
            "I couldn't verify an answer with valid citations from the available "
            "documents. Please try a more specific question."
        )
        sources_formatted = "SOURCES:\nNo verified sources cited."
        cited_docs = []
        citations_valid = False
        citation_error = str(exc)

    print(f"  [Answer] Generated response ({len(final_answer)} chars, {len(tool_calls)} tool calls)")
    print(f"{'='*60}\n")

    return {
        "question": question,
        "answer": final_answer,
        "sources": sources_formatted,
        "evidence": evidence_list,
        "cited_evidence": cited_docs,
        "citations_valid": citations_valid,
        "citation_error": citation_error,
        "tool_calls": tool_calls,
    }

