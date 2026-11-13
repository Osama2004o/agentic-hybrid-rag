"""
Orchestrator — Complete Agentic Hybrid RAG Workflow
===================================================
Coordinates the full 3-agent pipeline as specified in the project architecture:
    1. Retriever Agent: Searches vector DB and keyword index to gather evidence chunks
    2. Analyst Agent:   Reasons over evidence (calculations, comparisons, tables) with
                        an autonomous feedback loop back to the Retriever if missing facts
    3. Answer Agent:    Synthesizes evidence and analysis into the final formatted response
                        with precise page citations and references.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from core.analyst_agent import run_analyst_agent
from core.answer_agent import run_answer_agent
from core.retriever_agent import run_retriever_agent


class Orchestrator:
    """
    Coordinates execution across the 3 specialized agents:
    Retriever Agent → Analyst Agent (with feedback loop) → Answer Agent.
    """

    def __init__(
        self,
        pipeline: Any = None,
        max_analyst_loops: int = 2,
        source_style: str = "numbered",
    ):
        """
        Args:
            pipeline: KnowledgeBasePipeline instance with ChromaDB and embeddings.
            max_analyst_loops: Maximum feedback loop iterations for the Analyst.
            source_style: Format for sources ('bullet', 'numbered', or 'table').
        """
        self.pipeline = pipeline
        self.max_analyst_loops = max_analyst_loops
        self.source_style = source_style

    def run(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Executes the end-to-end 3-agent RAG workflow for a user query.

        Args:
            query: The user's question.
            chat_history: Optional conversation history.

        Returns:
            Dict containing the final answer, evidence, analysis, and execution trace.
        """
        start_time = time.time()
        print(f"\n{'#'*70}")
        print(f"  STARTING ORCHESTRATOR WORKFLOW")
        print(f"  Query: {query}")
        print(f"{'#'*70}\n")

        # Step 1: Retriever Agent
        print(">>> [Phase 1/3] Invoking Retriever Agent...")
        evidence_docs = run_retriever_agent(
            query=query,
            pipeline=self.pipeline,
            chat_history=chat_history,
        )
        print(f">>> [Phase 1/3] Retriever Agent finished with {len(evidence_docs)} chunks.")

        # Step 2: Analyst Agent (with feedback loop to retriever)
        print("\n>>> [Phase 2/3] Invoking Analyst Agent...")
        analyst_bundle = run_analyst_agent(
            question=query,
            evidence=evidence_docs,
            pipeline=self.pipeline,
            max_loops=self.max_analyst_loops,
        )
        final_evidence = analyst_bundle.get("evidence", evidence_docs)
        analysis_text = analyst_bundle.get("analysis", "")
        print(f">>> [Phase 2/3] Analyst Agent completed analysis.")

        # Step 3: Answer Agent
        print("\n>>> [Phase 3/3] Invoking Answer Agent...")
        answer_bundle = run_answer_agent(
            question=query,
            analysis=analysis_text,
            evidence=final_evidence,
            source_style=self.source_style,
        )
        print(">>> [Phase 3/3] Answer Agent completed final response.")

        elapsed = round(time.time() - start_time, 2)
        print(f"\n{'#'*70}")
        print(f"  ORCHESTRATOR WORKFLOW COMPLETE ({elapsed}s)")
        print(f"{'#'*70}\n")

        return {
            "question": query,
            "answer": answer_bundle["answer"],
            "sources": answer_bundle["sources"],
            "evidence": final_evidence,
            "analysis": analysis_text,
            "analyst_tool_calls": analyst_bundle.get("tool_calls", []),
            "answer_tool_calls": answer_bundle.get("tool_calls", []),
            "execution_time_seconds": elapsed,
        }


def run_orchestrator(
    query: str,
    pipeline: Any = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
    max_analyst_loops: int = 2,
    source_style: str = "numbered",
) -> Dict[str, Any]:
    """Convenience function to run the complete 3-agent RAG workflow."""
    orchestrator = Orchestrator(
        pipeline=pipeline,
        max_analyst_loops=max_analyst_loops,
        source_style=source_style,
    )
    return orchestrator.run(query=query, chat_history=chat_history)

