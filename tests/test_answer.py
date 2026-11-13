"""
Unit & Integration Tests for Answer Agent and Answer Tools
==========================================================
Tests the 3 Answer Agent tools (citation_formatter, source_formatter,
response_formatter) and the Answer Agent synthesis workflow with fake LLMs.
Compatible with both pytest and python -m unittest / direct execution.
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from core.answer_tools import (
    citation_formatter,
    source_formatter,
    response_formatter,
    _format_comparison_dict,
    validate_evidence_citations,
)
from core.answer_agent import (
    format_evidence_bundle,
    format_input_bundle,
    run_answer_agent,
)
from core.orchestrator import Orchestrator, run_orchestrator


# ---------------------------------------------------------------------------
# Mocks & Test Fixtures
# ---------------------------------------------------------------------------

class FakeToolLLM(GenericFakeChatModel):
    """Fake chat model that replays scripted messages and accepts tools."""

    def bind_tools(self, tools, **kwargs):
        return self


def fake_llm(*replies):
    return FakeToolLLM(messages=iter(replies))


def tool_call(name, args, call_id="call_answer_1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def get_sample_evidence():
    return [
        Document(
            page_content="RAG optimizes LLM outputs by fetching up-to-date documents.",
            metadata={"source": "rag_overview.pdf", "page": 5},
        ),
        Document(
            page_content="Vector databases store embeddings for low-latency similarity search.",
            metadata={"source": "rag_overview.pdf", "page": 8},
        ),
        Document(
            page_content="CNN models achieve 92% accuracy on benchmark tests.",
            metadata={"source": "benchmark_report.pdf", "page": 12},
        ),
    ]


# ---------------------------------------------------------------------------
# Tool 1: Citation Formatter Tests
# ---------------------------------------------------------------------------

def test_citation_formatter_numeric():
    text = "RAG updates knowledge without fine-tuning [E1] and queries vector stores [E2]."
    formatted = citation_formatter.invoke({"text": text, "style": "numeric"})
    assert "[1]" in formatted
    assert "[2]" in formatted
    assert "[E1]" not in formatted
    assert "[E2]" not in formatted


def test_citation_formatter_bracket_style():
    text = "CNN achieves 92% accuracy [E1]."
    mapping = {"E1": {"source": "cnn_paper.pdf", "page_number": 4}}
    formatted = citation_formatter.invoke({
        "text": text,
        "evidence_mapping": mapping,
        "style": "bracket",
    })
    assert "[cnn_paper.pdf, p. 4]" in formatted


def test_citation_formatter_inline_style():
    text = "Model achieves high precision [E1]."
    mapping = {"E1": {"source": "report.pdf", "page_number": 10}}
    formatted = citation_formatter.invoke({
        "text": text,
        "evidence_mapping": mapping,
        "style": "inline",
    })
    assert "(Source: report.pdf, Page 10)" in formatted


def test_citation_formatter_empty():
    assert citation_formatter.invoke({"text": ""}) == ""


def test_citation_formatter_rejects_invalid_ids():
    text = "Valid statement [E1] and hallucinated claim [E99]."
    mapping = {"E1": {"source": "valid.pdf", "page": 2}}
    # Reject the draft rather than silently leaving a fabricated claim behind.
    with pytest.raises(ValueError, match="Unknown evidence ID"):
        citation_formatter.invoke({
            "text": text,
            "evidence_mapping": mapping,
            "style": "numeric",
        })


def test_duplicate_page_evidence_uses_same_number():
    docs = [
        Document(page_content="First chunk", metadata={"source": "a.pdf", "page": 0}),
        Document(page_content="Second chunk", metadata={"source": "a.pdf", "page": 0}),
        Document(page_content="Third chunk", metadata={"source": "b.pdf", "page": 3}),
    ]
    draft = "First fact [E1]. Second fact [E2]. Third fact [E3]."
    cited = citation_formatter.invoke({"text": draft, "evidence_mapping": docs})
    assert cited == "First fact [1]. Second fact [1]. Third fact [2]."
    sources = source_formatter.invoke({
        "sources": docs, "style": "numbered", "cited_ids": ["E1", "E2", "E3"]
    })
    assert sources == "SOURCES:\n[1] a.pdf, page 1\n[2] b.pdf, page 4"


def test_non_sequential_citation_order_and_sources_agree():
    docs = get_sample_evidence()
    draft = "CNN scored 92% [E3]. RAG retrieves context [E1]."
    answer = citation_formatter.invoke({"text": draft, "evidence_mapping": docs})
    sources = source_formatter.invoke({
        "sources": docs, "cited_ids": ["E3", "E1"], "style": "numbered"
    })
    assert answer == "CNN scored 92% [1]. RAG retrieves context [2]."
    assert sources.splitlines()[1:] == [
        "[1] benchmark_report.pdf, page 13",
        "[2] rag_overview.pdf, page 6",
    ]


def test_validation_rejects_numeric_and_missing_citations():
    docs = get_sample_evidence()
    with pytest.raises(ValueError, match="original"):
        validate_evidence_citations("Answer [1].", docs)
    with pytest.raises(ValueError, match="no verifiable"):
        validate_evidence_citations("Uncited answer.", docs)
    with pytest.raises(ValueError, match="E99"):
        validate_evidence_citations("Unsupported [E99].", docs)


def test_distinct_files_with_same_filename_do_not_merge():
    docs = [
        Document(page_content="North", metadata={"source": "a/report.pdf", "page": 0}),
        Document(page_content="South", metadata={"source": "b/report.pdf", "page": 0}),
    ]
    assert citation_formatter.invoke({
        "text": "North [E1]. South [E2].", "evidence_mapping": docs
    }) == "North [1]. South [2]."



# ---------------------------------------------------------------------------
# Tool 2: Source Formatter Tests
# ---------------------------------------------------------------------------

def test_source_formatter_bullet_deduplication(evidence_docs=None):
    docs = evidence_docs or get_sample_evidence()
    # Add a duplicate doc with same source and page
    docs_with_dup = docs + [
        Document(page_content="Duplicate chunk", metadata={"source": "rag_overview.pdf", "page": 5})
    ]
    result = source_formatter.invoke({"sources": docs_with_dup, "style": "bullet"})
    assert "SOURCES:" in result
    assert "• [1] rag_overview.pdf, page 6" in result
    assert "• [2] rag_overview.pdf, page 9" in result
    assert "• [3] benchmark_report.pdf, page 13" in result
    # Check deduplication: rag_overview.pdf, page 5 should appear only once
    assert result.count("rag_overview.pdf, page 6") == 1


def test_source_formatter_numbered(evidence_docs=None):
    docs = evidence_docs or get_sample_evidence()
    result = source_formatter.invoke({"sources": docs, "style": "numbered"})
    assert "SOURCES:" in result
    assert "[1] rag_overview.pdf, page 6" in result
    assert "[2] rag_overview.pdf, page 9" in result
    assert "[3] benchmark_report.pdf, page 13" in result


def test_source_formatter_table(evidence_docs=None):
    docs = evidence_docs or get_sample_evidence()
    result = source_formatter.invoke({"sources": docs, "style": "table"})
    assert "| Document | Page | Reference Excerpt |" in result
    assert "| 1 | rag_overview.pdf | 6 |" in result


def test_source_formatter_empty():
    result = source_formatter.invoke({"sources": []})
    assert "No verified sources cited" in result


def test_source_formatter_synchronizes_with_cited_ids():
    docs = get_sample_evidence()  # 3 docs: E1=rag, E2=rag, E3=benchmark
    # Only cite E3 and E1 (in that order)
    result = source_formatter.invoke({
        "sources": docs,
        "style": "numbered",
        "cited_ids": ["E3", "E1"],
    })
    # E3 was cited first, so it should be [1]
    assert "[1] benchmark_report.pdf, page 13" in result
    # E1 was cited second, so it should be [2]
    assert "[2] rag_overview.pdf" in result
    # E2 was never cited, so it should NOT be present in sources
    assert result.count("rag_overview.pdf") == 1


def test_correct_pdf_page_display():
    from core.answer_tools import _normalize_single_source
    # 0-based page index from PyMuPDF
    doc = Document(page_content="Test page", metadata={"source": "test.pdf", "page": 0})
    norm = _normalize_single_source(doc)
    assert norm["page"] == "1"  # 0-based page 0 converted to human page 1


# ---------------------------------------------------------------------------
# Tool 3: Response Formatter Tests
# ---------------------------------------------------------------------------

def test_response_formatter_structure():
    output = response_formatter.invoke({
        "direct_answer": "RAG enables dynamic updates to LLMs.",
        "key_points": [
            "Knowledge can be updated without retraining [1].",
            "External documents are retrieved dynamically [2].",
        ],
        "calculations": {"Average CNN Accuracy": "92%"},
        "comparisons": {"Accuracy": {"values": {"CNN": "92%", "LSTM": "88%"}, "same_in_all": False}},
        "sources_section": "SOURCES:\n• rag.pdf, page 5",
    })
    assert "RAG enables dynamic updates to LLMs." in output
    assert "- Knowledge can be updated without retraining [1]." in output
    assert "**Calculations & Quantitative Metrics:**" in output
    assert "**Average CNN Accuracy**: 92%" in output
    assert "**Comparative Analysis:**" in output
    assert "| Aspect |" in output
    assert "SOURCES:\n• rag.pdf, page 5" in output


def test_response_formatter_comparison_dict():
    comp = {
        "Accuracy": {"values": {"Paper A": "92%", "Paper B": "95%"}, "same_in_all": False},
        "Dataset": {"values": {"Paper A": "CIFAR-10", "Paper B": "CIFAR-10"}, "same_in_all": True},
    }
    table = _format_comparison_dict(comp)
    assert "| Aspect | Paper A | Paper B | Same Across All? |" in table
    assert "| Accuracy | 92% | 95% | ✗ |" in table
    assert "| Dataset | CIFAR-10 | CIFAR-10 | ✓ |" in table


# ---------------------------------------------------------------------------
# Agent Tests
# ---------------------------------------------------------------------------

def test_format_input_bundle():
    docs = get_sample_evidence()
    bundle = format_input_bundle(
        question="What are the advantages of RAG?",
        analysis="Average CNN accuracy is 92%.",
        evidence=docs,
    )
    assert "QUESTION:\nWhat are the advantages of RAG?" in bundle
    assert "ANALYSIS DATA:\nAverage CNN accuracy is 92%." in bundle
    assert "[E1] Source: rag_overview.pdf | Page: 6" in bundle


def test_answer_agent_basic_synthesis():
    docs = get_sample_evidence()
    llm = fake_llm(
        AIMessage(
            content=(
                "According to the documents, RAG allows dynamic retrieval [E1].\n\n"
                "SOURCES:\n[1] wrong_source.pdf, page 9"
            )
        )
    )
    result = run_answer_agent(
        question="What is RAG?",
        analysis="RAG retrieves relevant documents.",
        evidence=docs,
        custom_llm=llm,
    )
    assert "RAG allows dynamic retrieval [1]" in result["answer"]
    assert "SOURCES:\n[1] rag_overview.pdf, page 6" in result["answer"]
    assert "wrong_source.pdf" not in result["answer"]
    assert result["citations_valid"] is True
    assert len(result["evidence"]) == 3
    assert result["tool_calls"] == []


def test_answer_agent_rejects_answer_without_citations():
    docs = get_sample_evidence()[:1]
    llm = fake_llm(AIMessage(content="RAG optimizes outputs dynamically."))
    result = run_answer_agent(
        question="What does RAG do?",
        analysis="Dynamic optimization.",
        evidence=docs,
        custom_llm=llm,
    )
    assert result["citations_valid"] is False
    assert "RAG optimizes outputs dynamically" not in result["answer"]
    assert "No verified sources cited" in result["sources"]
    assert "rag_overview.pdf" not in result["answer"]


def test_answer_agent_rejects_premature_numeric_citations():
    result = run_answer_agent(
        question="What is RAG?",
        analysis="RAG retrieves context.",
        evidence=get_sample_evidence(),
        custom_llm=fake_llm(AIMessage(content="RAG retrieves context [1].")),
    )
    assert result["citations_valid"] is False
    assert "RAG retrieves context" not in result["answer"]



def test_answer_agent_with_tool_call():
    docs = get_sample_evidence()
    llm = fake_llm(
        tool_call("citation_formatter", {"text": "RAG improves models [E1]."}),
        AIMessage(content="RAG improves models [E1].\n\nSOURCES:\n[1] fake.pdf, page 30"),
    )
    result = run_answer_agent(
        question="How does RAG improve models?",
        analysis="Accuracy gains observed.",
        evidence=docs,
        custom_llm=llm,
    )
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["name"] == "citation_formatter"
    assert "RAG improves models [1]" in result["answer"]


def test_answer_agent_rejects_invalid_citations_without_publishing_claim():
    docs = get_sample_evidence()
    llm = fake_llm(AIMessage(content="Benchmark confirms CNN accuracy [E3]. Fabrication [E99]."))
    result = run_answer_agent(
        question="What is CNN accuracy?",
        analysis="Reported accuracy.",
        evidence=docs,
        custom_llm=llm,
    )
    assert result["citations_valid"] is False
    assert "Fabrication" not in result["answer"]
    assert "CNN accuracy" not in result["answer"]
    assert "No verified sources cited" in result["sources"]
    assert result["cited_evidence"] == []
    assert "E99" in result["citation_error"]


def test_answer_agent_rebuilds_sources_from_cited_evidence_only():
    docs = get_sample_evidence()
    result = run_answer_agent(
        question="What is CNN accuracy?",
        analysis="Benchmark reported 92%.",
        evidence=docs,
        custom_llm=fake_llm(AIMessage(content="CNN achieves 92% [E3].")),
    )
    assert result["citations_valid"] is True
    assert result["answer"] == "CNN achieves 92% [1].\n\nSOURCES:\n[1] benchmark_report.pdf, page 13"
    assert len(result["cited_evidence"]) == 1


def test_page_label_is_used_when_available():
    doc = Document(page_content="Example", metadata={
        "source": "manual.pdf", "page": 8, "page_label": "ix"
    })
    assert "Page: ix" in format_evidence_bundle([doc])
    assert "page ix" in source_formatter.invoke({"sources": [doc]})


def test_answer_agent_with_duplicate_document_pages():
    docs = [
        Document(page_content="A", metadata={"source": "shared.pdf", "page": 2}),
        Document(page_content="B", metadata={"source": "shared.pdf", "page": 2}),
    ]
    result = run_answer_agent(
        question="Two facts?", analysis="A and B", evidence=docs,
        custom_llm=fake_llm(AIMessage(content="A [E1]. B [E2].")),
    )
    assert result["answer"] == "A [1]. B [1].\n\nSOURCES:\n[1] shared.pdf, page 3"



# ---------------------------------------------------------------------------
# Orchestrator Integration Tests
# ---------------------------------------------------------------------------

def test_orchestrator_pipeline_flow():
    docs = get_sample_evidence()
    mock_retriever = MagicMock(return_value=docs)
    mock_analyst = MagicMock(return_value={
        "question": "Compare models",
        "evidence": docs,
        "analysis": "CNN achieved 92% accuracy, LSTM achieved 88%.",
        "tool_calls": [{"name": "calculator", "args": {"expression": "92 - 88"}}],
        "missing_evidence": [],
    })
    mock_answer = MagicMock(return_value={
        "question": "Compare models",
        "answer": "CNN outperformed LSTM by 4% [1].\n\nSOURCES:\n• benchmark_report.pdf, page 12",
        "sources": "SOURCES:\n• benchmark_report.pdf, page 12",
        "evidence": docs,
        "tool_calls": [],
    })

    with patch("core.orchestrator.run_retriever_agent", mock_retriever), \
         patch("core.orchestrator.run_analyst_agent", mock_analyst), \
         patch("core.orchestrator.run_answer_agent", mock_answer):

        result = run_orchestrator("Compare models", pipeline=MagicMock())

    assert mock_retriever.called
    assert mock_analyst.called
    assert mock_answer.called
    assert "CNN outperformed LSTM by 4% [1]" in result["answer"]
    assert "benchmark_report.pdf, page 12" in result["sources"]
    assert result["execution_time_seconds"] >= 0


if __name__ == "__main__":
    import unittest

    class TestAnswer(unittest.TestCase):
        def test_citations(self):
            test_citation_formatter_numeric()
            test_citation_formatter_bracket_style()
            test_citation_formatter_inline_style()
            test_citation_formatter_empty()
            test_citation_formatter_rejects_invalid_ids()
            test_duplicate_page_evidence_uses_same_number()
            test_non_sequential_citation_order_and_sources_agree()
            test_validation_rejects_numeric_and_missing_citations()
            test_distinct_files_with_same_filename_do_not_merge()

        def test_sources(self):
            test_source_formatter_bullet_deduplication()
            test_source_formatter_numbered()
            test_source_formatter_table()
            test_source_formatter_empty()
            test_source_formatter_synchronizes_with_cited_ids()
            test_correct_pdf_page_display()

        def test_response(self):
            test_response_formatter_structure()
            test_response_formatter_comparison_dict()

        def test_agent(self):
            test_format_input_bundle()
            test_answer_agent_basic_synthesis()
            test_answer_agent_rejects_answer_without_citations()
            test_answer_agent_rejects_premature_numeric_citations()
            test_answer_agent_with_tool_call()
            test_answer_agent_rejects_invalid_citations_without_publishing_claim()
            test_answer_agent_rebuilds_sources_from_cited_evidence_only()
            test_page_label_is_used_when_available()
            test_answer_agent_with_duplicate_document_pages()

        def test_orchestrator(self):
            test_orchestrator_pipeline_flow()

    print("Running Answer Agent tests...")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAnswer)
    runner = unittest.TextTestRunner(verbosity=2)
    res = runner.run(suite)
    if not res.wasSuccessful():
        exit(1)
    print("All tests passed successfully!")