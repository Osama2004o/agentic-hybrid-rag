"""
Unit & Integration Tests for Analyst Agent and Tools
====================================================
Tests the 5 analyst tools and the Analyst Agent, including the feedback
loop back to the Retriever. The LLM is replaced by a scripted fake model.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from core.analyst_tools import (
    calculator,
    data_analysis,
    document_comparison,
    table_extractor,
)
from core.analyst_agent import format_evidence, run_analyst_agent


# ---------------------------------------------------------------------------
# Fixtures & Mocks
# ---------------------------------------------------------------------------

class FakeToolLLM(GenericFakeChatModel):
    """Fake chat model that replays scripted messages and accepts tools."""

    def bind_tools(self, tools, **kwargs):
        return self


def fake_llm(*replies):
    return FakeToolLLM(messages=iter(replies))


def tool_call(name, args, call_id="call_1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


@pytest.fixture
def evidence_docs():
    return [
        Document(
            page_content="| Model | Accuracy |\n|---|---|\n| CNN | 92 |\n| LSTM | 88 |",
            metadata={"source": "paper_a.pdf", "page": 4},
        ),
        Document(
            page_content="Our CNN reaches 95% accuracy.",
            metadata={"source": "paper_b.pdf", "page": 6},
        ),
    ]


# ---------------------------------------------------------------------------
# Tool Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("expression, expected", [
    ("(92 + 95 + 89) / 3", "92.0"),
    ("mean(92, 95, 89)", "92.0"),
    ("pct_change(80, 92)", "15.0"),
    ("-2 ** 2", "-4"),
])
def test_calculator(expression, expected):
    assert calculator.invoke({"expression": expression}) == expected


@pytest.mark.parametrize("expression", ["__import__('os')", "open('x')", "1 / 0"])
def test_calculator_rejects_bad_input(expression):
    assert calculator.invoke({"expression": expression}).startswith("Error")


def test_table_extractor(evidence_docs):
    table = json.loads(table_extractor.invoke({"text": evidence_docs[0].page_content}))
    assert table["columns"] == ["Model", "Accuracy"]
    assert table["rows"] == [["CNN", "92"], ["LSTM", "88"]]


def test_table_extractor_no_table():
    assert table_extractor.invoke({"text": "Plain sentence."}) == "No table found."


def test_document_comparison():
    result = json.loads(document_comparison.invoke({"facts": {
        "a.pdf": {"accuracy": "92%", "dataset": "CIFAR-10"},
        "b.pdf": {"accuracy": "95%", "dataset": "CIFAR-10"},
    }}))
    assert result["dataset"]["same_in_all"] is True
    assert result["accuracy"]["same_in_all"] is False
    assert result["accuracy"]["values"]["b.pdf"] == "95%"


def test_document_comparison_missing_aspect():
    result = json.loads(document_comparison.invoke({"facts": {
        "a.pdf": {"params": "5M"}, "b.pdf": {},
    }}))
    assert result["params"]["values"]["b.pdf"] == "not mentioned"


def test_data_analysis():
    result = json.loads(data_analysis.invoke({"values": {"CNN": 92, "LSTM": 88, "GRU": 95}}))
    assert result["best"] == "GRU"
    assert result["ranking"] == ["GRU", "CNN", "LSTM"]
    assert result["gap_from_best"]["LSTM"] == -7


def test_data_analysis_lower_is_better():
    result = json.loads(data_analysis.invoke({
        "values": {"CNN": 30, "LSTM": 45}, "higher_is_better": False,
    }))
    assert result["best"] == "CNN"


def test_format_evidence(evidence_docs):
    text = format_evidence(evidence_docs)
    assert "[E1] Source: paper_a.pdf | Page: 4" in text
    assert "[E2] Source: paper_b.pdf | Page: 6" in text


# ---------------------------------------------------------------------------
# Agent Tests
# ---------------------------------------------------------------------------

def test_analyst_agent_uses_calculator(evidence_docs):
    llm = fake_llm(
        tool_call("calculator", {"expression": "mean(92, 95)"}),
        AIMessage(content="Average CNN accuracy is 93.5 [E1][E2]."),
    )
    with patch("core.analyst_agent._get_llm", return_value=llm):
        result = run_analyst_agent("Average CNN accuracy?", evidence_docs)

    assert result["analysis"] == "Average CNN accuracy is 93.5 [E1][E2]."
    assert result["tool_calls"][0]["name"] == "calculator"
    assert result["missing_evidence"] == []


def test_analyst_agent_feedback_loop(evidence_docs):
    """Agent requests more evidence → Retriever is called → second pass finishes."""
    extra = Document(page_content="The CNN has 5M parameters.", metadata={"source": "paper_a.pdf"})
    retriever_fn = MagicMock(return_value=[extra])

    llm = fake_llm(
        # Pass 1: asks for the missing information
        tool_call("request_more_evidence", {"query": "CNN number of parameters"}),
        AIMessage(content="Complexity is missing."),
        # Pass 2: has everything it needs
        AIMessage(content="CNN: 92-95% accuracy, 5M parameters [E3]."),
    )
    with patch("core.analyst_agent._get_llm", return_value=llm):
        result = run_analyst_agent("Compare CNN accuracy and size", evidence_docs,
                                   retriever_fn=retriever_fn)

    retriever_fn.assert_called_once_with("CNN number of parameters")
    assert len(result["evidence"]) == 3
    assert result["missing_evidence"] == []
    assert "5M parameters" in result["analysis"]


def test_analyst_agent_without_retriever_reports_missing(evidence_docs):
    llm = fake_llm(
        tool_call("request_more_evidence", {"query": "CNN training time"}),
        AIMessage(content="Training time not available."),
    )
    with patch("core.analyst_agent._get_llm", return_value=llm):
        result = run_analyst_agent("CNN training time?", evidence_docs)

    assert result["missing_evidence"] == ["CNN training time"]
