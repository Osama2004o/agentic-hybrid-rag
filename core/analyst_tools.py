"""
Analyst Agent Tools
===================
Five LangChain tools the Analyst Agent can call:
    1. Calculator              Exact arithmetic and statistics
    2. Table Extractor         Turns a markdown/pipe table into rows & columns
    3. Document Comparison     Lines up facts from several documents side by side
    4. Data Analysis           Stats and ranking over a set of values
    5. Request More Evidence   Asks the Retriever Agent for missing information
"""

import ast
import json
import math
import operator
import statistics
from typing import Dict

from langchain_core.tools import tool


# ===================================================================
# Tool 1: Calculator
# ===================================================================

_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}

_FUNCTIONS = {
    "mean": lambda *v: statistics.fmean(v),
    "median": lambda *v: statistics.median(v),
    "stdev": lambda *v: statistics.stdev(v),
    "sum": lambda *v: sum(v),
    "min": min,
    "max": max,
    "abs": abs,
    "round": round,
    "sqrt": math.sqrt,
    "pct_change": lambda old, new: (new - old) / old * 100,
}


def _evaluate(node):
    """Evaluates only numbers, + - * / **, and the functions above (no eval())."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_evaluate(node.operand))
    if isinstance(node, ast.Call) and getattr(node.func, "id", None) in _FUNCTIONS:
        return _FUNCTIONS[node.func.id](*[_evaluate(arg) for arg in node.args])
    raise ValueError("Unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Calculate a math expression exactly. Never do arithmetic in your head.
    Supports + - * / ** and mean, median, stdev, sum, min, max, abs, round,
    sqrt, pct_change(old, new). Example: "mean(92, 95, 89)"."""
    try:
        result = _evaluate(ast.parse(expression, mode="eval").body)
        return str(round(result, 4))
    except Exception as e:
        return f"Error: {e}"


# ===================================================================
# Tool 2: Table Extractor
# ===================================================================

@tool
def table_extractor(text: str) -> str:
    """Extract a table written with | separators from evidence text and return
    it as JSON with "columns" and "rows"."""
    rows = []
    for line in text.splitlines():
        if line.count("|") < 2:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if all(set(cell) <= set("-: ") for cell in cells):  # skip |---|---| lines
            continue
        rows.append(cells)

    if len(rows) < 2:
        return "No table found."
    return json.dumps({"columns": rows[0], "rows": rows[1:]})


# ===================================================================
# Tool 3: Document Comparison
# ===================================================================

@tool
def document_comparison(facts: Dict[str, Dict[str, str]]) -> str:
    """Compare documents side by side.
    `facts` maps document -> {aspect: value}, e.g.
    {"paper_a.pdf": {"accuracy": "92%", "dataset": "CIFAR-10"},
     "paper_b.pdf": {"accuracy": "95%", "dataset": "CIFAR-10"}}.
    Returns each aspect with every document's value and whether they agree."""
    aspects = {aspect for values in facts.values() for aspect in values}

    comparison = {}
    for aspect in sorted(aspects):
        values = {doc: facts[doc].get(aspect, "not mentioned") for doc in facts}
        comparison[aspect] = {
            "values": values,
            "same_in_all": len(set(values.values())) == 1,
        }
    return json.dumps(comparison)


# ===================================================================
# Tool 4: Data Analysis
# ===================================================================

@tool
def data_analysis(values: Dict[str, float], higher_is_better: bool = True) -> str:
    """Analyse one metric across items, e.g. {"CNN": 92, "LSTM": 88, "GRU": 95}.
    Returns mean, median, min, max, the ranking, the best item, and how far
    each item is from the best."""
    ranking = sorted(values, key=values.get, reverse=higher_is_better)
    best = ranking[0]
    numbers = list(values.values())

    return json.dumps({
        "mean": round(statistics.fmean(numbers), 4),
        "median": statistics.median(numbers),
        "min": min(numbers),
        "max": max(numbers),
        "ranking": ranking,
        "best": best,
        "gap_from_best": {name: round(values[name] - values[best], 4) for name in ranking},
    })


# ===================================================================
# Tool 5: Request More Evidence
# ===================================================================

@tool
def request_more_evidence(query: str) -> str:
    """Ask the Retriever Agent for information that is missing from the
    evidence (e.g. a metric or a paper you need). Use a specific search query.
    Never guess missing facts - request them instead."""
    return f"Request sent to the Retriever Agent: {query}"


ANALYST_TOOLS = [
    calculator,
    table_extractor,
    document_comparison,
    data_analysis,
    request_more_evidence,
]
