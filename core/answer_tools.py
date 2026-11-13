"""
Answer Agent Tools
==================
Three LangChain tools for synthesizing evidence and analysis:
    1. Citation Formatter    Validates and formats evidence citations
    2. Source Formatter      Builds a references section with matching numbers
    3. Response Formatter    Structures the answer with text, tables and metrics

The first two tools share the same source/page identity rule, so multiple
chunks from one PDF page always receive the same reference number.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Union

from langchain_core.documents import Document
from langchain_core.tools import tool


_EVIDENCE_REF = re.compile(r"\[E(\d+)\]", re.IGNORECASE)
_NUMERIC_REF = re.compile(r"\[(\d+)\]")


def _normalize_single_source(item: Any) -> Dict[str, Any]:
    """Normalize a Document/dict, preferring an explicit printed page label.

    PyMuPDFLoader sets integer `page` to a zero-based index, so it is displayed
    as page + 1. `page_label` and `page_number`, when present, are already
    human-readable; string `page` values are retained to avoid double shifts.
    """
    if isinstance(item, Document):
        data = item.metadata
        content = item.page_content
    elif isinstance(item, dict):
        data = item
        content = item.get("content", item.get("text", ""))
    else:
        return {"source": str(item), "page": "?", "chunk_id": "", "content": ""}

    source = str(data.get("source") or data.get("doc") or data.get("file") or "unknown")
    page_label = data.get("page_label")
    page_number = data.get("page_number")
    if page_label is not None and str(page_label).strip():
        page = str(page_label).strip()
    elif page_number is not None and str(page_number).strip():
        page = str(page_number).strip()
    elif data.get("page_index") is not None:
        raw_page = data["page_index"]
        page = str(int(raw_page) + 1) if str(raw_page).strip().isdigit() else str(raw_page)
    else:
        raw_page = data.get("page", "?")
        if isinstance(raw_page, int) and not isinstance(raw_page, bool):
            page = str(raw_page + 1)
        elif str(raw_page).strip() == "0":
            page = "1"
        else:
            page = str(raw_page)

    return {
        "source": source,
        "page": page,
        "chunk_id": data.get("chunk_id", ""),
        "content": content,
    }


def _normalize_sources(sources_input: Union[str, List[Any], Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Accept Documents, metadata dictionaries, JSON, or plain source lines."""
    if isinstance(sources_input, str):
        try:
            parsed = json.loads(sources_input)
        except (ValueError, TypeError):
            parsed = None
        if isinstance(parsed, (list, dict)):
            return _normalize_sources(parsed)
        items = []
        for line in sources_input.splitlines():
            line = line.strip().lstrip("-*•0123456789.) ").strip()
            if not line:
                continue
            match = re.search(r"([^,|]+)[,|]\s*(?:page|p\.?)\s*[:=]?\s*([\w?.-]+)", line, re.IGNORECASE)
            if match:
                items.append({"source": match.group(1).strip(), "page": match.group(2).strip()})
            else:
                items.append({"source": line, "page": "?"})
        return items
    if isinstance(sources_input, dict):
        return [_normalize_single_source(sources_input)]
    if isinstance(sources_input, list):
        return [_normalize_single_source(s) for s in sources_input]
    return []


def _source_key(meta: Dict[str, Any]) -> tuple[str, str]:
    """Identity used consistently by the citation and source formatters.

    The full path distinguishes distinct files with the same basename; the
    displayed filename is shortened only when rendering the source list.
    """
    source = str(meta.get("source", "unknown")).replace("\\", "/").strip().lower()
    page = str(meta.get("page", "?")).strip().lower()
    return source, page


def _evidence_map(evidence_mapping: Optional[Union[str, Dict[str, Any], List[Any]]]) -> Dict[str, Dict[str, Any]]:
    if isinstance(evidence_mapping, str):
        try:
            evidence_mapping = json.loads(evidence_mapping)
        except (ValueError, TypeError) as exc:
            raise ValueError("evidence_mapping must be valid JSON if passed as a string") from exc
    if isinstance(evidence_mapping, list):
        return {f"E{i}": _normalize_single_source(item) for i, item in enumerate(evidence_mapping, start=1)}
    if isinstance(evidence_mapping, dict):
        result = {}
        for key, value in evidence_mapping.items():
            normalized_key = str(key).strip("[]").upper()
            if normalized_key.isdigit():
                normalized_key = f"E{int(normalized_key)}"
            elif normalized_key.startswith("E") and normalized_key[1:].isdigit():
                normalized_key = f"E{int(normalized_key[1:])}"
            if isinstance(value, dict):
                result[normalized_key] = _normalize_single_source(value)
            else:
                result[normalized_key] = {"source": str(value), "page": "?"}
        return result
    return {}


def extract_evidence_ids(text: str) -> List[str]:
    """Return unique original [E#] IDs in order of first appearance."""
    seen = set()
    ids = []
    for match in _EVIDENCE_REF.finditer(text):
        identifier = f"E{int(match.group(1))}"
        if identifier not in seen:
            seen.add(identifier)
            ids.append(identifier)
    return ids


def validate_evidence_citations(text: str, evidence_list: List[Document]) -> List[str]:
    """Validate the original LLM draft before publishing any sourced claims.

    A reference marker such as [1] is not accepted in the draft: only original
    [E#] IDs can be independently validated against the retrieved Documents.
    This checks reference integrity, not the factual entailment of every claim.
    """
    if _NUMERIC_REF.search(text):
        raise ValueError("Use original [E1], [E2] evidence IDs, not [1], [2] citation numbers")
    ids = extract_evidence_ids(text)
    if not ids:
        raise ValueError("The answer contains no verifiable evidence citations")
    missing = [cid for cid in ids if not 1 <= int(cid[1:]) <= len(evidence_list)]
    if missing:
        raise ValueError(f"Unknown evidence IDs: {', '.join(missing)}")
    return ids


# ===================================================================
# Tool 1: Citation Formatter
# ===================================================================

@tool
def citation_formatter(
    text: str,
    evidence_mapping: Optional[Union[str, Dict[str, Any], List[Any]]] = None,
    style: str = "numeric",
) -> str:
    """Replace [E#] with precise, validated citations.

    When evidence_mapping is supplied, unknown IDs raise ValueError. Duplicate
    chunks from the same document page share a numeric citation. Without a
    mapping, sequential conversion is offered for standalone formatting only.
    """
    if not text:
        return ""
    if evidence_mapping is not None and _NUMERIC_REF.search(text):
        raise ValueError("Draft citations must use original [E#] identifiers")
    if style not in {"numeric", "bracket", "inline"}:
        raise ValueError(f"Unsupported citation style: {style}")

    mapping = _evidence_map(evidence_mapping)
    number_by_source: Dict[tuple[str, str], int] = {}

    def replace(match: re.Match) -> str:
        evidence_id = f"E{int(match.group(1))}"
        if evidence_mapping is not None and evidence_id not in mapping:
            raise ValueError(f"Unknown evidence ID: {evidence_id}")
        meta = mapping.get(evidence_id)
        if meta is None:
            key = (evidence_id, "")
        else:
            key = _source_key(meta)
        if key not in number_by_source:
            number_by_source[key] = len(number_by_source) + 1
        number = number_by_source[key]
        if style == "bracket" and meta:
            filename = str(meta.get("source", "unknown")).replace("\\", "/").split("/")[-1]
            return f"[{filename}, p. {meta.get('page', '?')}]"
        if style == "inline" and meta:
            filename = str(meta.get("source", "unknown")).replace("\\", "/").split("/")[-1]
            return f"(Source: {filename}, Page {meta.get('page', '?')})"
        return f"[{number}]"

    return _EVIDENCE_REF.sub(replace, text).strip()


# ===================================================================
# Tool 2: Source Formatter
# ===================================================================

@tool
def source_formatter(
    sources: Union[str, List[Any]],
    style: str = "numbered",
    cited_ids: Optional[Union[str, List[str]]] = None,
) -> str:
    """Format a references section using the same order and deduplication as citation_formatter.

    When cited_ids is supplied, references are emitted in the first-appearance
    order of the original [E#] markers. When omitted, all sources are formatted.
    Even 'bullet' or 'table' styles include reference numbers for inline [1].
    """
    if style not in {"bullet", "numbered", "table"}:
        raise ValueError(f"Unsupported source style: {style}")
    raw_sources = _normalize_sources(sources)

    if isinstance(cited_ids, str):
        try:
            parsed = json.loads(cited_ids)
        except (ValueError, TypeError):
            parsed = None
        if isinstance(parsed, list):
            cited_ids = parsed
        else:
            cited_ids = [c.strip() for c in cited_ids.split(",") if c.strip()]

    if cited_ids is not None:
        selected = []
        for identifier in cited_ids:
            token = str(identifier).strip("[]").upper()
            idx_str = token[1:] if token.startswith("E") else token
            if not idx_str.isdigit() or not 1 <= int(idx_str) <= len(raw_sources):
                raise ValueError(f"Unknown cited evidence ID: {identifier}")
            selected.append(raw_sources[int(idx_str) - 1])
        raw_sources = selected

    unique_sources = []
    seen = set()
    for src in raw_sources:
        key = _source_key(src)
        if key not in seen:
            seen.add(key)
            unique_sources.append(src)

    if not unique_sources:
        return "SOURCES:\nNo verified sources cited."

    if style == "table":
        lines = ["### Sources", "| # | Document | Page | Reference Excerpt |", "|---|---|---|---|"]
        for number, src in enumerate(unique_sources, 1):
            filename = str(src["source"]).replace("\\", "/").split("/")[-1]
            excerpt = str(src.get("content") or "").replace("\n", " ").strip()
            excerpt = (excerpt[:60] + "...") if len(excerpt) > 60 else (excerpt or "—")
            excerpt = excerpt.replace("|", "\\|")
            lines.append(f"| {number} | {filename} | {src['page']} | {excerpt} |")
        return "\n".join(lines)

    lines = ["SOURCES:"]
    for number, src in enumerate(unique_sources, 1):
        filename = str(src["source"]).replace("\\", "/").split("/")[-1]
        ref = f"[{number}] {filename}, page {src['page']}"
        lines.append(f"• {ref}" if style == "bullet" else ref)
    return "\n".join(lines)


# ===================================================================
# Tool 3: Response Formatter
# ===================================================================

@tool
def response_formatter(
    direct_answer: str,
    key_points: Optional[Union[str, List[str]]] = None,
    calculations: Optional[Union[str, Dict[str, Any], List[str]]] = None,
    comparisons: Optional[Union[str, Dict[str, Any]]] = None,
    sources_section: Optional[str] = None,
) -> str:
    """Structures and formats the final answer into a polished, executive-ready response.

    Args:
        direct_answer: The synthesized, easy-to-read direct answer to the user's question.
        key_points: List of bullet points or detailed findings.
        calculations: Any mathematical or statistical results from the Analyst calculator/data_analysis.
        comparisons: Comparison data (can be a markdown table, dictionary, or comparison text).
        sources_section: The formatted SOURCES block from source_formatter.

    Returns:
        A complete, structured Markdown document ready for the user.
    """
    sections = []

    # 1. Direct Answer
    if direct_answer:
        sections.append(direct_answer.strip())

    # 2. Key Points / Structured Insights
    if key_points:
        points_list = []
        if isinstance(key_points, str):
            try:
                parsed = json.loads(key_points)
                if isinstance(parsed, list):
                    points_list = parsed
                else:
                    points_list = [key_points]
            except Exception:
                points_list = [line.strip() for line in key_points.splitlines() if line.strip()]
        elif isinstance(key_points, list):
            points_list = key_points

        if points_list:
            formatted_points = []
            for pt in points_list:
                cleaned = str(pt).lstrip("-*•0123456789.) ").strip()
                if cleaned:
                    formatted_points.append(f"- {cleaned}")
            if formatted_points:
                sections.append("\n".join(formatted_points))

    # 3. Calculation Highlights
    if calculations:
        calc_text = ""
        if isinstance(calculations, str):
            try:
                calc_dict = json.loads(calculations)
                if isinstance(calc_dict, dict):
                    calc_lines = [f"- **{k}**: {v}" for k, v in calc_dict.items()]
                    calc_text = "\n".join(calc_lines)
                else:
                    calc_text = calculations
            except Exception:
                calc_text = calculations
        elif isinstance(calculations, dict):
            calc_lines = [f"- **{k}**: {v}" for k, v in calculations.items()]
            calc_text = "\n".join(calc_lines)
        elif isinstance(calculations, list):
            calc_text = "\n".join(f"- {c}" for c in calculations)

        if calc_text.strip():
            sections.append(f"**Calculations & Quantitative Metrics:**\n{calc_text}")

    # 4. Comparisons / Tabular Data
    if comparisons:
        comp_text = ""
        if isinstance(comparisons, str):
            try:
                comp_obj = json.loads(comparisons)
                if isinstance(comp_obj, dict):
                    comp_text = _format_comparison_dict(comp_obj)
                else:
                    comp_text = comparisons
            except Exception:
                comp_text = comparisons
        elif isinstance(comparisons, dict):
            comp_text = _format_comparison_dict(comparisons)

        if comp_text.strip():
            sections.append(f"**Comparative Analysis:**\n{comp_text}")

    # 5. Sources Section
    if sources_section and sources_section.strip():
        sections.append(sources_section.strip())

    return "\n\n".join(sections)


def _format_comparison_dict(comp_dict: Dict[str, Any]) -> str:
    """Converts a document_comparison dictionary into a clean markdown table."""
    # Check if format is aspect -> {"values": {doc: val}, "same_in_all": bool}
    first_val = next(iter(comp_dict.values()), None)
    if isinstance(first_val, dict) and "values" in first_val:
        # Extract all document names
        docs = sorted({doc for item in comp_dict.values() for doc in item.get("values", {})})
        header = "| Aspect | " + " | ".join(docs) + " | Same Across All? |"
        sep = "|---|" + "|".join(["---"] * len(docs)) + "|:---:|"
        rows = [header, sep]
        for aspect, data in comp_dict.items():
            vals = [str(data.get("values", {}).get(d, "—")) for d in docs]
            same = "✓" if data.get("same_in_all") else "✗"
            rows.append(f"| {aspect} | " + " | ".join(vals) + f" | {same} |")
        return "\n".join(rows)

    # General dict fallback
    lines = ["| Item | Details |", "|---|---|"]
    for k, v in comp_dict.items():
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines)


def extract_cited_evidence(
    text: str,
    evidence_list: List[Document],
) -> List[Document]:
    """Finds all valid [E1], [E2] tokens in text in order of appearance,
    and returns the corresponding Document objects from evidence_list (deduplicated)."""
    found = re.findall(r"\[E(\d+)\]", text, re.IGNORECASE)
    cited_docs: List[Document] = []
    seen_indices = set()
    for num_str in found:
        idx = int(num_str) - 1  # convert 1-based E index to 0-based
        if 0 <= idx < len(evidence_list):
            if idx not in seen_indices:
                seen_indices.add(idx)
                cited_docs.append(evidence_list[idx])
    return cited_docs


# ===================================================================
# Exported Tools List
# ===================================================================

ANSWER_TOOLS = [
    citation_formatter,
    source_formatter,
    response_formatter,
]
