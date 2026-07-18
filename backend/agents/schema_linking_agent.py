"""
schema_linking_agent.py — Schema-Linking & Question-Understanding Agent (Issue #17)

Runs BEFORE SQL generation. It reads the schema, any <domain_context> evidence,
and the user question, then emits a STRUCTURED PLAN that the SQL generator
follows. This forces the model to resolve entities → table.columns, map value
phrases, and (critically) resolve the JOIN PATH from the schema's foreign-key
section *before* writing any SQL — the step that single-shot generation skips on
moderate multi-hop questions.

The agent never writes SQL. Its output is a plain dict (the "query plan"), or
None when planning fails — in which case the pipeline degrades to direct
generation with no error.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from backend.agents.utils import PromptHarness

logger = logging.getLogger(__name__)

# Keys the planner is expected to emit. A parsed object must contain at least one
# of these to be treated as a usable plan.
PLAN_KEYS = (
    "target", "tables", "join_path", "filters", "evidence_predicates",
    "evidence_formula", "aggregation", "group_by", "having", "order_by", "set_op",
)

# Fields rendered as bullet lists vs. scalars in the <query_plan> block.
_LIST_FIELDS = ("tables", "join_path", "filters", "evidence_predicates")
_SCALAR_FIELDS = ("target", "evidence_formula", "aggregation", "group_by",
                  "having", "order_by", "set_op")


# ── value helpers ─────────────────────────────────────────────────────────────

def _is_set(value: Any) -> bool:
    """True when a plan field carries real content (not null/empty/'none')."""
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in ("", "null", "none", "(none)", "n/a")
    if isinstance(value, (list, tuple, dict)):
        return len(value) > 0
    return True


def _extract_json(raw: str) -> Optional[dict]:
    """
    Pull the JSON object out of an LLM response, tolerating ```json fences or
    surrounding prose. Returns the parsed dict, or None if nothing parses.
    """
    if not raw:
        return None

    # 1. Prefer an explicit ```json ... ``` (or bare ``` ... ```) fenced block.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1) if fenced else None

    # 2. Otherwise grab the outermost { ... } span.
    if candidate is None:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            candidate = raw[start : end + 1]

    if not candidate:
        return None
    try:
        obj = json.loads(candidate)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def render_query_plan(plan: Optional[dict]) -> str:
    """
    Render a parsed plan into the <query_plan> block injected into the generation
    prompt. Only non-empty fields are shown. Returns "" when there's nothing
    useful to render (so the generator falls back to plain decomposition).
    """
    if not plan or not isinstance(plan, dict):
        return ""

    lines: List[str] = ["<query_plan>"]

    def add_scalar(key: str) -> None:
        v = plan.get(key)
        if _is_set(v):
            lines.append(f"{key}: {v}")

    def add_list(key: str) -> None:
        v = plan.get(key)
        if not _is_set(v):
            return
        if isinstance(v, list):
            lines.append(f"{key}:")
            for item in v:
                if _is_set(item):
                    lines.append(f"  - {item}")
        else:
            lines.append(f"{key}: {v}")

    if _is_set(plan.get("target")):
        lines.append(f"target: {plan['target']}")
    add_list("tables")
    add_list("join_path")
    add_list("filters")
    add_list("evidence_predicates")
    # GH#49 P5 (TA-SQL): the evidence formula is transcribed, never paraphrased.
    if _is_set(plan.get("evidence_formula")):
        lines.append(f"evidence_formula (COPY VERBATIM): {plan['evidence_formula']}")
    for key in ("aggregation", "group_by", "having", "order_by", "set_op"):
        add_scalar(key)

    # Nothing beyond the opening tag → not worth injecting.
    if len(lines) <= 1:
        return ""

    lines.append("</query_plan>")
    lines.append(
        "Follow this plan: use exactly these tables and join conditions, apply "
        "every filter/predicate, and keep the aggregation/grouping/set-operation "
        "as specified. If an evidence_formula is given, transcribe it into the "
        "SQL VERBATIM — same operators, operands, and constants — never "
        "reinterpret it. Deviate only if the plan clearly contradicts the schema."
    )
    return "\n".join(lines)


# ─────────────────────────────────────────────
# Schema Linking Agent
# ─────────────────────────────────────────────

class SchemaLinkingAgent:
    """
    Produces a structured query plan from (question + schema + evidence) so the
    SQL generator can follow a resolved spec instead of deriving everything in
    one shot. Never raises to the caller — returns None on any failure.
    """

    def __init__(self, config: dict, llm: ChatOpenAI):
        self.llm = llm
        self.dialect = config.get("db", {}).get("dialect", "postgres")

    # ── public API ──────────────────────────────

    def analyze(
        self,
        query: str,
        schema_context: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Optional[dict]:
        """
        Return a query-plan dict, or None when planning is not possible
        (empty input, LLM error, unparseable output, or no useful fields).
        """
        if not query or not query.strip():
            return None

        prompt = self._build_prompt(query, schema_context, history or [])
        messages = [
            SystemMessage(content=self._system_prompt()),
            HumanMessage(content=prompt),
        ]
        try:
            raw = self.llm.invoke(messages).content
        except Exception as exc:
            logger.warning(f"[LINK] LLM call failed: {exc}")
            return None

        plan = _extract_json(raw)
        if not plan:
            logger.warning("[LINK] No JSON plan parsed from linker output")
            return None

        # Require at least one meaningful field, else treat as no plan.
        if not any(_is_set(plan.get(k)) for k in PLAN_KEYS):
            logger.warning("[LINK] Parsed plan had no usable fields")
            return None
        return plan

    # ── prompt construction ─────────────────────

    def _system_prompt(self) -> str:
        try:
            return PromptHarness.get("SCHEMA_LINKING_AGENT")
        except Exception:
            return (
                "You are a schema-linking module for a text-to-SQL system. Read "
                "the schema, the <domain_context> evidence, and the question, then "
                "output ONLY a JSON object with keys: target, tables, join_path, "
                "filters, evidence_predicates, evidence_formula (the evidence's "
                "arithmetic copied verbatim, else null), aggregation, group_by, "
                "having, order_by, set_op. Resolve joins only from the schema's foreign-key "
                "section. Use only tables/columns that appear in the schema. Do not "
                "write SQL."
            )

    def _build_prompt(
        self,
        query: str,
        schema_context: str,
        history: List[Dict[str, str]],
    ) -> str:
        history_block = "\n".join(
            f"{m.get('role', '').upper()}: {m.get('content', '')}" for m in history
        ) or "No prior context."
        return f"""
<schema_context>
{schema_context}
</schema_context>

<conversation_history>
{history_block}
</conversation_history>

<question>
{query}
</question>

Produce the JSON query plan now. Output ONLY the JSON object — no prose, no code fence.
""".strip()
