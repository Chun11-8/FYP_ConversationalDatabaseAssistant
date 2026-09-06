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

from backend.agents.usage import invoke_tracked
from backend.agents.utils import PromptHarness, strip_reasoning

logger = logging.getLogger(__name__)

# Keys the planner is expected to emit. A parsed object must contain at least one
# of these to be treated as a usable plan.
PLAN_KEYS = (
    "target", "tables", "join_path", "filters", "evidence_predicates",
    "evidence_formula", "aggregation", "group_by", "having", "order_by", "set_op",
    # GH#76: 52% of the failures that had the right tables AND joins differed
    # from gold only in which aggregate was applied, and 100% of them carried
    # evidence describing the computation. Naming the grain and justifying the
    # aggregate forces that evidence to be applied rather than paraphrased.
    "grain", "aggregation_rationale",
)

# Fields rendered as bullet lists vs. scalars in the <query_plan> block.
_LIST_FIELDS = ("tables", "join_path", "filters", "evidence_predicates")
_SCALAR_FIELDS = ("target", "evidence_formula", "aggregation", "group_by",
                  "having", "order_by", "set_op", "grain",
                  "aggregation_rationale")


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

    # GH#65: a reasoning model's think block routinely contains braces, which
    # makes the outermost-{...} fallback below span both the reasoning and the
    # answer and fail to parse — losing the query plan silently.
    raw = strip_reasoning(raw)

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


def _same_shape(a: dict, b: dict) -> bool:
    """True when two plans agree on the dimensions the alternative exists to
    vary. Anything else — aliases, column order, wording — is not a real
    disagreement and would spend candidates on a duplicate."""
    def norm(p: dict, k: str) -> str:
        return " ".join(str(p.get(k) or "").lower().split())
    return all(norm(a, k) == norm(b, k)
               for k in ("aggregation", "group_by", "having", "grain", "target"))


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
    # GH#76: grain is rendered BEFORE the aggregate deliberately — "the highest
    # monthly X" means one row per month, summed, then the largest, and stating
    # the row unit first is what stops MAX() over raw rows.
    add_scalar("grain")
    for key in ("aggregation", "group_by", "having", "order_by", "set_op"):
        add_scalar(key)
    add_scalar("aggregation_rationale")

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

    _RATIONALE_INSTRUCTION = (
        "\n\nADDITIONALLY, emit two more keys:\n"
        "  \"grain\": one phrase naming the row unit of the ANSWER — what one "
        "output row represents (e.g. \"one row per month\", \"a single total "
        "over all rows\", \"one row per customer\").\n"
        "  \"aggregation_rationale\": one sentence saying WHY that aggregate "
        "and that grain follow from the question and the <domain_context> "
        "evidence. If the evidence states how to compute the value, quote the "
        "relevant fragment verbatim inside this sentence.\n"
        "Decide the grain BEFORE the aggregate: \"the highest monthly X\" "
        "means one row per month, summed, then the largest — not the largest "
        "single row."
    )

    def __init__(self, config: dict, llm: ChatOpenAI, rationale: bool = False):
        self.llm = llm
        self.dialect = config.get("db", {}).get("dialect", "postgres")
        # GH#76, default OFF: changes what the planner emits, so its effect on
        # accuracy is a hypothesis until the BIRD ablation is run.
        self.rationale = bool(rationale)

    # GH#76 Fix 1: a SECOND reading of the same question, differing on the one
    # dimension that causes 52% of the structurally-correct failures. Not a
    # "try again" — an instruction to disagree, so execution voting receives
    # decorrelated inputs instead of N restatements of one guess.
    _ALTERNATIVE_INSTRUCTION = (
        "\n\nA FIRST reading of this question has already been produced:\n"
        "{first}\n\n"
        "Produce a DIFFERENT, equally defensible reading. It must differ from "
        "the one above in its GRAIN or its AGGREGATION — not merely in "
        "phrasing, aliases or column order. Typical disagreements worth "
        "surfacing: one row per group versus a single overall total; summing "
        "within a group then taking the extreme, versus an extreme over raw "
        "rows; counting rows versus counting distinct entities. Keep the same "
        "tables and join path unless the alternative reading genuinely "
        "requires otherwise. If the question truly admits only one reading, "
        "return exactly the same plan."
    )

    # ── public API ──────────────────────────────

    def analyze_alternative(
        self,
        query: str,
        schema_context: str,
        first_plan: dict,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Optional[dict]:
        """
        An alternative plan disagreeing with ``first_plan`` on grain or
        aggregation (GH#76). None when planning fails, or when the alternative
        came back identical — in which case there is nothing to diversify and
        the caller should not spend candidates on it.
        """
        if not query or not query.strip() or not isinstance(first_plan, dict):
            return None
        prompt = self._build_prompt(query, schema_context, history or [])
        system_p = self._system_prompt() + self._ALTERNATIVE_INSTRUCTION.format(
            first=json.dumps(first_plan, indent=1)[:1200])
        try:
            raw = invoke_tracked(
                self.llm,
                [SystemMessage(content=system_p), HumanMessage(content=prompt)],
                "link_schema_alt").content
        except Exception as exc:
            logger.warning(f"[LINK-ALT] LLM call failed: {exc}")
            return None
        alt = _extract_json(raw)
        if not alt or not any(_is_set(alt.get(k)) for k in PLAN_KEYS):
            return None
        if _same_shape(alt, first_plan):
            logger.info("[LINK-ALT] alternative matched the first plan — discarded")
            return None
        return alt

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
            raw = invoke_tracked(self.llm, messages, "link_schema").content
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
        return self._base_system_prompt() + (
            self._RATIONALE_INSTRUCTION if self.rationale else "")

    def _base_system_prompt(self) -> str:
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
