"""
db_agent.py — Minimal Flow with Full Thought Process Visibility
Location : backend/agents/db_agent.py
Prompts  : backend/agents/prompts.md

Flow:
  [1] Load prompt from prompts.md
  [2] Classify intent
  [3] Build prompt with schema
  [4] Generate top-K SQL candidates
  [5] Rerank → pick best
  [6] Validate (syntax → safety)
  [7] Return result

No memory, no schema retriever, no response optimizer.
Pass schema_context as a plain string for now.
"""

from __future__ import annotations

import re
import logging
from pathlib import Path
from functools import lru_cache
from typing import Any, Dict, List

import sqlglot
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# ─────────────────────────────────────────────
# Logging — shows thought process in terminal
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("db_agent")


def _step(label: str, detail: str = ""):
    """Print a visible step marker so you can follow the agent's flow."""
    print(f"\n{'='*60}")
    print(f"  STEP ▶  {label}")
    if detail:
        print(f"  {detail}")
    print(f"{'='*60}")


def _show(label: str, value: Any):
    """Print a labelled value for inspection."""
    print(f"\n  [{label}]\n  {str(value)[:600]}")


# ─────────────────────────────────────────────
# Prompt Harness — loads sections from prompts.md
# ─────────────────────────────────────────────

class PromptHarness:
    """
    Reads backend/agents/prompts.md once and caches all sections.
    Usage: PromptHarness.get("SQL_SYSTEM")
    """

    _PROMPTS_PATH = Path(__file__).parent / "prompts_demo.md"

    @classmethod
    @lru_cache(maxsize=1)
    def _load(cls) -> Dict[str, str]:
        _step("PromptHarness.load", f"Reading: {cls._PROMPTS_PATH}")

        raw = cls._PROMPTS_PATH.read_text()
        sections = re.split(r"##\s+\[SECTION:\s*(\w+)\]", raw)
        # sections = [preamble, name1, body1, name2, body2, ...]

        result = {}
        for i in range(1, len(sections), 2):
            name = sections[i].strip()
            body = re.sub(r"<!--.*?-->", "", sections[i + 1], flags=re.DOTALL).strip()
            body = re.sub(r"^---\s*$", "", body, flags=re.MULTILINE).strip()
            result[name] = body

        _show("Sections found", list(result.keys()))
        return result

    @classmethod
    def get(cls, section: str) -> str:
        data = cls._load()
        if section not in data:
            raise KeyError(
                f"Section '{section}' not found in prompts.md. "
                f"Available: {list(data.keys())}"
            )
        prompt = data[section]
        _show(f"PromptHarness.get({section})", prompt[:300] + "..." if len(prompt) > 300 else prompt)
        return prompt


# ─────────────────────────────────────────────
# Safety Patterns
# ─────────────────────────────────────────────

_BLOCKED_KW = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|REPLACE|MERGE)\b",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────
# SQL Validator Agent
# ─────────────────────────────────────────────

class SQLValidatorAgent:

    def __init__(self, dialect: str = "postgresql"):
        self.dialect = dialect

    def validate(self, sql: str) -> Dict[str, Any]:
        _step("SQLValidatorAgent.validate")
        _show("Input SQL", sql)

        if not sql:
            result = {"valid": False, "error": "Empty SQL received"}
            _show("Result", result)
            return result

        # ── Stage 1: Syntax ──
        print("\n  >> Stage 1: Syntax check (sqlglot)")
        ok, err = self._syntax(sql)
        _show("Syntax OK", ok)
        if not ok:
            result = {"valid": False, "error": f"Syntax: {err}"}
            _show("Result", result)
            return result

        # ── Stage 2: Safety ──
        print("\n  >> Stage 2: Safety check (DML block)")
        ok, err = self._safety(sql)
        _show("Safety OK", ok)
        if not ok:
            result = {"valid": False, "error": f"Safety: {err}"}
            _show("Result", result)
            return result

        result = {"valid": True, "error": None}
        _show("Result", result)
        return result

    def _syntax(self, sql: str):
        try:
            sqlglot.parse_one(sql, dialect=self.dialect)
            return True, None
        except sqlglot.errors.ParseError as e:
            return False, str(e)

    def _safety(self, sql: str):
        match = _BLOCKED_KW.search(sql)
        if match:
            return False, f"Blocked keyword detected: '{match.group()}'"
        return True, None


# ─────────────────────────────────────────────
# SQL Code Agent
# ─────────────────────────────────────────────

class SQLCodeAgent:

    TOP_K = 3

    def __init__(self, llm: ChatOpenAI, dialect: str = "postgres"):
        self.llm = llm
        self.dialect = dialect
        self.validator = SQLValidatorAgent(dialect)

    # ── Public entry point ───────────────────────
    
    
    def run(self, query: str, schema_context: str) -> Dict[str, Any]:
        """
        Full minimal flow with thought process printed at every stage.
        Returns: { intent, best_sql, all_sqls, validation }
        """
        print("\n" + "█" * 60)
        print("  db_agent.run() — Starting pipeline")
        print("█" * 60)
        _show("User Query", query)
        _show("Schema Context (truncated)", schema_context[:300])

        # ── Step 1: Classify intent ──────────────
        intent = self._classify_intent(query)
        if intent == "WRITE_BLOCKED":
            return {
                "intent": intent,
                "best_sql": "",
                "all_sqls": [],
                "validation": {
                    "valid": False,
                    "error": "Write operations are not permitted. This system is read-only."
                }
            }
        if intent == "GENERAL":
            return {
                "intent": intent,
                "best_sql": "",
                "all_sqls": [],
                "validation": {"valid": False, "error": "General question — no SQL needed."}
            }

        # ── Step 2: Build prompt ─────────────────
        prompt = self._build_prompt(query, schema_context)

        # ── Step 3: Generate SQL candidates ──────
        sqls = self._generate(prompt)
        if not sqls:
            return {
                "intent": intent,
                "best_sql": "",
                "all_sqls": [],
                "validation": {"valid": False, "error": "LLM returned no SQL blocks."}
            }

        # ── Step 4: Rerank ────────────────────────
        best = self._rerank(sqls, query) if len(sqls) > 1 else sqls[0]

        # ── Step 5: Validate ──────────────────────
        validation = self.validator.validate(best)

        print("\n" + "█" * 60)
        print("  db_agent.run() — Pipeline complete")
        print("█" * 60)

        return {
            "intent": intent,
            "best_sql": best,
            "all_sqls": sqls,
            "validation": validation,
        }

    # ── Stage implementations ────────────────────

    def _classify_intent(self, query: str) -> str:
        _step("Stage 1 — Intent Classification", "Asking LLM to classify query intent")

        system_prompt = PromptHarness.get("INTENT")

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query),
        ]

        print("\n  >> Calling LLM for intent...")
        raw = self.llm.invoke(messages).content.strip().upper()

        # Hard override: detect DML keywords regardless of LLM
        if _BLOCKED_KW.search(query):
            raw = "WRITE_BLOCKED"
            print("  >> Hard override: DML keyword detected — forcing WRITE_BLOCKED")

        valid_intents = {"READ", "GENERAL", "WRITE_BLOCKED", "AMBIGUOUS"}
        intent = raw if raw in valid_intents else "AMBIGUOUS"

        _show("Intent classified as", intent)
        return intent

    def _build_prompt(self, query: str, schema_context: str) -> str:
        _step("Stage 2 — Prompt Construction", "Injecting schema context into user prompt")

        prompt = (
            f"<schema>\n{schema_context}\n</schema>\n\n"
            f"User query: {query}\n\n"
            f"Generate {self.TOP_K} distinct SQL query candidates.\n"
            "Requirements:\n"
            "  - SELECT only, no DML\n"
            "  - Explicit column names, no SELECT *\n"
            "  - Include LIMIT 100\n"
            "  - Only use tables/columns from <schema>\n"
            "  - Wrap each in ```sql ... ``` fences"
        )

        _show("Built Prompt (truncated)", prompt[:400])
        return prompt

    def _generate(self, prompt: str) -> List[str]:
        _step("Stage 3 — SQL Generation", f"Requesting {self.TOP_K} SQL candidates from LLM")

        system_prompt = PromptHarness.get("SQL_SYSTEM")

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt),
        ]

        print("\n  >> Calling LLM for SQL generation...")
        raw = self.llm.invoke(messages).content
        _show("Raw LLM Output", raw)

        sqls = re.findall(r"```sql\s*(.*?)```", raw, re.DOTALL | re.IGNORECASE)
        sqls = [s.strip() for s in sqls if s.strip()]

        print(f"\n  >> Extracted {len(sqls)} SQL block(s)")
        for i, sql in enumerate(sqls):
            _show(f"Candidate {i+1}", sql)

        return sqls

    def _rerank(self, sqls: List[str], query: str) -> str:
        _step("Stage 4 — Reranking", f"LLM judge picking best from {len(sqls)} candidates")

        system_prompt = PromptHarness.get("SQL_RERANK")

        candidates = "\n\n".join(
            f"Option {i+1}:\n```sql\n{s}\n```" for i, s in enumerate(sqls)
        )
        user_message = (
            f"User query: {query}\n\n"
            f"{candidates}\n\n"
            "Which is best? Reply with only the number."
        )

        print("\n  >> Calling LLM for reranking judgment...")
        raw = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]).content.strip()

        _show("LLM judge raw answer", raw)

        try:
            idx = int(re.search(r"\d", raw).group()) - 1
            idx = max(0, min(idx, len(sqls) - 1))
        except Exception:
            print("  >> Could not parse judge answer — defaulting to Candidate 1")
            idx = 0

        _show(f"Selected", f"Candidate {idx+1}")
        _show("Best SQL", sqls[idx])
        return sqls[idx]

def pick_schema(query: str, schemas: dict, llm) -> tuple[str, str]:
        """LLM picks the most relevant schema based on the query."""
        
        schema_descriptions = "\n".join(
            f"{name}: {list(set(word.split('(')[0] for word in s.split()))[:5]}"
            for name, s in schemas.items()
        )
        
        raw = llm.invoke([
            SystemMessage(content=(
                "You are a schema router. Given a user query and available schemas, "
                "pick the most relevant schema name. "
                "Reply with ONLY the schema name. No explanation."
            )),
            HumanMessage(content=(
                f"Available schemas:\n{schema_descriptions}\n\n"
                f"User query: {query}\n\n"
                "Which schema name is most relevant?"
            )),
        ]).content.strip().lower()

        # Fallback to first schema if LLM returns something unexpected
        name = raw if raw in schemas else list(schemas.keys())[0]
        print(f"\n  [Schema Router] Query routed to → '{name}'")
        return name, schemas[name]
# ─────────────────────────────────────────────
# Manual test — python agents/db_agent.py
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv("backend/config/secrets.env")
    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=OPENAI_API_KEY,
    )

    agent = SQLCodeAgent(llm)

    SCHEMAS = {
    "university": """
        students(student_id INT PK, name TEXT, gpa FLOAT, department TEXT)
        enrollments(enrollment_id INT PK, student_id INT FK->students, course_id INT FK->courses, semester TEXT, grade TEXT)
        courses(course_id INT PK, course_name TEXT, credits INT)
    """,
    "ecommerce": """
        orders(order_id INT PK, customer_id INT FK->customers, total FLOAT, status TEXT, created_at DATE)
        customers(customer_id INT PK, name TEXT, email TEXT, country TEXT)
        products(product_id INT PK, name TEXT, price FLOAT, stock INT)
    """,
    "hr": """
        employees(employee_id INT PK, name TEXT, department TEXT, salary FLOAT, hire_date DATE)
        departments(department_id INT PK, name TEXT, manager_id INT FK->employees)
    """,
}


    TEST_QUERIES = [
        "Show me top 5 product sold",  # READ
        "DELETE FROM students WHERE student_id = 1",                      # WRITE_BLOCKED
        "How do I write a JOIN in PostgreSQL?",                           # GENERAL
    ]

    for q in TEST_QUERIES:
        schema_name, schema_context = pick_schema(q, SCHEMAS, llm)  # ← auto pick
        result = agent.run(query=q, schema_context=schema_context)
        print("\n" + "─" * 60)
        print("  FINAL RESULT")
        print("─" * 60)
        print(f"  Intent   : {result['intent']}")
        print(f"  Valid    : {result['validation']['valid']}")
        print(f"  Error    : {result['validation']['error']}")
        print(f"  Best SQL :\n{result['best_sql']}")
        input("\n  Press Enter for next query...\n")