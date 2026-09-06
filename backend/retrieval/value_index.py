"""
value_index.py — CHESS-style database-value grounding (GH#49 P3).

The generation prompt already shows `values: {…}` for LOW-cardinality TEXT
columns; misses like `'usa'` vs `'USA'` in higher-cardinality columns survive
and silently return 0 rows. This module keeps an in-memory distinct-value
index over TEXT columns and, given the question+evidence text, emits
"closest actual value" hints the generator copies verbatim.

Deterministic (no LLM), lazy (a table's values load on first use), and
best-effort (any DB failure degrades to no hints).
"""

from __future__ import annotations

import difflib
import logging
import re
import threading
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Per-column cap: columns with more distinct values than this are skipped —
# they are almost always free-text/name columns where fuzzy hints mislead.
MAX_DISTINCT_PER_COLUMN = 2000
# Spans we try to ground: quoted strings, Capitalized multi-words, ALL-CAPS /
# alphanumeric codes (e.g. 'USA', F-2, 201208).
_SPAN_RES = (
    re.compile(r"'([^']{2,60})'"),
    re.compile(r"\"([^\"]{2,60})\""),
    re.compile(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3})\b"),
    re.compile(r"\b([A-Z0-9][A-Z0-9\-/]{1,15})\b"),
)
# Words that match the Capitalized pattern but are never data values.
_STOPWORDS = {
    "what", "which", "who", "how", "many", "list", "show", "give", "the",
    "for", "and", "with", "from", "please", "state", "name", "names",
    "count", "average", "total", "number", "percentage", "difference",
}


def extract_value_spans(text: str) -> List[str]:
    """Candidate literal spans from question + evidence text, deduped."""
    seen: Dict[str, None] = {}
    for rx in _SPAN_RES:
        for m in rx.finditer(text or ""):
            span = m.group(1).strip()
            if len(span) < 2 or span.lower() in _STOPWORDS:
                continue
            seen.setdefault(span, None)
    return list(seen)


class ValueIndex:
    """
    Lazy distinct-value index over the live database's TEXT columns.

    ``conn_factory`` returns a psycopg2 connection (injected so tests can fake
    it and so the index reuses the store's read-only connection settings).
    """

    def __init__(self, conn_factory):
        self._conn_factory = conn_factory
        # table -> {column: [values]} (loaded once per table, thread-safe)
        self._by_table: Dict[str, Dict[str, List[str]]] = {}
        self._lock = threading.Lock()

    # ── loading ────────────────────────────────────────────────────────────

    def _load_table(self, table: str) -> Dict[str, List[str]]:
        with self._lock:
            if table in self._by_table:
                return self._by_table[table]
        cols: Dict[str, List[str]] = {}
        conn = None
        try:
            conn = self._conn_factory()
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema = 'public' AND table_name = %s "
                    "AND data_type IN ('text', 'character varying', 'character')",
                    (table,),
                )
                text_cols = [r[0] for r in cur.fetchall()]
                for col in text_cols:
                    # +1 over the cap detects overflow without counting all rows
                    cur.execute(
                        f'SELECT DISTINCT "{col}" FROM "{table}" '
                        f'WHERE "{col}" IS NOT NULL LIMIT %s',
                        (MAX_DISTINCT_PER_COLUMN + 1,),
                    )
                    values = [str(r[0]) for r in cur.fetchall()]
                    if 0 < len(values) <= MAX_DISTINCT_PER_COLUMN:
                        cols[col] = values
        except Exception as exc:  # degraded mode: no hints for this table
            logger.warning(f"[VALUE_INDEX] load failed for '{table}': {exc}")
        finally:
            try:
                if conn is not None:
                    conn.close()
            except Exception:
                pass
        with self._lock:
            self._by_table[table] = cols
        return cols

    # ── grounding ──────────────────────────────────────────────────────────

    def _best_match(
        self, span: str, values: List[str]
    ) -> Optional[Tuple[str, float]]:
        low = span.casefold()
        exact = next((v for v in values if v.casefold() == low), None)
        if exact is not None:
            return exact, 1.0
        close = difflib.get_close_matches(span, values, n=1, cutoff=0.85)
        if close:
            ratio = difflib.SequenceMatcher(None, span, close[0]).ratio()
            return close[0], ratio
        return None

    def resolve_spans(self, text: str, tables: List[str]) -> Dict[str, Optional[str]]:
        """
        Every value-like span in *text* mapped to the closest real value in
        *tables*, or to None when nothing in the data comes close (GH#73).

        ``ground()`` deliberately returns nothing for a span that matches
        nothing — there is no casing to correct — but "this value does not
        exist in the database" is the more useful signal of the two, and
        silence throws it away. This method exposes both outcomes so a caller
        can tell "grounded to X" from "matches nothing at all".

        Never raises: an empty dict means the index could not answer, which
        callers must treat as "unknown", never as "nothing matched".
        """
        out: Dict[str, Optional[str]] = {}
        try:
            spans = extract_value_spans(text)
            if not spans:
                return {}
            # _load_table swallows a failed load and returns {}, so an outage
            # is indistinguishable from a table with no indexed columns. Both
            # mean "could not check" — reporting None for every span would
            # tell the user a value does not exist in the database on the
            # strength of a connection error.
            loaded = {t: self._load_table(t) for t in tables}
            if not any(cols for cols in loaded.values()):
                return {}
            for span in spans:
                out[span] = None
                for cols in loaded.values():
                    for _col, values in cols.items():
                        m = self._best_match(span, values)
                        if m:
                            out[span] = m[0]
                            break
                    if out[span] is not None:
                        break
        except Exception as exc:
            logger.warning(f"[VALUE_INDEX] span resolution failed: {exc}")
            return {}
        return out

    def ground(self, text: str, tables: List[str]) -> List[str]:
        """
        Return hint lines for spans in *text* whose closest real value in
        *tables* differs from how the user wrote it. Exact matches produce no
        hint (nothing to correct). Never raises.
        """
        hints: List[str] = []
        try:
            spans = extract_value_spans(text)
            if not spans:
                return []
            for table in tables:
                for col, values in self._load_table(table).items():
                    for span in spans:
                        m = self._best_match(span, values)
                        if not m:
                            continue
                        actual, score = m
                        if actual == span:
                            continue  # written correctly — no correction needed
                        hints.append(
                            f"'{span}' → use '{actual}' ({table}.{col})"
                        )
        except Exception as exc:
            logger.warning(f"[VALUE_INDEX] grounding failed: {exc}")
        # Dedup, cap to keep the prompt tight.
        return list(dict.fromkeys(hints))[:8]


def render_value_hints(hints: List[str]) -> str:
    """The <value_hints> block appended to the schema context ('' when empty)."""
    if not hints:
        return ""
    lines = ["<value_hints>",
             "The user's wording differs from the ACTUAL stored values. In WHERE "
             "predicates copy the actual value exactly:"]
    lines += [f"  - {h}" for h in hints]
    lines.append("</value_hints>")
    return "\n".join(lines)
