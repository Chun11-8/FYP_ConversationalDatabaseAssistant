"""
Tests: few-shot examples must not name a real evaluation database.

`generate()` substitutes DEFAULT_FEWSHOTS whenever the caller passes no
few-shots, and the BIRD harness always passes `few_shots=[]`. So these examples
are shown to EVERY question in EVERY database.

They previously used two real BIRD databases, debit_card_specializing
(`customers`, `yearmonth`, `transactions_1k`) and student_club (`event`,
`attendance`, `member`, `budget`). A thrombosis or california_schools question
was therefore handed worked examples naming tables that do not exist in its own
database, and models occasionally copied the names instead of the pattern —
producing SQL for the wrong database, rejected as "tables not found", which
reads as a validator bug because those tables really do exist somewhere.

Offline. Run from repo root:
    pytest backend/tests/test_fewshot_neutrality.py -v
"""

from __future__ import annotations

import re

from backend.agents.sql_code_agent import DEFAULT_FEWSHOTS
from backend.agents.sql_validator_agent import schema_table_names

# Every table named by BIRD mini-dev gold SQL, across all 11 databases.
BIRD_TABLES = {
    "frpm", "schools", "satscores",
    "cards", "sets", "set_translations", "foreign_data", "legalities", "rulings",
    "badges", "posts", "users", "votes", "comments", "tags", "posthistory", "postlinks",
    "customers", "gasstations", "products", "transactions_1k", "yearmonth",
    "league", "match", "player", "player_attributes", "team", "team_attributes",
    "country",
    "client", "district", "account", "card", "disp", "loan", "order", "trans",
    "champion_time", "circuits", "constructorresults", "constructors", "drivers",
    "driverstandings", "fastest_lap_times", "lap_times_in_seconds", "laptimes",
    "last_driver_incremental", "pitstops", "qualifying", "races", "results",
    "time_in_seconds", "constructorstandings", "seasons", "status",
    "attendance", "budget", "event", "expense", "income", "major", "member",
    "zip_code",
    "alignment", "colour", "hero_power", "race", "superhero", "superpower",
    "gender", "publisher", "attribute", "hero_attribute",
    "examination", "laboratory", "patient",
    "atom", "bond", "connected", "molecule",
}

_FROM_JOIN = re.compile(r"(?:FROM|JOIN)\s+([A-Za-z_]\w*)", re.IGNORECASE)
_NOISE = {"on", "as", "through", "select", "where"}


def _tables_in_fewshots() -> set[str]:
    found: set[str] = set()
    for example in DEFAULT_FEWSHOTS:
        for sql_block in re.findall(r"```sql(.*?)```", example, re.DOTALL):
            found |= {t.lower() for t in _FROM_JOIN.findall(sql_block)}
    return found - _NOISE


class TestFewshotsNameNoRealDatabase:

    def test_no_example_names_a_bird_table(self):
        """The load-bearing assertion. A name shared with an evaluation database
        is one a model can copy into a question about that database and have it
        look plausible."""
        clash = _tables_in_fewshots() & BIRD_TABLES
        assert not clash, f"few-shot examples name real BIRD tables: {sorted(clash)}"

    def test_the_examples_still_contain_sql(self):
        """Guards against a rewrite that satisfies the test above by deleting
        the examples."""
        assert len(DEFAULT_FEWSHOTS) >= 5
        assert len(_tables_in_fewshots()) >= 4

    def test_the_teaching_patterns_survive(self):
        """Each example exists to teach one failure mode. Losing any of these
        markers means the rewrite dropped the lesson, not just the names."""
        blob = "\n".join(DEFAULT_FEWSHOTS)
        for marker in ("GROUP BY", "HAVING", "EXCEPT", "NULLIF", "CASE WHEN",
                       "INNER JOIN", "ORDER BY", "LIMIT 1", "''"):
            assert marker in blob, f"lost the pattern taught by {marker!r}"

    def test_evidence_following_is_still_demonstrated(self):
        """Several examples exist specifically to teach copying the BIRD
        `evidence` hint verbatim rather than paraphrasing it."""
        assert sum("evidence:" in ex for ex in DEFAULT_FEWSHOTS) >= 3


class TestSchemaTableNames:
    """The hint that makes a rejection actionable — it names what IS available."""

    def test_reads_the_bird_harness_format(self):
        ctx = ("Table: atom\nColumns:\n  - atom_id: TEXT [PK]\n\n"
               "Table: molecule\nColumns:\n  - molecule_id: TEXT [PK]")
        assert schema_table_names(ctx) == ["atom", "molecule"]

    def test_reads_the_live_m_schema_format(self):
        ctx = ("【DB_ID】 toxicology\n# Table: atom\n[\n(atom_id:TEXT)\n]\n\n"
               "【DB_ID】 toxicology\n# Table: connected\n[\n(atom_id:TEXT)\n]")
        assert schema_table_names(ctx) == ["atom", "connected"]

    def test_reads_raw_ddl(self):
        ctx = 'CREATE TABLE "frpm" ("CDSCode" TEXT);\nCREATE TABLE schools (x TEXT);'
        assert schema_table_names(ctx) == ["frpm", "schools"]

    def test_is_best_effort_and_never_raises(self):
        """An unparseable context must omit the hint, not fail validation."""
        for ctx in ("", None, "nothing table-shaped here", "Table:"):
            assert isinstance(schema_table_names(ctx), list)

    def test_duplicates_are_collapsed_and_the_list_is_capped(self):
        ctx = "\n".join(f"Table: t{i}" for i in range(80)) + "\nTable: t0"
        names = schema_table_names(ctx, limit=10)
        assert len(names) == 10 and len(set(names)) == 10


class TestTheRejectionIsActionable:

    def test_a_wrong_database_table_is_rejected_with_the_real_list(self):
        """The exact failure seen in the benchmark: a question about toxicology
        answered with debit_card_specializing tables."""
        from backend.agents.sql_validator_agent import SQLValidatorAgent
        ctx = ("Table: atom\nColumns:\n  - atom_id: TEXT [PK]\n\n"
               "Table: molecule\nColumns:\n  - molecule_id: TEXT [PK]")
        r = SQLValidatorAgent({"db": {"dialect": "sqlite"}}).validate(
            "SELECT customer_id FROM customers LIMIT 1", ctx)
        assert r["valid"] is False
        assert "customers" in r["error"]
        assert "atom" in r["error"] and "molecule" in r["error"]

    def test_a_valid_query_carries_no_hint(self):
        from backend.agents.sql_validator_agent import SQLValidatorAgent
        ctx = "Table: atom\nColumns:\n  - atom_id: TEXT [PK]"
        r = SQLValidatorAgent({"db": {"dialect": "sqlite"}}).validate(
            "SELECT atom_id FROM atom LIMIT 1", ctx)
        assert r["valid"] is True and not r.get("error")
