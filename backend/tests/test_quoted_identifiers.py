"""
Tests for GH#122 — quoted identifiers in the semantic check.

The validator's semantic check confirms that the tables and columns a query
names actually appear in the schema. It guarded against partial-identifier
matches with `\\b`, which only asserts a word/non-word TRANSITION — so it means
nothing next to a character that is not a word character.

A quoted SQL identifier often ends in one. `"Enrollment (K-12)"` ends in `)`,
and the next character in a CREATE TABLE is `"`. Both are non-word, there is no
transition, and the pattern can never match: the column is reported missing
from a schema that plainly defines it, the query is rejected before it ever
executes, and it scores 0.

Measured consequence: every prediction against BIRD's california_schools
database was rejected this way, across all seven benchmarked models, while the
SQL was correct. It is the one database whose columns are parenthesised.

All offline — no LLM, no database.

Run from repo root:
    pytest backend/tests/test_quoted_identifiers.py -v
"""

from __future__ import annotations

import pytest

from backend.agents.sql_validator_agent import SQLValidatorAgent, _boundary_pattern

# Faithful to california_schools: parenthesised columns, spaced columns, plain ones.
SCHEMA = (
    'CREATE TABLE frpm ("CDSCode" TEXT, "Enrollment (K-12)" REAL, '
    '"Enrollment (Ages 5-17)" REAL, "Low Grade" TEXT, "School Name" TEXT);\n'
    'CREATE TABLE schools ("CDSCode" TEXT, "School" TEXT, "DOC" TEXT);'
)


@pytest.fixture(scope="module")
def validator() -> SQLValidatorAgent:
    return SQLValidatorAgent({"db": {"dialect": "sqlite"}})


class TestBoundaryPattern:

    def test_a_name_ending_in_punctuation_is_still_found(self):
        import re
        pattern = _boundary_pattern("enrollment (k-12)")
        assert re.search(pattern, SCHEMA.lower()), "the \\b bug is back"

    def test_a_plain_name_is_still_found(self):
        import re
        assert re.search(_boundary_pattern("cdscode"), SCHEMA.lower())

    def test_a_partial_identifier_still_does_not_match(self):
        """The guard has to keep doing its original job: `doc` must not match
        inside `document_id`, or the check stops catching hallucinated columns."""
        import re
        assert not re.search(_boundary_pattern("doc"), "create table t (document_id text);")
        assert not re.search(_boundary_pattern("school"), "create table t (schooling text);")

    def test_regex_metacharacters_are_escaped(self):
        """A name is a literal, never a pattern — otherwise `(k-12)` would be
        read as a capture group and match the wrong thing."""
        import re
        assert not re.search(_boundary_pattern("a.c"), "abc")
        assert re.search(_boundary_pattern("a.c"), "a.c")


class TestValidatorAcceptsQuotedIdentifiers:
    """The exact predictions that scored 0 in the simplev2 benchmark run."""

    @pytest.mark.parametrize("sql", [
        # BIRD mini-dev question 11, as six of seven models wrote it.
        'SELECT CDSCode FROM frpm '
        'WHERE ("Enrollment (K-12)" + "Enrollment (Ages 5-17)") > 500',
        # Backtick quoting, which the BIRD gold SQL itself uses.
        'SELECT CDSCode FROM frpm WHERE `Enrollment (K-12)` > 500',
        # Question 46: a join ordering on a parenthesised column.
        'SELECT t1.School FROM schools AS t1 INNER JOIN frpm AS t2 '
        'ON t1.CDSCode = t2.CDSCode WHERE t1.DOC = 31 '
        'ORDER BY t2."Enrollment (K-12)" DESC LIMIT 1',
        # A space alone always worked; pinned so the fix cannot regress it.
        'SELECT "Low Grade", "School Name" FROM frpm',
    ])
    def test_valid_sql_over_awkward_column_names_is_accepted(self, validator, sql):
        result = validator.validate(sql, SCHEMA)
        assert result["valid"] is True, result.get("error")

    def test_a_plain_query_is_unaffected(self, validator):
        assert validator.validate("SELECT CDSCode FROM schools", SCHEMA)["valid"] is True


class TestTheCheckStillRejects:
    """
    Loosening a guard is only safe if it still catches what it was built for.
    """

    def test_a_hallucinated_column_is_still_rejected(self, validator):
        r = validator.validate("SELECT Nonexistent FROM frpm", SCHEMA)
        assert r["valid"] is False and "Columns not found" in r["error"]

    def test_a_hallucinated_table_is_still_rejected(self, validator):
        r = validator.validate("SELECT CDSCode FROM teachers", SCHEMA)
        assert r["valid"] is False and "Tables not found" in r["error"]

    def test_a_near_miss_column_is_still_rejected(self, validator):
        """`DOCX` must not pass just because `DOC` is in the schema."""
        r = validator.validate("SELECT DOCX FROM schools", SCHEMA)
        assert r["valid"] is False

    def test_a_write_is_still_blocked(self, validator):
        """Read-only is the domain-critical rule; it must survive any change
        to the semantic check."""
        r = validator.validate('DELETE FROM frpm WHERE "Enrollment (K-12)" > 0', SCHEMA)
        assert r["valid"] is False
