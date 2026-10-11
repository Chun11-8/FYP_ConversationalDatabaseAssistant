"""
Tests: M-Schema schema-chunk format (XiYan-SQL) and its
compatibility with the downstream grain-demotion + identifier-reconciliation
passes that parse the schema context. All offline.

Run from repo root:
    pytest backend/tests/test_mschema_format.py -v
"""

from __future__ import annotations

from backend.retrieval.embedding import build_mschema_chunk, build_table_chunk
from backend.agents.sql_code_agent import (
    reconcile_identifiers, _special_schema_columns, SQLCodeAgent,
)


TABLE = {
    "db_name": "FYP",
    "table_name": "players",
    "columns": [
        {"name": "player_id", "type": "INTEGER", "primary_key": True,
         "not_null": True, "description": "unique id", "distinct_values": None},
        {"name": "name", "type": "TEXT", "primary_key": False,
         "not_null": True, "description": "player name", "distinct_values": None},
        {"name": "position", "type": "TEXT", "primary_key": False,
         "not_null": False, "description": "field position",
         "distinct_values": ["GK", "DEF", "MID", "FWD"]},
        {"name": "Overall Rating", "type": "INTEGER", "primary_key": False,
         "not_null": False, "description": "0-100 skill", "distinct_values": None},
    ],
    "foreign_keys": [
        {"from_col": "team_id", "to_table": "teams", "to_col": "id"},
    ],
    "sample_rows": [{"player_id": 1, "name": "Messi", "position": "FWD"}],
}


class TestMSchemaRenderer:

    def test_structure_and_markers(self):
        text = build_mschema_chunk(TABLE)["text"]
        assert "【DB_ID】 FYP" in text
        assert "# Table: players" in text
        assert "(player_id:INTEGER, Primary Key, unique id)" in text
        assert "Examples: [GK, DEF, MID, FWD]" in text
        assert "【Foreign keys】" in text
        assert "players.team_id=teams.id" in text

    def test_metadata_matches_verbose(self):
        m = build_mschema_chunk(TABLE)
        v = build_table_chunk(TABLE)
        for k in ("chunk_id", "chunk_type", "db_name", "table_name"):
            assert m[k] == v[k]
        assert m["text"] != v["text"]        # only the rendered text differs

    def test_column_comma_convention(self):
        text = build_mschema_chunk(TABLE)["text"]
        lines = [l for l in text.splitlines() if l.startswith("(")]
        assert all(l.endswith("),") for l in lines[:-1])   # trailing comma…
        assert lines[-1].endswith(")") and not lines[-1].endswith("),")  # …but last


class TestDownstreamCompatibility:
    """The grain/identifier passes must keep working under M-Schema markers."""

    MSCHEMA = build_mschema_chunk(TABLE)["text"]

    def test_special_columns_detected(self):
        # "Overall Rating" (has a space) must be found for reconciliation.
        specials = _special_schema_columns(self.MSCHEMA)
        assert "Overall Rating" in specials

    def test_identifier_reconciliation_under_mschema(self):
        # Model mangles the spaced column → must be rewritten to the quoted name.
        sql = "SELECT name FROM players ORDER BY Overall_Rating DESC"
        out = reconcile_identifiers(sql, self.MSCHEMA)
        assert '"Overall Rating"' in out

    def test_pk_grain_penalty_under_mschema(self):
        agent = SQLCodeAgent({"db": {"dialect": "sqlite"}}, object())
        # Grouping ONLY by the name column when a PK exists → penalty 1.
        bad = "SELECT name, COUNT(*) FROM players GROUP BY name"
        assert agent._grain_penalty(bad, self.MSCHEMA) == 1
        # Grouping by the PK → no penalty.
        good = "SELECT player_id, COUNT(*) FROM players GROUP BY player_id"
        assert agent._grain_penalty(good, self.MSCHEMA) == 0
