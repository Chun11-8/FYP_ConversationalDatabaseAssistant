"""
db_catalog.py — which logical database does each table belong to?

Every BIRD mini-dev database was loaded into ONE PostgreSQL schema (`public`),
so 75 tables from 11 unrelated domains — customers, Patient, superhero,
circuits, frpm … — sit side by side. `indexing build --db FYP` then stamped
every schema chunk with the same `db_name = 'FYP'`, which made
`pgvector_store.retrieve()`'s existing `WHERE db_name = %s` filter a no-op:
the chat UI retrieves its top-5 tables out of all 75, while the BIRD harness
hands the model the 4 tables of the question's own `db_id`. Same question,
strictly harder retrieval problem — which is why the two disagreed.

This module reads a BIRD `tables.json` (the `dev_tables.json` shipped with
mini-dev) and answers "which db_id owns this table?", so the indexer can tag
each chunk with its real domain and retrieval can be scoped to one.

Nothing here touches the database: it only changes the LABEL written on a
vector row. No table is created, moved, renamed or dropped.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

# Repo root = backend/retrieval/db_catalog.py → ../../
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Searched in order when no explicit path is configured.
DEFAULT_CATALOG_PATHS = (
    "outputs/dev_tables.json",
    "dataset/minidev/MINIDEV/dev_tables.json",
    "dataset/dev_tables.json",
)


def resolve_catalog_path(path: Optional[str] = None) -> Optional[Path]:
    """
    Locate the BIRD tables.json. Precedence: explicit *path* → $BIRD_TABLES_JSON
    → the default locations. Returns None when no file is found — the caller is
    expected to degrade to the un-tagged behaviour rather than fail.
    """
    candidates: List[str] = []
    if path:
        candidates.append(path)
    env = os.getenv("BIRD_TABLES_JSON")
    if env:
        candidates.append(env)
    candidates.extend(DEFAULT_CATALOG_PATHS)

    for cand in candidates:
        p = Path(cand)
        if not p.is_absolute():
            p = _REPO_ROOT / p
        if p.is_file():
            return p
    return None


class DBCatalog:
    """
    Immutable table → db_id map. Matching is case-insensitive because
    PostgreSQL folds unquoted identifiers to lower case, so BIRD's
    `Player_Attributes` is stored as `player_attributes`.
    """

    def __init__(self, table_to_db: Dict[str, str], source: Optional[Path] = None):
        self._map = {k.casefold(): v for k, v in table_to_db.items()}
        self.source = source

    # ── construction ───────────────────────────────────────────────────────

    @classmethod
    def from_tables_json(cls, path: Path) -> "DBCatalog":
        """
        Parse a BIRD tables.json: a list of entries with `db_id` and
        `table_names_original`. A table claimed by two db_ids is a real
        ambiguity — it cannot exist twice in one `public` schema — so the
        first claim wins and the collision is logged rather than guessed at.
        """
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError(f"{path}: expected a list of database entries")

        mapping: Dict[str, str] = {}
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            db_id = entry.get("db_id")
            if not db_id:
                continue
            names = entry.get("table_names_original") or entry.get("table_names") or []
            for name in names:
                if not isinstance(name, str) or not name.strip():
                    continue
                key = name.strip().casefold()
                existing = mapping.get(key)
                if existing and existing != db_id:
                    logger.warning(
                        f"[CATALOG] table '{name}' claimed by both '{existing}' "
                        f"and '{db_id}'; keeping '{existing}'"
                    )
                    continue
                mapping[key] = db_id
        return cls(mapping, source=Path(path))

    @classmethod
    def empty(cls) -> "DBCatalog":
        """A catalog that knows nothing — every lookup falls back."""
        return cls({}, source=None)

    # ── lookup ─────────────────────────────────────────────────────────────

    def __bool__(self) -> bool:
        return bool(self._map)

    def __len__(self) -> int:
        return len(self._map)

    def db_for_table(self, table_name: str, default: str) -> str:
        """The db_id owning *table_name*, or *default* when it is unknown."""
        if not table_name:
            return default
        return self._map.get(table_name.strip().strip('"').casefold(), default)

    def databases(self) -> List[str]:
        """Every db_id in the catalog, sorted."""
        return sorted(set(self._map.values()))

    def tables_for_db(self, db_id: str) -> List[str]:
        """Table names owned by *db_id* (as stored, lower-cased), sorted."""
        target = (db_id or "").casefold()
        return sorted(t for t, d in self._map.items() if d.casefold() == target)


# ── process-wide cache ────────────────────────────────────────────────────────

_lock = threading.Lock()
_cached: Dict[str, DBCatalog] = {}


def load_catalog(path: Optional[str] = None) -> DBCatalog:
    """
    Load (and memoise) the catalog. Never raises: a missing or malformed file
    yields an empty catalog, which reproduces the previous behaviour of
    tagging everything with the build's `--db` name.
    """
    resolved = resolve_catalog_path(path)
    key = str(resolved) if resolved else ""
    with _lock:
        if key in _cached:
            return _cached[key]
    if resolved is None:
        logger.info(
            "[CATALOG] no tables.json found "
            f"(looked for {', '.join(DEFAULT_CATALOG_PATHS)}); "
            "schema chunks will all share one db_name"
        )
        catalog = DBCatalog.empty()
    else:
        try:
            catalog = DBCatalog.from_tables_json(resolved)
            logger.info(
                f"[CATALOG] {len(catalog)} tables across "
                f"{len(catalog.databases())} databases from {resolved}"
            )
        except Exception as exc:
            logger.warning(f"[CATALOG] could not read {resolved}: {exc}")
            catalog = DBCatalog.empty()
    with _lock:
        _cached[key] = catalog
    return catalog


def reset_cache() -> None:
    """Drop the memoised catalogs (tests, and after editing the JSON)."""
    with _lock:
        _cached.clear()


def group_by_db(table_names: Iterable[str], default: str,
                catalog: Optional[DBCatalog] = None) -> Dict[str, List[str]]:
    """{db_id: [table, ...]} for a set of table names — used for build reports."""
    cat = catalog if catalog is not None else load_catalog()
    out: Dict[str, List[str]] = {}
    for name in table_names or []:
        out.setdefault(cat.db_for_table(name, default), []).append(name)
    return {k: sorted(v) for k, v in sorted(out.items())}
