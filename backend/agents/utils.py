"""
utils.py — Shared utilities for agents
Includes: PromptHarness, SQL helpers, token estimation
"""

from __future__ import annotations
import re
from functools import lru_cache
from pathlib import Path

PROMPTS_PATH = Path(__file__).parent.parent / "context" / "prompts.md"


class PromptHarness:
    """Loads named sections from prompts.md. Cached after first load."""

    _cache: dict[str, str] = {}

    @classmethod
    @lru_cache(maxsize=1)
    def _load_all(cls) -> dict[str, str]:
        raw = PROMPTS_PATH.read_text()
        sections = re.split(r"##\s+\[SECTION:\s*(\w+)\]", raw)
        # sections = [pre, name1, body1, name2, body2, ...]
        result = {}
        for i in range(1, len(sections), 2):
            name = sections[i].strip()
            body = sections[i + 1].strip()
            # Strip markdown comment lines
            body = re.sub(r"<!--.*?-->", "", body, flags=re.DOTALL).strip()
            result[name] = body
        return result

    @classmethod
    def get(cls, section: str) -> str:
        data = cls._load_all()
        if section not in data:
            raise KeyError(f"Prompt section '{section}' not found in prompts.md")
        return data[section]


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token (GPT-style)."""
    return len(text) // 4


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to approximate token limit."""
    max_chars = max_tokens * 4
    return text[:max_chars] if len(text) > max_chars else text


def clean_sql(sql: str) -> str:
    """Normalize whitespace in SQL string."""
    return re.sub(r"\s+", " ", sql).strip()