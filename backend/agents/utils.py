"""
utils.py — Shared utilities for agents
Includes: PromptHarness, SQL helpers, token estimation
"""

from __future__ import annotations
import re
from functools import lru_cache
from pathlib import Path

# Section-based master prompt file (backend/agents/prompt.md)
PROMPTS_PATH = Path(__file__).parent / "prompt.md"
# Directory for standalone per-feature prompt files
PROMPTS_DIR = Path(__file__).parent / "prompts"


class PromptHarness:
    """
    Loads named sections from prompt.md (get) or standalone .md files (get_file).
    Both are cached after first load.
    """

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
            body = re.sub(r"<!--.*?-->", "", body, flags=re.DOTALL).strip()
            result[name] = body
        return result

    @classmethod
    def get(cls, section: str) -> str:
        """Load a named section from backend/agents/prompt.md."""
        data = cls._load_all()
        if section not in data:
            raise KeyError(f"Prompt section '{section}' not found in prompt.md")
        return data[section]

    @classmethod
    @lru_cache(maxsize=16)
    def get_file(cls, filename: str) -> str:
        """Load a standalone prompt file from backend/agents/prompts/<filename>."""
        path = PROMPTS_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        return path.read_text()


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