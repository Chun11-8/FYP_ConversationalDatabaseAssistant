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


# ── reasoning-block stripping (GH#65) ─────────────────────────────────────────
#
# Open-weight reasoning models (DeepSeek-R1, QwQ, Qwen3-thinking, GPT-OSS) emit
# their chain of thought before the answer. Every parser in this project reads
# free-text LLM output, and each one breaks differently on that prelude:
#
#   * condense_query accepts the rewrite only if len < 500 — a think block
#     blows past it, the condensation is silently dropped, and multi-turn
#     regresses to the GH#63 symptom.
#   * _extract_json falls back to the outermost { … } span; a brace inside the
#     reasoning makes that span cross both regions and fail to parse.
#   * _extract_sqls would happily pick up a ```sql fence the model was only
#     thinking about, not proposing.
#   * _normalize_intent can see several intent labels discussed before the
#     verdict.
#
# Stripping once, at the top of each parser, fixes all four rather than
# special-casing per model.

_REASONING_BLOCK_RE = re.compile(
    r"<(think|thinking|reasoning|thought)\b[^>]*>.*?</\1\s*>",
    re.DOTALL | re.IGNORECASE,
)
# An unterminated block: the model hit its token cap mid-thought. Everything
# after the opening tag is reasoning, so drop the remainder.
_UNCLOSED_REASONING_RE = re.compile(
    r"<(think|thinking|reasoning|thought)\b[^>]*>.*\Z",
    re.DOTALL | re.IGNORECASE,
)
# GPT-OSS harmony channels: analysis is scratch, final is the answer.
_HARMONY_FINAL_RE = re.compile(
    r"<\|channel\|>\s*final\s*<\|message\|>(.*?)(?:<\|end\|>|\Z)",
    re.DOTALL | re.IGNORECASE,
)


def strip_reasoning(raw: str) -> str:
    """
    Drop a reasoning model's chain-of-thought, returning just the answer.

    A no-op for ordinary replies, so it is safe to apply unconditionally. If
    stripping would leave nothing at all the original text is returned instead
    — better to hand a parser the messy full response than an empty string.
    """
    if not raw or "<" not in raw:
        return raw

    harmony = _HARMONY_FINAL_RE.search(raw)
    if harmony:
        final = harmony.group(1).strip()
        if final:
            return final

    out = _REASONING_BLOCK_RE.sub("", raw)
    out = _UNCLOSED_REASONING_RE.sub("", out)
    out = out.strip()
    return out or raw


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