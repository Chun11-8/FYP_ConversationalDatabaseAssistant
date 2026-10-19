"""
Tests: every file this project reads at runtime must be read as
UTF-8 explicitly.

Python without an explicit `encoding=` uses the LOCALE default. On a Windows
machine that is cp1252, and every text file here is UTF-8. The failure mode is
nasty because it is usually SILENT: most UTF-8 byte sequences decode to
mojibake under cp1252 rather than raising, so `prompt.md`'s box-drawing
characters have been quietly corrupting the system prompt on Windows. It only
crashes when a byte lands on one of the five positions cp1252 leaves undefined
— 0x81, 0x8d, 0x8f, 0x90, 0x9d — which is what a ⚠️ in config.yaml did.

Run from repo root:
    pytest backend/tests/test_encoding.py -v
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
# Files loaded while the app runs, as opposed to test fixtures.
RUNTIME_TEXT_FILES = [
    _BACKEND / "config" / "config.yaml",
    _BACKEND / "agents" / "prompt.md",
    _BACKEND / "pipelines" / "prompt_response.md",
    *sorted((_BACKEND / "agents" / "prompts").glob("*.md")),
]
# Undefined in cp1252: decoding these RAISES rather than mojibaking.
CP1252_UNDEFINED = {0x81, 0x8D, 0x8F, 0x90, 0x9D}


@pytest.mark.parametrize("path", RUNTIME_TEXT_FILES, ids=lambda p: p.name)
def test_every_runtime_file_is_valid_utf8(path: Path):
    path.read_bytes().decode("utf-8")      # raises if it is not


class TestRuntimeReadsDeclareEncoding:
    """The modules that read those files, checked at the source level — a
    missing `encoding=` is invisible on Linux and breaks on Windows, so CI on
    one platform cannot catch it."""

    @pytest.mark.parametrize("module,call", [
        ("config/config_loader.py", "_CONFIG_YAML_PATH.open("),
        ("agents/utils.py", "PROMPTS_PATH.read_text("),
        ("pipelines/postprocessing.py", "prompt_response.md"),
    ], ids=["config_loader", "prompt_harness", "response_optimizer"])
    def test_the_read_passes_an_explicit_encoding(self, module, call):
        src = (_BACKEND / module).read_text(encoding="utf-8")
        idx = src.index(call)
        # The encoding kwarg belongs to the same call, within a few lines.
        window = src[idx:idx + 400]
        assert 'encoding="utf-8"' in window, (
            f"{module}: the read near {call!r} has no explicit encoding — it "
            f"will use the locale default and mojibake on Windows")

    def test_no_runtime_module_reads_text_without_an_encoding(self):
        """Catches a NEW unencoded read anywhere in the runtime tree."""
        offenders = []
        for path in _BACKEND.rglob("*.py"):
            if "__pycache__" in str(path) or path.parent.name == "tests":
                continue
            for lineno, line in enumerate(
                    path.read_text(encoding="utf-8").splitlines(), start=1):
                if re.search(r"\.read_text\(\s*\)", line):
                    offenders.append(f"{path.relative_to(_BACKEND)}:{lineno}")
        assert offenders == [], (
            "read_text() with no encoding: " + ", ".join(offenders))


class TestConfigIsWindowsSafe:
    """config.yaml is edited by hand, sometimes in an editor that saves as the
    system codepage. Keeping it free of the bytes cp1252 cannot represent means
    a stray locale-default read degrades to mojibake rather than a crash."""

    def test_config_yaml_has_no_cp1252_undefined_bytes(self):
        raw = (_BACKEND / "config" / "config.yaml").read_bytes()
        bad = sorted({b for b in raw if b in CP1252_UNDEFINED})
        assert bad == [], (
            "config.yaml contains bytes undefined in cp1252 "
            f"({[hex(b) for b in bad]}) — a locale-default read on Windows "
            "will raise UnicodeDecodeError, as it has before")

    def test_config_loads_under_a_utf8_read(self):
        import yaml
        with (_BACKEND / "config" / "config.yaml").open(encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        assert cfg["llm"]["model"]
        assert cfg["llm"]["providers"]["openrouter"]["api_key_env"] == \
            "OPENROUTER_API_KEY"
