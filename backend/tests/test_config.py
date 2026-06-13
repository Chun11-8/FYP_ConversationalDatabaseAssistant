"""
Tests for backend/config/config_loader.py (Issue #0).

Run from repo root:
    pytest backend/tests/test_config.py -v
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from backend.config.config_loader import load_config, _REQUIRED_SECRETS


# ── helpers ────────────────────────────────────────────────────────────────────

_MOCK_ENV = {
    "OPENAI_API_KEY": "sk-test-key",
    "PG_PASSWORD":    "test-password",
}


def _env(**overrides):
    """Return a mock env dict with required secrets, allowing overrides."""
    return {**_MOCK_ENV, **overrides}


# ── tests ──────────────────────────────────────────────────────────────────────

class TestLoadConfig:

    def test_returns_dict_with_required_keys(self, tmp_path, monkeypatch):
        """load_config() returns a dict with llm and db sections."""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(textwrap.dedent("""\
            llm:
              model: gpt-4o-mini
            db:
              dialect: postgres
              host: "127.0.0.1"
              port: 5432
              name: FYP
              user: postgres
        """))

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setenv("PG_PASSWORD", "test-password")

        with patch("backend.config.config_loader._CONFIG_YAML_PATH", yaml_file), \
             patch("backend.config.config_loader._SECRETS_PATH", tmp_path / "secrets.env"):
            cfg = load_config()

        assert cfg["llm"]["model"] == "gpt-4o-mini"
        assert cfg["llm"]["api_key"] == "sk-test-key"
        assert cfg["db"]["dialect"] == "postgres"
        assert cfg["db"]["password"] == "test-password"
        assert cfg["db"]["host"] == "127.0.0.1"
        assert cfg["db"]["port"] == 5432

    def test_raises_when_openai_key_missing(self, tmp_path, monkeypatch):
        """load_config() raises EnvironmentError when OPENAI_API_KEY is absent."""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("llm:\n  model: gpt-4o-mini\n")

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("PG_PASSWORD", "test-password")

        with patch("backend.config.config_loader._CONFIG_YAML_PATH", yaml_file), \
             patch("backend.config.config_loader._SECRETS_PATH", tmp_path / "secrets.env"):
            with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
                load_config()

    def test_raises_when_pg_password_missing(self, tmp_path, monkeypatch):
        """load_config() raises EnvironmentError when PG_PASSWORD is absent."""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("llm:\n  model: gpt-4o-mini\n")

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.delenv("PG_PASSWORD", raising=False)

        with patch("backend.config.config_loader._CONFIG_YAML_PATH", yaml_file), \
             patch("backend.config.config_loader._SECRETS_PATH", tmp_path / "secrets.env"):
            with pytest.raises(EnvironmentError, match="PG_PASSWORD"):
                load_config()

    def test_env_vars_override_yaml_defaults(self, tmp_path, monkeypatch):
        """Environment variables override yaml values for db connection fields."""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(textwrap.dedent("""\
            db:
              host: "127.0.0.1"
              port: 5432
              name: FYP
              user: postgres
        """))

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setenv("PG_PASSWORD", "secret")
        monkeypatch.setenv("PG_HOST", "db.example.com")
        monkeypatch.setenv("PG_PORT", "5433")
        monkeypatch.setenv("PG_DB", "mydb")
        monkeypatch.setenv("PG_USER", "admin")

        with patch("backend.config.config_loader._CONFIG_YAML_PATH", yaml_file), \
             patch("backend.config.config_loader._SECRETS_PATH", tmp_path / "secrets.env"):
            cfg = load_config()

        assert cfg["db"]["host"] == "db.example.com"
        assert cfg["db"]["port"] == 5433
        assert cfg["db"]["name"] == "mydb"
        assert cfg["db"]["user"] == "admin"

    def test_port_is_int(self, tmp_path, monkeypatch):
        """db.port is always an int even when PG_PORT is a string env var."""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("db:\n  port: 5432\n")

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setenv("PG_PASSWORD", "secret")
        monkeypatch.setenv("PG_PORT", "5434")

        with patch("backend.config.config_loader._CONFIG_YAML_PATH", yaml_file), \
             patch("backend.config.config_loader._SECRETS_PATH", tmp_path / "secrets.env"):
            cfg = load_config()

        assert isinstance(cfg["db"]["port"], int)
        assert cfg["db"]["port"] == 5434

    def test_missing_config_yaml_raises(self, tmp_path, monkeypatch):
        """load_config() raises FileNotFoundError when config.yaml is absent."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setenv("PG_PASSWORD", "secret")

        with patch("backend.config.config_loader._CONFIG_YAML_PATH", tmp_path / "missing.yaml"), \
             patch("backend.config.config_loader._SECRETS_PATH", tmp_path / "secrets.env"):
            with pytest.raises(FileNotFoundError):
                load_config()

    def test_secrets_env_loaded_from_file(self, tmp_path, monkeypatch):
        """load_config() reads secrets from secrets.env when not set in shell."""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("llm:\n  model: gpt-4o-mini\n")

        secrets_file = tmp_path / "secrets.env"
        secrets_file.write_text("OPENAI_API_KEY=sk-from-file\nPG_PASSWORD=pw-from-file\n")

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("PG_PASSWORD", raising=False)

        with patch("backend.config.config_loader._CONFIG_YAML_PATH", yaml_file), \
             patch("backend.config.config_loader._SECRETS_PATH", secrets_file):
            cfg = load_config()

        assert cfg["llm"]["api_key"] == "sk-from-file"
        assert cfg["db"]["password"] == "pw-from-file"

    def test_required_secrets_list(self):
        """_REQUIRED_SECRETS must include both critical keys."""
        assert "OPENAI_API_KEY" in _REQUIRED_SECRETS
        assert "PG_PASSWORD" in _REQUIRED_SECRETS
