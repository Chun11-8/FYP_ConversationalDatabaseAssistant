"""
config_loader.py
----------------
Single source of truth for all configuration.

Merges non-secret defaults from config.yaml with secrets from secrets.env
(or environment variables already set in the shell). Environment variables
always win over yaml defaults.

Usage:
    from backend.config.config_loader import load_config
    config = load_config()

Returned shape:
    config["llm"]["model"]        str   e.g. "gpt-4o-mini"
    config["llm"]["api_key"]      str   from OPENAI_API_KEY
    config["db"]["dialect"]       str   e.g. "postgres"
    config["db"]["host"]          str
    config["db"]["port"]          int
    config["db"]["name"]          str
    config["db"]["user"]          str
    config["db"]["password"]      str   from PG_PASSWORD
    config["retrieval"]["vector_store"]   str
    config["retrieval"]["top_k_tables"]   int
    config["retrieval"]["top_k_kb"]       int
    config["response"]["inline_row_limit"] int
    config["response"]["table_row_limit"]  int
    config["response"]["csv_row_limit"]    int
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).parent
_SECRETS_PATH = _CONFIG_DIR / "secrets.env"
_CONFIG_YAML_PATH = _CONFIG_DIR / "config.yaml"

_REQUIRED_SECRETS = ("OPENAI_API_KEY", "PG_PASSWORD")


def load_config() -> dict:
    """
    Load and return the merged configuration dict.

    Raises:
        EnvironmentError: if any required secret env var is absent after
                          loading secrets.env.
        FileNotFoundError: if config.yaml is missing.
    """
    # Load secrets.env without overriding vars already set in the shell
    if _SECRETS_PATH.exists():
        load_dotenv(_SECRETS_PATH, override=False)
    else:
        logger.warning(
            "secrets.env not found at %s — relying on environment variables.",
            _SECRETS_PATH,
        )

    # Load non-secret defaults from YAML
    if not _CONFIG_YAML_PATH.exists():
        raise FileNotFoundError(f"config.yaml not found at {_CONFIG_YAML_PATH}")

    with _CONFIG_YAML_PATH.open() as fh:
        cfg: dict = yaml.safe_load(fh) or {}

    # Validate required secrets are present
    missing = [k for k in _REQUIRED_SECRETS if not os.getenv(k)]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {missing}. "
            f"Copy secrets.env.example → secrets.env and fill in real values."
        )

    # Inject secrets and env-var overrides (env always beats yaml)
    llm = cfg.setdefault("llm", {})
    llm["api_key"] = os.environ["OPENAI_API_KEY"]

    db = cfg.setdefault("db", {})
    db["host"]     = os.getenv("PG_HOST",     str(db.get("host", "127.0.0.1")))
    db["port"]     = int(os.getenv("PG_PORT", db.get("port", 5432)))
    db["name"]     = os.getenv("PG_DB",       db.get("name", "FYP"))
    db["user"]     = os.getenv("PG_USER",     db.get("user", "postgres"))
    db["password"] = os.environ["PG_PASSWORD"]

    logger.debug(
        "Config loaded: model=%s db=%s@%s:%s",
        llm.get("model"), db["name"], db["host"], db["port"],
    )
    return cfg
