"""
registry.py — Tool registry / dispatcher.

Maintains a name → Tool mapping. The orchestrator registers tools at startup
and dispatches by name at runtime.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from backend.tools.base import Tool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Simple name-keyed registry for Tool implementations."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool. Raises ValueError on name collision."""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        self._tools[tool.name] = tool
        logger.debug(f"[REGISTRY] registered tool: {tool.name}")

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def run(self, name: str, args: Dict[str, Any]) -> str:
        """Dispatch to a named tool. Returns an error string if not found."""
        tool = self._tools.get(name)
        if tool is None:
            return f"Tool '{name}' not found in registry."
        try:
            return tool.run(args)
        except Exception as exc:
            logger.warning(f"[REGISTRY] tool '{name}' raised: {exc}")
            return f"Tool '{name}' encountered an error: {exc}"

    def list_tools(self) -> list[Dict[str, str]]:
        return [{"name": t.name, "description": t.description} for t in self._tools.values()]
