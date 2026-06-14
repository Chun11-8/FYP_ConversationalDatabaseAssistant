"""
base.py — Tool interface for the tool registry.

All tools implement the `Tool` protocol: a name, a one-line description, and a
`run(args)` method that accepts a plain dict and returns a string result.
"""

from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable


@runtime_checkable
class Tool(Protocol):
    name: str
    description: str

    def run(self, args: Dict[str, Any]) -> str:
        """Execute the tool and return a string result."""
        ...
