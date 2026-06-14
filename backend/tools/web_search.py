"""
web_search.py — Web search tool using DuckDuckGo (duckduckgo-search).

Returns a concise snippet list suitable for LLM synthesis. No API key required.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_MAX_RESULTS = 5
_SNIPPET_MAX = 300  # chars per snippet before truncation


class WebSearchTool:
    """Search the web via DuckDuckGo and return a list of result snippets."""

    name = "web_search"
    description = (
        "Search the web for information about database concepts, SQL syntax, "
        "or other general questions. Input: {query: str, max_results?: int}."
    )

    def run(self, args: Dict[str, Any]) -> str:
        """
        Args:
            args: {"query": str, "max_results": int (optional)}
        Returns:
            Formatted string of search snippets, or an error message.
        """
        query: str = args.get("query", "").strip()
        if not query:
            return "No query provided."

        max_results: int = min(int(args.get("max_results", _MAX_RESULTS)), _MAX_RESULTS)

        try:
            from duckduckgo_search import DDGS
            results: List[Dict] = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append(r)
        except ImportError:
            logger.error("[WEB_SEARCH] duckduckgo-search not installed.")
            return "Web search is unavailable (missing duckduckgo-search package)."
        except Exception as exc:
            logger.warning(f"[WEB_SEARCH] search failed: {exc}")
            return f"Web search failed: {exc}"

        if not results:
            return "No web results found for this query."

        lines = [f"Web search results for: {query}\n"]
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            body = r.get("body", "") or r.get("snippet", "")
            href = r.get("href", "") or r.get("url", "")
            if len(body) > _SNIPPET_MAX:
                body = body[:_SNIPPET_MAX] + "…"
            lines.append(f"{i}. **{title}**")
            if body:
                lines.append(f"   {body}")
            if href:
                lines.append(f"   Source: {href}")
        return "\n".join(lines)
