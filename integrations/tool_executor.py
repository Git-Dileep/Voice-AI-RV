"""Local tool execution — direct function calls replacing the HTTP Action API.

Tools are executed in-process instead of routing through FastAPI/HTTP.
This eliminates network latency for tool calls in the headless agent.
"""

import logging
from typing import Any, Dict

from .db_mock import update_record

logger = logging.getLogger(__name__)


def execute_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a registered tool by name with the given arguments.

    Args:
        name: Tool name (must match a key in _TOOL_HANDLERS).
        args: Tool arguments dictionary.

    Returns:
        Result dictionary with an ``ok`` field and either ``result`` or ``error``.
    """
    handler = _TOOL_HANDLERS.get(name)
    if not handler:
        return {"ok": False, "error": f"Unknown tool: {name}"}
    try:
        return handler(args)
    except Exception as e:
        logger.error("Tool '%s' execution error: %s", name, e)
        return {"ok": False, "error": str(e)}


# ── Handler Implementations ──


def _handle_update_record(args: Dict[str, Any]) -> Dict[str, Any]:
    return update_record(args["record_id"], args["value"])


def _handle_get_weather(args: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ok": True,
        "location": args.get("location", "unknown"),
        "weather": "Partly cloudy",
        "temperature": "28°C",
    }


# ── Handler Registry ──

_TOOL_HANDLERS = {
    "update_record": _handle_update_record,
    "get_weather": _handle_get_weather,
}
