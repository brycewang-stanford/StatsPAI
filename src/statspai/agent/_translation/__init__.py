"""Stata → StatsPAI / R → StatsPAI command translators.

Public surface
--------------

* :func:`from_stata` — translate a Stata command string to a ready-to-
  dispatch ``{tool, arguments, python_code, notes}`` dict.
* :func:`from_r` — same shape, for R / fixest / felm / did syntax.

The translators are exposed as MCP tools so an agent can hand a user's
Stata or R snippet directly to the server and receive a verified
StatsPAI call back. The output carries both:

* ``python_code`` — a string the agent can paste into a chat reply
  (``sp.fixest("y ~ x", fe=["id"], data=df, cluster="id")``).
* ``tool_call`` — a JSON-RPC-ready ``{tool, arguments}`` payload the
  agent can dispatch via ``tools/call`` to actually run the
  translation.

Failure modes are non-fatal: an unrecognised command returns
``{tool: null, error: "...", suggestions: [...]}`` rather than
raising. Calls translate one command at a time — multi-command
``do`` files should be split by the caller.
"""
from __future__ import annotations

from ._stata import from_stata, STATA_COMMAND_MAP
from ._r import from_r, R_FUNCTION_MAP


__all__ = [
    "from_stata",
    "from_r",
    "STATA_COMMAND_MAP",
    "R_FUNCTION_MAP",
]
