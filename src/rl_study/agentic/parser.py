"""A deliberately small action language with a strict JSON tool-call parser."""

from __future__ import annotations

import json
import re

from rl_study.agentic.types import AgentAction, ToolSchema

_CALL = re.compile(r"^CALL ([A-Za-z_][A-Za-z0-9_]*) (\{.*\})$")
_MAX_ACTION_CHARACTERS = 256


def _invalid(raw_text: str, error: str) -> AgentAction:
    return AgentAction(kind="invalid", raw_text=raw_text or "<empty>", error=error)


def parse_action(raw_text: str, visible_tools: tuple[ToolSchema, ...]) -> AgentAction:
    """Parse ``FINAL text`` or a single-argument ``CALL name {...}`` action.

    Invalid model output is data, not a Python exception: the environment can assign a
    learning signal and continue until its explicit step limit.
    """

    text = raw_text.strip()
    if not text:
        return _invalid(raw_text, "empty_action")
    if len(text) > _MAX_ACTION_CHARACTERS:
        return _invalid(text, "action_too_long")
    if text.startswith("FINAL "):
        answer = text.removeprefix("FINAL ").strip()
        if not answer:
            return _invalid(text, "empty_final_answer")
        return AgentAction(kind="final", raw_text=text, final_answer=answer)
    match = _CALL.fullmatch(text)
    if match is None:
        return _invalid(text, "expected_FINAL_or_CALL")
    name, serialized = match.groups()
    schemas = {schema.name: schema for schema in visible_tools}
    schema = schemas.get(name)
    if schema is None:
        return _invalid(text, "tool_not_visible")
    try:
        arguments = json.loads(serialized)
    except json.JSONDecodeError:
        return _invalid(text, "invalid_json")
    if not isinstance(arguments, dict) or set(arguments) != {schema.argument}:
        return _invalid(text, "tool_schema_mismatch")
    value = arguments[schema.argument]
    if not isinstance(value, str) or not value:
        return _invalid(text, "tool_argument_must_be_nonempty_string")
    return AgentAction(
        kind="tool",
        raw_text=text,
        tool_name=name,
        argument_name=schema.argument,
        argument_value=value,
    )
