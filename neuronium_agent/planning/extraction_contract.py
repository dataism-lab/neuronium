"""Contracts for extraction and clarification in supervised planning."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from pydantic import BaseModel, Field


class ExtractedIntent(BaseModel):
    """Intent classification extracted from user objective."""

    task_type: str = "generic_task"
    confidence: float = 0.0


class MissingField(BaseModel):
    """A required field that is missing for safe execution."""

    field: str
    reason: str = ""
    critical: bool = True


class ExtractionEnvelope(BaseModel):
    """Structured extraction output used by planner backend."""

    intent: ExtractedIntent = Field(default_factory=ExtractedIntent)
    inputs: dict[str, Any] = Field(default_factory=dict)
    missing_fields: list[MissingField] = Field(default_factory=list)
    extras: dict[str, Any] = Field(default_factory=dict)


class ClarificationQuestion(BaseModel):
    """Single clarification question for user escalation."""

    key: str
    prompt: str
    path: str | None = None
    expected_schema: dict[str, Any] = Field(default_factory=dict)
    expected_type: str = "string"
    required: bool = True
    examples: list[str] = Field(default_factory=list)


class ClarificationRequest(BaseModel):
    """Escalation package asking user for missing parameters."""

    request_id: str
    missing_fields: list[MissingField] = Field(default_factory=list)
    questions: list[ClarificationQuestion] = Field(default_factory=list)
    candidate_evidence: list[dict[str, Any]] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)


class ClarificationResponse(BaseModel):
    """User-provided response to a clarification request."""

    request_artifact_id: str
    answers: dict[str, Any] = Field(default_factory=dict)
    answer_text: str | None = None


def extraction_envelope_json_schema(
    *,
    input_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return an OpenAI-compatible JSON Schema without $ref/$defs.

    Pydantic's `model_json_schema()` uses `$defs/$ref`, which can trigger
    `400 Bad Request` for `response_format=json_schema` on some OpenAI models.
    """
    resolved_inputs_schema: dict[str, Any]
    missing_field_enum: list[str] | None
    if input_schema is None:
        resolved_inputs_schema = {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "url",
                "urls",
                "doc_paths",
                "language",
                "output_format",
                "summary_length",
                "output_filename",
                "output_text",
            ],
            "properties": {
                "url": {"type": ["string", "null"]},
                "urls": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "doc_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "language": {"type": ["string", "null"]},
                "output_format": {"type": ["string", "null"]},
                "summary_length": {"type": ["string", "null"]},
                "output_filename": {
                    "type": ["string", "null"],
                    # Bare filename only (no path), e.g. "summary.html"
                    "pattern": "^(?![Nn]one$)(?![Nn][Uu][Ll][Ll]$)[A-Za-z0-9_.-]{1,128}$",
                },
                "output_text": {"type": ["string", "null"]},
            },
        }
        missing_field_enum = [
            "url",
            "urls",
            "doc_paths",
            "language",
            "output_format",
            "summary_length",
            "output_filename",
            "output_text",
        ]
    else:
        raw_required = input_schema.get("required", [])
        required = sorted({str(x) for x in raw_required if isinstance(x, str)})
        raw_properties = input_schema.get("properties", {})
        properties: dict[str, Any] = {}
        if isinstance(raw_properties, dict):
            for key in sorted(raw_properties):
                value = raw_properties[key]
                if isinstance(value, dict):
                    properties[key] = deepcopy(value)
        resolved_inputs_schema = {
            "type": "object",
            "additionalProperties": False,
            "required": required,
            "properties": properties,
        }
        missing_field_enum = required

    missing_field_schema: dict[str, Any] = {"type": "string"}
    if missing_field_enum:
        missing_field_schema["enum"] = missing_field_enum

    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["intent", "inputs", "missing_fields", "extras"],
        "properties": {
            "intent": {
                "type": "object",
                "additionalProperties": False,
                "required": ["task_type", "confidence"],
                "properties": {
                    "task_type": {"type": "string"},
                    "confidence": {"type": "number"},
                },
            },
            "inputs": resolved_inputs_schema,
            "missing_fields": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["field", "reason", "critical"],
                    "properties": {
                        "field": missing_field_schema,
                        "reason": {"type": "string"},
                        "critical": {"type": "boolean"},
                    },
                },
            },
            "extras": {
                "type": "object",
                "additionalProperties": False,
                "properties": {},
            },
        },
    }
