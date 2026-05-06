"""Phase 11 model-review request and structured-output helpers.

The integration is intentionally provider-neutral. RunPod may receive structured
model outputs from any LLM/Vision service, while this module builds the exact
request package and normalizes the returned JSON before Phase 6 validation.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any


DEFAULT_REVIEW_PROVIDER = "external_structured"
DEFAULT_REVIEW_MODEL = "identity-review-structured-v1"
GOOGLE_GEMMA_PROVIDER = "google_gemma_api"
GOOGLE_GEMMA_PROVIDERS = {
    GOOGLE_GEMMA_PROVIDER,
    "google_gemma",
    "gemini_gemma_api",
}
DEFAULT_GOOGLE_GEMMA_MODEL = "gemma-4-31b-it"
GOOGLE_GENERATE_CONTENT_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)

IDENTITY_REVIEW_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["case_id", "status", "verdict", "confidence", "reason", "evidence"],
    "properties": {
        "case_id": {"type": "string"},
        "status": {"type": "string", "enum": ["reviewed"]},
        "verdict": {
            "type": "string",
            "enum": [
                "same_player",
                "different_player",
                "goalkeeper",
                "not_goalkeeper",
                "unresolved",
            ],
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reason": {"type": "string"},
        "evidence": {
            "type": "array",
            "items": {"type": "object"},
        },
    },
}


def _as_int(value: Any, fallback: int = 0) -> int:
    """Convert value to int with a stable fallback."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _case_map(payload: dict[str, Any], key: str = "cases") -> dict[str, dict[str, Any]]:
    """Return cases keyed by case id."""
    return {
        str(case.get("case_id")): case
        for case in payload.get(key, []) or []
        if isinstance(case, dict) and case.get("case_id") is not None
    }


def _compact_crop_evidence(crop_case: dict[str, Any]) -> list[dict[str, Any]]:
    """Return compact crop metadata for model prompts."""
    evidence: list[dict[str, Any]] = []
    for crop in crop_case.get("crop_requests", []) or []:
        if not isinstance(crop, dict):
            continue
        evidence.append(
            {
                "crop_id": crop.get("crop_id"),
                "track_id": crop.get("track_id"),
                "raw_track_id": crop.get("raw_track_id"),
                "source_frame_idx": crop.get("source_frame_idx"),
                "role": crop.get("role"),
                "detected_role": crop.get("detected_role"),
                "display_label": crop.get("display_label"),
                "display_role": crop.get("display_role"),
                "crop_path": crop.get("crop_path"),
            }
        )
    return evidence


def _case_prompt(case: dict[str, Any], crop_case: dict[str, Any]) -> str:
    """Build a concise provider-neutral prompt for one identity case."""
    question = str(case.get("question") or "identity_review")
    case_id = str(case.get("case_id") or "unknown_case")
    if question == "goalkeeper_identity_fragmentation":
        return (
            "Review the contact sheet and crop metadata for case "
            f"{case_id}. Decide whether all shown goalkeeper crops belong to "
            "the same real goalkeeper. Return only JSON matching the schema. "
            "Use same_player only when visual evidence is strong; otherwise "
            "return unresolved."
        )
    return (
        f"Review identity case {case_id}. Decide only the requested identity "
        "question from the provided crops and metadata. Return only JSON "
        "matching the schema; do not infer jersey numbers."
    )


def _is_google_gemma_provider(provider: str | None) -> bool:
    """Return true when provider should use Google's Gemma/Gemini API."""
    return str(provider or "").strip().lower() in GOOGLE_GEMMA_PROVIDERS


def _google_api_key() -> str | None:
    """Return the configured Google AI API key, if available."""
    return (
        os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_AI_API_KEY")
    )


def _resolved_google_model(model: str | None = None) -> str:
    """Return the configured Gemma model id for the Google API provider."""
    return (
        model
        or os.environ.get("IDENTITY_REVIEW_MODEL")
        or os.environ.get("GOOGLE_GEMMA_MODEL")
        or DEFAULT_GOOGLE_GEMMA_MODEL
    )


def build_identity_model_review_request(
    *,
    vision_review_queue: dict[str, Any],
    player_crop_index: dict[str, Any],
    provider: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Build the Phase 11 request package for an external LLM/Vision reviewer."""
    crop_cases = _case_map(player_crop_index)
    cases: list[dict[str, Any]] = []
    for case in vision_review_queue.get("cases", []) or []:
        if not isinstance(case, dict):
            continue
        case_id = str(case.get("case_id") or "unknown_case")
        crop_case = crop_cases.get(case_id, {})
        crop_evidence = _compact_crop_evidence(crop_case)
        cases.append(
            {
                "case_id": case_id,
                "question": case.get("question"),
                "priority": case.get("priority"),
                "status": "ready_for_model_review" if crop_evidence else "missing_crop_evidence",
                "prompt": _case_prompt(case, crop_case),
                "contact_sheet_path": crop_case.get("contact_sheet_path"),
                "crop_count": len(crop_evidence),
                "target_count": _as_int(crop_case.get("target_count"), 0),
                "crop_evidence": crop_evidence,
                "output_schema": IDENTITY_REVIEW_OUTPUT_SCHEMA,
            }
        )

    return {
        "schema_version": "1.0",
        "phase": "phase_11_identity_model_review_request",
        "provider": provider or DEFAULT_REVIEW_PROVIDER,
        "model": model or DEFAULT_REVIEW_MODEL,
        "case_count": len(cases),
        "cases": cases,
        "instructions": [
            "Return one JSON object per case.",
            "Do not mutate track IDs or colors.",
            "Use unresolved when evidence is weak or ambiguous.",
            "Any resolved result will still pass deterministic validators.",
        ],
    }


def _case_payload_for_prompt(case: dict[str, Any]) -> dict[str, Any]:
    """Return compact case data safe to send to an identity-review model."""
    return {
        "case_id": case.get("case_id"),
        "question": case.get("question"),
        "priority": case.get("priority"),
        "crop_count": case.get("crop_count"),
        "target_count": case.get("target_count"),
        "crop_evidence": case.get("crop_evidence", []),
    }


def _google_prompt(case: dict[str, Any]) -> str:
    """Build the strict JSON-only prompt for one identity case."""
    schema = json.dumps(IDENTITY_REVIEW_OUTPUT_SCHEMA, ensure_ascii=False)
    payload = json.dumps(_case_payload_for_prompt(case), ensure_ascii=False, indent=2)
    return (
        "You are a cautious football identity-review model.\n"
        "Decide only the identity question in this one case.\n"
        "Return exactly one JSON object. No markdown. No extra text.\n"
        "Allowed verdict values: same_player, different_player, goalkeeper, "
        "not_goalkeeper, unresolved.\n"
        "Use unresolved when the evidence is weak, cropped poorly, or ambiguous.\n"
        "Do not rename track IDs. Do not change render colors.\n\n"
        f"Required JSON schema:\n{schema}\n\n"
        f"Case data:\n{payload}\n\n"
        f"Task:\n{case.get('prompt') or 'Review this identity case.'}"
    )


def _image_part(path_value: Any) -> dict[str, Any] | None:
    """Return a Gemini inline image part for a local contact sheet."""
    if not path_value:
        return None
    path = Path(str(path_value))
    if not path.exists() or not path.is_file():
        return None
    max_bytes = int(os.environ.get("IDENTITY_REVIEW_MAX_IMAGE_BYTES", str(6 * 1024 * 1024)))
    if path.stat().st_size > max_bytes:
        return None
    mime_type = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"inline_data": {"mime_type": mime_type, "data": data}}


def _extract_google_text(payload: dict[str, Any]) -> str:
    """Extract text content from a Google generateContent response."""
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return ""
    content = candidates[0].get("content") if isinstance(candidates[0], dict) else {}
    parts = content.get("parts") if isinstance(content, dict) else []
    texts: list[str] = []
    for part in parts or []:
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            texts.append(part["text"])
    return "\n".join(texts).strip()


def _parse_json_object(text: str) -> dict[str, Any] | None:
    """Parse a JSON object from model text, including fenced output."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`").strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def _clamp_confidence(value: Any) -> float:
    """Clamp model confidence into the expected 0..1 range."""
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, confidence))


def _unresolved_provider_output(
    *,
    case_id: str,
    reason: str,
    evidence: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return a conservative unresolved output for provider failures."""
    return {
        "case_id": case_id,
        "status": "reviewed",
        "verdict": "unresolved",
        "confidence": 0.0,
        "reason": reason,
        "evidence": evidence or [],
    }


def _sanitize_provider_output(case_id: str, output: dict[str, Any]) -> dict[str, Any]:
    """Coerce a provider JSON object into the Phase 6 output schema."""
    verdict = str(output.get("verdict") or "unresolved")
    allowed_verdicts = set(IDENTITY_REVIEW_OUTPUT_SCHEMA["properties"]["verdict"]["enum"])
    if verdict not in allowed_verdicts:
        verdict = "unresolved"
    evidence = output.get("evidence")
    return {
        "case_id": str(output.get("case_id") or case_id),
        "status": "reviewed",
        "verdict": verdict,
        "confidence": _clamp_confidence(output.get("confidence")),
        "reason": str(output.get("reason") or ""),
        "evidence": evidence if isinstance(evidence, list) else [],
    }


def _invoke_google_gemma_case(
    *,
    case: dict[str, Any],
    api_key: str,
    model: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Invoke Google's Gemma/Gemini API for one identity case."""
    import requests

    case_id = str(case.get("case_id") or "unknown_case")
    parts: list[dict[str, Any]] = [{"text": _google_prompt(case)}]
    image_part = _image_part(case.get("contact_sheet_path"))
    if image_part is not None:
        parts.append(image_part)

    request_payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": 0.0,
            "response_mime_type": "application/json",
        },
    }
    try:
        response = requests.post(
            GOOGLE_GENERATE_CONTENT_URL.format(model=model),
            params={"key": api_key},
            json=request_payload,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        text = _extract_google_text(response.json())
        parsed = _parse_json_object(text)
        if parsed is None:
            return _unresolved_provider_output(
                case_id=case_id,
                reason="Google Gemma API returned non-JSON output.",
                evidence=[{"type": "raw_text", "text": text[:500]}] if text else [],
            )
        return _sanitize_provider_output(case_id, parsed)
    except Exception as exc:  # noqa: BLE001 - provider failures must fail closed.
        return _unresolved_provider_output(
            case_id=case_id,
            reason=f"Google Gemma API failed: {exc}",
            evidence=[{"type": "provider_error", "provider": GOOGLE_GEMMA_PROVIDER}],
        )


def invoke_identity_review_provider(
    *,
    request: dict[str, Any],
    provider: str | None,
    model: str | None,
    provider_enabled: bool,
    model_outputs: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Invoke a configured identity-review provider, returning safe structured outputs.

    Existing caller-supplied model outputs always win. Live provider failures are
    converted to unresolved case outputs so the downstream validator never has to
    guess or mutate identity from a failed LLM call.
    """
    normalized_existing = normalize_identity_review_model_outputs(model_outputs)
    provider_name = str(provider or "").strip() or None
    model_name = str(model or "").strip() or None
    base_meta: dict[str, Any] = {
        "provider": provider_name,
        "model": model_name,
        "enabled": bool(provider_enabled),
        "case_count": int(request.get("case_count") or 0),
    }

    if normalized_existing is not None:
        return {
            "status": "supplied_outputs",
            "model_outputs": normalized_existing,
            **base_meta,
            "output_count": len(normalized_existing),
        }
    if not provider_enabled:
        return {
            "status": "disabled",
            "model_outputs": None,
            **base_meta,
            "output_count": 0,
        }
    if not _is_google_gemma_provider(provider_name):
        return {
            "status": "unsupported_provider",
            "model_outputs": None,
            **base_meta,
            "output_count": 0,
        }

    api_key = _google_api_key()
    model_name = _resolved_google_model(model_name)
    base_meta["model"] = model_name
    cases = [
        case
        for case in request.get("cases", []) or []
        if isinstance(case, dict) and case.get("status") == "ready_for_model_review"
    ]
    if not cases:
        return {
            "status": "no_ready_cases",
            "model_outputs": [],
            **base_meta,
            "output_count": 0,
        }
    if not api_key:
        outputs = [
            _unresolved_provider_output(
                case_id=str(case.get("case_id") or "unknown_case"),
                reason="Google API key is not configured.",
                evidence=[{"type": "missing_env", "env": "GOOGLE_API_KEY"}],
            )
            for case in cases
        ]
        return {
            "status": "missing_api_key",
            "model_outputs": outputs,
            **base_meta,
            "output_count": len(outputs),
        }

    timeout_seconds = float(os.environ.get("IDENTITY_REVIEW_PROVIDER_TIMEOUT", "45"))
    outputs = [
        _invoke_google_gemma_case(
            case=case,
            api_key=api_key,
            model=model_name,
            timeout_seconds=timeout_seconds,
        )
        for case in cases
    ]
    return {
        "status": "invoked",
        "model_outputs": outputs,
        **base_meta,
        "output_count": len(outputs),
    }


def normalize_identity_review_model_outputs(value: Any) -> list[dict[str, Any]] | None:
    """Normalize RunPod/user-supplied model outputs into a list of dictionaries."""
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            return []
    if isinstance(value, dict):
        if isinstance(value.get("model_outputs"), list):
            value = value["model_outputs"]
        elif isinstance(value.get("results"), list):
            value = value["results"]
        else:
            value = [value]
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def resolve_identity_review_model_config(
    *,
    provider: str | None,
    model: str | None,
    provider_enabled: bool | None,
    model_outputs: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Return stable provider config for Phase 11/Phase 6."""
    requested_provider = str(provider or "").strip() or None
    has_google_key = bool(_google_api_key())
    explicitly_disabled = provider_enabled is False and model_outputs is None
    explicit_google_provider = bool(
        requested_provider and _is_google_gemma_provider(requested_provider) and has_google_key
    )
    should_default_google = bool(provider_enabled and has_google_key and not requested_provider)
    enabled = bool(
        model_outputs is not None
        or (
            not explicitly_disabled
            and bool(provider_enabled or explicit_google_provider)
        )
    )
    resolved_provider = (
        requested_provider
        or (GOOGLE_GEMMA_PROVIDER if should_default_google else None)
        or (DEFAULT_REVIEW_PROVIDER if enabled else None)
    )
    if model:
        resolved_model = model
    elif _is_google_gemma_provider(resolved_provider):
        resolved_model = _resolved_google_model(None)
    elif enabled:
        resolved_model = DEFAULT_REVIEW_MODEL
    else:
        resolved_model = None
    return {
        "provider_enabled": enabled,
        "provider": resolved_provider,
        "model": resolved_model,
        "model_output_count": len(model_outputs or []),
    }
