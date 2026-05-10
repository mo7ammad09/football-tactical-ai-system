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
import re
import time
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
OPENROUTER_PROVIDER = "openrouter"
OPENROUTER_PROVIDERS = {
    OPENROUTER_PROVIDER,
    "openrouter_gemma",
    "openrouter_gemma_api",
}
DEFAULT_OPENROUTER_GEMMA_MODEL = "google/gemma-4-31b-it"
OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
PROVIDER_ERROR_REDACTION_PATTERNS = (
    re.compile(r"key=[^&\s)]+"),
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),
    re.compile(r"sk-or-v1-[0-9A-Za-z_\-]+"),
    re.compile(r"Bearer\s+[0-9A-Za-z_\-\.]+", flags=re.IGNORECASE),
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
                "player",
                "referee",
                "team_1",
                "team_2",
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
                "display_team": crop.get("display_team"),
                "team": crop.get("team"),
                "team_color": crop.get("team_color"),
                "display_color": crop.get("display_color"),
                "bbox": crop.get("bbox"),
                "confidence": crop.get("confidence"),
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
    if question == "team_assignment_uncertain":
        return (
            "Review the audit evidence, crop metadata, and contact sheet for case "
            f"{case_id}. Decide whether the track has strong evidence for team_1 "
            "or team_2. Do not treat KMeans/team counts as ground truth when they "
            "are split or weak. Compare the crops from each reported team segment; "
            "if they look like different people, return unresolved and start the "
            "reason with 'segment_split_required:'. If the same real player is "
            "split across separate display track IDs, start the reason with "
            "'identity_cluster_required:'. Return team_1 or team_2 only when "
            "numeric and visual evidence are both strong."
        )
    if question == "role_stability_flicker":
        return (
            "Review the audit evidence, crop metadata, and contact sheet for case "
            f"{case_id}. Decide whether the visible role should be player, "
            "referee, goalkeeper, or unresolved. Compare crops from every role "
            "segment, especially minority/problem frames. If one display track "
            "contains multiple real people, return unresolved and start the reason "
            "with 'segment_split_required:'. Treat isolated flicker as uncertain "
            "unless neighboring metadata and visual evidence strongly support one role."
        )
    return (
        f"Review identity case {case_id}. Decide only the requested identity "
        "question from the provided crops and metadata. Return only JSON "
        "matching the schema; do not infer jersey numbers."
    )


def _is_google_gemma_provider(provider: str | None) -> bool:
    """Return true when provider should use Google's Gemma/Gemini API."""
    return str(provider or "").strip().lower() in GOOGLE_GEMMA_PROVIDERS


def _is_openrouter_provider(provider: str | None) -> bool:
    """Return true when provider should use OpenRouter chat completions."""
    return str(provider or "").strip().lower() in OPENROUTER_PROVIDERS


def _google_api_key() -> str | None:
    """Return the configured Google AI API key, if available."""
    return (
        os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_AI_API_KEY")
    )


def _openrouter_api_key() -> str | None:
    """Return the configured OpenRouter API key, if available."""
    return os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_KEY")


def _resolved_google_model(model: str | None = None) -> str:
    """Return the configured Gemma model id for the Google API provider."""
    return (
        model
        or os.environ.get("IDENTITY_REVIEW_MODEL")
        or os.environ.get("GOOGLE_GEMMA_MODEL")
        or DEFAULT_GOOGLE_GEMMA_MODEL
    )


def _resolved_openrouter_model(model: str | None = None) -> str:
    """Return the configured OpenRouter model id for Gemma review."""
    return (
        model
        or os.environ.get("IDENTITY_REVIEW_MODEL")
        or os.environ.get("OPENROUTER_MODEL")
        or os.environ.get("OPENROUTER_GEMMA_MODEL")
        or DEFAULT_OPENROUTER_GEMMA_MODEL
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
                "reason": case.get("reason"),
                "audit_evidence": case.get("audit_evidence", {}),
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
        "reason": case.get("reason"),
        "crop_count": case.get("crop_count"),
        "target_count": case.get("target_count"),
        "audit_evidence": case.get("audit_evidence", {}),
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
        "not_goalkeeper, player, referee, team_1, team_2, unresolved.\n"
        "Use unresolved when the evidence is weak, cropped poorly, or ambiguous.\n"
        "If the evidence shows one track contains multiple real people, return "
        "verdict unresolved and begin reason with 'segment_split_required:'.\n"
        "If the evidence shows one real player is split across multiple track IDs, "
        "return verdict unresolved and begin reason with 'identity_cluster_required:'.\n"
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


def _openrouter_image_part(path_value: Any) -> dict[str, Any] | None:
    """Return an OpenAI-compatible image_url part for OpenRouter."""
    image_part = _image_part(path_value)
    if not image_part:
        return None
    inline_data = image_part.get("inline_data") or {}
    mime_type = inline_data.get("mime_type") or "image/jpeg"
    data = inline_data.get("data")
    if not isinstance(data, str) or not data:
        return None
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{data}"},
    }


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


def _extract_openrouter_text(payload: dict[str, Any]) -> str:
    """Extract text content from an OpenRouter chat-completions response."""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else {}
    content = message.get("content") if isinstance(message, dict) else ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts: list[str] = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                texts.append(part["text"])
        return "\n".join(texts).strip()
    return ""


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


def sanitize_provider_error_text(value: Any) -> str:
    """Return provider text safe to persist in public review artifacts."""
    text = str(value or "")
    for pattern in PROVIDER_ERROR_REDACTION_PATTERNS:
        if pattern.pattern.startswith("key="):
            replacement = "key=REDACTED"
        elif pattern.pattern.lower().startswith("bearer"):
            replacement = "Bearer API_KEY_REDACTED"
        else:
            replacement = "API_KEY_REDACTED"
        text = pattern.sub(replacement, text)
    # Requests HTTPError strings include the full URL; keep only the useful status.
    text = re.sub(
        r" for url: https://generativelanguage\.googleapis\.com/[^\s)]+",
        " for Google Generative Language API",
        text,
    )
    text = re.sub(
        r" for url: https://openrouter\.ai/[^\s)]+",
        " for OpenRouter API",
        text,
    )
    return text


def classify_provider_error_text(value: Any) -> str | None:
    """Classify common provider failures so downstream reports are explicit."""
    text = str(value or "").lower()
    if "429" in text or "too many requests" in text or "rate limit" in text:
        return "provider_rate_limited"
    if "403" in text or "forbidden" in text or "permission" in text:
        return "provider_forbidden"
    if "timeout" in text or "timed out" in text:
        return "provider_timeout"
    if "api key" in text or "key is not configured" in text:
        return "provider_missing_or_invalid_key"
    return None


def _provider_failure_reason(
    exc: Exception,
    *,
    attempt: int,
    max_attempts: int,
    provider_label: str,
) -> str:
    """Return a sanitized, retry-aware failure reason for a provider exception."""
    exc_name = exc.__class__.__name__
    detail = sanitize_provider_error_text(exc)
    return (
        f"{provider_label} failed "
        f"after attempt {attempt}/{max_attempts}: {exc_name}: {detail}"
    )


def _retry_delay_seconds(exc: Exception, *, fallback_delay: float, attempt: int) -> float:
    """Return provider retry delay, honoring Retry-After when present."""
    response = getattr(exc, "response", None)
    retry_after = None
    if response is not None:
        retry_after = getattr(response, "headers", {}).get("Retry-After")
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except (TypeError, ValueError):
            pass
    return max(0.0, float(fallback_delay) * float(attempt))


def _unresolved_provider_output(
    *,
    case_id: str,
    reason: str,
    evidence: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return a conservative unresolved output for provider failures."""
    failure_category = classify_provider_error_text(reason) or "provider_error"
    return {
        "case_id": case_id,
        "status": "reviewed",
        "verdict": "unresolved",
        "confidence": 0.0,
        "reason": sanitize_provider_error_text(reason),
        "failure_category": failure_category,
        "evidence": [
            {"type": "provider_failure_category", "category": failure_category},
            *(evidence or []),
        ],
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
        "reason": sanitize_provider_error_text(output.get("reason") or ""),
        "evidence": evidence if isinstance(evidence, list) else [],
    }


def _invoke_google_gemma_case(
    *,
    case: dict[str, Any],
    api_key: str,
    model: str,
    timeout_seconds: float,
    max_attempts: int,
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
    max_attempts = max(1, int(max_attempts))
    retry_delay = float(os.environ.get("IDENTITY_REVIEW_PROVIDER_RETRY_BACKOFF", "1.5"))
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
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
            last_exc = exc
            if attempt < max_attempts:
                time.sleep(_retry_delay_seconds(exc, fallback_delay=retry_delay, attempt=attempt))

    reason = (
        _provider_failure_reason(
            last_exc,
            attempt=max_attempts,
            max_attempts=max_attempts,
            provider_label="Google Gemma API",
        )
        if last_exc is not None
        else "Google Gemma API failed without a provider exception."
    )
    return _unresolved_provider_output(
        case_id=case_id,
        reason=reason,
        evidence=[
            {
                "type": "provider_error",
                "provider": GOOGLE_GEMMA_PROVIDER,
                "attempt_count": max_attempts,
            }
        ],
    )


def _invoke_openrouter_case(
    *,
    case: dict[str, Any],
    api_key: str,
    model: str,
    timeout_seconds: float,
    max_attempts: int,
) -> dict[str, Any]:
    """Invoke OpenRouter's chat-completions API for one identity case."""
    import requests

    case_id = str(case.get("case_id") or "unknown_case")
    content: list[dict[str, Any]] = [{"type": "text", "text": _google_prompt(case)}]
    image_part = _openrouter_image_part(case.get("contact_sheet_path"))
    if image_part is not None:
        content.append(image_part)

    request_payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    max_tokens = os.environ.get("IDENTITY_REVIEW_PROVIDER_MAX_TOKENS")
    if max_tokens:
        request_payload["max_tokens"] = max(1, int(max_tokens))

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    referer = os.environ.get("OPENROUTER_HTTP_REFERER")
    title = os.environ.get("OPENROUTER_X_TITLE", "Football Tactical AI Identity Review")
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title

    max_attempts = max(1, int(max_attempts))
    retry_delay = float(os.environ.get("IDENTITY_REVIEW_PROVIDER_RETRY_BACKOFF", "3.0"))
    last_exc: Exception | None = None
    url = os.environ.get("OPENROUTER_CHAT_COMPLETIONS_URL", OPENROUTER_CHAT_COMPLETIONS_URL)
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=request_payload,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            text = _extract_openrouter_text(response.json())
            parsed = _parse_json_object(text)
            if parsed is None:
                return _unresolved_provider_output(
                    case_id=case_id,
                    reason="OpenRouter Gemma API returned non-JSON output.",
                    evidence=[{"type": "raw_text", "text": text[:500]}] if text else [],
                )
            return _sanitize_provider_output(case_id, parsed)
        except Exception as exc:  # noqa: BLE001 - provider failures must fail closed.
            last_exc = exc
            if attempt < max_attempts:
                time.sleep(_retry_delay_seconds(exc, fallback_delay=retry_delay, attempt=attempt))

    reason = (
        _provider_failure_reason(
            last_exc,
            attempt=max_attempts,
            max_attempts=max_attempts,
            provider_label="OpenRouter Gemma API",
        )
        if last_exc is not None
        else "OpenRouter Gemma API failed without a provider exception."
    )
    return _unresolved_provider_output(
        case_id=case_id,
        reason=reason,
        evidence=[
            {
                "type": "provider_error",
                "provider": OPENROUTER_PROVIDER,
                "attempt_count": max_attempts,
            }
        ],
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
    uses_google = _is_google_gemma_provider(provider_name)
    uses_openrouter = _is_openrouter_provider(provider_name)
    if not uses_google and not uses_openrouter:
        return {
            "status": "unsupported_provider",
            "model_outputs": None,
            **base_meta,
            "output_count": 0,
        }

    if uses_openrouter:
        api_key = _openrouter_api_key()
        model_name = _resolved_openrouter_model(model_name)
        missing_env = "OPENROUTER_API_KEY"
        missing_reason = "OpenRouter API key is not configured."
    else:
        api_key = _google_api_key()
        model_name = _resolved_google_model(model_name)
        missing_env = "GOOGLE_API_KEY"
        missing_reason = "Google API key is not configured."

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
                reason=missing_reason,
                evidence=[{"type": "missing_env", "env": missing_env}],
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
    max_attempts = int(os.environ.get("IDENTITY_REVIEW_PROVIDER_RETRIES", "4"))
    default_case_delay = "3.0" if uses_openrouter else "0.0"
    case_delay = float(os.environ.get("IDENTITY_REVIEW_PROVIDER_CASE_DELAY", default_case_delay))
    invoke_case = _invoke_openrouter_case if uses_openrouter else _invoke_google_gemma_case
    outputs = []
    for index, case in enumerate(cases):
        outputs.append(
            invoke_case(
                case=case,
                api_key=api_key,
                model=model_name,
                timeout_seconds=timeout_seconds,
                max_attempts=max_attempts,
            )
        )
        if index < len(cases) - 1 and case_delay > 0:
            time.sleep(case_delay)
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
    has_openrouter_key = bool(_openrouter_api_key())
    explicitly_disabled = provider_enabled is False and model_outputs is None
    explicit_google_provider = bool(
        requested_provider and _is_google_gemma_provider(requested_provider) and has_google_key
    )
    explicit_openrouter_provider = bool(
        requested_provider and _is_openrouter_provider(requested_provider) and has_openrouter_key
    )
    should_default_openrouter = bool(
        provider_enabled and has_openrouter_key and not requested_provider
    )
    should_default_google = bool(
        provider_enabled and has_google_key and not requested_provider and not should_default_openrouter
    )
    enabled = bool(
        model_outputs is not None
        or (
            not explicitly_disabled
            and bool(provider_enabled or explicit_google_provider or explicit_openrouter_provider)
        )
    )
    resolved_provider = (
        requested_provider
        or (OPENROUTER_PROVIDER if should_default_openrouter else None)
        or (GOOGLE_GEMMA_PROVIDER if should_default_google else None)
        or (DEFAULT_REVIEW_PROVIDER if enabled else None)
    )
    if model:
        resolved_model = model
    elif _is_openrouter_provider(resolved_provider):
        resolved_model = _resolved_openrouter_model(None)
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
