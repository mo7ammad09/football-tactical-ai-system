"""Release-candidate gate for RunPod identity-correction images."""

from __future__ import annotations

from typing import Any

from src.identity.deployment_preflight import REQUIRED_RUNPOD_ARTIFACTS
from src.identity.pre_render_audit import KNOWN_BAD_RUNPOD_IMAGES, RUNPOD_BASELINE_IMAGE


DEFAULT_RELEASE_TEST_COMMANDS: tuple[str, ...] = (
    ".venv/bin/python scripts/check_runpod_phase9_preflight.py",
    (
        ".venv/bin/python -m pytest "
        "tests/test_runpod_phase9_preflight.py "
        "tests/test_identity_reporting.py "
        "tests/test_pre_render_identity_audit.py "
        "tests/test_identity_reconciliation.py "
        "tests/test_identity_quality_audit.py"
    ),
)


def _known_bad_image_set() -> set[str]:
    """Return known-bad image refs."""
    return {
        str(item.get("image"))
        for item in KNOWN_BAD_RUNPOD_IMAGES
        if item.get("image")
    }


def _preflight_passed(preflight_result: dict[str, Any] | None) -> bool:
    """Return whether Phase 9 preflight passed."""
    return bool(preflight_result and preflight_result.get("verdict") == "PASS")


def _test_results_passed(test_results: list[dict[str, Any]] | None) -> bool:
    """Return whether all declared release tests passed."""
    if not test_results:
        return False
    return all(str(result.get("verdict")) == "PASS" for result in test_results)


def build_runpod_release_candidate_manifest(
    *,
    candidate_image: str | None,
    source_commit: str | None,
    source_dirty: bool,
    preflight_result: dict[str, Any] | None,
    test_results: list[dict[str, Any]] | None,
    baseline_image: str = RUNPOD_BASELINE_IMAGE,
) -> dict[str, Any]:
    """Build Phase 10 release-candidate manifest.

    This gate does not build or push an image. It records whether a candidate is
    safe enough to test on RunPod, and blocks known-bad or non-reproducible refs.
    """
    candidate_image = str(candidate_image).strip() if candidate_image else None
    source_commit = str(source_commit).strip() if source_commit else None
    known_bad_images = _known_bad_image_set()
    blockers: list[str] = []
    warnings: list[str] = []

    if not candidate_image:
        blockers.append("candidate_image_not_provided")
    elif candidate_image in known_bad_images:
        blockers.append("candidate_image_is_known_bad_regression")
    elif candidate_image == baseline_image:
        warnings.append("candidate_image_matches_current_baseline")

    if not source_commit:
        blockers.append("source_commit_not_available")
    if source_dirty:
        blockers.append("source_tree_has_uncommitted_changes")
    if not _preflight_passed(preflight_result):
        blockers.append("phase9_preflight_not_passed")
    if not _test_results_passed(test_results):
        blockers.append("release_tests_not_passed_or_not_recorded")

    release_status = "candidate_ready" if not blockers else "blocked"
    return {
        "schema_version": "1.0",
        "phase": "phase_10_runpod_release_candidate_gate",
        "release_status": release_status,
        "candidate_image": candidate_image,
        "baseline_image": baseline_image,
        "known_bad_images": list(KNOWN_BAD_RUNPOD_IMAGES),
        "source": {
            "commit": source_commit,
            "dirty": bool(source_dirty),
        },
        "preflight": {
            "verdict": (preflight_result or {}).get("verdict"),
            "failure_count": (preflight_result or {}).get("failure_count"),
            "warning_count": (preflight_result or {}).get("warning_count"),
        },
        "tests": {
            "required_commands": list(DEFAULT_RELEASE_TEST_COMMANDS),
            "results": test_results or [],
            "all_passed": _test_results_passed(test_results),
        },
        "required_artifacts": list(REQUIRED_RUNPOD_ARTIFACTS),
        "blockers": blockers,
        "warnings": warnings,
        "required_runpod_smoke_checks": [
            "Run a short job on the candidate image using a known sample video.",
            "Confirm final_render_identity_manifest_json_url is present.",
            "Confirm release_status is identity_trusted or review_required, not invalid_render.",
            "Confirm defenders/field players do not inherit GK label/color from sha-5f969e6 regression.",
            "Confirm raw_tracklets_jsonl and identity_debug_json are uploaded.",
        ],
    }
