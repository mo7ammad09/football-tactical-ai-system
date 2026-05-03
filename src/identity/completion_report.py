"""Local completion report for the identity-correction project."""

from __future__ import annotations

from typing import Any


COMPLETED_PHASES: tuple[dict[str, str], ...] = (
    {"phase": "phase_1", "name": "pre_render_audit"},
    {"phase": "phase_2", "name": "dry_run_correction_plan"},
    {"phase": "phase_3", "name": "safe_deterministic_apply"},
    {"phase": "phase_4", "name": "vision_review_queue"},
    {"phase": "phase_5", "name": "crop_and_contact_sheet_evidence"},
    {"phase": "phase_6", "name": "vision_review_results_no_guessing"},
    {"phase": "phase_7", "name": "final_render_identity_manifest"},
    {"phase": "phase_8", "name": "ui_api_status_surfacing"},
    {"phase": "phase_9", "name": "runpod_packaging_preflight"},
    {"phase": "phase_10", "name": "release_candidate_gate"},
)


def build_identity_project_completion_report(
    *,
    preflight_result: dict[str, Any],
    release_manifest: dict[str, Any],
    verification: dict[str, Any],
) -> dict[str, Any]:
    """Build a compact local completion status report."""
    blockers: list[str] = []
    if preflight_result.get("verdict") != "PASS":
        blockers.append("phase9_preflight_failed")
    if release_manifest.get("release_status") != "candidate_ready":
        blockers.append("runpod_candidate_not_ready")
    if verification.get("verdict") != "PASS":
        blockers.append("verification_not_passed")

    local_status = "ready_for_candidate_build" if not blockers else "implementation_complete_deploy_blocked"
    return {
        "schema_version": "1.0",
        "project": "pre_render_identity_correction",
        "local_status": local_status,
        "completed_phase_count": len(COMPLETED_PHASES),
        "completed_phases": list(COMPLETED_PHASES),
        "preflight": {
            "phase": preflight_result.get("phase"),
            "verdict": preflight_result.get("verdict"),
            "failure_count": preflight_result.get("failure_count"),
            "warning_count": preflight_result.get("warning_count"),
        },
        "release_gate": {
            "phase": release_manifest.get("phase"),
            "release_status": release_manifest.get("release_status"),
            "candidate_image": release_manifest.get("candidate_image"),
            "blockers": release_manifest.get("blockers", []),
        },
        "verification": verification,
        "blockers": blockers,
        "next_required_steps": [
            "Commit the implementation changes intentionally.",
            "Build a new RunPod image from the committed source.",
            "Create a Phase 10 release manifest for the candidate image.",
            "Run a short RunPod smoke job before replacing the endpoint image.",
        ],
    }
