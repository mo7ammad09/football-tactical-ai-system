"""Deployment preflight checks for RunPod identity artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.identity.pre_render_audit import KNOWN_BAD_RUNPOD_IMAGES, RUNPOD_BASELINE_IMAGE


REQUIRED_RUNPOD_ARTIFACTS: tuple[str, ...] = (
    "annotated_video",
    "report_json",
    "report_csv",
    "raw_tracklets_jsonl",
    "identity_debug_json",
    "identity_events_json",
    "render_audit_before_json",
    "render_audit_after_json",
    "correction_candidates_json",
    "correction_plan_json",
    "correction_applied_json",
    "vision_review_queue_json",
    "player_crop_index_json",
    "vision_review_results_json",
    "final_render_identity_manifest_json",
    "vision_contact_sheets_zip",
)

REQUIRED_SOURCE_FILES: tuple[str, ...] = (
    "src/identity/__init__.py",
    "src/identity/pre_render_audit.py",
    "src/identity/reporting.py",
    "src/identity/deployment_preflight.py",
    "src/processing/batch_analyzer.py",
    "runpod/handler.py",
    "runpod/Dockerfile",
    ".github/workflows/build-runpod-worker.yml",
)


def _read(repo_root: Path, relative_path: str) -> str:
    """Read a required repo file."""
    return (repo_root / relative_path).read_text(encoding="utf-8")


def _contains_any_prefix(patterns: list[str], prefix: str) -> bool:
    """Return whether dockerignore patterns contain a risky broad prefix."""
    prefix = prefix.rstrip("/") + "/"
    return any(pattern.rstrip("/") + "/" == prefix for pattern in patterns)


def _dockerignore_patterns(repo_root: Path) -> list[str]:
    """Return non-comment dockerignore patterns."""
    path = repo_root / ".dockerignore"
    if not path.exists():
        return []
    patterns = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            patterns.append(stripped)
    return patterns


def run_phase9_preflight(repo_root: str | Path) -> dict[str, Any]:
    """Validate RunPod packaging and artifact surfacing before image build."""
    root = Path(repo_root).resolve()
    failures: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []

    for relative_path in REQUIRED_SOURCE_FILES:
        if not (root / relative_path).exists():
            failures.append(
                {
                    "code": "missing_required_source_file",
                    "path": relative_path,
                    "message": f"Required file is missing: {relative_path}",
                }
            )

    dockerfile = _read(root, "runpod/Dockerfile") if (root / "runpod/Dockerfile").exists() else ""
    if "COPY src/ ./src/" not in dockerfile:
        failures.append(
            {
                "code": "runpod_dockerfile_missing_src_copy",
                "path": "runpod/Dockerfile",
                "message": "RunPod worker image must copy src/ so identity modules are included.",
            }
        )
    if "COPY runpod/handler.py ." not in dockerfile:
        failures.append(
            {
                "code": "runpod_dockerfile_missing_handler_copy",
                "path": "runpod/Dockerfile",
                "message": "RunPod worker image must copy runpod/handler.py.",
            }
        )

    dockerignore = _dockerignore_patterns(root)
    if _contains_any_prefix(dockerignore, "src"):
        failures.append(
            {
                "code": "dockerignore_excludes_src",
                "path": ".dockerignore",
                "message": ".dockerignore must not exclude src/ for RunPod worker builds.",
            }
        )
    if "runpod/handler.py" in dockerignore:
        failures.append(
            {
                "code": "dockerignore_excludes_runpod_handler",
                "path": ".dockerignore",
                "message": ".dockerignore must not exclude runpod/handler.py.",
            }
        )

    handler_text = _read(root, "runpod/handler.py") if (root / "runpod/handler.py").exists() else ""
    batch_text = (
        _read(root, "src/processing/batch_analyzer.py")
        if (root / "src/processing/batch_analyzer.py").exists()
        else ""
    )
    for artifact in REQUIRED_RUNPOD_ARTIFACTS:
        if f'"{artifact}"' not in handler_text:
            failures.append(
                {
                    "code": "handler_missing_artifact_upload",
                    "path": "runpod/handler.py",
                    "message": f"RunPod handler does not upload artifact: {artifact}",
                }
            )
        if f'"{artifact}"' not in batch_text:
            failures.append(
                {
                    "code": "batch_analyzer_missing_artifact_export",
                    "path": "src/processing/batch_analyzer.py",
                    "message": f"Batch analyzer does not expose artifact: {artifact}",
                }
            )

    workflow_text = (
        _read(root, ".github/workflows/build-runpod-worker.yml")
        if (root / ".github/workflows/build-runpod-worker.yml").exists()
        else ""
    )
    if "runpod/Dockerfile" not in workflow_text or "runpod/handler.py" not in workflow_text:
        failures.append(
            {
                "code": "workflow_missing_runpod_paths",
                "path": ".github/workflows/build-runpod-worker.yml",
                "message": "RunPod build workflow must be triggered by Dockerfile and handler changes.",
            }
        )
    if "src/**" not in workflow_text:
        failures.append(
            {
                "code": "workflow_missing_src_trigger",
                "path": ".github/workflows/build-runpod-worker.yml",
                "message": "RunPod build workflow must be triggered by src/** changes.",
            }
        )

    known_bad_images = {item["image"] for item in KNOWN_BAD_RUNPOD_IMAGES}
    if not RUNPOD_BASELINE_IMAGE.endswith(":sha-bbe8dec"):
        failures.append(
            {
                "code": "unexpected_baseline_image",
                "path": "src/identity/pre_render_audit.py",
                "message": "RunPod baseline image must remain sha-bbe8dec until a new release is verified.",
            }
        )
    if "ghcr.io/mo7ammad09/football-tactical-ai-runpod:sha-5f969e6" not in known_bad_images:
        failures.append(
            {
                "code": "bad_image_not_registered",
                "path": "src/identity/pre_render_audit.py",
                "message": "Known bad sha-5f969e6 image must stay registered as a regression.",
            }
        )

    if "models" in dockerignore:
        warnings.append(
            {
                "code": "models_excluded_from_build_context",
                "path": ".dockerignore",
                "message": "Models are excluded from Docker context; RunPod must supply model_path or baked weights separately.",
            }
        )

    return {
        "schema_version": "1.0",
        "phase": "phase_9_runpod_packaging_preflight",
        "baseline_image": RUNPOD_BASELINE_IMAGE,
        "known_bad_images": list(KNOWN_BAD_RUNPOD_IMAGES),
        "required_artifacts": list(REQUIRED_RUNPOD_ARTIFACTS),
        "checked_files": list(REQUIRED_SOURCE_FILES),
        "failure_count": len(failures),
        "warning_count": len(warnings),
        "failures": failures,
        "warnings": warnings,
        "verdict": "PASS" if not failures else "FAIL",
    }
