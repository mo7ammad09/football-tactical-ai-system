from pathlib import Path

from src.identity.deployment_preflight import (
    REQUIRED_RUNPOD_ARTIFACTS,
    run_phase9_preflight,
)


def test_phase9_preflight_passes_current_runpod_packaging():
    repo_root = Path(__file__).resolve().parents[1]

    result = run_phase9_preflight(repo_root)

    assert result["phase"] == "phase_9_runpod_packaging_preflight"
    assert result["verdict"] == "PASS"
    assert result["failure_count"] == 0
    assert "models_excluded_from_build_context" in {
        warning["code"] for warning in result["warnings"]
    }
    assert set(REQUIRED_RUNPOD_ARTIFACTS).issubset(result["required_artifacts"])


def test_phase9_preflight_tracks_final_identity_manifest_artifact():
    repo_root = Path(__file__).resolve().parents[1]

    result = run_phase9_preflight(repo_root)

    assert "final_render_identity_manifest_json" in result["required_artifacts"]
    assert "identity_review_decisions_json" in result["required_artifacts"]
    assert "vision_review_results_json" in result["required_artifacts"]
    assert result["baseline_image"].endswith(":sha-bbe8dec")
    assert any(
        item["image"].endswith(":sha-5f969e6")
        for item in result["known_bad_images"]
    )
