from src.identity.deployment_preflight import REQUIRED_RUNPOD_ARTIFACTS
from src.identity.release_gate import build_runpod_release_candidate_manifest


def _passing_preflight():
    return {"verdict": "PASS", "failure_count": 0, "warning_count": 1}


def _passing_tests():
    return [
        {"command": "preflight", "verdict": "PASS"},
        {"command": "pytest", "verdict": "PASS"},
    ]


def test_phase10_blocks_known_bad_candidate_image():
    manifest = build_runpod_release_candidate_manifest(
        candidate_image="ghcr.io/mo7ammad09/football-tactical-ai-runpod:sha-5f969e6",
        source_commit="5f969e643494",
        source_dirty=False,
        preflight_result=_passing_preflight(),
        test_results=_passing_tests(),
    )

    assert manifest["phase"] == "phase_10_runpod_release_candidate_gate"
    assert manifest["release_status"] == "blocked"
    assert "candidate_image_is_known_bad_regression" in manifest["blockers"]


def test_phase10_blocks_dirty_source_even_when_tests_pass():
    manifest = build_runpod_release_candidate_manifest(
        candidate_image="ghcr.io/mo7ammad09/football-tactical-ai-runpod:sha-newgood1",
        source_commit="abc123def456",
        source_dirty=True,
        preflight_result=_passing_preflight(),
        test_results=_passing_tests(),
    )

    assert manifest["release_status"] == "blocked"
    assert "source_tree_has_uncommitted_changes" in manifest["blockers"]


def test_phase10_marks_candidate_ready_after_clean_preflight_and_tests():
    manifest = build_runpod_release_candidate_manifest(
        candidate_image="ghcr.io/mo7ammad09/football-tactical-ai-runpod:sha-newgood1",
        source_commit="abc123def456",
        source_dirty=False,
        preflight_result=_passing_preflight(),
        test_results=_passing_tests(),
    )

    assert manifest["release_status"] == "candidate_ready"
    assert manifest["blockers"] == []
    assert set(REQUIRED_RUNPOD_ARTIFACTS).issubset(manifest["required_artifacts"])
    assert manifest["required_runpod_smoke_checks"]
