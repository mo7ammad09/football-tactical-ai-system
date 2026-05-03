from src.identity.completion_report import build_identity_project_completion_report


def test_completion_report_is_deploy_blocked_until_candidate_is_ready():
    report = build_identity_project_completion_report(
        preflight_result={"phase": "phase_9", "verdict": "PASS", "failure_count": 0, "warning_count": 1},
        release_manifest={
            "phase": "phase_10",
            "release_status": "blocked",
            "candidate_image": None,
            "blockers": ["candidate_image_not_provided"],
        },
        verification={"verdict": "PASS"},
    )

    assert report["completed_phase_count"] == 10
    assert report["local_status"] == "implementation_complete_deploy_blocked"
    assert "runpod_candidate_not_ready" in report["blockers"]


def test_completion_report_ready_when_all_gates_pass():
    report = build_identity_project_completion_report(
        preflight_result={"phase": "phase_9", "verdict": "PASS", "failure_count": 0, "warning_count": 1},
        release_manifest={
            "phase": "phase_10",
            "release_status": "candidate_ready",
            "candidate_image": "ghcr.io/example/image:sha-good",
            "blockers": [],
        },
        verification={"verdict": "PASS"},
    )

    assert report["local_status"] == "ready_for_candidate_build"
    assert report["blockers"] == []
