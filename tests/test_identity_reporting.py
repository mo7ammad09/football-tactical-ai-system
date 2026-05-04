from src.identity.reporting import build_identity_artifact_links, summarize_pre_render_identity


def test_identity_summary_reports_review_required_and_links_manifest():
    results = {
        "pre_render_identity_correction": {
            "phase": "phase_7_final_render_integration",
            "release_status": "review_required",
            "output_identity_mode": "review_output_with_identity_artifacts",
            "render_policy": "produce_video_with_identity_review_artifacts",
            "render_audit_after_verdict": "REVIEW",
            "render_audit_after_score": 82,
            "vision_review_queue_count": 1,
            "vision_review_unresolved_count": 1,
            "correction_applied": False,
            "vision_model_invoked": False,
            "final_manifest_validation": {"verdict": "PASS"},
        },
        "final_render_identity_manifest_json_url": "https://cdn.example/manifest.json",
        "vision_review_results_json_url": "https://cdn.example/vision_results.json",
    }

    summary = summarize_pre_render_identity(results)

    assert summary["severity"] == "warning"
    assert summary["release_status"] == "review_required"
    assert summary["unresolved_case_count"] == 1
    assert summary["validation_verdict"] == "PASS"
    assert {
        "key": "final_render_identity_manifest_json_url",
        "label": "Final identity manifest",
        "url": "https://cdn.example/manifest.json",
    } in summary["artifact_links"]


def test_identity_artifact_links_read_runpod_artifact_metadata():
    results = {
        "artifacts": {
        "final_render_identity_manifest_json": {
            "public_url": "https://cdn.example/manifest.json"
        },
        "identity_review_decisions_json": {
            "public_url": "https://cdn.example/review.json"
        },
        "vision_contact_sheets_zip": {
            "public_url": "https://cdn.example/sheets.zip"
        },
        }
    }

    links = build_identity_artifact_links(results)

    assert [link["key"] for link in links] == [
        "final_render_identity_manifest_json_url",
        "identity_review_decisions_json_url",
        "vision_contact_sheets_zip_url",
    ]
