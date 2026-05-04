from src.identity.review_engine import build_identity_review_decisions


def test_identity_review_engine_trusts_clean_render_without_pending_cases():
    decisions = build_identity_review_decisions(
        identity_debug={"candidate_links": [], "warnings": []},
        render_audit_after={
            "verdict": "PASS",
            "score": 100,
            "summary": {},
            "issues": [],
            "goalkeeper_display_segments": [
                {"track_id": 29, "raw_track_id": 130},
            ],
        },
        correction_applied={"correction_applied": False},
        vision_review_queue={"case_count": 0, "cases": []},
        vision_review_results={
            "vision_model_invoked": False,
            "results": [],
        },
        final_render_identity_manifest={"release_status": "identity_trusted"},
    )

    assert decisions["phase"] == "phase_8_identity_review_engine"
    assert decisions["recommendation"] == "identity_trusted"
    assert decisions["render_safe"] is True
    assert decisions["final_identity_needs_review"] is False
    assert decisions["summary"]["goalkeeper_display_track_ids"] == [29]


def test_identity_review_engine_keeps_goalkeeper_fragmentation_for_vision_or_llm():
    decisions = build_identity_review_decisions(
        identity_debug={
            "warnings": [
                {
                    "code": "goalkeeper_identity_fragmentation",
                    "display_id_count": 3,
                }
            ],
            "candidate_links": [],
        },
        render_audit_after={
            "verdict": "PASS",
            "score": 92,
            "summary": {},
            "issues": [
                {
                    "issue_type": "identity_debug_role_flicker",
                    "severity": "medium",
                }
            ],
            "goalkeeper_display_segments": [
                {"track_id": 21, "raw_track_id": 114},
                {"track_id": 29, "raw_track_id": 131},
                {"track_id": 31, "raw_track_id": 131},
            ],
        },
        correction_applied={
            "correction_applied": True,
            "applied_action_count": 12,
            "updated_record_count": 29,
        },
        vision_review_queue={
            "case_count": 1,
            "cases": [{"case_id": "candidate_goalkeeper_identity_fragmentation"}],
        },
        vision_review_results={
            "vision_model_invoked": False,
            "results": [
                {
                    "case_id": "candidate_goalkeeper_identity_fragmentation",
                    "question": "goalkeeper_identity_fragmentation",
                    "priority": "medium",
                    "verdict": "unresolved",
                    "reason": "Vision provider is not configured/enabled.",
                }
            ],
        },
        final_render_identity_manifest={"release_status": "review_required"},
        player_crop_index={"case_count": 1},
    )

    assert decisions["recommendation"] == "review_required"
    assert decisions["render_safe"] is True
    assert decisions["final_identity_needs_review"] is True
    assert decisions["summary"]["unresolved_case_count"] == 1
    assert decisions["summary"]["goalkeeper_display_track_ids"] == [21, 29, 31]
    assert any(
        decision["status"] == "needs_vision_or_llm"
        for decision in decisions["decisions"]
    )


def test_identity_review_engine_blocks_high_risk_render():
    decisions = build_identity_review_decisions(
        identity_debug={},
        render_audit_after={
            "verdict": "FAIL",
            "score": 20,
            "summary": {},
            "issues": [
                {
                    "issue_type": "gk_false_positive_segment",
                    "severity": "critical",
                }
            ],
        },
        correction_applied={},
        vision_review_queue={"case_count": 0, "cases": []},
        vision_review_results={"results": []},
        final_render_identity_manifest={"release_status": "review_required"},
    )

    assert decisions["recommendation"] == "render_not_safe"
    assert decisions["render_safe"] is False
    assert decisions["next_step"] == "fix_render_identity_risks"
