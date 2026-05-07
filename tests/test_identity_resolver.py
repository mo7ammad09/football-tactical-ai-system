from src.identity.resolver import (
    LOCK_DISPLAY_ROLE_ACTION,
    LOCK_DISPLAY_TEAM_ACTION,
    MARK_SEGMENT_SPLIT_REQUIRED_ACTION,
    MERGE_GOALKEEPER_ACTION,
    READY_STATUS,
    build_identity_resolution_plan,
    validate_identity_resolution_plan,
)


def _review_decisions(*, vision_model_invoked: bool = False):
    return {
        "phase": "phase_8_identity_review_engine",
        "vision_model_invoked": vision_model_invoked,
        "render_safe": True,
        "summary": {"goalkeeper_display_track_ids": [21, 29, 31]},
        "decisions": [
            {
                "decision_id": "unresolved_candidate_goalkeeper_identity_fragmentation",
                "decision_type": "goalkeeper_identity_fragmentation",
                "status": "needs_vision_or_llm",
                "evidence": {
                    "case_id": "candidate_goalkeeper_identity_fragmentation",
                    "priority": "medium",
                },
            }
        ],
    }


def _queue():
    return {
        "case_count": 1,
        "cases": [
            {
                "case_id": "candidate_goalkeeper_identity_fragmentation",
                "question": "goalkeeper_identity_fragmentation",
                "priority": "medium",
            }
        ],
    }


def _crop_index():
    return {
        "case_count": 1,
        "cases": [
            {
                "case_id": "candidate_goalkeeper_identity_fragmentation",
                "contact_sheet_path": "/tmp/candidate_goalkeeper_identity_fragmentation.jpg",
                "crop_request_count": 3,
                "crop_requests": [
                    {"track_id": 21, "raw_track_id": 114, "source_frame_idx": 275},
                    {"track_id": 29, "raw_track_id": 114, "source_frame_idx": 485},
                    {"track_id": 31, "raw_track_id": 131, "source_frame_idx": 778},
                ],
            }
        ],
    }


def test_identity_resolver_defers_uninvoked_goalkeeper_fragmentation():
    plan = build_identity_resolution_plan(
        identity_review_decisions=_review_decisions(),
        vision_review_queue=_queue(),
        vision_review_results={
            "vision_model_invoked": False,
            "results": [
                {
                    "case_id": "candidate_goalkeeper_identity_fragmentation",
                    "question": "goalkeeper_identity_fragmentation",
                    "status": "not_invoked",
                    "verdict": "unresolved",
                    "confidence": 0.0,
                }
            ],
        },
        player_crop_index=_crop_index(),
        render_audit_after={"verdict": "PASS", "goalkeeper_display_segments": []},
        final_render_identity_manifest={"release_status": "review_required"},
    )

    assert plan["phase"] == "phase_9_identity_review_resolver"
    assert plan["recommendation"] == "keep_review_required_until_evidence"
    assert plan["validation"]["verdict"] == "PASS"
    assert plan["summary"]["ready_for_safe_apply_count"] == 0
    proposal = plan["resolution_proposals"][0]
    assert proposal["status"] == "deferred_needs_evidence"
    assert proposal["proposed_action"] == "keep_unresolved_no_identity_mutation"
    assert proposal["target_track_ids"] == [21, 29, 31]


def test_identity_resolver_proposes_goalkeeper_merge_after_positive_vision():
    plan = build_identity_resolution_plan(
        identity_review_decisions=_review_decisions(vision_model_invoked=True),
        vision_review_queue=_queue(),
        vision_review_results={
            "vision_model_invoked": True,
            "results": [
                {
                    "case_id": "candidate_goalkeeper_identity_fragmentation",
                    "question": "goalkeeper_identity_fragmentation",
                    "status": "reviewed",
                    "verdict": "same_player",
                    "confidence": 0.91,
                    "evidence": [
                        {"track_id": 21, "raw_track_id": 114, "source_frame_idx": 275},
                        {"track_id": 29, "raw_track_id": 114, "source_frame_idx": 485},
                        {"track_id": 31, "raw_track_id": 131, "source_frame_idx": 778},
                    ],
                    "model_evidence": [
                        {
                            "claim": "same goalkeeper kit and body shape across crops",
                            "tracks": [21, 29, 31],
                        }
                    ],
                }
            ],
        },
        player_crop_index=_crop_index(),
        render_audit_after={"verdict": "PASS"},
        final_render_identity_manifest={"release_status": "review_required"},
    )

    assert plan["recommendation"] == "safe_apply_available"
    assert plan["validation"]["verdict"] == "PASS"
    proposal = plan["resolution_proposals"][0]
    assert proposal["status"] == READY_STATUS
    assert proposal["proposed_action"] == MERGE_GOALKEEPER_ACTION
    assert proposal["confidence"] == 0.91
    assert proposal["target_track_ids"] == [21, 29, 31]


def test_identity_resolver_reads_resolved_vision_case_without_unresolved_decision():
    decisions = _review_decisions(vision_model_invoked=True)
    decisions["decisions"] = []

    plan = build_identity_resolution_plan(
        identity_review_decisions=decisions,
        vision_review_queue=_queue(),
        vision_review_results={
            "vision_model_invoked": True,
            "results": [
                {
                    "case_id": "candidate_goalkeeper_identity_fragmentation",
                    "question": "goalkeeper_identity_fragmentation",
                    "status": "reviewed",
                    "verdict": "same_player",
                    "confidence": 0.9,
                    "evidence": [
                        {"track_id": 21, "raw_track_id": 114, "source_frame_idx": 275},
                        {"track_id": 29, "raw_track_id": 114, "source_frame_idx": 485},
                    ],
                    "model_evidence": [{"claim": "same player"}],
                }
            ],
        },
        player_crop_index=_crop_index(),
        render_audit_after={"verdict": "PASS"},
        final_render_identity_manifest={"release_status": "identity_trusted"},
    )

    assert plan["summary"]["ready_for_safe_apply_count"] == 1
    assert plan["resolution_proposals"][0]["proposed_action"] == MERGE_GOALKEEPER_ACTION


def test_identity_resolver_defers_candidate_links_without_model_evidence():
    decisions = {
        "phase": "phase_8_identity_review_engine",
        "vision_model_invoked": False,
        "render_safe": True,
        "summary": {},
        "decisions": [
            {
                "decision_id": "candidate_link_35_14",
                "decision_type": "candidate_identity_link",
                "status": "needs_text_or_visual_review",
                "evidence": {
                    "source_id": 35,
                    "target_id": 14,
                    "reid_distance": 0.242,
                    "auto_reid_threshold": 0.12,
                    "roles": ["player", "player"],
                    "teams": [1, 1],
                },
            }
        ],
    }

    plan = build_identity_resolution_plan(
        identity_review_decisions=decisions,
        vision_review_queue={"case_count": 0, "cases": []},
        vision_review_results={"vision_model_invoked": False, "results": []},
    )

    assert plan["recommendation"] == "keep_review_required_until_evidence"
    assert plan["summary"]["candidate_link_proposal_count"] == 1
    proposal = plan["resolution_proposals"][0]
    assert proposal["status"] == "deferred_needs_evidence"
    assert proposal["target_track_ids"] == [14, 35]


def test_identity_resolution_validator_rejects_unsafe_ready_merge():
    validation = validate_identity_resolution_plan(
        {
            "render_safe": True,
            "resolution_proposals": [
                {
                    "proposal_id": "resolve_gk",
                    "status": READY_STATUS,
                    "proposed_action": MERGE_GOALKEEPER_ACTION,
                    "confidence": 0.92,
                    "target_track_ids": [21, 29],
                    "evidence": {
                        "vision_status": "reviewed",
                        "vision_verdict": "same_player",
                        "model_evidence_count": 0,
                    },
                    "apply_policy": {
                        "dry_run_only": True,
                        "requires_safe_apply_validator": True,
                    },
                }
            ],
        }
    )

    assert validation["verdict"] == "FAIL"
    assert validation["rejected_proposal_count"] == 1
    assert "goalkeeper_merge_requires_model_evidence" in validation[
        "rejected_proposals"
    ][0]["reasons"]


def test_identity_resolver_proposes_safe_display_team_lock():
    decisions = {"render_safe": False, "decisions": []}
    queue = {
        "cases": [
            {
                "case_id": "review_display_team_flicker_16",
                "question": "team_assignment_uncertain",
                "audit_evidence": {
                    "issue_type": "display_team_flicker",
                    "track_id": 16,
                    "player_team_counts": {"2": 415, "1": 23},
                    "player_team_confidence": 0.947,
                    "max_minor_player_team_segment_ratio": 0.052,
                    "max_minor_player_team_segment_frames": 23,
                    "raw_role_confidence": 1.0,
                },
            }
        ]
    }
    results = {
        "vision_model_invoked": True,
        "results": [
            {
                "case_id": "review_display_team_flicker_16",
                "question": "team_assignment_uncertain",
                "status": "reviewed",
                "verdict": "team_2",
                "confidence": 0.95,
                "evidence": [{"track_id": 16, "source_frame_idx": 210}],
                "model_evidence": [{"claim": "light kit throughout"}],
            }
        ],
    }

    plan = build_identity_resolution_plan(
        identity_review_decisions=decisions,
        vision_review_queue=queue,
        vision_review_results=results,
    )

    assert plan["validation"]["verdict"] == "PASS"
    assert plan["summary"]["ready_for_safe_apply_count"] == 1
    proposal = plan["resolution_proposals"][0]
    assert proposal["status"] == READY_STATUS
    assert proposal["proposed_action"] == LOCK_DISPLAY_TEAM_ACTION
    assert proposal["target_track_ids"] == [16]
    assert proposal["evidence"]["target_display_team"] == 2


def test_identity_resolver_defers_team_lock_when_model_detects_track_switch():
    decisions = {"render_safe": False, "decisions": []}
    queue = {
        "cases": [
            {
                "case_id": "review_display_team_flicker_7",
                "question": "team_assignment_uncertain",
                "audit_evidence": {
                    "issue_type": "display_team_flicker",
                    "track_id": 7,
                    "player_team_counts": {"2": 252, "1": 136},
                    "player_team_confidence": 0.649,
                    "max_minor_player_team_segment_ratio": 0.35,
                    "max_minor_player_team_segment_frames": 136,
                    "raw_role_confidence": 1.0,
                },
            }
        ]
    }
    results = {
        "vision_model_invoked": True,
        "results": [
            {
                "case_id": "review_display_team_flicker_7",
                "question": "team_assignment_uncertain",
                "status": "reviewed",
                "verdict": "unresolved",
                "confidence": 1.0,
                "reason": "The track contains a track switch between different players.",
                "evidence": [{"track_id": 7, "source_frame_idx": 1240}],
                "model_evidence": [{"claim": "track switch"}],
            }
        ],
    }

    plan = build_identity_resolution_plan(
        identity_review_decisions=decisions,
        vision_review_queue=queue,
        vision_review_results=results,
    )

    proposal = plan["resolution_proposals"][0]
    assert proposal["status"] == "deferred_needs_evidence"
    assert proposal["proposal_type"] == "segment_split_required"
    assert proposal["proposed_action"] == MARK_SEGMENT_SPLIT_REQUIRED_ACTION
    assert plan["summary"]["segment_split_required_count"] == 1


def test_identity_resolver_proposes_safe_display_role_lock():
    decisions = {"render_safe": False, "decisions": []}
    queue = {
        "cases": [
            {
                "case_id": "review_display_role_flicker_24",
                "question": "role_stability_flicker",
                "audit_evidence": {
                    "issue_type": "display_role_flicker",
                    "track_id": 24,
                    "dominant_raw_role": "player",
                    "raw_role_confidence": 0.996,
                    "max_minor_raw_role_segment_frames": 1,
                },
            }
        ]
    }
    results = {
        "vision_model_invoked": True,
        "results": [
            {
                "case_id": "review_display_role_flicker_24",
                "question": "role_stability_flicker",
                "status": "reviewed",
                "verdict": "player",
                "confidence": 1.0,
                "evidence": [{"track_id": 24, "source_frame_idx": 140}],
                "model_evidence": [{"claim": "isolated referee flicker"}],
            }
        ],
    }

    plan = build_identity_resolution_plan(
        identity_review_decisions=decisions,
        vision_review_queue=queue,
        vision_review_results=results,
    )

    proposal = plan["resolution_proposals"][0]
    assert proposal["status"] == READY_STATUS
    assert proposal["proposed_action"] == LOCK_DISPLAY_ROLE_ACTION
    assert proposal["evidence"]["target_display_role"] == "player"
