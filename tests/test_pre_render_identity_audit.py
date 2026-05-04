from src.identity.pre_render_audit import (
    KNOWN_BAD_RUNPOD_IMAGES,
    RUNPOD_BASELINE_IMAGE,
    apply_safe_correction_plan_to_raw_records,
    build_correction_candidates,
    build_dry_run_correction_plan,
    build_final_render_identity_manifest,
    build_identity_events,
    build_player_crop_index_plan,
    build_render_identity_audit,
    build_vision_review_results,
    build_vision_review_queue,
    post_fix_audit_improved,
    validate_correction_plan,
    validate_final_render_identity_manifest,
    validate_vision_review_results,
)


def _row(
    track_id,
    raw_track_id,
    sample,
    frame,
    role,
    team,
    *,
    display_label=None,
    display_role=None,
    display_team=None,
    display_color=None,
):
    team_color = [27, 35, 31] if team == 1 else [255, 0, 255] if team == 0 else None
    return {
        "sample_number": sample,
        "source_frame_idx": frame,
        "object_type": "player",
        "track_id": track_id,
        "raw_track_id": raw_track_id,
        "role": role,
        "detected_role": role,
        "team": team,
        "display_label": display_label,
        "display_role": display_role,
        "display_team": display_team,
        "display_color": display_color,
        "team_color": team_color,
        "bbox": [0, 0, 10, 20],
        "reid_available": True,
        "reid_dim": 512,
    }


def test_render_audit_flags_bad_gk_display_spread_to_player_tracks():
    rows = []
    rows.extend(
        _row(14, 36, sample, sample * 10, "player", 1)
        for sample in range(1, 7)
    )
    rows.extend(
        _row(
            14,
            36,
            sample,
            sample * 10,
            "player",
            1,
            display_label="GK",
            display_role="goalkeeper",
            display_team=0,
            display_color=[255, 0, 255],
        )
        for sample in range(7, 10)
    )
    rows.extend(
        _row(
            29,
            130,
            sample,
            sample * 10,
            "goalkeeper",
            0,
            display_label="GK",
            display_role="goalkeeper",
            display_team=0,
            display_color=[255, 0, 255],
        )
        for sample in range(10, 14)
    )

    audit = build_render_identity_audit(rows)

    assert audit["verdict"] == "FAIL"
    assert audit["baseline_image"] == RUNPOD_BASELINE_IMAGE
    issue_types = {issue["issue_type"] for issue in audit["issues"]}
    assert "gk_false_positive_segment" in issue_types
    assert "unsafe_gk_display_spread" in issue_types
    assert audit["summary"]["gk_false_positive_segment_count"] == 1


def test_render_audit_passes_single_goalkeeper_display_on_goalkeeper_track():
    rows = []
    rows.extend(
        _row(sample, sample, sample, sample * 10, "player", 1)
        for sample in range(1, 4)
    )
    rows.extend(
        _row(
            29,
            130,
            sample,
            sample * 10,
            "goalkeeper",
            0,
            display_label="GK",
            display_role="goalkeeper",
            display_team=0,
            display_color=[255, 0, 255],
        )
        for sample in range(4, 10)
    )

    audit = build_render_identity_audit(rows)

    assert audit["verdict"] == "PASS"
    assert audit["issues"] == []


def test_render_audit_downgrades_raw_role_flicker_when_display_is_clean():
    rows = [
        _row(
            29,
            130,
            1,
            10,
            "goalkeeper",
            0,
            display_label="GK",
            display_role="goalkeeper",
            display_team=0,
            display_color=[255, 0, 255],
        )
    ]
    identity_debug = {
        "role_stability": {
            "role_flicker_tracklet_count": 2,
        }
    }

    audit = build_render_identity_audit(rows, identity_debug)

    assert audit["verdict"] == "PASS"
    assert audit["issues"][0]["issue_type"] == "identity_debug_role_flicker"
    assert audit["issues"][0]["severity"] == "medium"
    assert audit["issues"][0]["advisory"] is True


def test_pre_render_artifacts_create_events_and_dry_run_candidates():
    rows = []
    rows.extend(
        _row(14, 36, sample, sample * 10, "player", 1)
        for sample in range(1, 5)
    )
    rows.append(
        _row(
            14,
            36,
            5,
            50,
            "player",
            1,
            display_label="GK",
            display_role="goalkeeper",
            display_team=0,
            display_color=[255, 0, 255],
        )
    )

    audit = build_render_identity_audit(rows)
    events = build_identity_events(rows, audit)
    candidates = build_correction_candidates(audit, {"warnings": []})

    assert events["event_count"] >= 1
    assert candidates["candidate_count"] >= 1
    assert candidates["candidates"][0]["candidate_type"] == "display_override"


def test_dry_run_correction_plan_is_validated_and_never_applied():
    rows = []
    rows.extend(
        _row(14, 36, sample, sample * 10, "player", 1)
        for sample in range(1, 5)
    )
    rows.append(
        _row(
            14,
            36,
            5,
            50,
            "player",
            1,
            display_label="GK",
            display_role="goalkeeper",
            display_team=0,
            display_color=[255, 0, 255],
        )
    )

    audit = build_render_identity_audit(rows)
    events = build_identity_events(rows, audit)
    candidates = build_correction_candidates(audit, {"warnings": []})
    plan = build_dry_run_correction_plan(
        correction_candidates=candidates,
        render_audit=audit,
        identity_events=events,
    )

    assert plan["phase"] == "phase_2_dry_run"
    assert plan["correction_applied"] is False
    assert plan["summary"]["safe_fix_count"] == 1
    assert plan["validation"]["verdict"] == "PASS"
    assert plan["actions"][0]["set_display_role"] == "player"
    assert plan["actions"][0]["set_display_label"] == "14"


def test_phase3_applies_safe_display_override_and_improves_post_fix_audit():
    rows = []
    rows.extend(
        _row(14, 36, sample, sample * 10, "player", 1)
        for sample in range(1, 5)
    )
    rows.append(
        _row(
            14,
            36,
            5,
            50,
            "player",
            1,
            display_label="GK",
            display_role="goalkeeper",
            display_team=0,
            display_color=[255, 0, 255],
        )
    )

    before = build_render_identity_audit(rows)
    events = build_identity_events(rows, before)
    candidates = build_correction_candidates(before, {"warnings": []})
    plan = build_dry_run_correction_plan(
        correction_candidates=candidates,
        render_audit=before,
        identity_events=events,
    )

    corrected_rows, applied = apply_safe_correction_plan_to_raw_records(rows, plan)
    after = build_render_identity_audit(corrected_rows)

    assert applied["correction_applied"] is True
    assert applied["applied_action_count"] == 1
    assert applied["updated_record_count"] == 1
    assert corrected_rows[-1]["display_role"] == "player"
    assert corrected_rows[-1]["display_label"] == "14"
    assert corrected_rows[-1]["display_color"] == [27, 35, 31]
    assert after["summary"]["gk_false_positive_segment_count"] == 0
    assert post_fix_audit_improved(before, after) is True


def test_phase3_restores_dominant_team_color_for_misclassified_gk_flash():
    rows = [
        _row(14, 14, 1, 93, "player", 1),
        _row(14, 36, 2, 100, "player", 1),
        _row(14, 36, 3, 103, "player", 1),
        _row(14, 36, 4, 105, "player", 1),
        _row(14, 36, 5, 108, "player", 1),
        _row(
            14,
            138,
            6,
            705,
            "goalkeeper",
            0,
            display_label="GK",
            display_role="goalkeeper",
            display_team=0,
            display_color=[255, 0, 255],
        ),
        _row(
            14,
            138,
            7,
            708,
            "goalkeeper",
            0,
            display_role="player",
            display_color=None,
        ),
    ]

    before = build_render_identity_audit(rows)
    events = build_identity_events(rows, before)
    candidates = build_correction_candidates(before, {"warnings": []})
    plan = build_dry_run_correction_plan(
        correction_candidates=candidates,
        render_audit=before,
        identity_events=events,
    )

    corrected_rows, applied = apply_safe_correction_plan_to_raw_records(rows, plan)
    after = build_render_identity_audit(corrected_rows)

    corrected_flash = corrected_rows[5]
    suppressed_flash = corrected_rows[6]
    assert corrected_flash["display_role"] == "player"
    assert corrected_flash["display_label"] == "14"
    assert corrected_flash["display_team"] == 1
    assert corrected_flash["display_color"] == [27, 35, 31]
    assert suppressed_flash["display_color"] == [27, 35, 31]
    assert applied["sanitized_display_color_count"] == 1
    assert after["summary"]["gk_false_positive_segment_count"] == 0


def test_phase3_does_not_apply_rejected_plan():
    rows = [
        _row(
            14,
            36,
            5,
            50,
            "player",
            1,
            display_label="GK",
            display_role="goalkeeper",
            display_team=0,
            display_color=[255, 0, 255],
        )
    ]
    plan = {
        "validation": {
            "verdict": "FAIL",
            "accepted_action_ids": [],
        },
        "actions": [
            {
                "action_id": "unsafe",
                "action_type": "display_override",
                "status": "safe_fix_dry_run",
                "track_id": 14,
                "first_source_frame_idx": 50,
                "last_source_frame_idx": 50,
                "set_display_role": "player",
                "set_display_label": "14",
                "set_display_color_policy": "team",
                "evidence_ids": ["gk_false_positive_14_50_50"],
            }
        ],
    }

    corrected_rows, applied = apply_safe_correction_plan_to_raw_records(rows, plan)

    assert applied["correction_applied"] is False
    assert applied["skipped_reason"] == "plan_validation_not_pass"
    assert corrected_rows[0]["display_label"] == "GK"


def test_phase4_queues_goalkeeper_fragmentation_for_vision_not_auto_apply():
    rows = [
        _row(
            29,
            130,
            1,
            10,
            "goalkeeper",
            0,
            display_label="GK",
            display_role="goalkeeper",
            display_team=0,
            display_color=[255, 0, 255],
        )
    ]
    identity_debug = {
        "warnings": [
            {
                "code": "goalkeeper_identity_fragmentation",
                "display_id_count": 3,
            }
        ]
    }
    audit = build_render_identity_audit(rows, identity_debug)
    events = build_identity_events(rows, audit)
    candidates = build_correction_candidates(audit, identity_debug)
    plan = build_dry_run_correction_plan(
        correction_candidates=candidates,
        render_audit=audit,
        identity_events=events,
    )
    queue = build_vision_review_queue(
        correction_plan=plan,
        render_audit_before=audit,
        render_audit_after=audit,
        correction_applied={"correction_applied": False},
    )

    assert plan["summary"]["safe_fix_count"] == 0
    assert plan["summary"]["needs_vision_count"] == 1
    assert queue["phase"] == "phase_4_vision_on_demand_queue"
    assert queue["vision_model_invoked"] is False
    assert queue["case_count"] == 1
    assert queue["cases"][0]["question"] == "goalkeeper_identity_fragmentation"
    assert "player_crop_index.json" in queue["cases"][0]["required_artifacts"]
    assert "contact_sheets" in queue["cases"][0]["required_artifacts"]


def test_phase4_skips_vision_case_when_source_issue_was_fixed():
    plan = {
        "needs_vision": [
            {
                "case_id": "candidate_unsafe_gk_display_spread",
                "source_issue_id": "unsafe_gk_display_spread",
                "question": "resolve_identity_or_goalkeeper_display_conflict",
                "reason": "Fixed by deterministic display override.",
            }
        ]
    }
    before_audit = {
        "issues": [
            {
                "issue_id": "unsafe_gk_display_spread",
                "issue_type": "unsafe_gk_display_spread",
                "severity": "critical",
            }
        ]
    }
    after_audit = {"issues": []}

    queue = build_vision_review_queue(
        correction_plan=plan,
        render_audit_before=before_audit,
        render_audit_after=after_audit,
        correction_applied={"correction_applied": True},
    )

    assert queue["case_count"] == 0
    assert queue["cases"] == []


def test_phase5_crop_index_plan_targets_queued_goalkeeper_segments():
    rows = [
        _row(
            29,
            130,
            1,
            10,
            "goalkeeper",
            0,
            display_label="GK",
            display_role="goalkeeper",
            display_team=0,
            display_color=[255, 0, 255],
        ),
        _row(
            31,
            130,
            2,
            20,
            "goalkeeper",
            0,
            display_label="GK",
            display_role="goalkeeper",
            display_team=0,
            display_color=[255, 0, 255],
        ),
    ]
    identity_debug = {
        "warnings": [{"code": "goalkeeper_identity_fragmentation"}]
    }
    audit = build_render_identity_audit(rows, identity_debug)
    events = build_identity_events(rows, audit)
    candidates = build_correction_candidates(audit, identity_debug)
    plan = build_dry_run_correction_plan(
        correction_candidates=candidates,
        render_audit=audit,
        identity_events=events,
    )
    queue = build_vision_review_queue(
        correction_plan=plan,
        render_audit_before=audit,
        render_audit_after=audit,
        correction_applied={"correction_applied": False},
    )

    crop_plan = build_player_crop_index_plan(
        raw_tracklet_rows=rows,
        vision_review_queue=queue,
        render_audit=audit,
    )

    assert crop_plan["phase"] == "phase_5_crop_evidence_plan"
    assert crop_plan["case_count"] == 1
    assert crop_plan["total_crop_request_count"] == 2
    requests = crop_plan["cases"][0]["crop_requests"]
    assert {request["track_id"] for request in requests} == {29, 31}
    assert all(request["crop_path"] is None for request in requests)


def test_phase6_keeps_cases_unresolved_when_vision_provider_is_not_enabled():
    rows = [
        _row(
            29,
            130,
            1,
            10,
            "goalkeeper",
            0,
            display_label="GK",
            display_role="goalkeeper",
            display_team=0,
            display_color=[255, 0, 255],
        ),
        _row(
            31,
            130,
            2,
            20,
            "goalkeeper",
            0,
            display_label="GK",
            display_role="goalkeeper",
            display_team=0,
            display_color=[255, 0, 255],
        ),
    ]
    identity_debug = {
        "warnings": [{"code": "goalkeeper_identity_fragmentation"}]
    }
    audit = build_render_identity_audit(rows, identity_debug)
    events = build_identity_events(rows, audit)
    candidates = build_correction_candidates(audit, identity_debug)
    plan = build_dry_run_correction_plan(
        correction_candidates=candidates,
        render_audit=audit,
        identity_events=events,
    )
    queue = build_vision_review_queue(
        correction_plan=plan,
        render_audit_before=audit,
        render_audit_after=audit,
        correction_applied={"correction_applied": False},
    )
    crop_plan = build_player_crop_index_plan(
        raw_tracklet_rows=rows,
        vision_review_queue=queue,
        render_audit=audit,
    )

    results = build_vision_review_results(
        vision_review_queue=queue,
        player_crop_index=crop_plan,
    )

    assert results["phase"] == "phase_6_vision_review_results"
    assert results["vision_model_invoked"] is False
    assert results["case_count"] == 1
    assert results["unresolved_count"] == 1
    assert results["validation"]["verdict"] == "PASS"
    result = results["results"][0]
    assert result["status"] == "not_invoked"
    assert result["verdict"] == "unresolved"
    assert result["confidence"] == 0.0
    assert result["recommended_action"] == "keep_unresolved_no_identity_mutation"
    assert result["crop_count"] == 2


def test_phase6_validator_rejects_visual_verdict_when_vision_was_not_invoked():
    queue = {
        "cases": [
            {
                "case_id": "case_1",
                "question": "goalkeeper_identity_fragmentation",
            }
        ]
    }
    results = {
        "vision_model_invoked": False,
        "results": [
            {
                "case_id": "case_1",
                "status": "reviewed",
                "verdict": "same_player",
                "confidence": 0.92,
                "evidence": [],
                "model_evidence": ["claimed_crop_similarity"],
            }
        ],
    }

    validation = validate_vision_review_results(
        results,
        vision_review_queue=queue,
    )

    assert validation["verdict"] == "FAIL"
    reasons = validation["rejected_results"][0]["reasons"]
    assert "not_invoked_requires_not_invoked_status" in reasons
    assert "not_invoked_must_stay_unresolved" in reasons
    assert "not_invoked_confidence_must_be_zero" in reasons


def test_phase7_manifest_is_trusted_when_audit_passes_and_no_cases_remain():
    audit = build_render_identity_audit(
        [
            _row(
                29,
                130,
                1,
                10,
                "goalkeeper",
                0,
                display_label="GK",
                display_role="goalkeeper",
                display_team=0,
                display_color=[255, 0, 255],
            )
        ]
    )
    queue = {"case_count": 0, "cases": []}
    vision_results = {
        "vision_model_invoked": False,
        "case_count": 0,
        "unresolved_count": 0,
        "results": [],
        "validation": {"verdict": "PASS"},
    }

    manifest = build_final_render_identity_manifest(
        render_audit_after=audit,
        correction_applied={"correction_applied": False, "kept": False},
        vision_review_queue=queue,
        vision_review_results=vision_results,
        rendered_output_frames=12,
    )

    assert manifest["phase"] == "phase_7_final_render_integration"
    assert manifest["release_status"] == "identity_trusted"
    assert manifest["output_identity_mode"] == "final_identity_output"
    assert manifest["render_policy"] == "produce_standard_final_video"
    assert manifest["validation"]["verdict"] == "PASS"


def test_phase7_manifest_marks_review_required_for_unresolved_vision_cases():
    audit = build_render_identity_audit(
        [
            _row(
                29,
                130,
                1,
                10,
                "goalkeeper",
                0,
                display_label="GK",
                display_role="goalkeeper",
                display_team=0,
                display_color=[255, 0, 255],
            )
        ]
    )
    queue = {
        "case_count": 1,
        "cases": [{"case_id": "case_1", "question": "goalkeeper_identity_fragmentation"}],
    }
    vision_results = {
        "vision_model_invoked": False,
        "case_count": 1,
        "unresolved_count": 1,
        "results": [
            {
                "case_id": "case_1",
                "question": "goalkeeper_identity_fragmentation",
                "priority": "high",
                "status": "not_invoked",
                "verdict": "unresolved",
                "confidence": 0.0,
                "reason": "Vision provider is not configured/enabled.",
                "evidence": [],
            }
        ],
        "validation": {"verdict": "PASS"},
    }

    manifest = build_final_render_identity_manifest(
        render_audit_after=audit,
        correction_applied={"correction_applied": False, "kept": False},
        vision_review_queue=queue,
        vision_review_results=vision_results,
        rendered_output_frames=12,
    )

    assert manifest["final_video_produced"] is True
    assert manifest["release_status"] == "review_required"
    assert manifest["output_identity_mode"] == "review_output_with_identity_artifacts"
    assert manifest["render_policy"] == "produce_video_with_identity_review_artifacts"
    assert manifest["vision_review"]["unresolved_case_count"] == 1
    assert "vision_review_cases_unresolved" in manifest["review_reasons"]
    assert manifest["validation"]["verdict"] == "PASS"


def test_phase7_validator_rejects_trusted_manifest_with_unresolved_cases():
    manifest = {
        "final_video_produced": True,
        "release_status": "identity_trusted",
        "output_identity_mode": "final_identity_output",
        "render_policy": "produce_standard_final_video",
        "blockers": [],
        "review_reasons": [],
        "vision_review": {"unresolved_case_count": 1},
        "audit": {"high_risk_issue_count": 0},
    }

    validation = validate_final_render_identity_manifest(manifest)

    assert validation["verdict"] == "FAIL"
    assert "trusted_output_cannot_have_unresolved_cases" in validation["reasons"]


def test_phase4_queues_rollback_for_review_when_post_fix_not_kept():
    plan = {"needs_vision": []}
    audit = build_render_identity_audit([])
    queue = build_vision_review_queue(
        correction_plan=plan,
        render_audit_before=audit,
        render_audit_after=audit,
        correction_applied={
            "candidate_correction_applied": True,
            "correction_applied": False,
            "rollback_reason": "post_fix_audit_not_improved",
        },
    )

    assert queue["case_count"] == 1
    assert queue["cases"][0]["case_id"] == "rollback_post_fix_audit_not_improved"
    assert queue["cases"][0]["priority"] == "high"


def test_validator_rejects_unsafe_non_goalkeeper_gk_display_override():
    audit = {
        "issues": [
            {
                "issue_id": "gk_false_positive_14_50_50",
                "issue_type": "gk_false_positive_segment",
                "severity": "critical",
            }
        ]
    }
    events = {"events": []}
    plan = {
        "actions": [
            {
                "action_id": "unsafe_keep_gk",
                "action_type": "display_override",
                "confidence": 0.9,
                "track_id": 14,
                "first_source_frame_idx": 50,
                "last_source_frame_idx": 50,
                "set_display_role": "player",
                "set_display_label": "GK",
                "set_display_color_policy": "team",
                "evidence_ids": ["gk_false_positive_14_50_50"],
            }
        ]
    }

    validation = validate_correction_plan(plan, render_audit=audit, identity_events=events)

    assert validation["verdict"] == "FAIL"
    assert validation["rejected_actions"][0]["reasons"] == ["non_goalkeeper_keeps_gk_label"]


def test_bad_runpod_image_is_registered_as_regression_not_baseline():
    bad_images = {item["image"] for item in KNOWN_BAD_RUNPOD_IMAGES}

    assert RUNPOD_BASELINE_IMAGE.endswith(":sha-bbe8dec")
    assert "ghcr.io/mo7ammad09/football-tactical-ai-runpod:sha-5f969e6" in bad_images
