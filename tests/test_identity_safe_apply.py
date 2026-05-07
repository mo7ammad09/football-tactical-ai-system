from src.identity.resolver import (
    LOCK_DISPLAY_ROLE_ACTION,
    LOCK_DISPLAY_TEAM_ACTION,
    MERGE_GOALKEEPER_ACTION,
    READY_STATUS,
)
from src.identity.safe_apply import (
    apply_identity_resolution_plan_to_annotation_states,
    apply_identity_resolution_plan_to_raw_records,
    validate_identity_safe_apply_plan,
)


def _ready_gk_plan():
    return {
        "schema_version": "1.0",
        "phase": "phase_9_identity_review_resolver",
        "render_safe": True,
        "validation": {"verdict": "PASS"},
        "resolution_proposals": [
            {
                "proposal_id": "resolve_candidate_goalkeeper_identity_fragmentation",
                "case_id": "candidate_goalkeeper_identity_fragmentation",
                "status": READY_STATUS,
                "proposed_action": MERGE_GOALKEEPER_ACTION,
                "confidence": 0.91,
                "target_track_ids": [21, 29, 31],
                "evidence": {
                    "vision_status": "reviewed",
                    "vision_verdict": "same_player",
                    "model_evidence_count": 1,
                },
                "apply_policy": {
                    "dry_run_only": True,
                    "requires_safe_apply_validator": True,
                },
            }
        ],
    }


def test_phase10_noops_without_ready_resolution_proposals():
    rows, applied = apply_identity_resolution_plan_to_raw_records(
        [
            {
                "track_id": 21,
                "display_role": "goalkeeper",
                "display_label": "GK",
            }
        ],
        {
            "validation": {"verdict": "PASS"},
            "resolution_proposals": [
                {
                    "proposal_id": "resolve_gk",
                    "status": "deferred_needs_evidence",
                    "proposed_action": "keep_unresolved_no_identity_mutation",
                    "target_track_ids": [21, 29, 31],
                    "apply_policy": {
                        "dry_run_only": True,
                        "requires_safe_apply_validator": True,
                    },
                }
            ],
        },
    )

    assert applied["phase"] == "phase_10_identity_safe_apply"
    assert applied["safe_apply_status"] == "no_ready_proposals"
    assert applied["updated_record_count"] == 0
    assert rows[0].get("resolved_identity_id") is None


def test_phase10_applies_goalkeeper_resolution_metadata_only_to_gk_rows():
    raw_rows = [
        {
            "track_id": 21,
            "display_role": "goalkeeper",
            "display_label": "GK",
            "role": "goalkeeper",
        },
        {
            "track_id": 21,
            "display_role": "player",
            "display_label": "21",
            "role": "player",
        },
        {
            "track_id": 14,
            "display_role": "player",
            "display_label": "14",
            "role": "player",
        },
        {
            "track_id": 31,
            "display_role": "goalkeeper",
            "display_label": "GK",
            "role": "player",
        },
    ]

    rows, applied = apply_identity_resolution_plan_to_raw_records(
        raw_rows,
        _ready_gk_plan(),
    )

    assert applied["safe_apply_status"] == "applied"
    assert applied["applied_proposal_count"] == 1
    assert applied["updated_record_count"] == 2
    assert rows[0]["resolved_identity_id"] == "gk"
    assert rows[0]["resolved_identity_label"] == "GK"
    assert rows[1].get("resolved_identity_id") is None
    assert rows[2].get("resolved_identity_id") is None
    assert rows[3]["identity_resolution_status"] == "phase10_safe_applied"


def test_phase10_applies_goalkeeper_resolution_to_annotation_metadata():
    states = [
        {
            "players": {
                21: {"display_role": "goalkeeper", "display_label": "GK"},
                14: {"display_role": "player", "display_label": "14"},
            }
        },
        {
            "players": {
                31: {"display_role": "goalkeeper", "display_label": "GK"},
            }
        },
    ]

    updated = apply_identity_resolution_plan_to_annotation_states(
        states,
        _ready_gk_plan(),
    )

    assert updated == 2
    assert states[0]["players"][21]["resolved_identity_id"] == "gk"
    assert states[0]["players"][14].get("resolved_identity_id") is None
    assert states[1]["players"][31]["identity_resolution_confidence"] == 0.91


def test_phase10_validator_rejects_invalid_phase9_plan():
    validation = validate_identity_safe_apply_plan(
        {
            "validation": {"verdict": "FAIL"},
            "resolution_proposals": [
                {
                    "proposal_id": "resolve_gk",
                    "status": READY_STATUS,
                    "proposed_action": MERGE_GOALKEEPER_ACTION,
                    "confidence": 0.91,
                    "target_track_ids": [21, 29],
                    "evidence": {
                        "vision_status": "reviewed",
                        "vision_verdict": "same_player",
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
    assert "phase9_resolution_plan_failed_validation" in validation[
        "rejected_proposals"
    ][0]["reasons"]


def test_phase10_applies_display_team_lock_to_raw_and_annotation_rows():
    plan = {
        "validation": {"verdict": "PASS"},
        "resolution_proposals": [
            {
                "proposal_id": "resolve_team_16",
                "case_id": "review_display_team_flicker_16",
                "status": READY_STATUS,
                "proposed_action": LOCK_DISPLAY_TEAM_ACTION,
                "confidence": 0.95,
                "target_track_ids": [16],
                "evidence": {
                    "vision_status": "reviewed",
                    "vision_verdict": "team_2",
                    "target_display_team": 2,
                },
                "apply_policy": {
                    "dry_run_only": True,
                    "requires_safe_apply_validator": True,
                    "display_only": True,
                },
            }
        ],
    }
    rows, applied = apply_identity_resolution_plan_to_raw_records(
        [
            {"track_id": 16, "team": 2, "team_color": [240, 240, 240]},
            {"track_id": 16, "team": 1, "team_color": [20, 20, 20]},
            {"track_id": 7, "team": 1, "team_color": [20, 20, 20]},
        ],
        plan,
    )

    assert applied["safe_apply_status"] == "applied"
    assert applied["updated_record_count"] == 2
    assert rows[0]["display_team"] == 2
    assert rows[1]["display_team"] == 2
    assert rows[1]["display_color"] == [240, 240, 240]
    assert rows[2].get("display_team") is None

    states = [{"players": {16: {"team_color": (240, 240, 240)}}}]
    updated = apply_identity_resolution_plan_to_annotation_states(states, plan)

    assert updated == 1
    assert states[0]["players"][16]["display_team"] == 2


def test_phase10_applies_display_role_lock_to_player_rows():
    plan = {
        "validation": {"verdict": "PASS"},
        "resolution_proposals": [
            {
                "proposal_id": "resolve_role_24",
                "case_id": "review_display_role_flicker_24",
                "status": READY_STATUS,
                "proposed_action": LOCK_DISPLAY_ROLE_ACTION,
                "confidence": 1.0,
                "target_track_ids": [24],
                "evidence": {
                    "vision_status": "reviewed",
                    "vision_verdict": "player",
                    "target_display_role": "player",
                },
                "apply_policy": {
                    "dry_run_only": True,
                    "requires_safe_apply_validator": True,
                    "display_only": True,
                },
            }
        ],
    }
    rows, applied = apply_identity_resolution_plan_to_raw_records(
        [
            {
                "track_id": 24,
                "role": "referee",
                "display_role": "referee",
                "team": 1,
                "display_team": 0,
                "team_color": [10, 10, 10],
            }
        ],
        plan,
    )

    assert applied["safe_apply_status"] == "applied"
    assert rows[0]["display_role"] == "player"
    assert rows[0]["display_team"] == 1
    assert rows[0]["display_color"] == [10, 10, 10]
