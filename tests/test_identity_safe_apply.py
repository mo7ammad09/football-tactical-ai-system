from src.identity.resolver import MERGE_GOALKEEPER_ACTION, READY_STATUS
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
