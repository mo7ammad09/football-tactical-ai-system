import importlib.util
from pathlib import Path


def _load_audit_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "audit_identity_quality.py"
    spec = importlib.util.spec_from_file_location("audit_identity_quality", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _row(track_id, raw_track_id, frame, role, team=1, object_type="player"):
    return {
        "sample_number": frame,
        "source_frame_idx": frame,
        "object_type": object_type,
        "track_id": track_id,
        "raw_track_id": raw_track_id,
        "role": role,
        "detected_role": role,
        "team": team,
        "reid_available": True,
        "reid_dim": 512,
        "bbox": [0, 0, 10, 20],
    }


def test_audit_identity_quality_flags_goalkeeper_fragmentation_and_color_flicker():
    audit_mod = _load_audit_module()
    rows = []
    rows.extend(_row(21, 113, frame, "goalkeeper", team=0) for frame in range(10, 16))
    rows.extend(_row(9, 129, frame, "goalkeeper", team=0) for frame in range(20, 26))
    rows.extend(_row(29, 129, frame, "goalkeeper", team=0) for frame in range(26, 34))
    rows.extend(_row(14, 35, frame, "player", team=1) for frame in range(40, 44))
    rows.extend(_row(14, 35, frame, "goalkeeper", team=0) for frame in range(44, 48))
    rows.extend(_row(14, 35, frame, "player", team=1) for frame in range(48, 52))

    audit = audit_mod.audit_identity_quality(rows, {"summary": {"profiles_with_reid": 4}})

    assert audit["verdict"] == "FAIL"
    codes = {finding["code"] for finding in audit["findings"]}
    assert "goalkeeper_identity_fragmentation" in codes
    assert "role_color_flicker" in codes
    assert "raw_track_display_fragmentation" in codes
    assert [segment["track_id"] for segment in audit["goalkeeper_timeline"][:3]] == [21, 9, 29]


def test_audit_identity_quality_passes_stable_single_goalkeeper():
    audit_mod = _load_audit_module()
    rows = []
    rows.extend(_row(1, 1, frame, "player", team=1) for frame in range(0, 20))
    rows.extend(_row(29, 129, frame, "goalkeeper", team=0) for frame in range(0, 20))
    rows.extend(_row(2, 2, frame, "player", team=2) for frame in range(0, 20))

    audit = audit_mod.audit_identity_quality(rows, {"summary": {"profiles_with_reid": 3}})

    assert audit["verdict"] == "PASS"
    assert audit["findings"] == []


def test_audit_identity_quality_treats_locked_gk_label_as_client_facing_identity():
    audit_mod = _load_audit_module()
    rows = []
    for frame, track_id in enumerate([21, 9, 29, 14, 29, 31], start=10):
        row = _row(track_id, 100 + track_id, frame, "player", team=1)
        row["display_label"] = "GK"
        row["display_role"] = "goalkeeper"
        row["display_team"] = 0
        row["goalkeeper_display_locked"] = True
        rows.append(row)

    audit = audit_mod.audit_identity_quality(rows, {"summary": {"profiles_with_reid": 6}})

    codes = {finding["code"] for finding in audit["findings"]}
    assert "goalkeeper_identity_fragmentation" not in codes
    assert audit["verdict"] == "PASS"
    assert {segment["track_id"] for segment in audit["goalkeeper_timeline"]} == {"GK"}
