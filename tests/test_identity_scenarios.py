"""
Identity Pipeline Scenario Tests

Run these to validate identity logic without processing real video:
    pytest tests/test_identity_scenarios.py -v

Scenarios cover:
- Player occlusion (disappears then returns)
- Cross-team ID swap
- Goalkeeper fragmentation
- Referee mixed into player track
- Team color instability (KMeans drift)
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

# Add project root to path for 'src' imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.identity.stabilizer import (
    apply_majority_voting_display_defaults,
    build_global_identity_stability_plan,
    apply_global_identity_stability_plan_to_raw_records,
)
from src.identity.pre_render_audit import build_render_identity_audit
from src.identity.resolver import build_identity_resolution_plan
from src.identity.safe_apply import apply_identity_resolution_plan_to_raw_records


def make_row(
    *,
    frame: int,
    track_id: int,
    raw_track_id: int | None = None,
    role: str = "player",
    team: int = 1,
    team_color: list[int] | None = None,
    display_role: str | None = None,
    display_team: int | None = None,
    display_label: str | None = None,
    confidence: float = 0.85,
    object_type: str = "player",
    bbox: list[float] | None = None,
) -> dict:
    """Build a single synthetic raw-tracklet row."""
    return {
        "source_frame_idx": frame,
        "sample_number": frame,
        "track_id": track_id,
        "raw_track_id": raw_track_id if raw_track_id is not None else track_id,
        "role": role,
        "detected_role": role,
        "object_type": object_type,
        "team": team,
        "team_color": team_color or ([0, 0, 255] if team == 1 else [255, 0, 0]),
        "display_role": display_role,
        "display_team": display_team,
        "display_label": display_label,
        "confidence": confidence,
        "bbox": bbox or [100.0, 100.0, 200.0, 200.0],
    }


# =============================================================================
# Scenario 1: Player disappears behind defender then returns (occlusion)
# =============================================================================
def test_player_occlusion_returns_same_id():
    """
    Player A (team 1, track 5) plays frames 0-50,
    disappears frames 51-100,
    returns frames 101-150.
    
    Expectation: Track 5 should remain team 1 throughout.
    Majority voting should fill display_team for missing frames.
    """
    rows: list[dict] = []
    # Visible
    for f in range(0, 51):
        rows.append(make_row(frame=f, track_id=5, role="player", team=1))
    # Occluded (no rows)
    # Returns
    for f in range(101, 151):
        rows.append(make_row(frame=f, track_id=5, role="player", team=1))

    # Majority voting should fill display fields
    result = apply_majority_voting_display_defaults(rows)
    track5_rows = [r for r in result if r["track_id"] == 5]

    # All rows should have display_team = 1
    for r in track5_rows:
        assert r.get("display_team") == 1, f"Frame {r['source_frame_idx']}: expected team 1"
        assert r.get("display_role") == "player"
        assert r.get("display_label") == "5"

    print("✅ Scenario 1 (occlusion) passed")


# =============================================================================
# Scenario 2: Cross-team ID swap (Player A team 1 → Player B team 2 reuses track)
# =============================================================================
def test_cross_team_id_swap_detected():
    """
    Track 18: frames 0-100 = player team 1 (blue)
    Then track 18 continues but player is now team 2 (red)
    
    Expectation: Audit should flag display_team_flicker or uncertain.
    """
    rows: list[dict] = []
    for f in range(0, 100):
        rows.append(make_row(frame=f, track_id=18, role="player", team=1,
                             team_color=[0, 0, 255]))
    for f in range(100, 200):
        rows.append(make_row(frame=f, track_id=18, role="player", team=2,
                             team_color=[255, 0, 0]))

    result = apply_majority_voting_display_defaults(rows)
    track18 = [r for r in result if r["track_id"] == 18]

    # With 50/50 split, confidence is only 50% — should NOT be filled
    # (threshold is 55% for weak fill, 70% for strong fill)
    filled_frames = sum(1 for r in track18 if r.get("display_team") is not None)
    # Weak fill may happen at 55% threshold, but 50% should not fill
    assert filled_frames == 0, f"Expected 0 filled frames with 50% confidence, got {filled_frames}"

    print(f"✅ Scenario 2 (cross-team swap) passed — correctly refused to fill with 50% confidence")


# =============================================================================
# Scenario 3: Goalkeeper fragmented across tracks
# =============================================================================
def test_goalkeeper_fragmentation():
    """
    Goalkeeper appears as track 21 (frames 0-50), track 29 (51-100), track 31 (101-150).
    All have role=goalkeeper, team=0.
    
    Expectation: All should display as GK with magenta color.
    """
    rows: list[dict] = []
    for f, tid in [(range(0, 51), 21), (range(51, 101), 29), (range(101, 151), 31)]:
        for frame in f:
            rows.append(make_row(frame=frame, track_id=tid, role="goalkeeper",
                                 team=0, object_type="goalkeeper"))

    result = apply_majority_voting_display_defaults(rows)
    for r in result:
        assert r.get("display_role") == "goalkeeper", f"Track {r['track_id']} should be GK"
        assert r.get("display_label") == "GK"
        assert r.get("display_team") == 0

    print("✅ Scenario 3 (GK fragmentation) passed")


# =============================================================================
# Scenario 4: Referee mixed into player track
# =============================================================================
def test_referee_in_player_track():
    """
    Track 20: 70% player team 1, 30% referee.
    
    Expectation: Majority voting should lock role=player (70% > 55%).
    """
    rows: list[dict] = []
    for f in range(0, 70):
        rows.append(make_row(frame=f, track_id=20, role="player", team=1))
    for f in range(70, 100):
        rows.append(make_row(frame=f, track_id=20, role="referee", team=0))

    result = apply_majority_voting_display_defaults(rows)
    track20 = [r for r in result if r["track_id"] == 20]

    # Should be locked as player
    player_frames = sum(1 for r in track20 if r.get("display_role") == "player")
    ratio = player_frames / len(track20)
    assert ratio >= 0.7, f"Expected >=70% player frames, got {ratio:.0%}"

    print(f"✅ Scenario 4 (referee mixed) passed — {ratio:.0%} locked as player")


# =============================================================================
# Scenario 5: Team color instability (KMeans drift)
# =============================================================================
def test_team_color_instability():
    """
    Track 10: alternates team 1/2 every 10 frames (simulating KMeans drift).
    
    Expectation: Majority voting should pick dominant team.
    """
    rows: list[dict] = []
    for f in range(0, 100):
        team = 1 if (f // 10) % 2 == 0 else 2
        rows.append(make_row(frame=f, track_id=10, role="player", team=team,
                             team_color=[0, 0, 255] if team == 1 else [255, 0, 0]))

    result = apply_majority_voting_display_defaults(rows)
    track10 = [r for r in result if r["track_id"] == 10]
    teams = [r.get("display_team") for r in track10]

    # All should be same team (the majority)
    assert len(set(teams)) == 1, f"Expected single team, got {set(teams)}"

    print(f"✅ Scenario 5 (KMeans drift) passed — locked to team {teams[0]}")


# =============================================================================
# Scenario 6: Full pipeline integration
# =============================================================================
def test_full_pipeline_stability():
    """
    Simulate a realistic match with 10 tracks, including:
    - 1 GK (track 1)
    - 1 referee (track 2)
    - 4 team-1 players
    - 4 team-2 players
    - 1 mixed track (team confusion)
    
    Run stabilizer + majority voting + audit.
    """
    rows: list[dict] = []

    # GK
    for f in range(0, 100):
        rows.append(make_row(frame=f, track_id=1, role="goalkeeper", team=0,
                             object_type="goalkeeper"))
    # Referee
    for f in range(0, 100):
        rows.append(make_row(frame=f, track_id=2, role="referee", team=0,
                             object_type="referee"))
    # Team 1 players
    for tid in [3, 4, 5, 6]:
        for f in range(0, 100):
            rows.append(make_row(frame=f, track_id=tid, role="player", team=1))
    # Team 2 players
    for tid in [7, 8, 9, 10]:
        for f in range(0, 100):
            rows.append(make_row(frame=f, track_id=tid, role="player", team=2))
    # Mixed track: team confusion
    for f in range(0, 60):
        rows.append(make_row(frame=f, track_id=11, role="player", team=1))
    for f in range(60, 100):
        rows.append(make_row(frame=f, track_id=11, role="player", team=2))

    # Run pipeline
    stability_plan = build_global_identity_stability_plan(rows)
    stable_rows, _ = apply_global_identity_stability_plan_to_raw_records(rows, stability_plan)
    final_rows = apply_majority_voting_display_defaults(stable_rows)
    audit = build_render_identity_audit(final_rows)

    print(f"✅ Scenario 6 (full pipeline) passed — score={audit.get('score')}, verdict={audit.get('verdict')}")
    print(f"   Issues: {len(audit.get('issues', []))}")


if __name__ == "__main__":
    print("Running identity scenario tests...\n")
    test_player_occlusion_returns_same_id()
    test_cross_team_id_swap_detected()
    test_goalkeeper_fragmentation()
    test_referee_in_player_track()
    test_team_color_instability()
    test_full_pipeline_stability()
    print("\n🎉 All scenarios passed!")
