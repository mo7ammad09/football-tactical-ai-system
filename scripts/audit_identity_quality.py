#!/usr/bin/env python3
"""Audit raw tracklets and identity debug output for club-grade ID quality."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Iterable


GOALKEEPER_ROLE = "goalkeeper"
PERSON_OBJECT_TYPES = {"player", "referee", "goalkeeper"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read newline-delimited JSON records."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as jsonl_f:
        for line in jsonl_f:
            if line.strip():
                records.append(json.loads(line))
    return records


def _read_json(path: Path | None) -> dict[str, Any]:
    """Read a JSON object, returning an empty dict when no path is supplied."""
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _counter_dict(counter: Counter) -> dict[str, int]:
    """Convert counter keys to strings for JSON stability."""
    return {str(key): int(value) for key, value in counter.items()}


def _compact_segment(segment: dict[str, Any]) -> dict[str, Any]:
    """Return a stable JSON shape for a role segment."""
    return {
        "role": str(segment.get("role", "unknown")),
        "team": int(segment.get("team", 0) or 0),
        "object_type": str(segment.get("object_type", "unknown")),
        "raw_track_id": (
            int(segment["raw_track_id"])
            if segment.get("raw_track_id") is not None
            else None
        ),
        "first_source_frame_idx": int(segment.get("first_source_frame_idx", 0)),
        "last_source_frame_idx": int(segment.get("last_source_frame_idx", 0)),
        "frames_seen": int(segment.get("frames_seen", 0)),
    }


def _dominant(counter: Counter, fallback: Any = None) -> tuple[Any, float]:
    """Return the dominant counter value and confidence."""
    total = sum(counter.values())
    if total <= 0:
        return fallback, 0.0
    value, count = counter.most_common(1)[0]
    return value, float(count) / float(total)


def _iter_person_rows(rows: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    """Yield non-ball rows."""
    for row in rows:
        if str(row.get("object_type", "")) in PERSON_OBJECT_TYPES:
            yield row


def _is_goalkeeper_like(row: dict[str, Any]) -> bool:
    """Return whether a row is visually treated as a goalkeeper."""
    return _visible_role(row) == GOALKEEPER_ROLE


def _visible_role(row: dict[str, Any]) -> str:
    """Return the client-facing role for a row."""
    return str(row.get("display_role") or row.get("role") or "unknown")


def _visible_team(row: dict[str, Any]) -> int:
    """Return the client-facing team/color bucket for a row."""
    if row.get("display_team") is not None:
        return int(row.get("display_team") or 0)
    return int(row.get("team", 0) or 0)


def _visible_identity(row: dict[str, Any]) -> Any:
    """Return the client-facing identity label for a row."""
    if row.get("display_label"):
        return str(row["display_label"])
    return int(row["track_id"])


def _identity_sort_key(identity: Any) -> tuple[int, Any]:
    """Sort numeric IDs before text labels while keeping JSON-friendly output."""
    if isinstance(identity, int):
        return (0, identity)
    try:
        return (0, int(identity))
    except Exception:
        return (1, str(identity))


def _build_track_profiles(rows: list[dict[str, Any]]) -> dict[Any, dict[str, Any]]:
    """Build per-visible-ID profiles with role/team/raw-track segments."""
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in _iter_person_rows(rows):
        if row.get("track_id") is not None:
            grouped[_visible_identity(row)].append(row)

    profiles: dict[Any, dict[str, Any]] = {}
    for visible_id, track_rows in grouped.items():
        ordered = sorted(track_rows, key=lambda item: int(item.get("source_frame_idx", 0)))
        roles = Counter(_visible_role(row) for row in ordered)
        teams = Counter(_visible_team(row) for row in ordered)
        objects = Counter(str(row.get("object_type", "unknown")) for row in ordered)
        source_track_ids = Counter(
            int(row["track_id"])
            for row in ordered
            if row.get("track_id") is not None
        )
        raw_ids = Counter(
            int(row["raw_track_id"])
            for row in ordered
            if row.get("raw_track_id") is not None
        )
        source_frames = [int(row.get("source_frame_idx", 0)) for row in ordered]

        segments: list[dict[str, Any]] = []
        for row in ordered:
            role = _visible_role(row)
            team = _visible_team(row)
            object_type = str(row.get("object_type", "unknown"))
            raw_track_id = (
                int(row["raw_track_id"])
                if row.get("raw_track_id") is not None
                else None
            )
            source_frame_idx = int(row.get("source_frame_idx", 0))
            if (
                not segments
                or segments[-1]["role"] != role
                or int(segments[-1]["team"]) != team
                or segments[-1].get("raw_track_id") != raw_track_id
                or str(segments[-1]["object_type"]) != object_type
            ):
                segments.append(
                    {
                        "role": role,
                        "team": team,
                        "object_type": object_type,
                        "raw_track_id": raw_track_id,
                        "first_source_frame_idx": source_frame_idx,
                        "last_source_frame_idx": source_frame_idx,
                        "frames_seen": 1,
                    }
                )
            else:
                segments[-1]["last_source_frame_idx"] = source_frame_idx
                segments[-1]["frames_seen"] += 1

        dominant_role, role_confidence = _dominant(roles, "unknown")
        dominant_team, team_confidence = _dominant(teams, 0)
        dominant_raw, raw_confidence = _dominant(raw_ids, None)
        goalkeeper_segments = [
            _compact_segment(segment)
            for segment in segments
            if str(segment.get("role")) == GOALKEEPER_ROLE
        ]
        goalkeeper_frames = sum(segment["frames_seen"] for segment in goalkeeper_segments)
        role_transition_count = sum(
            1
            for previous, current in zip(segments, segments[1:])
            if previous["role"] != current["role"]
        )

        profiles[visible_id] = {
            "id": visible_id,
            "frames_seen": len(ordered),
            "first_source_frame_idx": min(source_frames) if source_frames else 0,
            "last_source_frame_idx": max(source_frames) if source_frames else 0,
            "dominant_role": str(dominant_role),
            "role_confidence": float(role_confidence),
            "dominant_team": int(dominant_team or 0),
            "team_confidence": float(team_confidence),
            "dominant_raw_track_id": int(dominant_raw) if dominant_raw is not None else None,
            "raw_track_confidence": float(raw_confidence),
            "role_counts": _counter_dict(roles),
            "team_counts": _counter_dict(teams),
            "object_type_counts": _counter_dict(objects),
            "source_track_id_counts": _counter_dict(Counter(dict(source_track_ids.most_common(10)))),
            "raw_track_counts": _counter_dict(Counter(dict(raw_ids.most_common(10)))),
            "segment_count": len(segments),
            "role_transition_count": int(role_transition_count),
            "goalkeeper_frame_count": int(goalkeeper_frames),
            "goalkeeper_segments": goalkeeper_segments[:20],
            "goalkeeper_segments_truncated": len(goalkeeper_segments) > 20,
        }
    return profiles


def _build_raw_track_fragmentation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Find raw tracker IDs that appear under multiple display IDs."""
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in _iter_person_rows(rows):
        if row.get("raw_track_id") is not None:
            grouped[int(row["raw_track_id"])].append(row)

    fragments: list[dict[str, Any]] = []
    for raw_track_id, raw_rows in grouped.items():
        display_ids = Counter(int(row["track_id"]) for row in raw_rows if row.get("track_id") is not None)
        if len(display_ids) <= 1:
            continue
        roles = Counter(_visible_role(row) for row in raw_rows)
        source_frames = [int(row.get("source_frame_idx", 0)) for row in raw_rows]
        goalkeeper_rows = [row for row in raw_rows if _is_goalkeeper_like(row)]
        fragments.append(
            {
                "raw_track_id": int(raw_track_id),
                "display_id_count": len(display_ids),
                "display_ids": _counter_dict(display_ids),
                "role_counts": _counter_dict(roles),
                "first_source_frame_idx": min(source_frames) if source_frames else 0,
                "last_source_frame_idx": max(source_frames) if source_frames else 0,
                "goalkeeper_frame_count": len(goalkeeper_rows),
                "risk": (
                    "high"
                    if len(goalkeeper_rows) >= 3 or len(display_ids) >= 3
                    else "medium"
                ),
            }
        )
    return sorted(
        fragments,
        key=lambda item: (
            item["risk"] != "high",
            -int(item["goalkeeper_frame_count"]),
            -int(item["display_id_count"]),
        ),
    )


def _build_goalkeeper_timeline(profiles: dict[Any, dict[str, Any]]) -> list[dict[str, Any]]:
    """Build a chronological goalkeeper segment timeline."""
    timeline: list[dict[str, Any]] = []
    for visible_id, profile in profiles.items():
        for segment in profile["goalkeeper_segments"]:
            timeline.append({"track_id": visible_id, **segment})
    return sorted(
        timeline,
        key=lambda item: (
            int(item.get("first_source_frame_idx", 0)),
            _identity_sort_key(item.get("track_id")),
        ),
    )


def _score_audit(findings: list[dict[str, Any]], identity_debug: dict[str, Any]) -> tuple[int, str]:
    """Assign a founder-style quality score and verdict."""
    score = 100
    for finding in findings:
        severity = finding.get("severity")
        if severity == "critical":
            score -= 30
        elif severity == "high":
            score -= 18
        elif severity == "medium":
            score -= 8
        elif severity == "low":
            score -= 3

    warnings = identity_debug.get("warnings") or []
    if any(warning.get("code") == "missing_reid_embeddings" for warning in warnings):
        score -= 20

    score = max(0, min(100, score))
    critical_count = sum(1 for finding in findings if finding.get("severity") == "critical")
    high_count = sum(1 for finding in findings if finding.get("severity") == "high")
    if critical_count:
        return score, "FAIL"
    if high_count or score < 80:
        return score, "REVIEW"
    return score, "PASS"


def audit_identity_quality(
    raw_tracklet_rows: list[dict[str, Any]],
    identity_debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit raw tracklets and identity debug data for product-grade risks."""
    identity_debug = identity_debug or {}
    person_rows = list(_iter_person_rows(raw_tracklet_rows))
    profiles = _build_track_profiles(raw_tracklet_rows)
    goalkeeper_timeline = _build_goalkeeper_timeline(profiles)
    raw_fragments = _build_raw_track_fragmentation(raw_tracklet_rows)

    reid_available_rows = sum(1 for row in person_rows if row.get("reid_available"))
    object_types = Counter(str(row.get("object_type", "unknown")) for row in raw_tracklet_rows)
    role_counts = Counter(str(row.get("role", "unknown")) for row in person_rows)

    findings: list[dict[str, Any]] = []
    role_flickers = [
        profile
        for profile in profiles.values()
        if profile["role_transition_count"] > 0
        or (
            profile["dominant_role"] != GOALKEEPER_ROLE
            and profile["goalkeeper_frame_count"] >= 3
        )
        or (
            profile["frames_seen"] >= 20
            and profile["role_confidence"] < 0.85
        )
    ]
    if role_flickers:
        findings.append(
            {
                "severity": "high",
                "code": "role_color_flicker",
                "title": "Display IDs change role/color over time",
                "why_it_matters": (
                    "The annotated video can show the same visible ID switching "
                    "between team color and goalkeeper/referee color."
                ),
                "tracklet_count": len(role_flickers),
                "examples": sorted(
                    role_flickers,
                    key=lambda item: (
                        -int(item["goalkeeper_frame_count"]),
                        -int(item["role_transition_count"]),
                    ),
                )[:12],
            }
        )

    substantial_goalkeeper_ids = {
        segment["track_id"]
        for segment in goalkeeper_timeline
        if int(segment.get("frames_seen", 0)) >= 3
    }
    goalkeeper_switch_count = sum(
        1
        for previous, current in zip(goalkeeper_timeline, goalkeeper_timeline[1:])
        if previous["track_id"] != current["track_id"]
    )
    if len(substantial_goalkeeper_ids) > 1:
        findings.append(
            {
                "severity": "critical",
                "code": "goalkeeper_identity_fragmentation",
                "title": "Goalkeeper appears under multiple display IDs",
                "why_it_matters": (
                    "Coaches notice goalkeeper ID jumps immediately; this must "
                    "be fixed or clearly marked before a client-facing export."
                ),
                "display_id_count": len(substantial_goalkeeper_ids),
                "display_ids": sorted(substantial_goalkeeper_ids, key=_identity_sort_key),
                "switch_count": int(goalkeeper_switch_count),
                "timeline": goalkeeper_timeline[:40],
                "timeline_truncated": len(goalkeeper_timeline) > 40,
            }
        )

    high_raw_fragments = [item for item in raw_fragments if item["risk"] == "high"]
    if high_raw_fragments:
        findings.append(
            {
                "severity": "high",
                "code": "raw_track_display_fragmentation",
                "title": "Raw tracker identity is split across display IDs",
                "why_it_matters": (
                    "The display-ID layer is remapping a tracker identity, which "
                    "can create visible number jumps even when the tracker stayed "
                    "mostly consistent."
                ),
                "fragment_count": len(high_raw_fragments),
                "examples": high_raw_fragments[:15],
            }
        )

    unstable_raw_profiles = [
        profile
        for profile in profiles.values()
        if profile["frames_seen"] >= 50 and profile["raw_track_confidence"] < 0.75
    ]
    if unstable_raw_profiles:
        findings.append(
            {
                "severity": "medium",
                "code": "display_id_multiple_raw_tracks",
                "title": "Display IDs are stitched from many raw tracker IDs",
                "why_it_matters": (
                    "Some stitching is useful, but low raw-track dominance means "
                    "the visible ID may combine different people."
                ),
                "tracklet_count": len(unstable_raw_profiles),
                "examples": sorted(
                    unstable_raw_profiles,
                    key=lambda item: item["raw_track_confidence"],
                )[:12],
            }
        )

    short_goalkeeper_tracks = [
        profile
        for profile in profiles.values()
        if 0 < profile["goalkeeper_frame_count"] <= 3
    ]
    if short_goalkeeper_tracks:
        findings.append(
            {
                "severity": "medium",
                "code": "single_frame_goalkeeper_flickers",
                "title": "Very short goalkeeper flashes exist",
                "why_it_matters": (
                    "Short role flashes create one-frame pink/black color jumps "
                    "that look unprofessional in exported video."
                ),
                "tracklet_count": len(short_goalkeeper_tracks),
                "examples": sorted(
                    short_goalkeeper_tracks,
                    key=lambda item: -int(item["goalkeeper_frame_count"]),
                )[:12],
            }
        )

    debug_summary = identity_debug.get("summary", {})
    if debug_summary:
        if int(debug_summary.get("profiles_with_reid", 0)) == 0:
            findings.append(
                {
                    "severity": "critical",
                    "code": "missing_reid",
                    "title": "Identity debug has no ReID profiles",
                    "why_it_matters": "Long-range identity decisions are color/position only.",
                }
            )
        if int(debug_summary.get("accepted_auto_link_count", 0)) > int(debug_summary.get("candidate_link_count", 0)):
            findings.append(
                {
                    "severity": "high",
                    "code": "identity_debug_inconsistent_counts",
                    "title": "Identity debug counts are internally inconsistent",
                    "why_it_matters": "Quality gates cannot trust malformed identity reports.",
                }
            )

    score, verdict = _score_audit(findings, identity_debug)
    return {
        "verdict": verdict,
        "score": score,
        "summary": {
            "raw_record_count": len(raw_tracklet_rows),
            "person_record_count": len(person_rows),
            "tracklet_count": len(profiles),
            "reid_available_person_rows": int(reid_available_rows),
            "reid_coverage": (
                float(reid_available_rows) / float(len(person_rows))
                if person_rows
                else 0.0
            ),
            "object_type_counts": _counter_dict(object_types),
            "role_counts": _counter_dict(role_counts),
            "identity_debug_summary": debug_summary,
        },
        "findings": findings,
        "goalkeeper_timeline": goalkeeper_timeline[:80],
        "goalkeeper_timeline_truncated": len(goalkeeper_timeline) > 80,
        "raw_track_fragmentation": raw_fragments[:80],
        "raw_track_fragmentation_truncated": len(raw_fragments) > 80,
    }


def render_markdown(audit: dict[str, Any]) -> str:
    """Render an audit report for founder/product review."""
    lines = [
        f"# Identity Quality Audit: {audit['verdict']} ({audit['score']}/100)",
        "",
        "## Summary",
    ]
    summary = audit["summary"]
    lines.extend(
        [
            f"- Raw records: {summary['raw_record_count']}",
            f"- Person records: {summary['person_record_count']}",
            f"- Tracklets: {summary['tracklet_count']}",
            f"- ReID coverage: {summary['reid_coverage']:.1%}",
            f"- Object types: {summary['object_type_counts']}",
            f"- Roles: {summary['role_counts']}",
            "",
            "## Founder Findings",
        ]
    )
    if not audit["findings"]:
        lines.append("- No founder-grade identity risks detected.")
    for finding in audit["findings"]:
        lines.extend(
            [
                f"- [{finding['severity'].upper()}] {finding['code']}: {finding['title']}",
                f"  Why it matters: {finding['why_it_matters']}",
            ]
        )
        if "display_ids" in finding:
            lines.append(f"  Display IDs: {finding['display_ids']}")
        if "tracklet_count" in finding:
            lines.append(f"  Tracklets: {finding['tracklet_count']}")
        if "switch_count" in finding:
            lines.append(f"  Switches: {finding['switch_count']}")

    lines.extend(["", "## Goalkeeper Timeline"])
    for segment in audit["goalkeeper_timeline"][:30]:
        lines.append(
            "- "
            f"ID {segment['track_id']} "
            f"frames {segment['first_source_frame_idx']}-{segment['last_source_frame_idx']} "
            f"n={segment['frames_seen']} raw={segment.get('raw_track_id')}"
        )
    if audit.get("goalkeeper_timeline_truncated"):
        lines.append("- Timeline truncated.")

    lines.extend(["", "## Next Product Actions"])
    if audit["verdict"] == "PASS":
        lines.append("- Result is safe enough for review export. Continue with tactical validation.")
    else:
        lines.extend(
            [
                "- Add goalkeeper display-locking before exporting client-facing video.",
                "- Suppress short role/color flips inside long player tracklets.",
                "- Re-run this audit after every tracker/settings change.",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-tracklets", required=True, type=Path)
    parser.add_argument("--identity-debug", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()

    audit = audit_identity_quality(
        _read_jsonl(args.raw_tracklets),
        _read_json(args.identity_debug),
    )
    markdown = render_markdown(audit)

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(markdown, encoding="utf-8")

    print(markdown)
    return 0 if audit["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
