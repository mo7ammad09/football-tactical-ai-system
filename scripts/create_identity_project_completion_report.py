#!/usr/bin/env python3
"""Create a local completion report for the identity-correction project."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.identity.completion_report import build_identity_project_completion_report
from src.identity.deployment_preflight import run_phase9_preflight
from src.identity.release_gate import build_runpod_release_candidate_manifest


def _git_value(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return completed.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-image", default=None)
    parser.add_argument("--tests-passed", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    preflight = run_phase9_preflight(REPO_ROOT)
    release_manifest = build_runpod_release_candidate_manifest(
        candidate_image=args.candidate_image,
        source_commit=_git_value("rev-parse", "--short=12", "HEAD"),
        source_dirty=bool(_git_value("status", "--short")),
        preflight_result=preflight,
        test_results=[{"command": "operator_asserted", "verdict": "PASS"}]
        if args.tests_passed
        else None,
    )
    verification = {
        "verdict": "PASS" if args.tests_passed else "NOT_RECORDED",
        "note": "Use --tests-passed only after running the documented verification suite.",
    }
    report = build_identity_project_completion_report(
        preflight_result=preflight,
        release_manifest=release_manifest,
        verification=verification,
    )
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    print(encoded)

    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = REPO_ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(encoded + "\n", encoding="utf-8")

    return 0 if report["local_status"] == "ready_for_candidate_build" else 1


if __name__ == "__main__":
    sys.exit(main())
