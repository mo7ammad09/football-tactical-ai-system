#!/usr/bin/env python3
"""Create a Phase 10 RunPod release-candidate manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.identity.deployment_preflight import run_phase9_preflight
from src.identity.release_gate import (
    DEFAULT_RELEASE_TEST_COMMANDS,
    build_runpod_release_candidate_manifest,
)


def _git_value(*args: str) -> str | None:
    """Run a read-only git command and return stdout."""
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


def _source_dirty() -> bool:
    """Return whether the git worktree has uncommitted changes."""
    status = _git_value("status", "--short")
    return bool(status)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-image", default=None)
    parser.add_argument(
        "--tests-passed",
        action="store_true",
        help="Record the standard Phase 10 release tests as passing.",
    )
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    preflight = run_phase9_preflight(REPO_ROOT)
    test_results = None
    if args.tests_passed:
        test_results = [
            {
                "command": command,
                "verdict": "PASS",
                "source": "operator_asserted_current_session",
            }
            for command in DEFAULT_RELEASE_TEST_COMMANDS
        ]

    manifest = build_runpod_release_candidate_manifest(
        candidate_image=args.candidate_image,
        source_commit=_git_value("rev-parse", "--short=12", "HEAD"),
        source_dirty=_source_dirty(),
        preflight_result=preflight,
        test_results=test_results,
    )
    encoded = json.dumps(manifest, ensure_ascii=False, indent=2)
    print(encoded)

    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = REPO_ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(encoded + "\n", encoding="utf-8")

    return 0 if manifest["release_status"] == "candidate_ready" else 1


if __name__ == "__main__":
    sys.exit(main())
