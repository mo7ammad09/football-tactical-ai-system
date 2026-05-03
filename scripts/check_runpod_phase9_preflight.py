#!/usr/bin/env python3
"""Run Phase 9 RunPod packaging preflight checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.identity.deployment_preflight import run_phase9_preflight


def main() -> int:
    result = run_phase9_preflight(REPO_ROOT)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
