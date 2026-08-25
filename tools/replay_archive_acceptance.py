from __future__ import annotations

import argparse
import json
from pathlib import Path

from labvision_evidence.replay_acceptance import (
    build_archive_regression_snapshot,
    compare_archive_snapshot,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay VisionCortex archive acceptance from durable JSON only."
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8-sig"))
    snapshot = build_archive_regression_snapshot(args.archive)
    result = compare_archive_snapshot(snapshot, baseline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output),
                "video_files_opened": 0,
                "clock_csv_files_opened": 0,
            },
            ensure_ascii=False,
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
