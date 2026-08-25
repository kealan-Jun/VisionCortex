from __future__ import annotations

import argparse
import json
from pathlib import Path

from labvision_evidence.yolo_evaluation import evaluate_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate VisionCortex YOLO predictions against independent box GT."
    )
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()
    report = evaluate_files(
        args.predictions,
        args.ground_truth,
        args.output,
        confidence_threshold=args.confidence,
    )
    print(json.dumps({key: report[key] for key in ("status", "image_count", "macro")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
