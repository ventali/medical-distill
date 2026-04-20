from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from medical_distill.metrics import compute_basic_metrics  # noqa: E402
from medical_distill.utils import ensure_parent_dir, load_json, read_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a prediction JSONL file.")
    parser.add_argument("--config", required=True, help="Path to a JSON config file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    rows = read_jsonl(config["input_path"])

    metrics = compute_basic_metrics(
        rows=rows,
        prediction_field=config["prediction_field"],
        reference_field=config["reference_field"],
        compare_field=config.get("compare_field"),
        abstain_values=config.get("abstain_values", []),
        positive_labels=config.get("positive_labels", []),
        span_fields=config.get("span_fields", []),
        span_positive_compare_field=config.get("span_positive_compare_field"),
        span_positive_labels=config.get("span_positive_labels", []),
    )

    print(json.dumps(metrics, indent=2, sort_keys=True))
    output_path = config.get("output_path")
    if output_path:
        ensure_parent_dir(output_path)
        Path(output_path).write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
        print(f"Wrote metrics to {output_path}")


if __name__ == "__main__":
    main()

