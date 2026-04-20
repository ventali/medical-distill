from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from medical_distill.utils import get_nested_value, load_json, read_jsonl, write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter synthetic teacher outputs.")
    parser.add_argument("--config", required=True, help="Path to a JSON config file.")
    return parser.parse_args()


def make_dedupe_key(value: Any) -> str:
    if isinstance(value, str):
        return value.strip().lower()
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    rows = read_jsonl(config["input_path"])

    payload_field = config.get("payload_field", "teacher_output")
    confidence_field = config.get("confidence_field", "confidence")
    evidence_field = config.get("evidence_field", "evidence")
    required_fields = config.get("required_fields", [])
    min_confidence = float(config.get("min_confidence", 0.0))
    dedupe_on = config.get("dedupe_on", "user_input")
    drop_if_missing_evidence = bool(config.get("drop_if_missing_evidence", True))
    source_field = config.get("source_field", "user_input")
    require_evidence_overlap = float(config.get("require_evidence_overlap", 0.0))

    def word_overlap(a: Any, b: Any) -> float:
        def tokens(value: Any) -> set[str]:
            if not isinstance(value, str):
                return set()
            return {t for t in re.split(r"\W+", value.lower()) if len(t) >= 3}

        evidence_tokens = tokens(a)
        source_tokens = tokens(b)
        if not evidence_tokens:
            return 0.0
        return len(evidence_tokens & source_tokens) / len(evidence_tokens)

    filtered_rows: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    stats = {
        "total": len(rows),
        "kept": 0,
        "invalid_payload": 0,
        "missing_required_fields": 0,
        "below_confidence": 0,
        "missing_evidence": 0,
        "low_evidence_overlap": 0,
        "duplicates": 0,
    }

    for row in rows:
        payload = row.get(payload_field)
        if not isinstance(payload, dict):
            stats["invalid_payload"] += 1
            continue

        missing = [field for field in required_fields if get_nested_value(payload, field) in (None, "", [])]
        if missing:
            stats["missing_required_fields"] += 1
            continue

        confidence = get_nested_value(payload, confidence_field)
        if confidence is not None and float(confidence) < min_confidence:
            stats["below_confidence"] += 1
            continue

        evidence = get_nested_value(payload, evidence_field)
        if drop_if_missing_evidence and evidence in (None, "", []):
            stats["missing_evidence"] += 1
            continue

        if require_evidence_overlap > 0.0:
            source_text = row.get(source_field) or get_nested_value(row, source_field)
            if word_overlap(evidence, source_text) < require_evidence_overlap:
                stats["low_evidence_overlap"] += 1
                continue

        dedupe_value = row.get(dedupe_on)
        if dedupe_value is None:
            dedupe_value = get_nested_value(payload, dedupe_on)
        dedupe_key = make_dedupe_key(dedupe_value)
        if dedupe_key in seen_keys:
            stats["duplicates"] += 1
            continue

        seen_keys.add(dedupe_key)
        filtered_rows.append(row)

    stats["kept"] = len(filtered_rows)
    write_jsonl(config["output_path"], filtered_rows)
    print(json.dumps(stats, indent=2, sort_keys=True))
    print(f"Wrote filtered dataset to {config['output_path']}")


if __name__ == "__main__":
    main()

