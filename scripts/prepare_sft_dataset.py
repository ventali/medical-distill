from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from medical_distill.utils import json_dumps, load_json, read_jsonl, render_user_input, write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare an SFT-ready JSONL dataset.")
    parser.add_argument("--config", required=True, help="Path to a JSON config file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    rows = read_jsonl(config["input_path"])

    assistant_field = config.get("assistant_field", "teacher_output")
    user_field = config.get("user_field", "user_input")
    system_prompt = config["system_prompt"]

    sft_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        assistant_payload = row.get(assistant_field)
        if not isinstance(assistant_payload, dict):
            continue

        user_input = row.get(user_field)
        if not isinstance(user_input, str) or not user_input.strip():
            user_input = render_user_input(row.get("seed_record", row))

        sft_rows.append(
            {
                "id": row.get("seed_id", f"example-{index}"),
                "prompt": user_input,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input},
                    {
                        "role": "assistant",
                        "content": json_dumps(assistant_payload),
                    },
                ],
                "reference": assistant_payload,
                "metadata": {
                    "teacher_model": row.get("teacher_model"),
                    "source_id": row.get("seed_id"),
                },
            }
        )

    write_jsonl(config["output_path"], sft_rows)
    print(f"Wrote {len(sft_rows)} SFT rows to {config['output_path']}")


if __name__ == "__main__":
    main()

