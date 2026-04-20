from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from medical_distill.utils import json_dumps, load_json, read_jsonl, write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Vertex open-model tuning datasets in messages JSONL format."
    )
    parser.add_argument("--config", required=True, help="Path to a JSON config file.")
    return parser.parse_args()


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content", "")
        if not isinstance(content, str):
            content = json_dumps(content)
        normalized.append({"role": role, "content": content})
    return normalized


def build_validation_messages(
    row: dict[str, Any],
    system_prompt: str,
    prompt_field: str,
    reference_field: str,
) -> list[dict[str, str]]:
    prompt = row.get(prompt_field)
    reference = row.get(reference_field, {})
    if not isinstance(prompt, str) or not prompt.strip():
        return []

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    messages.append({"role": "assistant", "content": json_dumps(reference)})
    return messages


def main() -> None:
    args = parse_args()
    config = load_json(args.config)

    train_rows = read_jsonl(config["train_input_path"])
    train_output_rows: list[dict[str, Any]] = []
    for row in train_rows:
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            continue
        train_output_rows.append({"messages": normalize_messages(messages)})

    validation_rows = read_jsonl(config["validation_input_path"])
    validation_output_rows: list[dict[str, Any]] = []
    for row in validation_rows:
        messages = build_validation_messages(
            row=row,
            system_prompt=config.get("system_prompt", ""),
            prompt_field=config.get("validation_prompt_field", "prompt"),
            reference_field=config.get("validation_reference_field", "reference"),
        )
        if messages:
            validation_output_rows.append({"messages": messages})

    write_jsonl(config["train_output_path"], train_output_rows)
    write_jsonl(config["validation_output_path"], validation_output_rows)
    print(
        f"Wrote {len(train_output_rows)} train rows to {config['train_output_path']} "
        f"and {len(validation_output_rows)} validation rows to "
        f"{config['validation_output_path']}"
    )


if __name__ == "__main__":
    main()

