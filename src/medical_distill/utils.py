from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        return []

    rows: list[dict[str, Any]] = []
    for line in file_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        rows.append(json.loads(stripped))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    Path(path).write_text(payload)


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def render_user_input(record: dict[str, Any]) -> str:
    prompt = record.get("user_input") or record.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()
    return json_dumps(record)


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
    stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def extract_json_block(text: str) -> Any:
    stripped = strip_code_fences(text)
    candidates = [stripped]
    match = re.search(r"(\{.*\}|\[.*\])", stripped, re.DOTALL)
    if match:
        candidates.append(match.group(1))

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def render_messages_fallback(messages: list[dict[str, Any]]) -> str:
    rendered_parts: list[str] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if not isinstance(content, str):
            content = json_dumps(content)
        rendered_parts.append(f"<|{role}|>\n{content}".strip())
    return "\n\n".join(rendered_parts)


def load_few_shot_examples(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    file_path = Path(path)
    if file_path.suffix == ".jsonl":
        return read_jsonl(file_path)
    return json.loads(file_path.read_text())


def get_nested_value(data: Any, dotted_field: str) -> Any:
    current = data
    for part in dotted_field.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current
