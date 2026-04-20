from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from medical_distill.utils import (  # noqa: E402
    extract_json_block,
    json_dumps,
    load_few_shot_examples,
    load_json,
    read_jsonl,
    render_user_input,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic teacher outputs.")
    parser.add_argument("--config", required=True, help="Path to a JSON config file.")
    return parser.parse_args()


def message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def build_messages(task_config: dict[str, Any], seed_record: dict[str, Any], few_shots: list[dict[str, Any]]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": task_config["system_prompt"]},
    ]

    for example in few_shots:
        user_text = example.get("user") or json_dumps(example.get("input", {}))
        assistant_value = example.get("assistant") or example.get("output", {})
        assistant_text = assistant_value if isinstance(assistant_value, str) else json_dumps(assistant_value)
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": assistant_text})

    user_sections = [
        "Generate one high-quality teacher answer for the biomedical example below.",
        "Source example:",
        render_user_input(seed_record),
        "Structured seed record:",
        json_dumps(seed_record),
        "Output contract:",
        task_config["output_format_instructions"],
    ]
    messages.append({"role": "user", "content": "\n\n".join(user_sections)})
    return messages


def build_openai_client(teacher_config: dict[str, Any]):
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("openai is not installed. Run `pip install -e .` first.") from exc

    api_key = os.getenv(teacher_config["api_key_env"])
    if not api_key and teacher_config.get("base_url"):
        api_key = "DUMMY_KEY"
    if not api_key:
        raise SystemExit(
            f"Missing API key in environment variable {teacher_config['api_key_env']!r}."
        )

    client_kwargs = {"api_key": api_key}
    if teacher_config.get("base_url"):
        client_kwargs["base_url"] = teacher_config["base_url"]
    return OpenAI(**client_kwargs)


def refresh_google_access_token() -> tuple[str, str | None]:
    try:
        import google.auth
        import google.auth.transport.requests
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Google authentication libraries are missing. Run `pip install -e .` first."
        ) from exc

    credentials, discovered_project_id = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    request = google.auth.transport.requests.Request()
    credentials.refresh(request)
    return credentials.token, discovered_project_id


def resolve_vertex_project_id(teacher_config: dict[str, Any], discovered_project_id: str | None) -> str:
    project_id = teacher_config.get("project_id")
    if project_id:
        return project_id

    project_id_env = teacher_config.get("project_id_env")
    if project_id_env:
        project_id = os.getenv(project_id_env)
        if project_id:
            return project_id

    if discovered_project_id:
        return discovered_project_id

    raise SystemExit(
        "Could not resolve Google Cloud project ID. Set `teacher.project_id`, "
        "`teacher.project_id_env`, or configure ADC with a default project."
    )


def build_vertex_openai_client(teacher_config: dict[str, Any]):
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("openai is not installed. Run `pip install -e .` first.") from exc

    token, discovered_project_id = refresh_google_access_token()
    location = teacher_config.get("location", "us-central1")
    project_id = resolve_vertex_project_id(teacher_config, discovered_project_id)
    base_url = (
        f"https://{location}-aiplatform.googleapis.com/v1/projects/"
        f"{project_id}/locations/{location}/endpoints/openapi"
    )
    client = OpenAI(api_key=token, base_url=base_url)
    return client, project_id, location


def build_client_bundle(teacher_config: dict[str, Any]) -> dict[str, Any]:
    provider = teacher_config.get("provider", "openai_compatible")
    if provider == "vertex_openai":
        client, project_id, location = build_vertex_openai_client(teacher_config)
        return {
            "provider": provider,
            "client": client,
            "project_id": project_id,
            "location": location,
        }
    return {
        "provider": provider,
        "client": build_openai_client(teacher_config),
    }


def refresh_client_bundle_if_needed(teacher_config: dict[str, Any], client_bundle: dict[str, Any]) -> dict[str, Any]:
    if client_bundle["provider"] != "vertex_openai":
        return client_bundle
    return build_client_bundle(teacher_config)


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    client_bundle = build_client_bundle(config["teacher"])

    input_rows = read_jsonl(config["input"]["seed_examples_path"])
    max_examples = config["input"].get("max_examples")
    if isinstance(max_examples, int):
        input_rows = input_rows[:max_examples]

    few_shots = load_few_shot_examples(config["task"].get("few_shot_examples_path"))
    output_path = config["output"]["path"]
    resume = bool(config["output"].get("resume"))

    existing_rows = read_jsonl(output_path) if resume else []
    existing_ids = {row.get("seed_id") for row in existing_rows}
    new_rows: list[dict[str, Any]] = []

    for index, seed_record in enumerate(input_rows, start=1):
        seed_id = seed_record.get("id", f"seed-{index}")
        if seed_id in existing_ids:
            continue

        messages = build_messages(config["task"], seed_record, few_shots)
        client_bundle = refresh_client_bundle_if_needed(config["teacher"], client_bundle)
        response = client_bundle["client"].chat.completions.create(
            model=config["teacher"]["model"],
            messages=messages,
            temperature=config["teacher"].get("temperature", 0.2),
            max_tokens=config["teacher"].get("max_tokens", 900),
        )
        raw_text = message_content_to_text(response.choices[0].message.content)
        parsed_payload = extract_json_block(raw_text)

        if not isinstance(parsed_payload, dict):
            parsed_payload = {
                "answer": "abstain",
                "drug": "",
                "event": "",
                "evidence": "",
                "short_justification": "Teacher output could not be parsed as JSON.",
                "confidence": 0.0,
            }

        new_rows.append(
            {
                "seed_id": seed_id,
                "user_input": render_user_input(seed_record),
                "seed_record": seed_record,
                "teacher_model": config["teacher"]["model"],
                "teacher_provider": client_bundle["provider"],
                "teacher_project_id": client_bundle.get("project_id"),
                "teacher_location": client_bundle.get("location"),
                "teacher_output": parsed_payload,
                "raw_response_text": raw_text,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        print(f"[{index}/{len(input_rows)}] generated {seed_id}", flush=True)

        if len(new_rows) % 25 == 0:
            write_jsonl(output_path, existing_rows + new_rows)

    merged_rows = existing_rows + new_rows
    write_jsonl(output_path, merged_rows)
    print(
        f"Wrote {len(new_rows)} new rows to {output_path} "
        f"({len(merged_rows)} total records)."
    )


if __name__ == "__main__":
    main()
