from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from medical_distill.utils import load_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit a Vertex AI open-model tuning job.")
    parser.add_argument("--config", required=True, help="Path to a JSON config file.")
    return parser.parse_args()


def resolve_project_id(vertex_config: dict[str, Any]) -> str:
    project_id = vertex_config.get("project_id")
    if project_id:
        return project_id
    project_id_env = vertex_config.get("project_id_env")
    if project_id_env:
        project_id = os.getenv(project_id_env)
        if project_id:
            return project_id
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if project_id:
        return project_id
    raise SystemExit(
        "Could not resolve Google Cloud project ID. Set `vertex.project_id`, "
        "`vertex.project_id_env`, or GOOGLE_CLOUD_PROJECT."
    )


def compact_job_summary(job: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for attribute in [
        "resource_name",
        "name",
        "display_name",
        "state",
        "experiment",
        "tuned_model_name",
        "tuned_model_endpoint_name",
    ]:
        value = getattr(job, attribute, None)
        if value is not None:
            summary[attribute] = value
    return summary


def main() -> None:
    args = parse_args()
    config = load_json(args.config)

    try:
        import vertexai
        from vertexai.preview.tuning import SourceModel, sft
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Vertex AI SDK is not installed. Run `python3 -m pip install -e '.[gcp]'` first."
        ) from exc

    vertex_config = config["vertex"]
    project_id = resolve_project_id(vertex_config)
    location = vertex_config.get("location", "us-central1")

    vertexai.init(project=project_id, location=location)

    source_model_config = config["source_model"]
    custom_base_model = source_model_config.get("custom_base_model")
    source_model = SourceModel(
        base_model=source_model_config["base_model"],
        custom_base_model=custom_base_model if custom_base_model else None,
    )

    tuning_config = config["tuning"]
    train_kwargs: dict[str, Any] = {
        "source_model": source_model,
        "train_dataset": vertex_config["train_dataset_gcs_uri"],
        "validation_dataset": vertex_config.get("validation_dataset_gcs_uri"),
        "tuned_model_display_name": vertex_config["display_name"],
        "tuning_mode": tuning_config.get("mode", "PEFT_ADAPTER"),
        "epochs": tuning_config.get("epochs", 3),
        "learning_rate_multiplier": tuning_config.get("learning_rate_multiplier"),
        "adapter_size": tuning_config.get("adapter_size"),
        "output_uri": vertex_config.get("output_uri"),
        "labels": config.get("labels"),
    }
    train_kwargs = {key: value for key, value in train_kwargs.items() if value is not None}

    tuning_job = sft.preview_train(**train_kwargs)
    summary = compact_job_summary(tuning_job)
    print(json.dumps(summary or {"job_repr": repr(tuning_job)}, indent=2, sort_keys=True))
    print(
        "Vertex tuning job submitted. Track progress in the Vertex AI console or "
        "through the returned resource name."
    )


if __name__ == "__main__":
    main()
