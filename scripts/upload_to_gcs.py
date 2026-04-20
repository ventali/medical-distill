from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from medical_distill.utils import load_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload local files to Google Cloud Storage.")
    parser.add_argument("--config", required=True, help="Path to a JSON config file.")
    return parser.parse_args()


def resolve_project_id(config: dict) -> str | None:
    project_id = config.get("project_id")
    if project_id:
        return project_id
    project_id_env = config.get("project_id_env")
    if project_id_env:
        return os.getenv(project_id_env)
    return os.getenv("GOOGLE_CLOUD_PROJECT")


def main() -> None:
    args = parse_args()
    config = load_json(args.config)

    try:
        from google.cloud import storage
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "google-cloud-storage is not installed. Run `python3 -m pip install -e '.[gcp]'` first."
        ) from exc

    client = storage.Client(project=resolve_project_id(config))
    uploads = config.get("uploads", [])
    if not uploads:
        raise SystemExit("Config file must include a non-empty `uploads` list.")

    for upload in uploads:
        local_path = Path(upload["local_path"])
        if not local_path.exists():
            raise SystemExit(f"Local path does not exist: {local_path}")
        bucket = client.bucket(upload["bucket"])
        blob = bucket.blob(upload["blob"])
        blob.upload_from_filename(str(local_path))
        print(f"Uploaded {local_path} to gs://{upload['bucket']}/{upload['blob']}")


if __name__ == "__main__":
    main()
