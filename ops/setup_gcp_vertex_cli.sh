#!/usr/bin/env bash
set -euo pipefail

# CLI-first setup for the Vertex teacher path.
#
# Required environment variables:
#   PROJECT_ID
#   BUCKET_NAME
#
# Optional environment variables:
#   PROJECT_NAME=medical-distill
#   CREATE_PROJECT=0
#   BILLING_ACCOUNT=
#   LOCATION=us-central1
#   BUCKET_LOCATION=us-central1
#   SERVICE_ACCOUNT_NAME=medical-distill-vertex
#   USER_PRINCIPAL=user:$(gcloud config get-value account)
#   GRANT_OPEN_MODEL_ENABLE_ROLE=0
#
# Notes:
# - This script can create and configure the project, bucket, service account,
#   and IAM roles through the GCP CLI.
# - As of April 19, 2026, enabling MaaS open models such as
#   llama-3.3-70b-instruct-maas still requires Model Garden enablement/consent
#   from the console. Google does not document a gcloud equivalent for that step.

: "${PROJECT_ID:?Set PROJECT_ID}"
: "${BUCKET_NAME:?Set BUCKET_NAME}"

PROJECT_NAME="${PROJECT_NAME:-medical-distill}"
CREATE_PROJECT="${CREATE_PROJECT:-0}"
BILLING_ACCOUNT="${BILLING_ACCOUNT:-}"
LOCATION="${LOCATION:-us-central1}"
BUCKET_LOCATION="${BUCKET_LOCATION:-${LOCATION}}"
SERVICE_ACCOUNT_NAME="${SERVICE_ACCOUNT_NAME:-medical-distill-vertex}"
GRANT_OPEN_MODEL_ENABLE_ROLE="${GRANT_OPEN_MODEL_ENABLE_ROLE:-0}"

if [[ -z "${USER_PRINCIPAL:-}" ]]; then
  ACTIVE_ACCOUNT="$(gcloud config get-value account 2>/dev/null || true)"
  if [[ -n "${ACTIVE_ACCOUNT}" && "${ACTIVE_ACCOUNT}" != "(unset)" ]]; then
    USER_PRINCIPAL="user:${ACTIVE_ACCOUNT}"
  else
    USER_PRINCIPAL=""
  fi
fi

SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
BUCKET_URI="gs://${BUCKET_NAME}"
PROJECT_ACTION="selection"
if [[ "${CREATE_PROJECT}" == "1" ]]; then
  PROJECT_ACTION="selection/creation"
fi

ensure_project() {
  if gcloud projects describe "${PROJECT_ID}" >/dev/null 2>&1; then
    return
  fi

  if [[ "${CREATE_PROJECT}" != "1" ]]; then
    echo "Project ${PROJECT_ID} does not exist and CREATE_PROJECT is not 1." >&2
    exit 1
  fi

  gcloud projects create "${PROJECT_ID}" --name="${PROJECT_NAME}" --set-as-default

  if [[ -n "${BILLING_ACCOUNT}" ]]; then
    gcloud billing projects link "${PROJECT_ID}" --billing-account="${BILLING_ACCOUNT}"
  else
    cat <<EOF
Project created, but no billing account was linked.
Set BILLING_ACCOUNT and re-run if you want this script to link billing:
  BILLING_ACCOUNT=XXXXXX-XXXXXX-XXXXXX PROJECT_ID=${PROJECT_ID} BUCKET_NAME=${BUCKET_NAME} CREATE_PROJECT=0 $0
EOF
  fi
}

ensure_bucket() {
  if gcloud storage buckets describe "${BUCKET_URI}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
    return
  fi

  gcloud storage buckets create "${BUCKET_URI}" \
    --project="${PROJECT_ID}" \
    --location="${BUCKET_LOCATION}" \
    --uniform-bucket-level-access \
    --public-access-prevention
}

ensure_service_account() {
  if gcloud iam service-accounts describe "${SERVICE_ACCOUNT_EMAIL}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
    return
  fi

  gcloud iam service-accounts create "${SERVICE_ACCOUNT_NAME}" \
    --project="${PROJECT_ID}" \
    --display-name="medical-distill Vertex service account"
}

add_project_role() {
  local member="$1"
  local role="$2"
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="${member}" \
    --role="${role}" \
    --quiet >/dev/null
}

add_bucket_role() {
  local member="$1"
  local role="$2"
  gcloud storage buckets add-iam-policy-binding "${BUCKET_URI}" \
    --member="${member}" \
    --role="${role}" >/dev/null
}

add_service_account_role() {
  local member="$1"
  local role="$2"
  gcloud iam service-accounts add-iam-policy-binding "${SERVICE_ACCOUNT_EMAIL}" \
    --member="${member}" \
    --role="${role}" \
    --project="${PROJECT_ID}" \
    --quiet >/dev/null
}

ensure_project
gcloud config set project "${PROJECT_ID}" >/dev/null

gcloud services enable \
  aiplatform.googleapis.com \
  compute.googleapis.com \
  iam.googleapis.com \
  iamcredentials.googleapis.com \
  storage.googleapis.com \
  --project="${PROJECT_ID}"

ensure_bucket
ensure_service_account

add_project_role "serviceAccount:${SERVICE_ACCOUNT_EMAIL}" "roles/aiplatform.user"
add_bucket_role "serviceAccount:${SERVICE_ACCOUNT_EMAIL}" "roles/storage.objectAdmin"

if [[ -n "${USER_PRINCIPAL}" ]]; then
  add_project_role "${USER_PRINCIPAL}" "roles/aiplatform.user"
  add_bucket_role "${USER_PRINCIPAL}" "roles/storage.objectAdmin"
  add_service_account_role "${USER_PRINCIPAL}" "roles/iam.serviceAccountUser"

  if [[ "${GRANT_OPEN_MODEL_ENABLE_ROLE}" == "1" ]]; then
    add_project_role "${USER_PRINCIPAL}" "roles/consumerprocurement.entitlementManager"
  fi
fi

cat <<EOF
GCP CLI setup complete.

Project:               ${PROJECT_ID}
Region:                ${LOCATION}
Bucket:                ${BUCKET_URI}
Service account:       ${SERVICE_ACCOUNT_EMAIL}
User principal:        ${USER_PRINCIPAL:-"(not set)"}

What this script handled through gcloud:
- project ${PROJECT_ACTION}
- API enablement
- bucket creation
- service account creation
- Vertex AI / GCS IAM bindings

ADC options:
- Local shell: gcloud auth application-default login
- Local shell: gcloud auth application-default set-quota-project ${PROJECT_ID}
- GCE VM: attach ${SERVICE_ACCOUNT_EMAIL} to the VM and ADC will use it automatically

Important exception:
- MaaS open-model enablement for Llama 3.3 still requires Model Garden console access.
  Console URL:
  https://console.cloud.google.com/vertex-ai/publishers/meta/model-garden/llama3-3?project=${PROJECT_ID}

After console enablement, you can continue with:
- python3 scripts/generate_synthetic.py --config configs/generation.vertex_llama33.ade.example.json
- python3 scripts/filter_dataset.py --config configs/filter.vertex_llama33.ade.example.json
- python3 scripts/prepare_sft_dataset.py --config configs/prepare_sft.vertex_llama33_to_llama31.ade.example.json
EOF
