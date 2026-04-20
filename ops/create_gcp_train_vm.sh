#!/usr/bin/env bash
set -euo pipefail

# Example:
#   PROJECT_ID=my-project \
#   ZONE=us-central1-a \
#   INSTANCE_NAME=medical-distill-train \
#   MACHINE_TYPE=a2-ultragpu-1g \
#   IMAGE_FAMILY=<gpu-ready-image-family> \
#   IMAGE_PROJECT=<image-project> \
#   REPO_URL=https://github.com/you/medical-distill.git \
#   REPO_REF=main \
#   TARGET_USER=$USER \
#   ./ops/create_gcp_train_vm.sh

: "${PROJECT_ID:?Set PROJECT_ID}"
: "${ZONE:?Set ZONE}"
: "${INSTANCE_NAME:?Set INSTANCE_NAME}"
: "${MACHINE_TYPE:?Set MACHINE_TYPE}"
: "${IMAGE_FAMILY:?Set IMAGE_FAMILY}"
: "${IMAGE_PROJECT:?Set IMAGE_PROJECT}"
: "${REPO_URL:?Set REPO_URL}"

REPO_REF="${REPO_REF:-main}"
TARGET_USER="${TARGET_USER:-${USER:-ubuntu}}"
BOOT_DISK_SIZE_GB="${BOOT_DISK_SIZE_GB:-200}"
STARTUP_SCRIPT_PATH="${STARTUP_SCRIPT_PATH:-ops/gcp_train_vm_startup.sh}"

gcloud compute instances create "${INSTANCE_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --machine-type="${MACHINE_TYPE}" \
  --maintenance-policy=TERMINATE \
  --boot-disk-size="${BOOT_DISK_SIZE_GB}GB" \
  --image-family="${IMAGE_FAMILY}" \
  --image-project="${IMAGE_PROJECT}" \
  --metadata-from-file="startup-script=${STARTUP_SCRIPT_PATH}" \
  --metadata="repo_url=${REPO_URL},repo_ref=${REPO_REF},target_user=${TARGET_USER}"
