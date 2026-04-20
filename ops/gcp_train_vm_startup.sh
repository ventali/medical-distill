#!/usr/bin/env bash
set -euo pipefail

# Intended for a CUDA-ready GCP VM startup script.
# Required metadata values:
#   repo_url: Git URL for this repository
# Optional metadata values:
#   repo_dir: checkout directory, default /opt/medical-distill
#   repo_ref: branch or commit to checkout
#   target_user: non-root user that should own the repo

metadata() {
  curl -sf -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" || true
}

TARGET_USER="$(metadata target_user)"
if [[ -z "${TARGET_USER}" ]]; then
  TARGET_USER="$(getent passwd 1000 | cut -d: -f1 || true)"
fi
if [[ -z "${TARGET_USER}" ]]; then
  TARGET_USER="ubuntu"
fi

REPO_URL="$(metadata repo_url)"
REPO_DIR="$(metadata repo_dir)"
REPO_REF="$(metadata repo_ref)"

if [[ -z "${REPO_URL}" ]]; then
  echo "Missing required instance metadata attribute: repo_url" >&2
  exit 1
fi

if [[ -z "${REPO_DIR}" ]]; then
  REPO_DIR="/opt/medical-distill"
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y git python3 python3-venv python3-pip tmux build-essential

mkdir -p "$(dirname "${REPO_DIR}")"
if [[ ! -d "${REPO_DIR}/.git" ]]; then
  git clone "${REPO_URL}" "${REPO_DIR}"
fi

cd "${REPO_DIR}"
git fetch --all --tags
if [[ -n "${REPO_REF}" ]]; then
  git checkout "${REPO_REF}"
fi

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[gpu,gcp]'

chown -R "${TARGET_USER}":"${TARGET_USER}" "${REPO_DIR}"

cat >/etc/profile.d/medical_distill_env.sh <<EOF
export MEDICAL_DISTILL_HOME="${REPO_DIR}"
EOF

cat <<EOF
medical-distill VM bootstrap complete.
Repo: ${REPO_DIR}
Next steps:
  1. sudo -u ${TARGET_USER} -H bash -lc 'cd ${REPO_DIR} && source .venv/bin/activate && nvidia-smi'
  2. Authenticate to Hugging Face if you need the Llama 3.1 8B student weights.
  3. Run the train or prediction script from the repo.
EOF

