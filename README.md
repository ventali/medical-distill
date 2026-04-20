# medical-distill

Starter scaffold for biomedical LLM distillation with a teacher-student workflow:

- generate synthetic supervision with a strong teacher
- filter and normalize that supervision
- convert it into an SFT dataset
- fine-tune a smaller student model
- score predictions on a held-out evaluation set

This repo intentionally starts with response distillation and task-specific evaluation. That is the fastest path to a reliable v1 for ADE extraction, clinical IE, or biomedical QA.

## Recommended v1

- Task: adverse drug event extraction or binary ADE QA
- Teacher: Vertex AI managed `llama-3.3-70b-instruct-maas`
- Student: `meta-llama/Llama-3.1-8B-Instruct`
- Training: supervised fine-tuning first, preference optimization later

## Recommended GCP Path

The repo now includes a concrete managed-teacher setup for:

- Teacher: Vertex AI `llama-3.3-70b-instruct-maas`
- Student: `meta-llama/Llama-3.1-8B-Instruct`
- Generation: managed Vertex AI OpenAI-compatible endpoint with Google ADC
- Fine-tuning: run the student training script on a GCP GPU VM

Relevant configs:

- [configs/generation.vertex_llama33.ade.example.json](/Users/ventalitan/medical-distill/configs/generation.vertex_llama33.ade.example.json)
- [configs/filter.vertex_llama33.ade.example.json](/Users/ventalitan/medical-distill/configs/filter.vertex_llama33.ade.example.json)
- [configs/prepare_sft.vertex_llama33_to_llama31.ade.example.json](/Users/ventalitan/medical-distill/configs/prepare_sft.vertex_llama33_to_llama31.ade.example.json)
- [configs/prepare_vertex_tuning.llama31.example.json](/Users/ventalitan/medical-distill/configs/prepare_vertex_tuning.llama31.example.json)
- [configs/train.gcp.llama31.student.example.json](/Users/ventalitan/medical-distill/configs/train.gcp.llama31.student.example.json)
- [configs/predict.gcp.llama31.student.example.json](/Users/ventalitan/medical-distill/configs/predict.gcp.llama31.student.example.json)
- [configs/eval.vertex_llama33_to_llama31.ade.example.json](/Users/ventalitan/medical-distill/configs/eval.vertex_llama33_to_llama31.ade.example.json)
- [configs/vertex_tuning.llama31.student.example.json](/Users/ventalitan/medical-distill/configs/vertex_tuning.llama31.student.example.json)
- [configs/gcs_upload.vertex_tuning.example.json](/Users/ventalitan/medical-distill/configs/gcs_upload.vertex_tuning.example.json)

The older same-family local-teacher path is still available if you want to self-host a Llama teacher:

- [configs/generation.llama31.ade.example.json](/Users/ventalitan/medical-distill/configs/generation.llama31.ade.example.json)
- [configs/filter.llama31.ade.example.json](/Users/ventalitan/medical-distill/configs/filter.llama31.ade.example.json)
- [configs/prepare_sft.llama31.ade.example.json](/Users/ventalitan/medical-distill/configs/prepare_sft.llama31.ade.example.json)
- [configs/train.llama31.student.example.json](/Users/ventalitan/medical-distill/configs/train.llama31.student.example.json)
- [configs/predict.llama31.student.example.json](/Users/ventalitan/medical-distill/configs/predict.llama31.student.example.json)
- [configs/eval.llama31.ade.example.json](/Users/ventalitan/medical-distill/configs/eval.llama31.ade.example.json)

## Repo layout

```text
configs/                 Example configs for generation, filtering, training, and evaluation
data/raw/                Seed inputs and gold examples
data/interim/            Raw teacher outputs
data/processed/          Filtered datasets and SFT-ready data
evals/                   Smoke-test eval sets or gold evaluation files
outputs/                 Training outputs, metrics, and model artifacts
scripts/                 Entry points for each stage
src/medical_distill/     Shared utilities and metrics
```

## Quickstart

1. Create a virtual environment and install the package:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
```

If you plan to use 4-bit QLoRA on a Linux/CUDA machine, install the optional GPU extras too:

```bash
python3 -m pip install -e '.[gpu]'
```

If you want Vertex SDK helpers available as well, install the GCP extras:

```bash
python3 -m pip install -e '.[gcp]'
```

2. Prepare Vertex AI access for the managed teacher:

```bash
gcloud services enable aiplatform.googleapis.com
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT=your-project-id
```

Before the first request, grant your project access to managed open models in Vertex AI Model Garden.
If you run generation on a GCP VM, an attached service account with Vertex AI access can replace `gcloud auth application-default login`.

## CLI-First GCP Setup

If you want to provision as much of the environment as possible through `gcloud`, use:

- [ops/setup_gcp_vertex_cli.sh](/Users/ventalitan/medical-distill/ops/setup_gcp_vertex_cli.sh)

Example:

```bash
chmod +x ops/setup_gcp_vertex_cli.sh

PROJECT_ID=your-project-id \
BUCKET_NAME=your-unique-bucket \
LOCATION=us-central1 \
CREATE_PROJECT=0 \
GRANT_OPEN_MODEL_ENABLE_ROLE=0 \
./ops/setup_gcp_vertex_cli.sh
```

What this script handles through the CLI:

- project selection, and optional project creation
- optional billing link if you pass `BILLING_ACCOUNT`
- API enablement
- bucket creation
- service account creation
- Vertex AI / GCS IAM bindings

What still is not fully CLI-driven in the current Google docs:

- enabling MaaS open-model access for `llama-3.3-70b-instruct-maas`

For that step, Google currently documents going to the Model Garden model page and enabling the model there. If you want the script to also grant the permission needed to enable open models, set:

```bash
GRANT_OPEN_MODEL_ENABLE_ROLE=1
```

That adds `roles/consumerprocurement.entitlementManager` to your active user principal, but you still need to complete the enablement/consent step in the console.

3. Generate synthetic teacher outputs with Vertex AI:

```bash
python3 scripts/generate_synthetic.py --config configs/generation.vertex_llama33.ade.example.json
```

The Vertex config uses Google ADC and the OpenAI-compatible Vertex endpoint automatically, so no Hugging Face token or local `vllm` server is needed.

4. Filter low-quality or unsupported teacher outputs:

```bash
python3 scripts/filter_dataset.py --config configs/filter.vertex_llama33.ade.example.json
```

5. Convert the filtered set into chat-style SFT data:

```bash
python3 scripts/prepare_sft_dataset.py --config configs/prepare_sft.vertex_llama33_to_llama31.ade.example.json
```

After that, choose one of two student-tuning routes.

## Route A: GCP GPU VM

Use this when you want full control over LoRA/QLoRA, local checkpoints, and custom training code.

Example helper scripts:

- [ops/create_gcp_train_vm.sh](/Users/ventalitan/medical-distill/ops/create_gcp_train_vm.sh)
- [ops/gcp_train_vm_startup.sh](/Users/ventalitan/medical-distill/ops/gcp_train_vm_startup.sh)

Create a GPU VM and attach the startup bootstrap:

```bash
chmod +x ops/create_gcp_train_vm.sh

PROJECT_ID=your-project-id \
ZONE=us-central1-a \
INSTANCE_NAME=medical-distill-train \
MACHINE_TYPE=a2-ultragpu-1g \
IMAGE_FAMILY=your-gpu-ready-image-family \
IMAGE_PROJECT=your-image-project \
REPO_URL=https://github.com/you/medical-distill.git \
REPO_REF=main \
TARGET_USER=$USER \
./ops/create_gcp_train_vm.sh
```

The helper intentionally leaves the image family and image project explicit because those change over time. Use a CUDA-ready image in your GCP project or a current Deep Learning VM image.

Once the VM is ready, train the student:

```bash
python3 scripts/train_student.py --config configs/train.gcp.llama31.student.example.json
```

The sample training config uses 4-bit loading. If your environment does not support `bitsandbytes`, set `"quantization": null` in the training config before running.
The student weights still come from Hugging Face in this scaffold, so the training VM needs access to `meta-llama/Llama-3.1-8B-Instruct`.

Then generate predictions and score them:

```bash
python3 scripts/generate_predictions.py --config configs/predict.gcp.llama31.student.example.json
python3 scripts/run_eval.py --config configs/eval.vertex_llama33_to_llama31.ade.example.json
```

## Route B: Managed Vertex tuning job

Use this when you want the 8B student to stay inside Vertex AI as much as possible.

1. Convert the local SFT dataset into a strict Vertex tuning JSONL:

```bash
python3 scripts/prepare_vertex_tuning_dataset.py --config configs/prepare_vertex_tuning.llama31.example.json
```

2. Upload the train and validation files to GCS:

```bash
python3 scripts/upload_to_gcs.py --config configs/gcs_upload.vertex_tuning.example.json
```

3. Submit the Vertex tuning job:

```bash
python3 scripts/submit_vertex_tuning_job.py --config configs/vertex_tuning.llama31.student.example.json
```

The tuning submission script uses the current preview open-model tuning path in the Vertex AI SDK. Fill in your bucket URIs and display name before launching.

## Local or Self-Hosted Teacher Path

If you still want to self-host the teacher instead of using Vertex AI:

```bash
python3 -m pip install -e '.[serve]'
export VLLM_API_KEY=token-abc123
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --api-key "$VLLM_API_KEY" \
  --dtype auto \
  --generation-config vllm

python3 scripts/generate_synthetic.py --config configs/generation.llama31.ade.example.json
python3 scripts/filter_dataset.py --config configs/filter.llama31.ade.example.json
python3 scripts/prepare_sft_dataset.py --config configs/prepare_sft.llama31.ade.example.json
python3 scripts/train_student.py --config configs/train.llama31.student.example.json
python3 scripts/generate_predictions.py --config configs/predict.llama31.student.example.json
python3 scripts/run_eval.py --config configs/eval.llama31.ade.example.json
```

## Working conventions

- Keep a small, clinician-reviewed eval set separate from all synthetic data.
- Store raw teacher outputs before filtering so audits are possible.
- Require evidence and confidence fields in teacher responses for biomedical tasks.
- Start with short, evidence-backed justifications instead of long chain-of-thought style rationales.
- Add DPO or other preference optimization only after the SFT baseline is stable.
- On GCP, prefer attached service accounts or ADC over long-lived API keys for teacher generation.

## Suggested next steps

- Replace the smoke-test eval file with a real held-out ADE set.
- Add task-specific metrics for span/entity extraction if ADE moves beyond binary QA.
- Add task-specific prompt variants for hard negatives, ambiguity, and abstention cases.
- Add clinician review sampling before any deployment-oriented use.
