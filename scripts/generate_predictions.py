from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from medical_distill.utils import extract_json_block, load_json, read_jsonl, render_messages_fallback, write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate predictions from a base model or LoRA adapter.")
    parser.add_argument("--config", required=True, help="Path to a JSON config file.")
    return parser.parse_args()


def maybe_build_quantization_config(model_config: dict[str, Any]):
    quantization = model_config.get("quantization")
    if quantization != "4bit":
        return {}

    try:
        import torch
        from transformers import BitsAndBytesConfig
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "4bit quantization requested, but BitsAndBytesConfig is unavailable."
        ) from exc

    return {
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        "device_map": "auto",
    }


def build_messages(system_prompt: str, prompt: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def render_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        fallback = render_messages_fallback(messages)
        return f"{fallback}\n\n<|assistant|>\n"


def main() -> None:
    args = parse_args()
    config = load_json(args.config)

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Inference dependencies are missing. Run `pip install -e .` first.") from exc

    model_config = config["model"]
    model_name = model_config["name_or_path"]
    model_kwargs = {
        "trust_remote_code": model_config.get("trust_remote_code", False),
    }
    model_kwargs.update(maybe_build_quantization_config(model_config))

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=model_config.get("trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    adapter_path = model_config.get("adapter_path")
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    rows = read_jsonl(config["input"]["path"])
    prompt_field = config["input"].get("prompt_field", "prompt")
    system_prompt = config["task"].get("system_prompt", "")

    output_rows: list[dict[str, Any]] = []
    for row in rows:
        prompt = row.get(prompt_field)
        if not isinstance(prompt, str) or not prompt.strip():
            continue

        messages = build_messages(system_prompt, prompt)
        prompt_text = render_prompt(tokenizer, messages)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config["generation"].get("max_new_tokens", 256),
                temperature=config["generation"].get("temperature", 0.0),
                do_sample=bool(config["generation"].get("temperature", 0.0) > 0),
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        raw_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        parsed = extract_json_block(raw_text)
        if not isinstance(parsed, dict):
            parsed = {"answer": "abstain", "raw_text": raw_text}

        output_row = dict(row)
        output_row["prediction"] = parsed
        output_row["raw_prediction_text"] = raw_text
        output_rows.append(output_row)

    write_jsonl(config["output"]["path"], output_rows)
    print(f"Wrote {len(output_rows)} predictions to {config['output']['path']}")


if __name__ == "__main__":
    main()
