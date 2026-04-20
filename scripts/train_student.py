from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from medical_distill.utils import json_dumps, load_json, render_messages_fallback  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a student model with SFT.")
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


def format_messages(example: dict[str, Any], tokenizer) -> dict[str, str]:
    messages = example.get("messages")
    if messages:
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            text = render_messages_fallback(messages)
    elif example.get("prompt") and example.get("reference") is not None:
        fallback_messages = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": json_dumps(example["reference"])},
        ]
        try:
            text = tokenizer.apply_chat_template(
                fallback_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            text = render_messages_fallback(fallback_messages)
    else:
        text = example.get("text", "")
    return {"text": text}


def main() -> None:
    args = parse_args()
    config = load_json(args.config)

    try:
        from datasets import load_dataset
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Training dependencies are missing. Run `pip install -e .` first.") from exc

    dataset_files = {"train": config["data"]["train_path"]}
    eval_path = config["data"].get("eval_path")
    if eval_path:
        dataset_files["eval"] = eval_path

    dataset = load_dataset("json", data_files=dataset_files)
    model_name = config["model"]["name_or_path"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=config["model"].get("trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": config["model"].get("trust_remote_code", False),
    }
    model_kwargs.update(maybe_build_quantization_config(config["model"]))
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if config["training"].get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if config["model"].get("quantization") == "4bit":
        model = prepare_model_for_kbit_training(model)

    if config["lora"].get("enabled", False):
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=config["lora"].get("r", 16),
            lora_alpha=config["lora"].get("alpha", 32),
            lora_dropout=config["lora"].get("dropout", 0.05),
            target_modules=config["lora"].get("target_modules"),
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    dataset = dataset.map(lambda row: format_messages(row, tokenizer))

    max_length = int(config["data"].get("max_length", 2048))

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    training_args = TrainingArguments(
        output_dir=config["training"]["output_dir"],
        num_train_epochs=config["training"].get("num_train_epochs", 1),
        learning_rate=config["training"].get("learning_rate", 2e-4),
        per_device_train_batch_size=config["training"].get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=config["training"].get("gradient_accumulation_steps", 8),
        warmup_ratio=config["training"].get("warmup_ratio", 0.03),
        weight_decay=config["training"].get("weight_decay", 0.01),
        logging_steps=config["training"].get("logging_steps", 10),
        save_strategy=config["training"].get("save_strategy", "epoch"),
        evaluation_strategy="epoch" if "eval" in tokenized else "no",
        report_to="none",
        bf16=config["training"].get("bf16", False),
        fp16=config["training"].get("fp16", False),
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["eval"] if "eval" in tokenized else None,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(config["training"]["output_dir"])
    tokenizer.save_pretrained(config["training"]["output_dir"])


if __name__ == "__main__":
    main()
