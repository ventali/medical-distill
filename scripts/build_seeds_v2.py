"""Build v2 ADE seed + eval files from ade_corpus_v2.

v2 adds:
- Larger seed pool (3000): 1200 positive + 1200 easy-negative + 600 hard-negative.
- Hard negatives are classification label=0 rows whose text mentions a drug name
  from the positive split's drug vocabulary, so the student sees "drug mentioned,
  no ADE" cases instead of always abstaining when a drug is named.
- Eval file (200 rows) is rewritten to contain both the raw `prompt` field (for
  generate_predictions.py) and an SFT-compatible `messages` list (for
  train_student.py's eval-epoch LM loss), so load_dataset doesn't barf on a
  schema mismatch between train and eval splits.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path


SYS_PROMPT = (
    "You are a careful biomedical assistant. For each case, return a compact "
    "JSON answer grounded in the provided evidence. If the evidence is "
    "insufficient, abstain."
)


def make_prompt(text: str) -> str:
    return (
        f"Case: {text}\n\n"
        "Is this consistent with a possible adverse drug event? "
        "Identify the drug and event if so, or abstain if the evidence is insufficient."
    )


def norm_token(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def build() -> None:
    from datasets import load_dataset

    random.seed(23)

    pos_ds = load_dataset(
        "ade-benchmark-corpus/ade_corpus_v2",
        "Ade_corpus_v2_drug_ade_relation",
        split="train",
    )
    cls_ds = load_dataset(
        "ade-benchmark-corpus/ade_corpus_v2",
        "Ade_corpus_v2_classification",
        split="train",
    )

    pos_rows = [
        {"text": r["text"], "drug": r["drug"], "effect": r["effect"]}
        for r in pos_ds
    ]
    seen = set()
    pos_unique = []
    for r in pos_rows:
        if r["text"] in seen:
            continue
        seen.add(r["text"])
        pos_unique.append(r)
    pos_rows = pos_unique
    random.shuffle(pos_rows)

    neg_rows = [{"text": r["text"]} for r in cls_ds if r["label"] == 0]
    seen = set()
    neg_unique = []
    for r in neg_rows:
        if r["text"] in seen:
            continue
        seen.add(r["text"])
        neg_unique.append(r)
    neg_rows = neg_unique
    random.shuffle(neg_rows)

    # Drug vocabulary from the positive split, tokenized to a case-insensitive set.
    drug_vocab = {norm_token(r["drug"]) for r in pos_rows if r["drug"]}
    drug_vocab = {d for d in drug_vocab if len(d) >= 4}  # drop noisy short tokens
    print(f"drug vocab size: {len(drug_vocab)}")

    def mentions_drug(text: str) -> bool:
        tokens = {norm_token(t) for t in re.split(r"\W+", text) if t}
        return bool(tokens & drug_vocab)

    # Hold out: 100 pos + 100 neg for eval (same as v1, for comparability).
    eval_pos = pos_rows[:100]
    eval_neg = neg_rows[:100]

    # Seed pool: exclude the eval holdout rows.
    remaining_pos = pos_rows[100:]
    remaining_neg = neg_rows[100:]

    seed_pos = remaining_pos[:1200]

    # Split remaining negatives into hard (drug-mentioning) and easy buckets.
    hard_negs = [r for r in remaining_neg if mentions_drug(r["text"])]
    easy_negs = [r for r in remaining_neg if not mentions_drug(r["text"])]
    print(f"hard-negative pool: {len(hard_negs)}  easy-negative pool: {len(easy_negs)}")

    seed_hard_neg = hard_negs[:600]
    seed_easy_neg = easy_negs[:1200]

    # Build seed records.
    seed_records = []
    for r in seed_pos:
        seed_records.append({
            "prompt": make_prompt(r["text"]),
            "metadata": {
                "task": "ade_binary_qa",
                "source": "ade_corpus_v2.drug_ade_relation",
                "class": "positive",
                "gold_label": "yes",
                "gold_drug": r["drug"],
                "gold_effect": r["effect"],
            },
        })
    for r in seed_hard_neg:
        seed_records.append({
            "prompt": make_prompt(r["text"]),
            "metadata": {
                "task": "ade_binary_qa",
                "source": "ade_corpus_v2.classification_label0",
                "class": "hard_negative",
                "gold_label": "no",
            },
        })
    for r in seed_easy_neg:
        seed_records.append({
            "prompt": make_prompt(r["text"]),
            "metadata": {
                "task": "ade_binary_qa",
                "source": "ade_corpus_v2.classification_label0",
                "class": "easy_negative",
                "gold_label": "no",
            },
        })
    random.shuffle(seed_records)
    for i, r in enumerate(seed_records, 1):
        r["id"] = f"seed-{i:05d}"

    # Eval file: prompt + messages + reference, compatible with both predict and train-eval.
    eval_records = []
    for r in eval_pos:
        gold = {
            "answer": "yes",
            "drug": r["drug"],
            "event": r["effect"],
            "evidence": "",
            "short_justification": "",
            "confidence": 1.0,
        }
        prompt = make_prompt(r["text"])
        eval_records.append({
            "prompt": prompt,
            "messages": [
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": json.dumps(gold, ensure_ascii=False, indent=2, sort_keys=True)},
            ],
            "reference": gold,
            "metadata": {"teacher_model": "gold_ade_corpus_v2"},
        })
    for r in eval_neg:
        gold = {
            "answer": "no",
            "drug": "",
            "event": "",
            "evidence": "",
            "short_justification": "",
            "confidence": 1.0,
        }
        prompt = make_prompt(r["text"])
        eval_records.append({
            "prompt": prompt,
            "messages": [
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": json.dumps(gold, ensure_ascii=False, indent=2, sort_keys=True)},
            ],
            "reference": gold,
            "metadata": {"teacher_model": "gold_ade_corpus_v2"},
        })
    random.shuffle(eval_records)
    for i, r in enumerate(eval_records, 1):
        r["id"] = f"eval-{i:05d}"

    seed_path = Path("data/raw/ade_seed_examples.jsonl")
    eval_path = Path("evals/ade_eval.jsonl")
    seed_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in seed_records) + "\n"
    )
    eval_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in eval_records) + "\n"
    )
    print(f"wrote {seed_path} with {len(seed_records)} seeds")
    print(f"wrote {eval_path} with {len(eval_records)} eval rows")


if __name__ == "__main__":
    build()
