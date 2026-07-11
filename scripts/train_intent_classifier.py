"""
scripts/train_intent_classifier.py — Fine-tune DistilBERT (+ LoRA) for intent
classification (v2 extension).

Reuses the maintenance_nlp recipe: DistilBERT base, LoRA adapters on the
attention projections, merge after training so inference needs only
transformers (no peft) — that is what src/router.classify_local loads.

CPU-friendly by design: 3 labels, ~170 short queries, 128-token inputs.
Trains in a few minutes on a laptop; no GPU required.

Requirements (local dev env, NOT in deploy requirements.txt):
    pip install torch transformers peft

Usage (from repo root):
    python scripts/build_intent_dataset.py            # if not built yet
    python scripts/train_intent_classifier.py         # trains + evaluates
    INTENT_CLASSIFIER=local uvicorn api.main:app      # serve with it

Outputs:
    models/intent_distilbert/          (gitignored — merged model + tokenizer)
    eval_results/intent_local_training.json
"""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "intent" / "intent_queries.jsonl"
OUT_DIR = ROOT / "models" / "intent_distilbert"
RESULTS = ROOT / "eval_results" / "intent_local_training.json"

BASE_MODEL = "distilbert-base-uncased"
LABELS = ["clarify", "interpret", "lookup"]  # alphabetical — keep in sync with src/router.py
SEED = 558
EPOCHS = 8
BATCH_SIZE = 16
LR = 2e-4  # LoRA-appropriate (adapters train hotter than full fine-tuning)


def load_split() -> tuple[list[dict], list[dict]]:
    rows = [json.loads(line) for line in DATA.read_text().splitlines() if line.strip()]
    train = [r for r in rows if r["split"] == "train"]
    holdout = [r for r in rows if r["split"] == "holdout"]
    return train, holdout


def main() -> None:
    import numpy as np
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train_rows, holdout_rows = load_split()
    print(f"train={len(train_rows)}  holdout={len(holdout_rows)}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    label2id = {label: i for i, label in enumerate(LABELS)}
    id2label = {i: label for label, i in label2id.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=len(LABELS), label2id=label2id, id2label=id2label
    )
    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_lin", "v_lin"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    class QueryDataset(Dataset):
        def __init__(self, rows: list[dict]):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            row = self.rows[idx]
            enc = tokenizer(
                row["query"], truncation=True, max_length=128,
                padding="max_length", return_tensors="pt",
            )
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": torch.tensor(label2id[row["intent"]]),
            }

    loader = DataLoader(QueryDataset(train_rows), batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = len(loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            out = model(**batch)
            out.loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += out.loss.item()
        print(f"epoch {epoch + 1}/{EPOCHS}  loss={epoch_loss / len(loader):.4f}")

    # Merge LoRA into the base weights → plain transformers model on disk
    merged = model.merge_and_unload()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"saved merged model → {OUT_DIR}")

    # Holdout evaluation (same numbers benchmark_intent.py will report)
    from src.stats import format_proportion  # late import: repo-root sys.path

    merged.eval()
    correct = 0
    confusion: dict[str, dict[str, int]] = {a: {b: 0 for b in LABELS} for a in LABELS}
    for row in holdout_rows:
        enc = tokenizer(row["query"], truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            pred = id2label[int(merged(**enc).logits.argmax(dim=-1).item())]
        confusion[row["intent"]][pred] += 1
        correct += int(pred == row["intent"])

    print("holdout accuracy:", format_proportion(correct, len(holdout_rows)))
    print("confusion (true → pred):", json.dumps(confusion, indent=2))

    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps({
        "base_model": BASE_MODEL,
        "method": "LoRA r=8 alpha=16 on q_lin/v_lin, merged",
        "epochs": EPOCHS,
        "train_n": len(train_rows),
        "holdout_n": len(holdout_rows),
        "holdout_correct": correct,
        "confusion": confusion,
        "labels": LABELS,
        "seed": SEED,
    }, indent=2) + "\n")
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(ROOT))
    main()
