"""
BART Large CNN baseline for Multi-Document News Summarization
Dataset: NewsSumm (Indian English)
Author: Sakshi Dnyaneshwar Patil
"""

import os
import json
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from rouge_score import rouge_scorer
from tqdm import tqdm

# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "facebook/bart-large-cnn"
MAX_INPUT_LENGTH = 1024
MAX_SUMMARY_LENGTH = 180
BATCH_SIZE = 2               # keep small for Mac/CPU
NUM_SAMPLES = 500            # evaluation subset (paper-safe)

TEST_FILE = "data/test.jsonl"
RESULT_FILE = "results/bart_results.json"

# --------------------------------------


def load_jsonl(path, limit=None):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            data.append(json.loads(line))
    return data


def compute_rouge(preds, refs):
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

    r1, r2, rl = [], [], []

    for p, r in zip(preds, refs):
        scores = scorer.score(r, p)
        r1.append(scores["rouge1"].fmeasure)
        r2.append(scores["rouge2"].fmeasure)
        rl.append(scores["rougeL"].fmeasure)

    return {
        "ROUGE-1": round(sum(r1) / len(r1), 4),
        "ROUGE-2": round(sum(r2) / len(r2), 4),
        "ROUGE-L": round(sum(rl) / len(rl), 4),
    }


def main():
    os.makedirs("results", exist_ok=True)

    print(f"Device: {DEVICE}")
    print("Loading BART model...")

    tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
    model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()

    print("Loading test data...")
    data = load_jsonl(TEST_FILE, limit=NUM_SAMPLES)

    predictions = []
    references = []

    print("Generating summaries...")

    for i in tqdm(range(0, len(data), BATCH_SIZE)):
        batch = data[i : i + BATCH_SIZE]

        texts = [
            " ".join(sample["articles"]) for sample in batch
        ]

        refs = [sample["summary"] for sample in batch]

        inputs = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=MAX_INPUT_LENGTH,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            summary_ids = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=MAX_SUMMARY_LENGTH,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )

        decoded = tokenizer.batch_decode(
            summary_ids, skip_special_tokens=True
        )

        predictions.extend(decoded)
        references.extend(refs)

    print("Computing ROUGE scores...")
    scores = compute_rouge(predictions, references)

    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)

    print("\n===== BART BASELINE RESULTS =====")
    for k, v in scores.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
