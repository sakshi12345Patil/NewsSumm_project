import json
import torch
from transformers import LEDTokenizer, LEDForConditionalGeneration
from rouge_score import rouge_scorer
from tqdm import tqdm
import os

# =========================
# Configuration
# =========================

MODEL_NAME = "allenai/led-base-16384"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_INPUT_LENGTH = 4096     # safe for CPU (not full 16k)
MAX_SUMMARY_LENGTH = 180
NUM_BEAMS = 4
NUM_SAMPLES = 50            # professional CPU evaluation size

TEST_FILE = "data/test.jsonl"
RESULT_DIR = "results"
RESULT_FILE = os.path.join(RESULT_DIR, "led_results.json")

os.makedirs(RESULT_DIR, exist_ok=True)

# =========================
# Utility functions
# =========================

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

# =========================
# Main evaluation
# =========================

def main():
    print(f"Device: {DEVICE}")
    print("Loading LED tokenizer and model...")

    tokenizer = LEDTokenizer.from_pretrained(MODEL_NAME)
    model = LEDForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()

    print("Loading test data...")
    data = load_jsonl(TEST_FILE, limit=NUM_SAMPLES)

    predictions = []
    references = []

    print("Generating summaries...")

    for sample in tqdm(data):
        # concatenate multi-document cluster
        input_text = " ".join(sample["articles"])

        inputs = tokenizer(
            input_text,
            truncation=True,
            max_length=MAX_INPUT_LENGTH,
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"].to(DEVICE)

        # Global attention on first token (LED requirement)
        global_attention_mask = torch.zeros_like(input_ids)
        global_attention_mask[:, 0] = 1

        with torch.no_grad():
            summary_ids = model.generate(
                input_ids,
                global_attention_mask=global_attention_mask,
                max_length=MAX_SUMMARY_LENGTH,
                num_beams=NUM_BEAMS,
                early_stopping=True
            )

        summary_text = tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        predictions.append(summary_text)
        references.append(sample["summary"])

    print("Computing ROUGE scores...")
    rouge_scores = compute_rouge(predictions, references)

    results = {
        "model": MODEL_NAME,
        "num_samples": NUM_SAMPLES,
        "max_input_length": MAX_INPUT_LENGTH,
        "max_summary_length": MAX_SUMMARY_LENGTH,
        "num_beams": NUM_BEAMS,
        "device": DEVICE,
        "rouge": rouge_scores
    }

    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n===== LED BASELINE RESULTS =====")
    for k, v in rouge_scores.items():
        print(f"{k}: {v}")

    print(f"\nResults saved to: {RESULT_FILE}")


if __name__ == "__main__":
    main()
