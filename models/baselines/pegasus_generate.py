import json
import torch
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from rouge_score import rouge_scorer
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_FILE = "data/test.jsonl"
RESULT_FILE = "results/pegasus_results.json"

MODEL_NAME = "google/pegasus-cnn_dailymail"
MAX_INPUT_LENGTH = 1024
MAX_SUMMARY_LENGTH = 180
SAMPLE_LIMIT = 50   # same as SRLCSF (important!)


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def compute_rouge(preds, refs):
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for p, r in zip(preds, refs):
        s = scorer.score(r, p)
        for k in scores:
            scores[k].append(s[k].fmeasure)

    return {
        "ROUGE-1": sum(scores["rouge1"]) / len(scores["rouge1"]),
        "ROUGE-2": sum(scores["rouge2"]) / len(scores["rouge2"]),
        "ROUGE-L": sum(scores["rougeL"]) / len(scores["rougeL"]),
    }


def main():
    print("Device:", DEVICE)

    print("Loading PEGASUS model...")
    tokenizer = PegasusTokenizer.from_pretrained(MODEL_NAME)
    model = PegasusForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    print("Loading test data...")
    data = load_jsonl(TEST_FILE)[:SAMPLE_LIMIT]

    predictions, references = [], []

    print("Generating PEGASUS summaries...")
    for sample in tqdm(data):
        articles = sample["articles"]
        reference = sample["summary"]

        # PEGASUS expects concatenated documents
        input_text = " ".join(articles)

        inputs = tokenizer(
            input_text,
            truncation=True,
            padding="longest",
            max_length=MAX_INPUT_LENGTH,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            summary_ids = model.generate(
                **inputs,
                max_length=MAX_SUMMARY_LENGTH,
                num_beams=4,
                early_stopping=True
            )

        summary = tokenizer.decode(
            summary_ids[0], skip_special_tokens=True
        )

        predictions.append(summary)
        references.append(reference)

    print("Computing ROUGE...")
    scores = compute_rouge(predictions, references)

    with open(RESULT_FILE, "w") as f:
        json.dump(scores, f, indent=2)

    print("\n===== PEGASUS RESULTS =====")
    for k, v in scores.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
