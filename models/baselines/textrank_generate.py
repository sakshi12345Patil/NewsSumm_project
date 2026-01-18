import json
from tqdm import tqdm
from rouge_score import rouge_scorer
from summa.summarizer import summarize

TEST_FILE = "data/test.jsonl"
RESULT_FILE = "results/textrank_results.json"
SAMPLE_LIMIT = 50


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def compute_rouge(preds, refs):
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    scores = {"r1": [], "r2": [], "rl": []}

    for p, r in zip(preds, refs):
        s = scorer.score(r, p)
        scores["r1"].append(s["rouge1"].fmeasure)
        scores["r2"].append(s["rouge2"].fmeasure)
        scores["rl"].append(s["rougeL"].fmeasure)

    return {
        "ROUGE-1": sum(scores["r1"]) / len(scores["r1"]),
        "ROUGE-2": sum(scores["r2"]) / len(scores["r2"]),
        "ROUGE-L": sum(scores["rl"]) / len(scores["rl"]),
    }


def main():
    print("Running TextRank baseline...")
    data = load_jsonl(TEST_FILE)[:SAMPLE_LIMIT]

    preds, refs = [], []

    for sample in tqdm(data):
        text = " ".join(sample["articles"])
        summary = summarize(text, ratio=0.2)

        if len(summary.strip()) == 0:
            summary = text[:500]

        preds.append(summary)
        refs.append(sample["summary"])

    scores = compute_rouge(preds, refs)

    with open(RESULT_FILE, "w") as f:
        json.dump(scores, f, indent=2)

    print("\n===== TEXTRANK RESULTS =====")
    for k, v in scores.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
