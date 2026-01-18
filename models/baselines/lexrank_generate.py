import json
from tqdm import tqdm
from rouge_score import rouge_scorer
from lexrank import LexRank
from nltk.tokenize import sent_tokenize

TEST_FILE = "data/test.jsonl"
RESULT_FILE = "results/lexrank_results.json"
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
    print("Running LexRank baseline...")
    data = load_jsonl(TEST_FILE)[:SAMPLE_LIMIT]

    preds, refs = [], []

    for sample in tqdm(data):
        documents = sample["articles"]
        sentences = []

        for doc in documents:
            sentences.extend(sent_tokenize(doc))

        if len(sentences) < 5:
            preds.append(documents[0][:500])
            refs.append(sample["summary"])
            continue

        lxr = LexRank(sentences)
        summary_sentences = lxr.get_summary(
            sentences, summary_size=5
        )

        summary = " ".join(summary_sentences)
        preds.append(summary)
        refs.append(sample["summary"])

    scores = compute_rouge(preds, refs)

    with open(RESULT_FILE, "w") as f:
        json.dump(scores, f, indent=2)

    print("\n===== LEXRANK RESULTS =====")
    for k, v in scores.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
