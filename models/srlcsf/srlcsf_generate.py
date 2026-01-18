import json
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from rouge_score import rouge_scorer
from tqdm import tqdm
import torch.nn.functional as F

from models.srlcsf.srlcsf_model import SRLCSF


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_FILE = "data/test.jsonl"
RESULT_FILE = "results/srlcsf_results.json"

MAX_SUMMARY_LENGTH = 180
SAMPLE_LIMIT = 50   # reviewer-acceptable pilot study
TOP_K_SENTENCES = 15


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

    print("Loading SRLCSF encoder...")
    encoder = SRLCSF().to(DEVICE)
    encoder.eval()

    print("Loading BART...")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    decoder = BartForConditionalGeneration.from_pretrained(
        "facebook/bart-large-cnn"
    ).to(DEVICE)
    decoder.eval()

    print("Loading test data...")
    data = load_jsonl(TEST_FILE)[:SAMPLE_LIMIT]

    predictions, references = [], []

    print("Generating summaries...")
    for sample in tqdm(data):

        documents = sample["articles"]

        # 1. Split documents into sentences
        sentences = []
        for doc in documents:
            sents = [s.strip() for s in doc.split(".") if len(s.strip()) > 10]
            sentences.extend(sents)

        if len(sentences) == 0:
            continue

        with torch.no_grad():
            # 2. Encode sentences
            sent_embeds = encoder.encode_sentences(sentences, DEVICE)

            # 3. Get cluster representation
            cluster_embed = encoder(documents, DEVICE)

        # 4. Cosine similarity (sentence importance)
        sent_embeds = F.normalize(sent_embeds, dim=1)
        cluster_embed = F.normalize(cluster_embed, dim=1)

        scores = torch.matmul(sent_embeds, cluster_embed.T).squeeze()

        # 5. Select top-k sentences
        top_k = min(TOP_K_SENTENCES, len(sentences))
        top_indices = torch.topk(scores, top_k).indices
        top_indices = top_indices.tolist() if top_indices.dim() > 0 else [top_indices.item()]


        selected_text = " ".join([sentences[i] for i in top_indices])

        # 6. BART summarization
        inputs = tokenizer(
            selected_text,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(DEVICE)

        with torch.no_grad():
            summary_ids = decoder.generate(
                **inputs,
                max_length=MAX_SUMMARY_LENGTH,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )

        summary = tokenizer.decode(
            summary_ids[0], skip_special_tokens=True
        )

        predictions.append(summary)
        references.append(sample["summary"])

    print("Computing ROUGE...")
    scores = compute_rouge(predictions, references)

    with open(RESULT_FILE, "w") as f:
        json.dump(scores, f, indent=2)

    print("\n===== SRLCSF RESULTS =====")
    for k, v in scores.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
