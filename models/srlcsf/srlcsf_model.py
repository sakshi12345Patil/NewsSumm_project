import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class SentenceEncoder(nn.Module):
    """
    Encodes individual sentences using a pretrained transformer.
    """
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.last_hidden_state.mean(dim=1)


class RedundancyAwareAggregator(nn.Module):
    """
    Reduces redundancy across sentence representations.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            hidden_size,
            num_heads=4,
            batch_first=True
        )

    def forward(self, sentence_embeddings):
        attn_output, _ = self.attn(
            sentence_embeddings,
            sentence_embeddings,
            sentence_embeddings
        )
        return attn_output.mean(dim=1)


class SRLCSF(nn.Module):
    """
    Structured Redundancy-Aware Long-Context Summarization Framework
    """
    def __init__(self):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.sent_encoder = SentenceEncoder()
        self.aggregator = RedundancyAwareAggregator(hidden_size=384)

        self.cluster_proj = nn.Linear(384, 384)

    def encode_sentences(self, sentences, device):
        enc = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)

        embeddings = self.sent_encoder(
            enc["input_ids"],
            enc["attention_mask"]
        )
        return embeddings

    def forward(self, documents, device):
        all_sentences = []
        for doc in documents:
            sents = [s.strip() for s in doc.split(".") if len(s.strip()) > 10]
            all_sentences.extend(sents)

        if len(all_sentences) == 0:
            return None

        sentence_embeddings = self.encode_sentences(all_sentences, device)
        sentence_embeddings = sentence_embeddings.unsqueeze(0)

        doc_rep = self.aggregator(sentence_embeddings)
        cluster_rep = self.cluster_proj(doc_rep)

        return cluster_rep


if __name__ == "__main__":
    model = SRLCSF()
    print("SRLCSF loaded successfully")
