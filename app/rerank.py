"""Step 6: a cross-encoder reads (question, chunk) together and gives a careful 0-1 score."""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class Reranker:
    def __init__(self, model_name: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).eval().to(self.device)

    @torch.no_grad()
    def score(self, query: str, passages, batch_size: int = 16):
        out = []
        for i in range(0, len(passages), batch_size):
            batch = passages[i:i + batch_size]
            enc = self.tokenizer([query] * len(batch), batch, padding=True, truncation=True,
                                 max_length=512, return_tensors="pt").to(self.device)
            logits = self.model(**enc).logits.view(-1)
            out.extend(torch.sigmoid(logits).cpu().tolist())  # sigmoid -> 0..1
        return out
