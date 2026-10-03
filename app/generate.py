
import re

REFUSAL = "I could not find this in the documents."

SYSTEM_PROMPT = (
    "You answer questions using ONLY the numbered context passages provided. "
    "Cite the passages you use with their numbers, like [1] or [2]. "
    f"If the context does not contain the answer, reply exactly: {REFUSAL}"
)


def build_context(chunks) -> str:
    return "\n\n".join(
        f"[{i}] (source: {c['doc_id']}, page {c['page']})\n{c['text']}"
        for i, c in enumerate(chunks, start=1)
    )


def extract_citations(answer: str, n_context: int):
    """Returns (valid_citation_numbers, invalid_citation_numbers)."""
    nums = sorted({int(x) for x in re.findall(r"\[(\d+)\]", answer)})
    valid = [n for n in nums if 1 <= n <= n_context]
    invalid = [n for n in nums if n not in valid]
    return valid, invalid


class Generator:
    def __init__(self, backend: str, model_name: str):
        self.backend = backend
        if backend == "local":
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)

    def chat(self, messages, max_new_tokens: int = 256) -> str:
        if self.backend != "local":
            raise RuntimeError("chat() needs LLM_BACKEND=local")
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def answer(self, question: str, chunks) -> str:
        if self.backend == "extractive":  
            return f"{chunks[0]['text']} [1]"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{build_context(chunks)}\n\nQuestion: {question}"},
        ]
        return self.chat(messages)
