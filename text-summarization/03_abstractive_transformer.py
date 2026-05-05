"""
Day 3: Abstractive summarization with pre-trained transformers.

Unlike TextRank (extractive - just picks existing sentences), abstractive
summarization paraphrases. We compare two off-the-shelf seq2seq models:
  - facebook/bart-large-cnn  (fine-tuned on CNN/DailyMail)
  - google/pegasus-xsum      (fine-tuned on XSum, more terse style)

Both are loaded via HuggingFace transformers. Inputs longer than the model's
max position get split into chunks; chunk summaries are then concatenated and
optionally re-summarized once (recursive collapse) for very long documents.
"""

import time
from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# Quick-switch registry; add more models here as needed.
MODEL_REGISTRY = {
    "bart": "facebook/bart-large-cnn",
    "pegasus": "google/pegasus-xsum",
    "t5-small": "t5-small",
}


@dataclass
class SummaryResult:
    text: str
    model: str
    runtime_s: float
    input_tokens: int
    output_tokens: int


class AbstractiveSummarizer:
    """Wraps a HF seq2seq model with sensible chunking + decoding defaults."""

    def __init__(self, model_key: str = "bart", device: Optional[str] = None):
        if model_key not in MODEL_REGISTRY:
            raise ValueError(f"unknown model_key={model_key!r}; pick from {list(MODEL_REGISTRY)}")
        self.model_name = MODEL_REGISTRY[model_key]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        # leave a little headroom under the model's hard limit
        self.max_input_tokens = min(getattr(self.model.config, "max_position_embeddings", 1024), 1024) - 16

    def _chunk_by_tokens(self, text: str) -> List[str]:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= self.max_input_tokens:
            return [text]
        chunks = []
        for start in range(0, len(ids), self.max_input_tokens):
            piece = ids[start : start + self.max_input_tokens]
            chunks.append(self.tokenizer.decode(piece, skip_special_tokens=True))
        return chunks

    @torch.no_grad()
    def _generate(self, text: str, min_len: int, max_len: int) -> str:
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.max_input_tokens
        ).to(self.device)
        out = self.model.generate(
            **inputs,
            min_length=min_len,
            max_length=max_len,
            num_beams=4,
            length_penalty=2.0,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    def summarize(self, text: str, min_len: int = 40, max_len: int = 140) -> SummaryResult:
        start = time.perf_counter()
        chunks = self._chunk_by_tokens(text)
        partials = [self._generate(c, min_len, max_len) for c in chunks]
        merged = " ".join(partials)
        # one recursive pass if the merged summary is itself too long
        if len(self.tokenizer.encode(merged)) > self.max_input_tokens:
            merged = self._generate(merged, min_len, max_len)
        runtime = time.perf_counter() - start
        return SummaryResult(
            text=merged,
            model=self.model_name,
            runtime_s=runtime,
            input_tokens=len(self.tokenizer.encode(text)),
            output_tokens=len(self.tokenizer.encode(merged)),
        )


def compare_models(text: str, model_keys: Iterable[str] = ("bart", "pegasus")) -> List[SummaryResult]:
    results = []
    for key in model_keys:
        summ = AbstractiveSummarizer(key)
        results.append(summ.summarize(text))
    return results


if __name__ == "__main__":
    sample = (
        "The James Webb Space Telescope, launched in December 2021, has begun returning "
        "unprecedented infrared observations of the early universe. Its 6.5-metre primary "
        "mirror and sun-shielded orbit at L2 enable detection of light from galaxies that "
        "formed within a few hundred million years of the Big Bang. Early science releases "
        "include detailed spectra of exoplanet atmospheres, deep-field images of galaxy "
        "clusters acting as gravitational lenses, and high-resolution imagery of star-forming "
        "regions. Scientists expect the mission to operate for at least 20 years."
    )
    for r in compare_models(sample, ("bart",)):
        print(f"[{r.model}]  ({r.runtime_s:.2f}s, {r.input_tokens}->{r.output_tokens} tokens)")
        print(r.text)
        print("-" * 60)
