"""
Day 4: ROUGE-based evaluation and head-to-head comparison.

We score the three approaches built earlier (TextRank extractive, BART, Pegasus)
against reference summaries using ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.

ROUGE measures n-gram overlap (R-1 unigrams, R-2 bigrams) and longest common
subsequence (R-L) between candidate and reference. It correlates reasonably
with human judgment for extractive summaries but is known to under-credit
abstractive systems that paraphrase well, so we report runtime and length
stats alongside the scores.
"""

import json
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

from rouge_score import rouge_scorer


METRICS = ("rouge1", "rouge2", "rougeL")
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


@dataclass
class EvalRow:
    system: str
    doc_id: str
    rouge1_f: float
    rouge2_f: float
    rougeL_f: float
    runtime_s: float
    n_words: int


def score_pair(scorer: rouge_scorer.RougeScorer, reference: str, candidate: str) -> Dict[str, float]:
    scores = scorer.score(reference, candidate)
    return {m: scores[m].fmeasure for m in METRICS}


def evaluate_system(
    name: str,
    summarize_fn: Callable[[str], str],
    docs: Sequence[Tuple[str, str, str]],  # (doc_id, source, reference)
) -> List[EvalRow]:
    """Run one summarizer over all docs and collect per-doc rows."""
    scorer = rouge_scorer.RougeScorer(list(METRICS), use_stemmer=True)
    rows: List[EvalRow] = []
    for doc_id, source, reference in docs:
        t0 = time.perf_counter()
        candidate = summarize_fn(source)
        elapsed = time.perf_counter() - t0
        f = score_pair(scorer, reference, candidate)
        rows.append(EvalRow(
            system=name,
            doc_id=doc_id,
            rouge1_f=f["rouge1"],
            rouge2_f=f["rouge2"],
            rougeL_f=f["rougeL"],
            runtime_s=elapsed,
            n_words=len(candidate.split()),
        ))
    return rows


def aggregate(rows: Iterable[EvalRow]) -> Dict[str, Dict[str, float]]:
    """Mean score per system across documents."""
    by_system: Dict[str, List[EvalRow]] = defaultdict(list)
    for r in rows:
        by_system[r.system].append(r)

    summary: Dict[str, Dict[str, float]] = {}
    for system, system_rows in by_system.items():
        summary[system] = {
            "rouge1_f": statistics.fmean(r.rouge1_f for r in system_rows),
            "rouge2_f": statistics.fmean(r.rouge2_f for r in system_rows),
            "rougeL_f": statistics.fmean(r.rougeL_f for r in system_rows),
            "avg_runtime_s": statistics.fmean(r.runtime_s for r in system_rows),
            "avg_words": statistics.fmean(r.n_words for r in system_rows),
            "n_docs": len(system_rows),
        }
    return summary


def format_leaderboard(agg: Dict[str, Dict[str, float]]) -> str:
    header = f"{'system':<14} {'R-1':>7} {'R-2':>7} {'R-L':>7} {'sec/doc':>8} {'words':>6}"
    lines = [header, "-" * len(header)]
    ranked = sorted(agg.items(), key=lambda kv: kv[1]["rougeL_f"], reverse=True)
    for system, m in ranked:
        lines.append(
            f"{system:<14} "
            f"{m['rouge1_f']:>7.3f} "
            f"{m['rouge2_f']:>7.3f} "
            f"{m['rougeL_f']:>7.3f} "
            f"{m['avg_runtime_s']:>8.2f} "
            f"{m['avg_words']:>6.0f}"
        )
    return "\n".join(lines)


def save_results(rows: List[EvalRow], agg: Dict[str, Dict[str, float]]) -> None:
    (RESULTS_DIR / "rows.json").write_text(
        json.dumps([asdict(r) for r in rows], indent=2)
    )
    (RESULTS_DIR / "summary.json").write_text(json.dumps(agg, indent=2))
    (RESULTS_DIR / "leaderboard.txt").write_text(format_leaderboard(agg) + "\n")


def main() -> None:
    # Lazy imports - only pay the model-loading cost when running the eval end-to-end.
    from importlib import import_module
    textrank = import_module("02_textrank_extractive")
    abstractive = import_module("03_abstractive_transformer")
    data = import_module("01_data_prep_preprocessing")

    # Small held-out eval set; using 50 docs keeps this script under ~10 min on CPU.
    docs = data.load_eval_split(n=50)

    bart = abstractive.AbstractiveSummarizer("bart")
    pegasus = abstractive.AbstractiveSummarizer("pegasus")

    systems = {
        "textrank": lambda src: textrank.summarize(src, num_sentences=3),
        "bart-large-cnn": lambda src: bart.summarize(src).text,
        "pegasus-xsum": lambda src: pegasus.summarize(src).text,
    }

    all_rows: List[EvalRow] = []
    for name, fn in systems.items():
        print(f"evaluating {name} on {len(docs)} docs...")
        all_rows.extend(evaluate_system(name, fn, docs))

    agg = aggregate(all_rows)
    print()
    print(format_leaderboard(agg))
    save_results(all_rows, agg)


if __name__ == "__main__":
    main()
