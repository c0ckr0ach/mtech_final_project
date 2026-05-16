"""
Stage 6: LLM & RAG Evaluation — RAGAS Framework
Evaluates the quality of the LLM threat analysis produced by Stage 4b
using the RAGAS (Retrieval-Augmented Generation Assessment) framework.

Four metrics are computed per analysis entry:
  • Faithfulness      — Are all claims grounded in the retrieved context?
                        Score < 0.7 is considered hallucinatory.
  • Answer Relevancy  — Is the response relevant to the anomaly question?
  • Context Precision — Are the most useful RAG docs ranked highest?
  • Context Recall    — Did the retriever surface all necessary information?

Key research contribution:
  A "no-RAG baseline" is also evaluated (same LLM, same questions, but
  retrieved_docs replaced with empty string). Comparing Faithfulness
  WITH RAG vs WITHOUT RAG quantitatively proves that the RAG pipeline
  reduces hallucinations — a compelling thesis finding.

Outputs:
  ├── llm_metrics.json               — Per-entry and aggregated RAGAS scores
  └── figures/ragas_radar_chart.png  — Radar chart of 4 metrics (with/without RAG)
  └── figures/faithfulness_comparison.png — Bar chart: RAG vs no-RAG Faithfulness
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from openai import OpenAI
from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.llms import llm_factory

# RAGAS v0.2+: metrics are classes that must be instantiated, not module singletons.
# Import from the correct v0.2+ location.
try:
    from ragas.metrics.collections import (
        Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
    )
    RAGAS_METRICS = [
        Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()
    ]
    METRIC_COLS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
except ImportError:
    # Older ragas ≤ v0.1 — metrics were singletons
    from ragas.metrics import (
        faithfulness, answer_relevancy, context_precision, context_recall
    )
    RAGAS_METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]
    METRIC_COLS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

RESULTS_JSON    = "/content/data/llm_results.json"
LLM_METRICS_JSON = "/content/data/llm_metrics.json"
FIGURES_DIR     = "/content/data/figures"
OLLAMA_MODEL    = "llama3"
OLLAMA_BASE_URL = "http://localhost:11434"

plt.style.use("dark_background")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _configure_ragas_llm(model: str = OLLAMA_MODEL):
    """
    Configure RAGAS LLM and embeddings for local Ollama (RAGAS v0.2+ API).

    LLM : llm_factory pointed at Ollama's OpenAI-compatible /v1 endpoint.
          api_key can be any non-empty string; Ollama ignores it.
    Emb : RAGAS's own HuggingFaceEmbeddings wrapper around all-MiniLM-L6-v2.
          sentence-transformers is already installed from Stage 3.
          This avoids routing embedding calls through Ollama entirely.
    """
    from ragas.embeddings import HuggingFaceEmbeddings as RagasHFEmbeddings

    ollama_client = OpenAI(
        base_url=f"{OLLAMA_BASE_URL}/v1",
        api_key="ollama",
    )
    llm = llm_factory(model=model, client=ollama_client)
    emb = RagasHFEmbeddings(model_name="all-MiniLM-L6-v2")

    return llm, emb


def _build_ragas_dataset(results: list[dict],
                         no_rag: bool = False) -> Dataset:
    """
    Convert llm_results.json entries into a HuggingFace Dataset
    in the format expected by RAGAS.

    RAGAS expects columns:
        question   : str          — the question/prompt
        answer     : str          — the LLM's generated answer
        contexts   : list[str]    — the retrieved passages used
        ground_truth: str         — reference answer (optional for some metrics)
    """
    rows = []
    for r in results:
        question = (
            f"Analyse this security anomaly: "
            f"EventID {r.get('event_id')} on host {r.get('hostname')}. "
            f"Topic keywords: {r.get('topic_keywords', '')}."
        )
        answer   = (
            f"Threat Analysis: {r.get('threat_analysis', '')}\n"
            f"MITRE Technique: {r.get('mitre_technique', '')}\n"
            f"Remediation: {r.get('remediation_steps', '')}\n"
            f"Severity: {r.get('severity_rating', '')}"
        )
        # For the no-RAG baseline, replace context with empty string
        context_text = r.get("retrieved_docs", "") if not no_rag else ""
        contexts = [context_text] if context_text else ["No context provided."]

        # Ground truth: use MITRE technique as a minimal reference
        ground_truth = r.get("mitre_technique", "Unknown technique")

        rows.append({
            "question"    : question,
            "answer"      : answer,
            "contexts"    : contexts,
            "ground_truth": ground_truth,
        })
    return Dataset.from_list(rows)


def _run_ragas(dataset: Dataset, llm, emb) -> dict:
    """
    Run RAGAS evaluation sequentially (max_workers=1) to avoid overwhelming
    a local Ollama instance with concurrent requests, which causes TimeoutErrors.
    Returns a dict of averaged metric scores.
    """
    run_cfg = RunConfig(
        max_workers=1,    # sequential — critical for local Ollama
        timeout=180,      # 3 min per call; llama3 on GPU should be well within this
        max_retries=2,
    )
    result = evaluate(
        dataset,
        metrics=RAGAS_METRICS,   # pre-instantiated metric objects
        llm=llm,
        embeddings=emb,
        run_config=run_cfg,
        raise_exceptions=False,
    )

    # RAGAS returns an EvaluationResult — use to_pandas() to get per-row scores
    df   = result.to_pandas()
    cols = [c for c in METRIC_COLS if c in df.columns]
    return {k: round(float(v), 4) if pd.notna(v) else 0.0
            for k, v in df[cols].mean(numeric_only=True).items()}


# ── Radar chart ───────────────────────────────────────────────────────────────

def plot_radar_chart(with_rag: dict, without_rag: dict, out_path: str):
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    labels  = ["Faithfulness", "Answer\nRelevancy", "Context\nPrecision", "Context\nRecall"]
    N       = len(metrics)
    angles  = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    vals_with    = [with_rag.get(m, 0) for m in metrics]    + [with_rag.get(metrics[0], 0)]
    vals_without = [without_rag.get(m, 0) for m in metrics] + [without_rag.get(metrics[0], 0)]

    fig, ax = plt.subplots(1, 1, figsize=(7, 7),
                           subplot_kw=dict(polar=True))
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    ax.plot(angles, vals_with,    "o-", linewidth=2.5, color="#00b4d8", label="With RAG")
    ax.fill(angles, vals_with,    alpha=0.2, color="#00b4d8")
    ax.plot(angles, vals_without, "s--", linewidth=2.5, color="#e94560", label="Without RAG")
    ax.fill(angles, vals_without, alpha=0.15, color="#e94560")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12, color="white")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", fontsize=9)
    ax.grid(color="grey", alpha=0.3)
    ax.set_title("RAGAS Evaluation — RAG vs No-RAG", fontsize=14, color="white", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_faithfulness_bar(with_rag: dict, without_rag: dict, out_path: str):
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    labels  = ["Faithfulness", "Answer Relevancy", "Context Precision", "Context Recall"]
    x       = np.arange(len(metrics))
    w       = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    bars1 = ax.bar(x - w/2, [with_rag.get(m, 0) for m in metrics],
                   w, label="With RAG", color="#00b4d8", alpha=0.9)
    bars2 = ax.bar(x + w/2, [without_rag.get(m, 0) for m in metrics],
                   w, label="Without RAG", color="#e94560", alpha=0.9)

    ax.set_xlabel("RAGAS Metric", fontsize=12, color="white")
    ax.set_ylabel("Score (0–1)", fontsize=12, color="white")
    ax.set_title("RAGAS Scores: With RAG Context vs Without RAG Context",
                 fontsize=13, color="white", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, color="white")
    ax.set_ylim(0, 1.1)
    ax.tick_params(colors="white")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.2)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", va="bottom",
                fontsize=9, color="white")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", va="bottom",
                fontsize=9, color="white")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def llm_eval_stage(results_path: str   = RESULTS_JSON,
                   metrics_path: str   = LLM_METRICS_JSON,
                   figures_dir: str    = FIGURES_DIR,
                   model: str          = OLLAMA_MODEL,
                   max_samples: int    = 10) -> dict:
    """
    End-to-end Stage 6 entry point.
    Evaluates LLM analysis quality with and without RAG context using RAGAS.

    Args:
        max_samples: Cap on entries evaluated. RAGAS makes multiple LLM calls
                     per row (4 metrics × N rows), so keep this small for local
                     Ollama. 10 samples = ~80 LLM calls, runtime ~15-30 min.
                     Increase to 30 for the final thesis run.
    """
    os.makedirs(figures_dir, exist_ok=True)

    print(f"Loading LLM results from {results_path} …")
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    # Subsample for evaluation speed
    if len(results) > max_samples:
        rng     = np.random.default_rng(42)
        indices = rng.choice(len(results), max_samples, replace=False)
        results = [results[i] for i in sorted(indices)]
    print(f"  Evaluating {len(results)} entries …")

    llm, emb = _configure_ragas_llm(model)

    # ── WITH RAG ─────────────────────────────────────────────────────────────
    print("\n── Evaluating WITH RAG context ────────────────────────────")
    ds_with  = _build_ragas_dataset(results, no_rag=False)
    scores_with = _run_ragas(ds_with, llm, emb)
    print("  Scores:", {k: f"{v:.4f}" for k, v in scores_with.items()})

    # ── WITHOUT RAG baseline ──────────────────────────────────────────────────
    print("\n── Evaluating WITHOUT RAG (baseline) ──────────────────────")
    ds_without  = _build_ragas_dataset(results, no_rag=True)
    scores_without = _run_ragas(ds_without, llm, emb)
    print("  Scores:", {k: f"{v:.4f}" for k, v in scores_without.items()})

    # ── Delta (RAG improvement) ───────────────────────────────────────────────
    delta = {k: round(scores_with.get(k, 0) - scores_without.get(k, 0), 4)
             for k in scores_with}
    print("\n── RAG Improvement (Δ) ─────────────────────────────────────")
    for k, v in delta.items():
        direction = "▲" if v >= 0 else "▼"
        print(f"  {k:25s}: {direction} {abs(v):.4f}")
    print("────────────────────────────────────────────────────────────")

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\n── Generating RAGAS figures ───────────────────────────────")
    plot_radar_chart(
        scores_with, scores_without,
        os.path.join(figures_dir, "ragas_radar_chart.png")
    )
    plot_faithfulness_bar(
        scores_with, scores_without,
        os.path.join(figures_dir, "faithfulness_comparison.png")
    )

    # ── Save JSON ─────────────────────────────────────────────────────────────
    output = {
        "with_rag"   : scores_with,
        "without_rag": scores_without,
        "delta"      : delta,
        "n_samples"  : len(results),
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n  LLM metrics saved → {metrics_path}")

    return output


if __name__ == "__main__":
    llm_eval_stage()
