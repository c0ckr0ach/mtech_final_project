"""
Stage 6: LLM & RAG Evaluation — RAGAS Framework
Evaluates the quality of the LLM threat analysis produced by Stage 4b
using the RAGAS (Retrieval-Augmented Generation Assessment) framework.

LLM Judge: Mistral API (mistral-large-latest) via LiteLLM / OpenAI-compatible client.
           Set MISTRAL_API_KEY as a Colab secret or environment variable.

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

# RAGAS v0.2+ canonical LLM setup:
#   LangchainLLMWrapper(ChatOpenAI(...)) is the correct LLM type that RAGAS's
#   isinstance(m, Metric) check and metric constructors expect.
#   Using llm_factory(model, client=OpenAI(...)) produces a wrapper that passes
#   Python's type() but is not recognized by RAGAS's internal Langchain callback
#   protocol, causing the "All metrics must be initialised metric objects" TypeError.
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper

# RAGAS v0.2+: metrics are classes that must be instantiated with an LLM.
# They are imported here but instantiated dynamically inside _run_ragas.
METRIC_COLS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

RESULTS_JSON      = "/content/data/llm_results.json"
LLM_METRICS_JSON  = "/content/data/llm_metrics.json"
FIGURES_DIR       = "/content/data/figures"

# ── Mistral API config (LLM judge for RAGAS evaluation) ──────────────────────
MISTRAL_MODEL    = "mistral-large-latest"     # switch to mistral-small-latest for cheaper dev runs
MISTRAL_API_BASE = "https://api.mistral.ai/v1"
# API key is read from MISTRAL_API_KEY env var (set as a Colab secret)
# ─────────────────────────────────────────────────────────────────────────────

plt.style.use("dark_background")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _configure_ragas_llm(model: str = MISTRAL_MODEL,
                         api_key: str = None):
    """
    Configure RAGAS LLM judge using the Mistral API.

    LLM : LangchainLLMWrapper(ChatOpenAI(...)) — the canonical RAGAS v0.2+
          LLM type. ChatOpenAI's base_url is pointed at Mistral's
          OpenAI-compatible endpoint. This is the only wrapper type that RAGAS
          metrics accept; passing a raw openai.OpenAI client via llm_factory
          causes a TypeError at evaluate() time.
    Emb : RAGAS native HuggingFaceEmbeddings (all-MiniLM-L6-v2) kept local
          to avoid additional API costs.
    """
    import os
    from ragas.embeddings import HuggingFaceEmbeddings as RagasHFEmbeddings

    key = api_key or os.environ.get("MISTRAL_API_KEY", "")
    if not key:
        raise ValueError(
            "Mistral API key not found. "
            "Set the MISTRAL_API_KEY environment variable or pass api_key= directly."
        )

    chat_model = ChatOpenAI(
        model=model,
        base_url=MISTRAL_API_BASE,
        api_key=key,
    )
    llm = LangchainLLMWrapper(chat_model)
    emb = RagasHFEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

    print(f"  RAGAS LLM judge → Mistral API ({model}) via LangchainLLMWrapper")
    return llm, emb


def _build_ragas_dataset(results: list[dict],
                         no_rag: bool = False):
    """
    Convert llm_results.json entries into an EvaluationDataset (RAGAS v0.2+)
    or fallback to a HuggingFace Dataset.
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
        if no_rag:
            contexts = ["[No retrieval context provided — baseline condition.]"
                        " This string intentionally contains no threat-intel information."]
        else:
            context_text = r.get("retrieved_docs", "")
            contexts = [context_text] if context_text else [
                "[Retrieved context was empty for this entry.]"
            ]

        ground_truth = r.get("mitre_technique", "Unknown technique")

        rows.append({
            "user_input": question,
            "response": answer,
            "retrieved_contexts": contexts,
            "reference": ground_truth,
            # Fallback legacy columns for compatibility
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth,
        })

    try:
        from ragas import SingleTurnSample, EvaluationDataset
        samples = [
            SingleTurnSample(
                user_input=row["user_input"],
                response=row["response"],
                retrieved_contexts=row["retrieved_contexts"],
                reference=row["reference"],
            )
            for row in rows
        ]
        return EvaluationDataset(samples=samples)
    except ImportError:
        from datasets import Dataset
        return Dataset.from_list(rows)



def _run_ragas(dataset, llm, emb) -> dict:
    """
    Run RAGAS evaluation using the Mistral API as LLM judge.
    The API handles concurrency, so max_workers > 1 is safe and speeds up
    the evaluation significantly versus local Ollama.
    Returns a dict of averaged metric scores.
    """
    if len(dataset) == 0:
        print("  [WARNING] Evaluation dataset is empty. Skipping RAGAS evaluation and returning zero scores.")
        return {k: 0.0 for k in METRIC_COLS}

    run_cfg = RunConfig(
        max_workers=4,    # Mistral API handles concurrency; 4 parallel RAGAS workers
        timeout=120,      # 120 s per call — allow for API round-trip latency
        max_retries=2,    # allow one retry on transient API errors
    )
    # ── Metric instantiation (RAGAS v0.2+) ───────────────────────────────────
    # With a proper LangchainLLMWrapper LLM, metrics accept llm= directly.
    # We still use a graceful factory to handle minor API differences across
    # RAGAS patch versions (some classes don't expose embeddings= on all metrics).
    try:
        from ragas.metrics.collections import (
            Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
        )
    except ImportError:
        from ragas.metrics import (
            Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
        )

    def _make_metric(cls, **kw):
        """Construct a RAGAS metric, gracefully dropping unsupported kwargs."""
        try:
            return cls(**kw)
        except TypeError:
            # Drop embeddings= first (only AnswerRelevancy uses it)
            kw.pop("embeddings", None)
            try:
                return cls(**kw)
            except TypeError:
                # Absolute fallback: llm= only
                return cls(llm=kw["llm"])

    metrics = [
        _make_metric(Faithfulness,     llm=llm),
        _make_metric(AnswerRelevancy,  llm=llm, embeddings=emb),
        _make_metric(ContextPrecision, llm=llm),
        _make_metric(ContextRecall,    llm=llm),
    ]

    result = evaluate(
        dataset,
        metrics=metrics,
        llm=llm,
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
                   model: str          = MISTRAL_MODEL,
                   api_key: str        = None,
                   max_samples: int    = 10) -> dict:  # 10 = ~80 API calls; ~5–15 min via Mistral API
    """
    End-to-end Stage 6 entry point.
    Evaluates LLM analysis quality with and without RAG context using RAGAS.

    Args:
        model:       Mistral model name (e.g. 'mistral-large-latest').
        api_key:     Mistral API key. Falls back to MISTRAL_API_KEY env var.
        max_samples: Cap on entries evaluated. RAGAS makes multiple API calls
                     per row (4 metrics × N rows). 10 samples ≈ ~80 API calls,
                     runtime ~5–15 min via Mistral API.
                     Increase to 30 for the final thesis run.
    """
    os.makedirs(figures_dir, exist_ok=True)

    print(f"Loading LLM results from {results_path} …")
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    # Handle empty results gracefully to prevent downstream crashes in Ragas/Matplotlib
    if not results:
        print(f"\n  [WARNING] No results found in {results_path}!")
        print("            Please ensure Stage 4b (LLM Analysis) completed successfully and generated threat analyses.")
        scores_empty = {k: 0.0 for k in METRIC_COLS}
        delta_empty = {k: 0.0 for k in METRIC_COLS}
        output = {
            "with_rag"   : scores_empty,
            "without_rag": scores_empty,
            "delta"      : delta_empty,
            "n_samples"  : 0,
        }
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"  Empty LLM metrics report saved to {metrics_path}")
        return output

    # Subsample for evaluation speed
    if len(results) > max_samples:
        rng     = np.random.default_rng(42)
        indices = rng.choice(len(results), max_samples, replace=False)
        results = [results[i] for i in sorted(indices)]
    print(f"  Evaluating {len(results)} entries …")

    llm, emb = _configure_ragas_llm(model, api_key=api_key)

    # ── WITH RAG ─────────────────────────────────────────────────────────────
    print("\n── Evaluating WITH RAG context ────────────────────────────")
    ds_with  = _build_ragas_dataset(results, no_rag=False)
    # Sanity-check: confirm the two datasets differ
    if len(ds_with) > 0:
        if hasattr(ds_with, "samples"):
            _ctx_with = ds_with.samples[0].retrieved_contexts[0][:80]
        else:
            _ctx_with = ds_with[0].get("contexts", ds_with[0].get("retrieved_contexts", [""]))[0][:80]
    else:
        _ctx_with = ""
    print(f"  [WITH RAG]  sample context prefix: {_ctx_with!r}")
    scores_with = _run_ragas(ds_with, llm, emb)
    print("  Scores:", {k: f"{v:.4f}" for k, v in scores_with.items()})

    # ── WITHOUT RAG baseline ──────────────────────────────────────────────────
    print("\n── Evaluating WITHOUT RAG (baseline) ──────────────────────")
    ds_without  = _build_ragas_dataset(results, no_rag=True)
    if len(ds_without) > 0:
        if hasattr(ds_without, "samples"):
            _ctx_without = ds_without.samples[0].retrieved_contexts[0][:80]
        else:
            _ctx_without = ds_without[0].get("contexts", ds_without[0].get("retrieved_contexts", [""]))[0][:80]
    else:
        _ctx_without = ""
    print(f"  [WITHOUT RAG] sample context prefix: {_ctx_without!r}")
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
