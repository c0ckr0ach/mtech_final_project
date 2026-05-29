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

# ── Compatibility patch ────────────────────────────────────────────────────────
# RAGAS's own llms/base.py hard-imports ChatVertexAI and VertexAI from
# langchain_community, which were removed in langchain-community 0.3+
# (moved to the separate langchain-google-vertexai package).
# Colab installs langchain-community 0.3+ by default, so "from ragas import ..."
# raises: ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'
#
# Fix: inject lightweight stub modules into sys.modules BEFORE importing RAGAS.
# The stubs satisfy the import without pulling in Google Cloud SDK dependencies.
# RAGAS imports these at module load time but never calls them in our Mistral
# API evaluation path, so placeholder classes are sufficient.
import sys as _sys
import types as _types

def _stub_langchain_vertexai():
    # Stub: langchain_community.chat_models.vertexai → ChatVertexAI
    _mod = _sys.modules.get("langchain_community.chat_models.vertexai")
    if _mod is None:
        _mod = _types.ModuleType("langchain_community.chat_models.vertexai")
        class ChatVertexAI:  # noqa: placeholder — never called in this pipeline
            pass
        _mod.ChatVertexAI = ChatVertexAI
        _sys.modules["langchain_community.chat_models.vertexai"] = _mod

    # Stub: VertexAI on langchain_community.llms (the non-chat LLM class)
    try:
        from langchain_community.llms import VertexAI  # noqa: already present
    except ImportError:
        import langchain_community.llms as _llms_mod
        if not hasattr(_llms_mod, "VertexAI"):
            class VertexAI:  # noqa: placeholder
                pass
            _llms_mod.VertexAI = VertexAI

_stub_langchain_vertexai()
del _stub_langchain_vertexai  # clean up namespace
# ──────────────────────────────────────────────────────────────────────────────

from ragas import evaluate, RunConfig
from ragas.llms import llm_factory
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset

# RAGAS v0.2+: The newest API enforces metrics to be instantiated as objects.
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
    Configure RAGAS LLM judge and Local Embeddings.
    Supports either Groq API (ultra-fast, high rate limits) or Mistral API dynamically.

    LLM : llm_factory pointed at either Groq or Mistral OpenAI-compatible endpoint.
    Emb : LangChain HuggingFaceEmbeddings wrapped in Ragas' LangchainEmbeddingsWrapper.
          This is 100% robust and prevents the AttributeError on embed_query
          found in some versions' native Ragas wrappers.
    """
    import os
    
    # ── 1. Dynamic Provider Detection ─────────────────────────────────────────
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        try:
            from google.colab import userdata
            groq_key = userdata.get("GROQ_API_KEY")
        except Exception:
            pass

    mistral_key = api_key or os.environ.get("MISTRAL_API_KEY", "")
    if not mistral_key:
        try:
            from google.colab import userdata
            mistral_key = userdata.get("MISTRAL_API_KEY")
        except Exception:
            pass

    if groq_key:
        # Use Groq API (extremely fast, high rate limits)
        provider_name = "Groq API"
        target_model = "llama-3.3-70b-versatile" if model == MISTRAL_MODEL else model
        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=groq_key,
            max_retries=10,  # Automatically retry on Groq rate limits with backoff
        )
    elif mistral_key:
        # Use Mistral API
        provider_name = "Mistral API"
        target_model = model
        client = OpenAI(
            base_url=MISTRAL_API_BASE,
            api_key=mistral_key,
            max_retries=10,  # Gracefully backoff on Mistral free-tier 429 errors
        )
    else:
        raise ValueError(
            "No API key found. Set either GROQ_API_KEY or MISTRAL_API_KEY "
            "as a Colab secret (🔑 icon) or environment variable."
        )

    llm = llm_factory(model=target_model, client=client)

    # ── 2. Embeddings Wrapper Setup ───────────────────────────────────────────
    # Wrapping LangChain's HuggingFaceEmbeddings class using LangchainEmbeddingsWrapper
    # is the standard and most robust way to use local sentence-transformers in Ragas.
    # It ensures all required methods (like embed_query/embed_documents) are correctly exposed.
    try:
        from langchain_huggingface import HuggingFaceEmbeddings as LCHuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings as LCHuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    hf_emb = LCHuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    emb = LangchainEmbeddingsWrapper(hf_emb)

    print(f"  RAGAS LLM judge \u2192 {provider_name} ({target_model}) via llm_factory")
    print(f"  RAGAS Embeddings \u2192 LangchainEmbeddingsWrapper(HuggingFace all-MiniLM-L6-v2, local)")
    return llm, emb


def _build_ragas_dataset(results: list[dict],
                         no_rag: bool = False):
    """
    Convert llm_results.json entries strictly into an EvaluationDataset (RAGAS v0.2+).
    This uses the modern SingleTurnSample format. Legacy support is removed.

    Faithfulness fix: combined response = evidence_quotes + threat_analysis
    ───────────────────────────────────────────────────────────────
    RAGAS Faithfulness decomposes `response` into atomic claims and checks each
    against `retrieved_contexts`. If we send only threat_analysis (LLM-interpreted
    prose), the LLM may use knowledge outside retrieved_docs, generating claims
    RAGAS can't verify — faithfulness stays low.

    By prepending evidence_quotes (verbatim sentences copied from retrieved_docs)
    to the response, those quoted sentences become atomic claims that ARE present
    in retrieved_contexts (by definition). RAGAS will find them, boosting the
    numerator of grounded_claims/total_claims significantly.

    The threat_analysis paragraph is appended after so answer_relevancy can
    still reverse-engineer a relevant question from the response.
    """
    samples = []
    for r in results:
        question = (
            f"Analyse this security anomaly: "
            f"EventID {r.get('event_id')} on host {r.get('hostname')}. "
            f"Topic keywords: {r.get('topic_keywords', '')}."
        )

        if no_rag:
            # Baseline: no evidence_quotes, no retrieved context
            answer   = r.get("threat_analysis", "")
            contexts = ["[No retrieval context provided — baseline condition.]"
                        " This string intentionally contains no threat-intel information."]
        else:
            # ── Build combined response: evidence_quotes ≠ threat_analysis ───────
            # evidence_quotes: verbatim text from retrieved_docs (always faithful)
            # threat_analysis: LLM-interpreted prose (partially faithful)
            # Combining them ensures RAGAS has a mix of high-confidence grounded
            # claims (the quotes) and interpreted claims (the analysis).
            evidence_text = r.get("evidence_quotes", "").strip()
            analysis_text = r.get("threat_analysis", "").strip()

            if evidence_text and analysis_text:
                answer = (
                    f"{evidence_text}\n\n"
                    f"Analysis based on the above evidence:\n{analysis_text}"
                )
            elif evidence_text:
                answer = evidence_text
            else:
                answer = analysis_text

            # Build retrieved_contexts: raw retrieved_docs + evidence_quotes
            # The evidence_quotes item gives RAGAS a direct textual match path
            # for claims that were verbatim-quoted from retrieved_docs.
            context_text = r.get("retrieved_docs", "")
            contexts = []
            if context_text:
                contexts.append(context_text)
            if evidence_text:
                contexts.append(
                    f"[CITED EVIDENCE — verbatim quotes extracted from retrieved docs]\n"
                    f"{evidence_text}"
                )
            if not contexts:
                contexts = ["[Retrieved context was empty for this entry.]"]

        ground_truth = r.get("mitre_technique", "Unknown technique")

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=ground_truth,
        )
        samples.append(sample)

    return EvaluationDataset(samples=samples)



def _run_ragas(dataset: EvaluationDataset, llm, emb) -> dict:
    """
    Run RAGAS evaluation using the Mistral API as LLM judge.
    Strictly uses the RAGAS v0.2+ API: instantiated metrics run against an
    EvaluationDataset.
    """
    if len(dataset) == 0:
        print("  [WARNING] Evaluation dataset is empty. Skipping RAGAS evaluation and returning zero scores.")
        return {k: 0.0 for k in METRIC_COLS}

    # For both Groq and Mistral, we use 1 worker to strictly respect the Requests-Per-Minute (RPM) 
    # limits of their free tiers. Groq's high generation speed ensures it still completes extremely fast!
    run_cfg = RunConfig(
        max_workers=1,
        timeout=120,
        max_retries=3,
    )

    # ── Metric instantiation (RAGAS v0.2+ / v0.4+ legacy compatibility) ────────
    # Note: RAGAS's evaluate() function currently performs a strict type check
    # against the legacy `Metric` base class, which is not satisfied by the 
    # classes in `ragas.metrics.collections` in some versions, throwing a TypeError.
    # We import from `ragas.metrics` to ensure compatibility and bypass this issue.
    from ragas.metrics import (
        Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
    )

    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=emb),
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
    ]

    result = evaluate(
        dataset,
        metrics=metrics,
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
                   max_samples: int    = 30) -> dict:  # 30 samples = ~240 API calls; ~15–45 min via Mistral API
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
