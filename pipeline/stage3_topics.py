"""
Stage 3: BERTopic — Topic Modelling on Anomalous Events
Cleans log text, embeds with sentence-transformers, clusters with HDBSCAN,
and visualises topics interactively.
"""
import os
import re
import nltk
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

# ── Config ────────────────────────────────────────────────────────────────────
ANOMALIES_PARQUET = "/content/data/anomalies.parquet"
TOPICS_PARQUET    = "/content/data/anomalies_with_topics.parquet"
TOPIC_MODEL_DIR   = "/content/data/bertopic_model"
EMBED_MODEL       = "all-MiniLM-L6-v2"
MIN_CLUSTER_SIZE  = 15
TOP_N_WORDS       = 10
# ──────────────────────────────────────────────────────────────────────────────


# ── Text cleaning ─────────────────────────────────────────────────────────────
_GUID_RE      = re.compile(r"\{[0-9a-fA-F\-]{8,}\}")
_HEX_RE       = re.compile(r"\b0x[0-9a-fA-F]+\b")
_TS_RE        = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?")
_PATH_RE      = re.compile(r"[A-Za-z]:\\(?:[^\s\r\n|,\\]+\\)*([^\s\r\n|,\\]+)")
_NUM_RE       = re.compile(r"\b\d+\b")
_WS_RE        = re.compile(r"\s+")


def _ensure_stopwords():
    try:
        from nltk.corpus import stopwords
        return set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords
        return set(stopwords.words("english"))


STOP_WORDS = _ensure_stopwords()

# Windows / Sysmon noise words that add no semantic value
DOMAIN_NOISE = {
    "rulename", "utctime", "processguid", "processid", "image",
    "targetprocessid", "targetprocessguid", "sourcename", "channel",
    "keywords", "opcodevalue", "severityvalue", "eventreceivedtime",
    "sourcemodulename", "sourcemoduletype", "version", "task",
    "threadid", "recordnumber", "executionprocessid", "providerguid",
    "timestamp", "version", "none", "null", "true", "false",
}


def clean_log_text(text: str) -> str:
    """
    Strip technical noise (GUIDs, hex, paths, timestamps, numbers)
    and return a bag-of-meaningful-words string.
    """
    # Replace paths with just the exe/filename token
    text = _PATH_RE.sub(lambda m: " " + m.group(1).lower() + " ", text)
    text = _GUID_RE.sub(" ", text)
    text = _HEX_RE.sub(" hexval ", text)
    text = _TS_RE.sub(" ", text)
    text = _NUM_RE.sub(" ", text)
    # Keep only alpha
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = _WS_RE.sub(" ", text).strip().lower()

    tokens = [
        w for w in text.split()
        if w not in STOP_WORDS
        and w not in DOMAIN_NOISE
        and len(w) > 2
    ]
    return " ".join(tokens) if tokens else "unknown_event"


# ── BERTopic pipeline ─────────────────────────────────────────────────────────

def build_topic_model() -> BERTopic:
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
        low_memory=True,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    vectorizer = CountVectorizer(
        stop_words="english",
        min_df=2,
        ngram_range=(1, 2),
        max_features=10_000,
    )
    embedding_model = SentenceTransformer(EMBED_MODEL)

    return BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        top_n_words=TOP_N_WORDS,
        calculate_probabilities=True,
        verbose=True,
    )


def topic_stage(anomalies_path: str = ANOMALIES_PARQUET,
                topics_path: str    = TOPICS_PARQUET,
                model_dir: str      = TOPIC_MODEL_DIR) -> tuple[pd.DataFrame, BERTopic]:
    """
    End-to-end Stage 3 entry point.
    Returns (annotated_df, topic_model).
    """
    os.makedirs(os.path.dirname(topics_path), exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print("📥  Loading anomalies …")
    df = pd.read_parquet(anomalies_path)
    print(f"    Shape: {df.shape}")

    # ── Prepare corpus ────────────────────────────────────────────────────────
    print("🧹  Cleaning log text …")

    def _col(name: str) -> pd.Series:
        """Safely retrieve a column; return empty strings if absent."""
        return df[name].fillna("") if name in df.columns else pd.Series("", index=df.index)

    corpus_raw = (
        _col("message") + " " +
        _col("image_base") + " " +
        _col("target_image_base") + " " +
        _col("target_object").apply(
            lambda x: x.split("\\")[-1].lower() if isinstance(x, str) and x else ""
        )
    )
    docs = corpus_raw.apply(clean_log_text).tolist()
    print(f"    Corpus size: {len(docs):,} documents")
    print(f"    Sample doc : {docs[0][:120]}")

    # ── Fit BERTopic ──────────────────────────────────────────────────────────
    print("\n🔬  Fitting BERTopic …")
    topic_model = build_topic_model()
    topics, probs = topic_model.fit_transform(docs)

    df = df.copy()
    df["topic"]       = topics
    df["topic_prob"]  = [float(p.max()) if hasattr(p, "max") else float(p)
                         for p in probs]

    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    print(f"\n✅  Discovered {n_topics} topics  (topic -1 = noise/outliers)")

    # ── Topic info ────────────────────────────────────────────────────────────
    topic_info = topic_model.get_topic_info()
    print("\nTop topics:\n", topic_info.head(12).to_string(index=False))

    # ── Visualisations ────────────────────────────────────────────────────────
    print("\n📊  Generating visualisations …")

    fig_bar = topic_model.visualize_barchart(
        top_n_topics=min(12, n_topics), n_words=8
    )
    fig_bar.update_layout(template="plotly_dark",
                          title="📊 Security Event Topics — Top Keywords")
    fig_bar.show()

    if n_topics >= 2:
        fig_map = topic_model.visualize_topics()
        fig_map.update_layout(template="plotly_dark",
                              title="🗺️ Inter-topic Distance Map")
        fig_map.show()

        fig_heat = topic_model.visualize_heatmap()
        fig_heat.update_layout(template="plotly_dark",
                               title="🔥 Topic Similarity Heatmap")
        fig_heat.show()

    # ── Save ──────────────────────────────────────────────────────────────────
    df.to_parquet(topics_path, index=False)
    topic_model.save(os.path.join(model_dir, "model.pkl"))
    print(f"\n💾  Saved annotated parquet → {topics_path}")
    print(f"💾  Saved BERTopic model    → {model_dir}")

    return df, topic_model


def get_topic_summary(topic_model: BERTopic, top_n: int = 15) -> dict:
    """
    Returns {topic_id: 'keyword1, keyword2, …'} for the RAG query builder.
    """
    summary = {}
    for tid in topic_model.get_topic_info()["Topic"].tolist():
        if tid == -1:
            continue
        words = topic_model.get_topic(tid)
        if words:
            summary[tid] = ", ".join([w for w, _ in words[:top_n]])
    return summary


if __name__ == "__main__":
    df, model = topic_stage()
    print("\nTopic keyword summary:")
    for tid, kws in list(get_topic_summary(model).items())[:5]:
        print(f"  Topic {tid}: {kws}")
