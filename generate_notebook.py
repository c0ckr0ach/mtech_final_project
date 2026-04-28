"""
generate_notebook.py
Assembles main.ipynb from the four pipeline stage modules.
Run: python generate_notebook.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10.0"},
    "accelerator": "GPU",
}

def md(src):   return nbf.v4.new_markdown_cell(src)
def code(src): return nbf.v4.new_code_cell(src)

# ─────────────────────────────────────────────────────────────────────────────
cells = []

# ── TITLE ────────────────────────────────────────────────────────────────────
cells.append(md("""# 🛡️ Security Log Analysis Pipeline
### APT29 Evaluation Dataset · Sysmon Events
**Pipeline**: Parse → Anomaly Detection → BERTopic → RAG + LLM (DSPy + Ollama)

> ⚡ **Runtime**: Set Colab to **GPU** (T4 or A100) before running.  
> 📁 **Dataset**: Upload `apt29_evals_day1_manual_2020-05-01225525.json` to `/content/` before running Stage 1.

---
"""))

# ── SETUP ────────────────────────────────────────────────────────────────────
cells.append(md("## ⚙️ Setup — Install Dependencies"))

cells.append(code("""\
# Install all pipeline dependencies
!pip install -q \\
    pandas pyarrow ijson \\
    scikit-learn umap-learn \\
    plotly kaleido \\
    bertopic sentence-transformers hdbscan \\
    chromadb \\
    langchain langchain-community \\
    dspy-ai \\
    mitreattack-python \\
    nltk pyyaml regex requests tqdm

import nltk
nltk.download('stopwords', quiet=True)
print("✅ All dependencies installed")
"""))

cells.append(md("### Clone pipeline modules from this repo"))

cells.append(code("""\
import os, sys

# If running from Google Drive, mount it
# from google.colab import drive
# drive.mount('/content/drive')

# Clone the pipeline repo (adjust URL to your repo)
REPO_URL = "https://github.com/YOUR_USERNAME/mtech_final_project.git"

if not os.path.exists("/content/mtech_final_project"):
    !git clone {REPO_URL} /content/mtech_final_project

sys.path.insert(0, "/content/mtech_final_project")
print("✅ Pipeline modules on path")
"""))

cells.append(md("### Global configuration"))

cells.append(code("""\
# ─── EDIT THESE PATHS IF NEEDED ───────────────────────────────────────────
DATA_PATH          = "/content/apt29_evals_day1_manual_2020-05-01225525.json"
NORMALIZED_PARQUET = "/content/data/normalized.parquet"
ANOMALIES_PARQUET  = "/content/data/anomalies.parquet"
TOPICS_PARQUET     = "/content/data/anomalies_with_topics.parquet"
CHROMA_DIR         = "/content/data/chroma_db"
RESULTS_JSON       = "/content/data/llm_results.json"
TOPIC_MODEL_DIR    = "/content/data/bertopic_model"

OLLAMA_MODEL       = "llama3"       # or "mistral", "phi3"
CONTAMINATION      = 0.05           # fraction flagged as anomalous
TOP_N_TOPICS       = 12             # topics passed to LLM
EVENTS_PER_TOPIC   = 3              # worst anomalies per topic

import os
os.makedirs("/content/data", exist_ok=True)
print("✅ Config ready")
"""))

# ── STAGE 1 ──────────────────────────────────────────────────────────────────
cells.append(md("""---
## 📂 Stage 1 — Parse & Normalize

Streams the 385 MB NDJSON file line-by-line (no OOM), normalises each Sysmon
event into a flat schema, engineers ML features, and saves `normalized.parquet`.

**Key features extracted**:
| Feature | Description |
|---|---|
| `event_id` | Sysmon event type (10=ProcessAccess, 11=FileCreate, 13=RegistrySet …) |
| `process_depth` | Depth of the process image path |
| `granted_access` | Hex access rights → int |
| `is_system` | Is the account NT AUTHORITY\\SYSTEM? |
| `hour_of_day` / `day_of_week` | Temporal features |
| `message_len` | Raw message character count |
| `eid_*` | One-hot top-15 EventIDs |
"""))

cells.append(code("""\
from pipeline.stage1_parse import parse_stage

df_norm = parse_stage(
    data_path  = DATA_PATH,
    out_path   = NORMALIZED_PARQUET,
    chunk_size = 50_000,
)
df_norm.head(3)
"""))

cells.append(code("""\
import pandas as pd
import plotly.express as px

df_norm = pd.read_parquet(NORMALIZED_PARQUET)

fig = px.histogram(
    df_norm, x="event_id",
    color="is_system",
    title="Event ID Distribution (System vs User Accounts)",
    template="plotly_dark",
    barmode="overlay",
    opacity=0.8,
    height=420,
)
fig.show()

print(f"Total events : {len(df_norm):,}")
print(f"Unique hosts : {df_norm['hostname'].nunique()}")
print(f"Event ID types: {df_norm['event_id'].nunique()}")
print(df_norm[['event_id','hostname','channel','severity','message_len']].describe())
"""))

# ── STAGE 2 ──────────────────────────────────────────────────────────────────
cells.append(md("""---
## 🚨 Stage 2 — Anomaly Detection (Isolation Forest + UMAP)

Trains an **Isolation Forest** on the engineered feature matrix (no labels needed).
Events in the bottom `CONTAMINATION` percentile are flagged as anomalous.
A UMAP 2-D projection is rendered as an interactive scatter plot.
"""))

cells.append(code("""\
from pipeline.stage2_anomaly import anomaly_stage

anomalies_df = anomaly_stage(
    normalized_path = NORMALIZED_PARQUET,
    anomalies_path  = ANOMALIES_PARQUET,
    contamination   = CONTAMINATION,
)
"""))

cells.append(code("""\
import pandas as pd

anomalies_df = pd.read_parquet(ANOMALIES_PARQUET)

# Top anomalous event types
top = (
    anomalies_df.groupby("event_id")
    .agg(count=("event_id","size"),
         avg_if_score=("anomaly_score","mean"),
         unique_hosts=("hostname","nunique"))
    .sort_values("avg_if_score")
    .head(10)
)
display(top)
"""))

# ── STAGE 3 ──────────────────────────────────────────────────────────────────
cells.append(md("""---
## 🔬 Stage 3 — BERTopic (Semantic Topic Modelling)

Runs BERTopic on the anomalous event corpus:
1. **Clean** log text (strip GUIDs, hex, paths, timestamps)
2. **Embed** with `all-MiniLM-L6-v2` on GPU
3. **Cluster** with HDBSCAN
4. **Label** topics with c-TF-IDF keywords
5. **Visualise** — bar chart, inter-topic map, heatmap
"""))

cells.append(code("""\
import os, re, nltk, numpy as np, pandas as pd
from tqdm.auto import tqdm
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

# ── Text cleaning helpers ────────────────────────────────────────────────────
_GUID_RE = re.compile(r"\\{[0-9a-fA-F\\-]{8,}\\}")
_HEX_RE  = re.compile(r"\\b0x[0-9a-fA-F]+\\b")
_TS_RE   = re.compile(r"\\d{4}-\\d{2}-\\d{2}[T ]\\d{2}:\\d{2}:\\d{2}(?:\\.\\d+)?")
_PATH_RE = re.compile(r"[A-Za-z]:\\\\\\\\(?:[^\\s\\r\\n|,\\\\\\\\]+\\\\\\\\)*([^\\s\\r\\n|,\\\\\\\\]+)")
_NUM_RE  = re.compile(r"\\b\\d+\\b")
_WS_RE   = re.compile(r"\\s+")

try:
    from nltk.corpus import stopwords as _sw
    _STOP = set(_sw.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords as _sw
    _STOP = set(_sw.words("english"))

_DOMAIN_NOISE = {
    "rulename", "utctime", "processguid", "processid", "image",
    "targetprocessid", "targetprocessguid", "sourcename", "channel",
    "keywords", "opcodevalue", "severityvalue", "eventreceivedtime",
    "sourcemodulename", "sourcemoduletype", "task", "threadid",
    "recordnumber", "executionprocessid", "providerguid",
    "timestamp", "version", "none", "null", "true", "false",
}

def _clean(text):
    text = _PATH_RE.sub(lambda m: " " + m.group(1).lower() + " ", text)
    text = _GUID_RE.sub(" ", text)
    text = _HEX_RE.sub(" hexval ", text)
    text = _TS_RE.sub(" ", text)
    text = _NUM_RE.sub(" ", text)
    text = re.sub(r"[^a-zA-Z\\s]", " ", text)
    text = _WS_RE.sub(" ", text).strip().lower()
    tokens = [w for w in text.split()
              if w not in _STOP and w not in _DOMAIN_NOISE and len(w) > 2]
    return " ".join(tokens) if tokens else "unknown_event"

def _col(df, name):
    '''Safe column accessor - returns empty strings if column is absent.'''
    return df[name].fillna("") if name in df.columns else pd.Series("", index=df.index)

# ── BERTopic pipeline ────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(TOPICS_PARQUET),   exist_ok=True)
os.makedirs(TOPIC_MODEL_DIR, exist_ok=True)

print("📥  Loading anomalies …")
df_a = pd.read_parquet(ANOMALIES_PARQUET)
print(f"    Shape: {df_a.shape}")
print(f"    Columns: {list(df_a.columns)}")

print("🧹  Cleaning log text …")
corpus_raw = (
    _col(df_a, "message") + " " +
    _col(df_a, "image_base") + " " +
    _col(df_a, "target_image_base") + " " +
    _col(df_a, "target_object").apply(
        lambda x: x.split("\\\\")[-1].lower() if isinstance(x, str) and x else ""
    )
)
docs = corpus_raw.apply(_clean).tolist()
print(f"    Corpus size: {len(docs):,}")
print(f"    Sample    : {docs[0][:120]}")

umap_m = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
              metric="cosine", random_state=42, low_memory=True)
hdbscan_m = HDBSCAN(min_cluster_size=15, metric="euclidean",
                    cluster_selection_method="eom", prediction_data=True)
vectorizer_m = CountVectorizer(stop_words="english", min_df=2,
                               ngram_range=(1, 2), max_features=10_000)
topic_model = BERTopic(
    embedding_model=SentenceTransformer("all-MiniLM-L6-v2"),
    umap_model=umap_m, hdbscan_model=hdbscan_m, vectorizer_model=vectorizer_m,
    top_n_words=10, calculate_probabilities=True, verbose=True,
)

print("\\n🔬  Fitting BERTopic …")
topics, probs = topic_model.fit_transform(docs)

df_a = df_a.copy()
df_a["topic"]      = topics
df_a["topic_prob"] = [float(p.max()) if hasattr(p, "max") else float(p) for p in probs]

n_topics = len(set(topics)) - (1 if -1 in topics else 0)
print(f"\\n✅  Discovered {n_topics} topics  (topic -1 = noise/outliers)")
print(topic_model.get_topic_info().head(12).to_string(index=False))

# Visualisations
fig_bar = topic_model.visualize_barchart(top_n_topics=min(12, n_topics), n_words=8)
fig_bar.update_layout(template="plotly_dark", title="📊 Security Event Topics — Top Keywords")
fig_bar.show()
if n_topics >= 2:
    topic_model.visualize_topics().show()
    topic_model.visualize_heatmap().show()

df_a.to_parquet(TOPICS_PARQUET, index=False)
topic_model.save(TOPIC_MODEL_DIR, serialization="safetensors",
                 save_ctfidf=True, save_embedding_model="all-MiniLM-L6-v2")
print(f"\\n💾  Saved → {TOPICS_PARQUET}")
topics_df    = df_a
"""))

cells.append(code("""\
# Merge topic keywords into the dataframe for Stage 4
from pipeline.stage3_topics import get_topic_summary
import pandas as pd

topics_df = pd.read_parquet(TOPICS_PARQUET)
kw_map    = get_topic_summary(topic_model, top_n=8)

topics_df["topic_keywords"] = topics_df["topic"].map(
    lambda t: kw_map.get(t, "unknown")
)
topics_df.to_parquet(TOPICS_PARQUET, index=False)

print("Topic keyword map (first 8 topics):")
for tid, kw in list(kw_map.items())[:8]:
    count = (topics_df["topic"] == tid).sum()
    print(f"  Topic {tid:3d} ({count:5,} events): {kw}")
"""))

# ── STAGE 4a ─────────────────────────────────────────────────────────────────
cells.append(md("""---
## 📚 Stage 4a — Build RAG Knowledge Base (ChromaDB)

Downloads and indexes **5 cybersecurity knowledge sources**:

| Collection | Source | Content |
|---|---|---|
| `mitre_attack` | MITRE ATT&CK v14 | ~700 techniques + descriptions |
| `mitre_d3fend` | MITRE D3FEND | Defensive countermeasures |
| `mitre_car` | MITRE CAR | Detection analytics & pseudocode |
| `cisa_kev` | CISA KEV | Known exploited CVEs + required actions |
| `sigma_rules` | SigmaHQ | 3000+ YAML detection rules |

> ⏱️ This cell takes **5–15 min** (cloning Sigma is the slow part). Run once; ChromaDB persists to disk.
"""))

cells.append(code("""\
from pipeline.stage4a_rag_kb import build_knowledge_base

chroma_client = build_knowledge_base()
"""))

cells.append(code("""\
# Verify collections
from pipeline.stage4a_rag_kb import _get_client, query_all_collections

chroma_client = _get_client()
print("ChromaDB collections:")
for col in chroma_client.list_collections():
    print(f"  {col.name:20s} → {col.count():,} docs")

# Quick test query
test_ctx = query_all_collections(chroma_client, "lsass process access credential dump", top_k=2)
print("\\nTest query (lsass credential dump):")
print(test_ctx[:800])
"""))

# ── STAGE 4b ─────────────────────────────────────────────────────────────────
cells.append(md("""---
## 🤖 Stage 4b — LLM Threat Analysis (DSPy + Ollama + RAG)

### What happens per anomalous event:
1. **Build context** string from event fields
2. **Retrieve** relevant docs from all ChromaDB collections (semantic search)
3. **DSPy `ChainOfThought`** reasons step-by-step over context + retrieved docs
4. **Ollama `llama3`** generates structured output:
   - `threat_analysis` — what the threat is
   - `mitre_technique` — ATT&CK ID + name
   - `remediation_steps` — numbered list
   - `severity_rating` — Critical / High / Medium / Low

> 🎯 **DSPy `BootstrapFewShot`** auto-optimises prompt selection using 2 APT29 labelled examples.
"""))

cells.append(code("""\
from pipeline.stage4b_llm import llm_analysis_stage

results = llm_analysis_stage(
    topics_path   = TOPICS_PARQUET,
    results_path  = RESULTS_JSON,
    skip_optimize = False,   # set True to skip BootstrapFewShot (faster)
)
"""))

# ── RESULTS DASHBOARD ────────────────────────────────────────────────────────
cells.append(md("""---
## 📊 Results Dashboard

Summary visualisations of the LLM analysis output.
"""))

cells.append(code("""\
import json, pandas as pd
import plotly.express as px
import plotly.graph_objects as go

with open(RESULTS_JSON) as f:
    results = json.load(f)

res_df = pd.DataFrame(results)
display(res_df[["topic_id","event_id","hostname","mitre_technique","severity_rating"]].head(20))
"""))

cells.append(code("""\
# Severity breakdown
sev_counts = res_df["severity_rating"].str.extract(r"(Critical|High|Medium|Low)")[0].value_counts()
fig = px.pie(
    values=sev_counts.values,
    names=sev_counts.index,
    title="🚦 Severity Distribution of Detected Anomalies",
    color=sev_counts.index,
    color_discrete_map={"Critical":"#e94560","High":"#f5a623","Medium":"#f0e130","Low":"#1db954"},
    template="plotly_dark",
    hole=0.4,
)
fig.update_layout(height=420)
fig.show()
"""))

cells.append(code("""\
# MITRE technique frequency
tech_counts = res_df["mitre_technique"].value_counts().head(12)
fig = px.bar(
    x=tech_counts.values,
    y=tech_counts.index,
    orientation="h",
    title="⚔️ Most Frequent MITRE ATT&CK Techniques",
    template="plotly_dark",
    color=tech_counts.values,
    color_continuous_scale="reds",
    height=500,
    labels={"x":"Count","y":"Technique"},
)
fig.update_layout(showlegend=False, yaxis=dict(autorange="reversed"))
fig.show()
"""))

cells.append(code("""\
# Per-topic severity heatmap
import numpy as np

pivot = res_df.assign(
    sev_num=res_df["severity_rating"].str.extract(r"(Critical|High|Medium|Low)")[0].map(
        {"Critical":4,"High":3,"Medium":2,"Low":1}
    )
).groupby(["topic_id","event_id"])["sev_num"].mean().unstack(fill_value=0)

fig = go.Figure(go.Heatmap(
    z=pivot.values,
    x=[str(c) for c in pivot.columns],
    y=[f"Topic {r}" for r in pivot.index],
    colorscale="Reds",
    colorbar_title="Severity",
))
fig.update_layout(
    title="🔥 Severity Heatmap — Topic × EventID",
    template="plotly_dark",
    height=max(300, len(pivot)*40),
)
fig.show()
"""))

cells.append(md("""---
## ✅ Pipeline Complete

All outputs saved to `/content/data/`:
| File | Description |
|---|---|
| `normalized.parquet` | All ~1M events, flat schema |
| `anomalies.parquet` | Anomalous events with IF scores |
| `anomalies_with_topics.parquet` | Anomalies + BERTopic labels |
| `chroma_db/` | Persistent vector store (5 collections) |
| `bertopic_model/` | Saved BERTopic model |
| `llm_results.json` | Structured LLM threat analysis per anomaly |
"""))

# ─────────────────────────────────────────────────────────────────────────────
nb.cells = cells

out_path = "main.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"[OK] Notebook written -> {out_path}  ({len(cells)} cells)")
