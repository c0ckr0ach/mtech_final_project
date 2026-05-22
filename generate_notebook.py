"""
generate_notebook.py
Assembles main.ipynb from all pipeline stage modules (Stages 0–6).
Run: python generate_notebook.py
"""
import nbformat as nbf
import os
import re

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10.0"},
    "accelerator": "GPU",
}

def md(src):   return nbf.v4.new_markdown_cell(src)
def code(src): return nbf.v4.new_code_cell(src)

def read_pipeline_module(module_name):
    """Read a pipeline python file, strip the __main__ block, and return as a string."""
    filepath = os.path.join("pipeline", module_name)
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    # Strip the __main__ block
    content = re.split(r'^if __name__ == "__main__":', content, flags=re.MULTILINE)[0]
    return content.strip()


cells = []

cells.append(md("""# Security Log Analysis Pipeline"""))


cells.append(code("""\
# Install all pipeline dependencies
!pip install -q \\
    pandas pyarrow \\
    scikit-learn umap-learn \\
    plotly \\
    bertopic sentence-transformers hdbscan \\
    chromadb \\
    dspy-ai \\
    nltk pyyaml requests tqdm \\
    torch \\
    ragas datasets \\
    matplotlib seaborn \\
    langchain-community langchain-ollama langchain-huggingface \\
    litellm openai

import nltk
nltk.download('stopwords', quiet=True)
"""))

cells.append(md("### Fetch and Extract Dataset"))

cells.append(code("""\
import os
import zipfile
import glob

DATA_REPO_URL = "https://raw.githubusercontent.com/OTRF/Security-Datasets/master/datasets/compound/apt29/day1/apt29_evals_day1_manual.zip"
ZIP_PATH = "/content/apt29_evals_day1_manual.zip"
EXTRACT_DIR = "/content/data_path"

os.makedirs(EXTRACT_DIR, exist_ok=True)
!wget -q {DATA_REPO_URL} -O {ZIP_PATH}

print("extracting dataset")
with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(EXTRACT_DIR)

DATA_PATH = glob.glob(f"{EXTRACT_DIR}/**/*.json", recursive=True)[0]
print(f"dataset loaded")
"""))

cells.append(md("### Global configuration"))

cells.append(code("""\
NORMALIZED_PARQUET = "/content/data/normalized.parquet"
LABELED_PARQUET    = "/content/data/labeled.parquet"
ANOMALIES_PARQUET  = "/content/data/anomalies.parquet"
TOPICS_PARQUET     = "/content/data/anomalies_with_topics.parquet"
CHROMA_DIR         = "/content/data/chroma_db"
RESULTS_JSON       = "/content/data/llm_results.json"
TOPIC_MODEL_DIR    = "/content/data/bertopic_model"
METRICS_JSON       = "/content/data/metrics_report.json"
LLM_METRICS_JSON   = "/content/data/llm_metrics.json"
FIGURES_DIR        = "/content/data/figures"

# Stage 4b — local Ollama model for threat analysis
OLLAMA_MODEL       = "llama3"
# Stage 6 — Mistral API model used as RAGAS LLM judge
MISTRAL_MODEL      = "mistral-large-latest"  # swap to mistral-small-latest for dev runs

CONTAMINATION      = 0.05           # fraction flagged as anomalous
TOP_N_TOPICS       = 12             # topics passed to LLM
EVENTS_PER_TOPIC   = 3              # worst anomalies per topic

import os
os.makedirs("/content/data", exist_ok=True)
os.makedirs(FIGURES_DIR,    exist_ok=True)
"""))

# ── STAGE 1 ──────────────────────────────────────────────────────────────────
cells.append(md("""---
## Stage 1 — Parse & Normalize
"""))

cells.append(code(read_pipeline_module("stage1_parse.py")))

cells.append(code("""\
df_norm = parse_stage(
    data_path  = DATA_PATH,
    out_path   = NORMALIZED_PARQUET,
    chunk_size = 50_000,
)
df_norm.head(3)
"""))

# ── STAGE 0 ──────────────────────────────────────────────────────────────────
cells.append(md("""---
## Stage 0 — Ground Truth Labeling

Assigns a `ground_truth` label to every event in `normalized.parquet` using
a **three-tier rule set** derived from the MITRE ATT&CK APT29 technique mapping:

| Tier | Rule | Label |
|---|---|---|
| 1 — High confidence | EventID 10 targeting `lsass.exe`, EventID 13 writing to `\\Run\\` keys, etc. | `1` (Malicious) |
| 2 — Heuristic | Known offensive tool image names (`mimikatz.exe`, `psexec.exe`, etc.) | `1` (Malicious) |
| 3 — Default | All other events | `0` (Benign) |

> The OTRF APT29 Mordor dataset is **semi-labeled by design** — both adversarial and
> normal endpoint events are captured in the same file. No second dataset is needed.
"""))

cells.append(code(read_pipeline_module("stage0_label.py")))

cells.append(code("""\
df_labeled = label_stage(
    normalized_path = NORMALIZED_PARQUET,
    labeled_path    = LABELED_PARQUET,
)
"""))

cells.append(code("""\
import pandas as pd
import plotly.express as px

df_labeled = pd.read_parquet(LABELED_PARQUET)

# Class distribution bar chart
counts = df_labeled["ground_truth"].value_counts().rename({0: "Benign", 1: "Malicious"})
fig = px.bar(
    x=counts.index,
    y=counts.values,
    color=counts.index,
    color_discrete_map={"Benign": "#00b4d8", "Malicious": "#e94560"},
    title="Ground Truth Class Distribution (APT29 Mordor Dataset)",
    labels={"x": "Class", "y": "Event Count"},
    template="plotly_dark",
    height=380,
)
fig.show()
print(f"Total events   : {len(df_labeled):,}")
print(f"Benign (0)     : {(df_labeled['ground_truth']==0).sum():,}")
print(f"Malicious (1)  : {(df_labeled['ground_truth']==1).sum():,}")
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
## Stage 2 — Comparative Anomaly Detection

Runs **three independent unsupervised anomaly detectors** on the same feature matrix:

| Model | Algorithm | Key Property |
|---|---|---|
| A | **Isolation Forest** | Ensemble tree-based; efficient on high-dimensional data |
| B | **One-Class SVM** | Kernel-based; non-linear decision boundary |
| C | **Deep Autoencoder** | Reconstruction-error; learns compact latent representation (GPU) |

Each model produces its own anomaly score column. The **union** of all three flags
forms the final anomaly set passed to downstream stages.

> Ground truth labels (from Stage 0) are carried through but **not used** for training —
> all models remain fully unsupervised. Labels are used only in Stage 5 evaluation.
"""))

cells.append(code(read_pipeline_module("stage2_anomaly.py")))

cells.append(code("""\
anomalies_df = anomaly_stage(
    labeled_path   = LABELED_PARQUET,
    anomalies_path = ANOMALIES_PARQUET,
    contamination  = CONTAMINATION,
)
"""))

cells.append(code("""\
import pandas as pd

anomalies_df = pd.read_parquet(ANOMALIES_PARQUET)

# Top anomalous event types — all model scores
top = (
    anomalies_df.groupby("event_id")
    .agg(
        count=("event_id", "size"),
        avg_if_score=("if_anomaly_score", "mean"),
        avg_ae_score=("ae_anomaly_score", "mean"),
        unique_hosts=("hostname", "nunique"),
    )
    .sort_values("avg_if_score")
    .head(10)
)
display(top)
"""))

# ── STAGE 3 ──────────────────────────────────────────────────────────────────
cells.append(md("""---
## Stage 3 — BERTopic (Semantic Topic Modelling)

Runs BERTopic on the anomalous event corpus:
1. **Clean** log text (strip GUIDs, hex, paths, timestamps)
2. **Embed** with `all-MiniLM-L6-v2` on GPU
3. **Cluster** with HDBSCAN
4. **Label** topics with c-TF-IDF keywords
5. **Visualise** — bar chart, inter-topic map, heatmap
"""))

cells.append(code(read_pipeline_module("stage3_topics.py")))

cells.append(code("""\
df_a, topic_model = topic_stage(
    anomalies_path = ANOMALIES_PARQUET,
    topics_path    = TOPICS_PARQUET,
    model_dir      = TOPIC_MODEL_DIR,
)
"""))

cells.append(code("""\
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
## Stage 4a — Build RAG Knowledge Base (ChromaDB)

Downloads and indexes **5 cybersecurity knowledge sources**:

| Collection | Source | Content |
|---|---|---|
| `mitre_attack` | MITRE ATT&CK v14 | ~700 techniques + descriptions |
| `mitre_d3fend` | MITRE D3FEND | Defensive countermeasures |
| `mitre_car` | MITRE CAR | Detection analytics & pseudocode |
| `cisa_kev` | CISA KEV | Known exploited CVEs + required actions |
| `sigma_rules` | SigmaHQ | 3000+ YAML detection rules |

> This cell takes **5–15 min** (cloning Sigma is the slow part). Run once; ChromaDB persists to disk.
"""))

cells.append(code(read_pipeline_module("stage4a_rag_kb.py")))

cells.append(code("""\
chroma_client = build_knowledge_base()
"""))

cells.append(code("""\
# Verify collections
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

"""))

cells.append(code("""\
# Install Ollama (used by Stage 4b for local llama3 inference)
!sudo apt-get install -y zstd
!curl -fsSL https://ollama.com/install.sh | sh

# Start the server in the background
import subprocess
import time
subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(4)

# Pull llama3 (~4.7 GB — takes 3–6 min on Colab)
!ollama pull llama3
"""))

cells.append(md("""---
## Stage 4b — LLM Threat Analysis (DSPy + Ollama + RAG)

### What happens per anomalous event:
1. **Build context** string from event fields
2. **Retrieve** relevant docs from all ChromaDB collections (semantic search)
3. **DSPy `ChainOfThought`** reasons step-by-step over context + retrieved docs
4. **Ollama `llama3`** generates structured output:
   - `threat_analysis` — what the threat is
   - `mitre_technique` — ATT&CK ID + name
   - `remediation_steps` — numbered list
   - `severity_rating` — Critical / High / Medium / Low

> **DSPy `BootstrapFewShot`** auto-optimises prompt selection using 2 APT29 labelled examples.
"""))

cells.append(code(read_pipeline_module("stage4b_llm.py")))

cells.append(code("""\
results = llm_analysis_stage(
    topics_path   = TOPICS_PARQUET,
    results_path  = RESULTS_JSON,
    skip_optimize = False,   # set True to skip BootstrapFewShot (faster)
)
"""))

# ── STAGE 5 ──────────────────────────────────────────────────────────────────
cells.append(md("""---
## Stage 5 — Quantitative ML Evaluation

Compares the three anomaly detectors against the ground truth labels (Stage 0)
using the following metrics:

| Metric | Description |
|---|---|
| **ROC-AUC** | Threshold-free discrimination ability |
| **PR-AUC** | Better than ROC-AUC for imbalanced security log data |
| **F1-Score** | Harmonic mean of Precision and Recall at optimal threshold |
| **Precision** | Fraction of flagged events that are actually malicious |
| **Recall** | Fraction of malicious events that were caught |
| **FPR** | False Positive Rate — the SOC analyst burden metric |

Generates thesis-quality figures: ROC curves, PR curves, and confusion matrices.
"""))

cells.append(code(read_pipeline_module("stage5_evaluation.py")))

cells.append(code("""\
metrics_report = evaluation_stage(
    anomalies_path = ANOMALIES_PARQUET,
    figures_dir    = FIGURES_DIR,
    metrics_path   = METRICS_JSON,
)
"""))

cells.append(code("""\
# Display metrics comparison table
import json, pandas as pd
from IPython.display import display, Image

with open(METRICS_JSON) as f:
    report = json.load(f)

metrics_df = pd.DataFrame(report).T
metrics_df.index.name = "Model"
print("\\n── Evaluation Metrics Summary ──────────────────────────")
display(metrics_df.style.background_gradient(cmap='RdYlGn', axis=0)
        .format("{:.4f}"))

# Show figures inline
print("\\n── ROC Curves ──")
display(Image(FIGURES_DIR + "/roc_curve_comparison.png"))
print("\\n── Precision-Recall Curves ──")
display(Image(FIGURES_DIR + "/pr_curve_comparison.png"))
print("\\n── Confusion Matrices ──")
display(Image(FIGURES_DIR + "/confusion_matrices.png"))
"""))

# ── STAGE 6 ──────────────────────────────────────────────────────────────────
cells.append(md("""---
## Stage 6 — LLM & RAG Evaluation (RAGAS)

Evaluates the quality of the LLM threat analyses using the
**RAGAS (Retrieval-Augmented Generation Assessment)** framework.

Two conditions are compared:
- **With RAG**: LLM answers generated using ChromaDB-retrieved context (MITRE ATT&CK, Sigma rules, etc.)
- **Without RAG** (baseline): Same questions, same LLM, but **no context** provided

| Metric | Interpretation |
|---|---|
| **Faithfulness** | Are claims grounded in retrieved docs? < 0.7 = hallucination |
| **Answer Relevancy** | Is the response on-topic for the anomaly? |
| **Context Precision** | Are the most useful RAG docs ranked highest? |
| **Context Recall** | Did the retriever surface all necessary information? |

> The Δ (delta) between with-RAG and without-RAG Faithfulness is the thesis's
> key quantitative contribution on the LLM side.
"""))

cells.append(code(read_pipeline_module("stage6_llm_eval.py")))

cells.append(code("""\
# ── Mistral API key setup (Stage 6) ─────────────────────────────────────────
# The key is stored as a Colab secret named MISTRAL_API_KEY.
# To add it: click the 🔑 (Secrets) icon in the left sidebar → New secret.
import os
try:
    from google.colab import userdata
    MISTRAL_API_KEY = userdata.get("MISTRAL_API_KEY")
except Exception:
    # Fallback: read from environment if already set (e.g. local run)
    MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
if not MISTRAL_API_KEY:
    raise ValueError(
        "MISTRAL_API_KEY is not set. "
        "Add it as a Colab secret named MISTRAL_API_KEY (the 🔑 icon in the sidebar)."
    )
os.environ["MISTRAL_API_KEY"] = MISTRAL_API_KEY
print("Mistral API key loaded ✔")
"""))

cells.append(code("""\
llm_metrics = llm_eval_stage(
    results_path = RESULTS_JSON,
    metrics_path = LLM_METRICS_JSON,
    figures_dir  = FIGURES_DIR,
    model        = MISTRAL_MODEL,        # mistral-large-latest via Mistral API
    api_key      = MISTRAL_API_KEY,
    max_samples  = 10,   # 10 samples ≈ 80 API calls; ~5–15 min via Mistral API
                         # increase to 30 for the final thesis run
)
"""))

cells.append(code("""\
import json
from IPython.display import display, Image

with open(LLM_METRICS_JSON) as f:
    llm_m = json.load(f)

print("── RAGAS Results ───────────────────────────────────────")
for condition in ["with_rag", "without_rag", "delta"]:
    print(f"\\n  [{condition.upper()}]")
    for k, v in llm_m[condition].items():
        print(f"    {k:25s}: {v:.4f}")
print("────────────────────────────────────────────────────────")

display(Image(FIGURES_DIR + "/ragas_radar_chart.png"))
display(Image(FIGURES_DIR + "/faithfulness_comparison.png"))
"""))

# ── RESULTS DASHBOARD ────────────────────────────────────────────────────────
cells.append(md("""---
## Results Dashboard

Summary visualisations of the LLM analysis output.
"""))

cells.append(code("""\
import json, pandas as pd
import plotly.express as px
import plotly.graph_objects as go

with open(RESULTS_JSON) as f:
    results = json.load(f)

res_df = pd.DataFrame(results)
res_df = res_df.rename(columns={"mitre_technique": "Attack Technique", "remediation_steps": "Fixes"})
display(res_df[["topic_id","event_id","hostname","Attack Technique","Fixes","severity_rating"]].head(20))
"""))

cells.append(code("""\
# Severity breakdown
sev_counts = res_df["severity_rating"].str.extract(r"(Critical|High|Medium|Low)")[0].value_counts()
fig = px.pie(
    values=sev_counts.values,
    names=sev_counts.index,
    title="Severity Distribution of Detected Anomalies",
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
tech_counts = res_df["Attack Technique"].value_counts().head(12)
fig = px.bar(
    x=tech_counts.values,
    y=tech_counts.index,
    orientation="h",
    title="Most Frequent MITRE ATT&CK Techniques",
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
    title="Severity Heatmap — Topic × EventID",
    template="plotly_dark",
    height=max(300, len(pivot)*40),
)
fig.show()
"""))

cells.append(md("""---
## Pipeline Complete

All outputs saved to `/content/data/`:
| File | Description |
|---|---|
| `normalized.parquet` | All ~1M events, flat schema |
| `labeled.parquet` | All events with `ground_truth` column (Stage 0) |
| `anomalies.parquet` | Anomalies with IF / OCSVM / AE scores (Stage 2) |
| `anomalies_with_topics.parquet` | Anomalies + BERTopic labels (Stage 3) |
| `chroma_db/` | Persistent vector store — 5 KB collections (Stage 4a) |
| `bertopic_model/` | Saved BERTopic model (Stage 3) |
| `llm_results.json` | Structured LLM threat analysis per anomaly (Stage 4b) |
| `metrics_report.json` | ROC-AUC / PR-AUC / F1 per model (Stage 5) |
| `llm_metrics.json` | RAGAS scores with/without RAG (Stage 6) |
| `figures/` | All thesis-ready PNG plots |
"""))

# ─────────────────────────────────────────────────────────────────────────────
nb.cells = cells

out_path = "main.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"[OK] Notebook written -> {out_path}  ({len(cells)} cells)")
