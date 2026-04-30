"""
generate_notebook.py
Assembles main.ipynb from the four pipeline stage modules.
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
    nltk pyyaml requests tqdm

import nltk
nltk.download('stopwords', quiet=True)
print("All dependencies installed")
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
print(f"Downloading dataset from {DATA_REPO_URL}...")
!wget -q {DATA_REPO_URL} -O {ZIP_PATH}

print("Extracting dataset...")
with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(EXTRACT_DIR)

DATA_PATH = glob.glob(f"{EXTRACT_DIR}/**/*.json", recursive=True)[0]
print(f"Found dataset: {DATA_PATH}")
"""))

cells.append(md("### Global configuration"))

cells.append(code("""\
# ─── EDIT THESE PATHS IF NEEDED ───────────────────────────────────────────

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
print("Config ready")
"""))

# ── STAGE 1 ──────────────────────────────────────────────────────────────────
cells.append(md("""---
## Stage 1 — Parse & Normalize

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

cells.append(code(read_pipeline_module("stage1_parse.py")))

cells.append(code("""\
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
## Stage 2 — Anomaly Detection (Isolation Forest + UMAP)

Trains an **Isolation Forest** on the engineered feature matrix (no labels needed).
Events in the bottom `CONTAMINATION` percentile are flagged as anomalous.
A UMAP 2-D projection is rendered as an interactive scatter plot.
"""))

cells.append(code(read_pipeline_module("stage2_anomaly.py")))

cells.append(code("""\
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
# Install Ollama
!sudo apt-get install -y zstd
!curl -fsSL https://ollama.com/install.sh | sh

# Start the server in the background
import subprocess
import time
subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(4)

# Pull the model (you will see the progress bar here!)
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
