"""
Stage 4b: LLM Analysis — DSPy + Ollama
Defines the DSPy signature, ChainOfThought module, and BootstrapFewShot
optimizer. Runs threat analysis on the top anomalous events per BERTopic cluster.
"""
import os, json
import pandas as pd
import dspy
import chromadb
from IPython.display import display, Markdown

try:
    from pipeline.stage4a_rag_kb import query_all_collections, _get_client, _get_embedder
except ImportError:
    pass  # In notebook mode, these functions are already in the global namespace

# ── Config ────────────────────────────────────────────────────────────────────
TOPICS_PARQUET  = "/content/data/anomalies_with_topics.parquet"
RESULTS_JSON    = "/content/data/llm_results.json"
OLLAMA_MODEL    = "llama3"           # swap to mistral / phi3 if preferred
OLLAMA_BASE_URL = "http://localhost:11434"
TOP_N_TOPICS    = 12                 # topics to analyse
EVENTS_PER_TOPIC = 3                 # worst-scoring events per topic
RAG_TOP_K       = 5
# ──────────────────────────────────────────────────────────────────────────────


# ── Ollama bootstrap ──────────────────────────────────────────────────────────

def configure_dspy(model: str = OLLAMA_MODEL):
    """Wire DSPy to the local Ollama endpoint."""
    lm = dspy.LM(
        model=f"ollama_chat/{model}",
        api_base=OLLAMA_BASE_URL,
        api_key="ollama",
        temperature=0.15,
        max_tokens=1024,
    )
    dspy.configure(lm=lm)
    print(f"  DSPy configured -> ollama/{model}")
    return lm


# ── DSPy Signature ────────────────────────────────────────────────────────────

class ThreatAnalysis(dspy.Signature):
    """
    You are a senior threat-intelligence analyst.
    Given a Windows security anomaly, related topic keywords, and retrieved
    threat-intel context, produce a structured analysis.
    """
    anomaly_context: str  = dspy.InputField(
        desc="EventID, process image, hostname, account, granted access, "
             "and the raw Sysmon message snippet."
    )
    topic_keywords: str   = dspy.InputField(
        desc="BERTopic cluster keywords that describe the group this event belongs to."
    )
    retrieved_docs: str   = dspy.InputField(
        desc="Relevant passages from MITRE ATT&CK, D3FEND, CAR, CISA KEV, "
             "and Sigma rules retrieved via semantic search."
    )

    threat_analysis: str    = dspy.OutputField(
        desc="1–3 sentence analysis of the likely threat this anomaly represents."
    )
    mitre_technique: str    = dspy.OutputField(
        desc="Most applicable MITRE ATT&CK technique, e.g. 'T1055 - Process Injection'."
    )
    remediation_steps: str  = dspy.OutputField(
        desc="Numbered list of 3–5 concrete detection or remediation steps."
    )
    severity_rating: str    = dspy.OutputField(
        desc="One of: Critical / High / Medium / Low — with a one-sentence rationale."
    )


# ── DSPy Module ───────────────────────────────────────────────────────────────

class SecurityAnalyzer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.analyze = dspy.ChainOfThought(ThreatAnalysis)

    def forward(self, anomaly_context: str,
                topic_keywords: str,
                retrieved_docs: str) -> dspy.Prediction:
        return self.analyze(
            anomaly_context=anomaly_context,
            topic_keywords=topic_keywords,
            retrieved_docs=retrieved_docs,
        )


# ── Few-shot examples for BootstrapFewShot ───────────────────────────────────

FEW_SHOT_EXAMPLES = [
    dspy.Example(
        anomaly_context=(
            "EventID: 10 | Process: C:\\Windows\\System32\\lsass.exe | "
            "Target: lsass.exe | GrantedAccess: 0x1FFFFF | "
            "Message: Process accessed lsass with full handle rights."
        ),
        topic_keywords="lsass, credential, access, memory, dump, mimikatz, process",
        retrieved_docs="[MITRE_ATTACK] T1003.001 - LSASS Memory: Adversaries may "
                       "attempt to access credential material stored in LSASS.",
        threat_analysis=(
            "This event strongly indicates credential dumping via direct LSASS "
            "memory access, consistent with tools like Mimikatz or ProcDump. "
            "Full handle rights (0x1FFFFF) are rarely required by legitimate processes."
        ),
        mitre_technique="T1003.001 - OS Credential Dumping: LSASS Memory",
        remediation_steps=(
            "1. Enable Credential Guard (Windows 10/11).\n"
            "2. Restrict LSASS access via Protected Process Light (PPL).\n"
            "3. Alert on GrantedAccess 0x1FFFFF targeting lsass.exe.\n"
            "4. Deploy Sigma rule 'win_lsass_access_non_system_account'.\n"
            "5. Review process lineage for the accessing process."
        ),
        severity_rating="Critical — Direct credential theft enables lateral movement.",
    ).with_inputs("anomaly_context", "topic_keywords", "retrieved_docs"),

    dspy.Example(
        anomaly_context=(
            "EventID: 13 | Process: reg.exe | "
            "TargetObject: HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run\\backdoor | "
            "Message: Registry value set for persistence."
        ),
        topic_keywords="registry, persistence, run, key, startup, autorun",
        retrieved_docs="[MITRE_ATTACK] T1547.001 - Registry Run Keys: Adversaries "
                       "may achieve persistence by adding a program to a Run key.",
        threat_analysis=(
            "A new Run key was written by reg.exe, a classic persistence mechanism. "
            "The key name 'backdoor' is highly suspicious and warrants immediate review."
        ),
        mitre_technique="T1547.001 - Boot or Logon Autostart: Registry Run Keys",
        remediation_steps=(
            "1. Remove the malicious Run key immediately.\n"
            "2. Alert on unexpected writes to HKLM\\SOFTWARE\\...\\Run\\.\n"
            "3. Audit reg.exe invocations not launched by administrators.\n"
            "4. Use Sigma rule 'win_registry_run_key_modification'.\n"
            "5. Investigate the parent process that spawned reg.exe."
        ),
        severity_rating="High — Persistence mechanism allows re-infection after reboot.",
    ).with_inputs("anomaly_context", "topic_keywords", "retrieved_docs"),
]


def optimize_analyzer(analyzer: SecurityAnalyzer,
                      examples: list = FEW_SHOT_EXAMPLES) -> SecurityAnalyzer:
    """Run BootstrapFewShot to auto-select best prompts."""
    print("  Running DSPy BootstrapFewShot optimisation...")
    optimizer = dspy.BootstrapFewShot(max_bootstrapped_demos=2,
                                      max_labeled_demos=2)
    # Metric: response is non-empty (adjust with a real eval if labels exist)
    def metric(example, pred, trace=None):
        return (bool(pred.threat_analysis) and
                bool(pred.mitre_technique) and
                bool(pred.remediation_steps))

    optimized = optimizer.compile(analyzer, trainset=examples, metric=metric)
    print("  Optimisation complete.")
    return optimized


# ── Context builder ───────────────────────────────────────────────────────────

def build_anomaly_context(row: pd.Series) -> str:
    ga = row.get("granted_access", 0)
    ga_hex = hex(int(ga)) if ga else "N/A"
    return (
        f"EventID: {row['event_id']} | "
        f"Hostname: {row['hostname']} | "
        f"Process: {row['image']} | "
        f"TargetImage: {row.get('target_image', '')} | "
        f"TargetObject: {str(row.get('target_object', ''))[:80]} | "
        f"Account: {row['account_name']} ({row['domain']}) | "
        f"GrantedAccess: {ga_hex} | "
        f"IsolationForest score: {row.get('anomaly_score', 'N/A'):.4f} | "
        f"Message: {str(row.get('message', ''))[:400]}"
    )


# ── Main analysis loop ────────────────────────────────────────────────────────

def display_result(result: dict, idx: int):
    md = f"""
---
### Analysis #{idx+1} — Topic {result['topic_id']}
**Event:** `{result['event_id']}` on `{result['hostname']}`
**Topic keywords:** _{result['topic_keywords']}_

**Threat Analysis**
{result['threat_analysis']}

**MITRE ATT&CK Technique**
`{result['mitre_technique']}`

**Remediation Steps**
{result['remediation_steps']}

**Severity:** {result['severity_rating']}
---
"""
    display(Markdown(md))


def llm_analysis_stage(topics_path: str    = TOPICS_PARQUET,
                       results_path: str   = RESULTS_JSON,
                       skip_optimize: bool = False) -> list[dict]:
    """
    End-to-end Stage 4b entry point.
    Returns list of result dicts, also saved to JSON.
    """
    # ── Setup ─────────────────────────────────────────────────────────────────
    configure_dspy(OLLAMA_MODEL)

    client   = _get_client()
    analyzer = SecurityAnalyzer()

    if not skip_optimize:
        try:
            analyzer = optimize_analyzer(analyzer)
        except Exception as e:
            print(f"  Optimisation skipped: {e}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\n  Loading {topics_path}...")
    df = pd.read_parquet(topics_path)
    valid_topics = sorted(
        [t for t in df["topic"].unique() if t != -1],
        key=lambda t: df[df["topic"] == t]["anomaly_score"].mean()
    )[:TOP_N_TOPICS]

    print(f"  Analysing {len(valid_topics)} topics × {EVENTS_PER_TOPIC} events each...\n")

    results = []

    for topic_id in valid_topics:
        t_df = df[df["topic"] == topic_id].nsmallest(EVENTS_PER_TOPIC, "anomaly_score")

        # Get topic keywords from BERTopic (stored in df if available, else from model)
        topic_kw = df[df["topic"] == topic_id]["topic_keywords"].iloc[0] \
                   if "topic_keywords" in df.columns else f"topic_{topic_id}"

        for _, row in t_df.iterrows():
            ctx   = build_anomaly_context(row)
            query = f"{ctx} {topic_kw}"
            docs  = query_all_collections(client, query, top_k=RAG_TOP_K)

            try:
                pred = analyzer(
                    anomaly_context=ctx,
                    topic_keywords=topic_kw,
                    retrieved_docs=docs,
                )
                rec = {
                    "topic_id"         : int(topic_id),
                    "topic_keywords"   : topic_kw,
                    "event_id"         : int(row["event_id"]),
                    "hostname"         : row["hostname"],
                    "anomaly_score"    : float(row.get("anomaly_score", 0)),
                    "threat_analysis"  : pred.threat_analysis,
                    "mitre_technique"  : pred.mitre_technique,
                    "remediation_steps": pred.remediation_steps,
                    "severity_rating"  : pred.severity_rating,
                }
                results.append(rec)
                display_result(rec, len(results) - 1)

            except Exception as e:
                print(f"  Topic {topic_id} event failed: {e}")

    # ── Save results ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved -> {results_path}")
    print(f"    Total analyses: {len(results)}")
    return results


if __name__ == "__main__":
    llm_analysis_stage()
