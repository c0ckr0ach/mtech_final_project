"""
Stage 4a: RAG Knowledge Base Builder
Downloads and indexes 5 cybersecurity knowledge sources into ChromaDB.
Sources: MITRE ATT&CK, MITRE D3FEND, MITRE CAR, CISA KEV, SigmaHQ Rules
"""
import os, re, json, subprocess
import requests
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_DIR   = "/content/data/chroma_db"
EMBED_MODEL  = "all-MiniLM-L6-v2"
BATCH_SIZE   = 128
KB_DIR       = "/content/data/kb_raw"
# ──────────────────────────────────────────────────────────────────────────────

_emb_model: SentenceTransformer = None

def _get_embedder():
    global _emb_model
    if _emb_model is None:
        print("Loading embedding model...")
        _emb_model = SentenceTransformer(EMBED_MODEL)
    return _emb_model


def _get_client():
    os.makedirs(CHROMA_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_DIR)


def _upsert_collection(client: chromadb.Client,
                       name: str,
                       docs: list[str],
                       ids: list[str],
                       metas: list[dict]):
    """Create or replace a ChromaDB collection and batch-embed docs."""
    try:
        client.delete_collection(name)
    except Exception:
        pass
    col = client.create_collection(name)
    embedder = _get_embedder()

    for i in tqdm(range(0, len(docs), BATCH_SIZE), desc=f"  Indexing {name}"):
        bd = docs[i:i+BATCH_SIZE]
        bi = ids[i:i+BATCH_SIZE]
        bm = metas[i:i+BATCH_SIZE]
        emb = embedder.encode(bd, show_progress_bar=False).tolist()
        col.add(documents=bd, ids=bi, embeddings=emb, metadatas=bm)

    print(f"  {name}: {col.count():,} docs indexed")
    return col


# ── Source 1: MITRE ATT&CK ───────────────────────────────────────────────────

def load_mitre_attack(client):
    print("\nMITRE ATT&CK v14...")
    url = ("https://raw.githubusercontent.com/mitre/cti/master/"
           "enterprise-attack/enterprise-attack.json")
    data = requests.get(url, timeout=120).json()

    docs, ids, metas = [], [], []
    for obj in data["objects"]:
        if obj.get("type") != "attack-pattern":
            continue
        tech_id, tactic = "", ""
        for ref in obj.get("external_references", []):
            if ref.get("source_name") == "mitre-attack":
                tech_id = ref.get("external_id", "")
        for phase in obj.get("kill_chain_phases", []):
            tactic = phase.get("phase_name", "")
        name = obj.get("name", "")
        desc = obj.get("description", "")[:1800]
        text = f"ATT&CK {tech_id} — {name}\nTactic: {tactic}\n{desc}"
        uid  = f"attack_{tech_id}_{obj['id'][-6:]}"
        docs.append(text); ids.append(uid)
        metas.append({"source": "mitre_attack", "tech_id": tech_id,
                      "tactic": tactic, "name": name})

    _upsert_collection(client, "mitre_attack", docs, ids, metas)


# ── Source 2: MITRE D3FEND ───────────────────────────────────────────────────

def load_d3fend(client):
    print("\nMITRE D3FEND...")
    url = "https://d3fend.mitre.org/api/technique/all.json"
    try:
        data = requests.get(url, timeout=60).json()
        techniques = data.get("techniques") or data.get("data") or []
    except Exception as e:
        print(f"  D3FEND fetch failed: {e}")
        return

    docs, ids, metas = [], [], []
    for t in techniques:
        tid   = t.get("id") or t.get("d3f_id") or "unknown"
        label = t.get("label") or t.get("name") or ""
        desc  = t.get("definition") or t.get("description") or ""
        text  = f"D3FEND {tid} — {label}\n{desc}"[:2000]
        uid   = f"d3fend_{re.sub(r'[^a-zA-Z0-9_]', '_', tid)}"
        docs.append(text); ids.append(uid)
        metas.append({"source": "d3fend", "d3fend_id": tid, "name": label})

    if docs:
        _upsert_collection(client, "mitre_d3fend", docs, ids, metas)
    else:
        print("  D3FEND returned 0 usable techniques.")


# ── Source 3: MITRE CAR ──────────────────────────────────────────────────────

def load_car(client):
    print("\nMITRE CAR analytics...")
    car_dir = os.path.join(KB_DIR, "car")
    if not os.path.exists(car_dir):
        subprocess.run(
            ["git", "clone", "--depth=1",
             "https://github.com/mitre-attack/car.git", car_dir],
            check=True, capture_output=True,
        )

    import yaml
    docs, ids, metas = [], [], []
    analytics_dir = os.path.join(car_dir, "analytics")
    if not os.path.exists(analytics_dir):
        print("  CAR analytics directory not found.")
        return

    for fname in os.listdir(analytics_dir):
        if not (fname.endswith(".yaml") or fname.endswith(".yml")):
            continue
        try:
            with open(os.path.join(analytics_dir, fname), "r", encoding="utf-8") as f:
                obj = yaml.safe_load(f)
            title = obj.get("title", fname)
            desc  = obj.get("description", "")
            impl  = " ".join(
                str(i.get("code", ""))
                for i in (obj.get("implementations") or [])
                if isinstance(i, dict)
            )
            text = f"CAR — {title}\n{desc}\nDetection logic: {impl}"[:2000]
            uid  = f"car_{fname.replace('.yaml','').replace('.yml','')[:60]}"
            docs.append(text); ids.append(uid)
            metas.append({"source": "mitre_car", "title": title, "file": fname})
        except Exception:
            continue

    if docs:
        _upsert_collection(client, "mitre_car", docs, ids, metas)


# ── Source 4: CISA KEV ───────────────────────────────────────────────────────

def load_cisa_kev(client):
    print("\nCISA Known Exploited Vulnerabilities...")
    url = ("https://www.cisa.gov/sites/default/files/feeds/"
           "known_exploited_vulnerabilities.json")
    data = requests.get(url, timeout=60).json()

    docs, ids, metas = [], [], []
    for v in data.get("vulnerabilities", []):
        cve  = v.get("cveID", "unknown")
        text = (
            f"CVE: {cve} | Vendor: {v.get('vendorProject','')} | "
            f"Product: {v.get('product','')} | "
            f"Vulnerability: {v.get('vulnerabilityName','')} | "
            f"Required Action: {v.get('requiredAction','')} | "
            f"Due Date: {v.get('dueDate','')} | "
            f"Notes: {v.get('notes','')}"
        )[:2000]
        uid = f"kev_{cve}"
        docs.append(text); ids.append(uid)
        metas.append({"source": "cisa_kev", "cve_id": cve,
                      "product": v.get("product", "")})

    _upsert_collection(client, "cisa_kev", docs, ids, metas)


# ── Source 5: SigmaHQ Rules ──────────────────────────────────────────────────

def load_sigma(client):
    print("\nSigmaHQ detection rules (sparse clone)...")
    sigma_dir = os.path.join(KB_DIR, "sigma")
    if not os.path.exists(sigma_dir):
        subprocess.run(
            ["git", "clone", "--depth=1", "--filter=blob:none", "--sparse",
             "https://github.com/SigmaHQ/sigma.git", sigma_dir],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "sparse-checkout", "set", "rules/"],
            cwd=sigma_dir, check=True, capture_output=True,
        )

    import yaml
    docs, ids, metas, seen = [], [], [], set()
    rules_dir = os.path.join(sigma_dir, "rules")

    for root, _, files in os.walk(rules_dir):
        for fname in files:
            if not fname.endswith(".yml"):
                continue
            uid = f"sigma_{fname[:60]}"
            if uid in seen:
                uid += f"_{len(seen)}"
            seen.add(uid)
            try:
                with open(os.path.join(root, fname), "r",
                          encoding="utf-8", errors="replace") as f:
                    raw = f.read()
                obj   = yaml.safe_load(raw) or {}
                title = obj.get("title", fname)
                desc  = obj.get("description", "")
                tags  = ", ".join(obj.get("tags") or [])
                detect = str(obj.get("detection") or "")
                text = (
                    f"Sigma Rule: {title}\n"
                    f"Tags: {tags}\n"
                    f"Description: {desc}\n"
                    f"Detection: {detect}"
                )[:2000]
                docs.append(text); ids.append(uid)
                metas.append({"source": "sigma", "title": title,
                               "file": fname, "tags": tags})
            except Exception:
                continue

    if docs:
        _upsert_collection(client, "sigma_rules", docs, ids, metas)


# ── Source 6: MSRC CVRF ──────────────────────────────────────────────────────

def load_msrc_cvrf(client):
    print("\nMicrosoft Security Update Summaries (MSRC CVRF) (2023+)...")
    url = "https://api.msrc.microsoft.com/cvrf/v3.0/updates"
    try:
        data = requests.get(url, timeout=60).json()
        updates = data.get("value", [])
    except Exception as e:
        print(f"  MSRC fetch failed: {e}")
        return

    docs, ids, metas = [], [], []
    for u in updates:
        id_str = u.get("ID", "")
        if not id_str:
            continue
        
        # Limit to 2023 onwards
        year_str = id_str[:4]
        try:
            if int(year_str) < 2023:
                continue
        except ValueError:
            continue

        doc_title = u.get("DocumentTitle", "")
        init_date = u.get("InitialReleaseDate", "")
        cvrf_url = u.get("CvrfUrl", "")

        text = (
            f"MSRC Update: {id_str} | "
            f"Title: {doc_title} | "
            f"Release Date: {init_date} | "
            f"CVRF URL: {cvrf_url}"
        )[:2000]

        uid = f"msrc_{id_str}"
        docs.append(text)
        ids.append(uid)
        metas.append({"source": "msrc_cvrf", "update_id": id_str, "title": doc_title})

    if docs:
        _upsert_collection(client, "msrc_cvrf", docs, ids, metas)


# ── Entry point ───────────────────────────────────────────────────────────────

def build_knowledge_base():
    """Download and index all knowledge sources."""
    os.makedirs(KB_DIR, exist_ok=True)
    client = _get_client()

    load_mitre_attack(client)
    load_d3fend(client)
    load_car(client)
    load_cisa_kev(client)
    load_sigma(client)
    load_msrc_cvrf(client)

    print("\nKnowledge base complete.")
    print(f"    Collections: {[c.name for c in client.list_collections()]}")
    return client


def query_all_collections(client: chromadb.Client,
                          query: str,
                          top_k: int = 5) -> str:
    """
    Semantic search across all indexed collections with cross-encoder re-ranking.

    Retrieval strategy (two-stage):
      1. Bi-encoder ANN search: retrieve top_k candidates PER COLLECTION using
         the same all-MiniLM-L6-v2 embeddings used at index time.
      2. Cross-encoder re-ranking: re-score all combined candidates jointly and
         REORDER by relevance — but keep ALL passages (no global truncation).

    Why no global truncation?
    ─────────────────────────
    RAGAS Context Recall checks whether the ground-truth technique is supported
    by ANY passage in retrieved_contexts. Truncating to a global top-k collapses
    6-collection × top_k passages down to a tiny window, hiding the passages that
    cover the correct MITRE technique and tanking recall (0.60 → 0.32 observed).

    By keeping ALL retrieved passages (just reordered by relevance), the LLM
    benefits from quality ordering (most relevant first = position bias) while
    RAGAS has maximum coverage to compute context recall.

    Faithfulness impact:
    ─────────────────────────
    Richer source+ID tags give the LLM a precise citation handle for each passage,
    enabling reliable evidence_quotes extraction.
    Longer passages (900 chars vs old 600) provide more quotable verbatim text.
    """
    try:
        from pipeline.stage4a_rerank import rerank_passages
    except ImportError:
        # Notebook mode: try direct import
        try:
            from stage4a_rerank import rerank_passages  # type: ignore
        except ImportError:
            rerank_passages = lambda q, p, k=None: p  # noqa: graceful no-op (no-truncate)

    embedder = _get_embedder()
    q_emb    = embedder.encode([query]).tolist()
    raw_parts: list[tuple[str, str]] = []  # (plain_text_for_reranking, formatted_passage)

    for col in client.list_collections():
        try:
            n = min(top_k, col.count())
            if n == 0:
                continue
            res = col.query(query_embeddings=q_emb, n_results=n)
            for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
                src = meta.get("source", col.name).upper()
                # Build a richer citation tag so the LLM has a precise handle
                # for each passage and can copy it verbatim into evidence_quotes.
                id_part = (
                    meta.get("tech_id")          # MITRE ATT&CK technique ID
                    or meta.get("d3fend_id")     # D3FEND ID
                    or meta.get("cve_id")        # CISA KEV CVE
                    or meta.get("update_id")     # MSRC update ID
                    or meta.get("title", "")[:40]  # CAR / Sigma title
                )
                tag       = f"[{src}: {id_part}]" if id_part else f"[{src}]"
                formatted = f"{tag}\n{doc[:900]}"   # 900 chars (up from 600)
                raw_parts.append((doc[:900], formatted))
        except Exception as e:
            print(f"  {col.name}: {e}")

    if not raw_parts:
        return ""

    # ── Cross-encoder re-ranking: REORDER all passages, do NOT truncate ────────
    # Keeping all passages ensures RAGAS context recall can find any MITRE
    # technique from any collection. The LLM naturally attends more to the
    # higher-ranked (earlier) passages due to position bias in attention.
    raw_texts     = [text for text, _   in raw_parts]
    raw_formatted = {text: fmt for text, fmt in raw_parts}

    # Pass top_k=None (or len) to rerank_passages so it reorders without truncating
    reranked_texts = rerank_passages(query, raw_texts, top_k=len(raw_texts))

    final_parts = [raw_formatted[t] for t in reranked_texts if t in raw_formatted]
    return "\n\n---\n\n".join(final_parts)


if __name__ == "__main__":
    build_knowledge_base()
