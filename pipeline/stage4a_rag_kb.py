"""
Stage 4a: RAG Knowledge Base Builder
Downloads and indexes 5 cybersecurity knowledge sources into ChromaDB.
Sources: MITRE ATT&CK, MITRE D3FEND, MITRE CAR, CISA KEV, SigmaHQ Rules
"""
import os, re, json, subprocess
import requests
import chromadb
import numpy as np
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
        print("🤖  Loading embedding model …")
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

    print(f"  ✅  {name}: {col.count():,} docs indexed")
    return col


# ── Source 1: MITRE ATT&CK ───────────────────────────────────────────────────

def load_mitre_attack(client):
    print("\n📥  MITRE ATT&CK v14 …")
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
    print("\n📥  MITRE D3FEND …")
    url = "https://d3fend.mitre.org/api/technique/all.json"
    try:
        data = requests.get(url, timeout=60).json()
        techniques = data.get("techniques") or data.get("data") or []
    except Exception as e:
        print(f"  ⚠️  D3FEND fetch failed: {e}")
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
        print("  ⚠️  D3FEND returned 0 usable techniques.")


# ── Source 3: MITRE CAR ──────────────────────────────────────────────────────

def load_car(client):
    print("\n📥  MITRE CAR analytics …")
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
        print("  ⚠️  CAR analytics directory not found.")
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
    print("\n📥  CISA Known Exploited Vulnerabilities …")
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
    print("\n📥  SigmaHQ detection rules (sparse clone) …")
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


# ── Entry point ───────────────────────────────────────────────────────────────

def build_knowledge_base():
    """Download and index all 5 knowledge sources."""
    os.makedirs(KB_DIR, exist_ok=True)
    client = _get_client()

    load_mitre_attack(client)
    load_d3fend(client)
    load_car(client)
    load_cisa_kev(client)
    load_sigma(client)

    print("\n🏁  Knowledge base complete.")
    print(f"    Collections: {[c.name for c in client.list_collections()]}")
    return client


def query_all_collections(client: chromadb.Client,
                          query: str,
                          top_k: int = 5) -> str:
    """
    Semantic search across all indexed collections.
    Returns a single concatenated context string.
    """
    embedder = _get_embedder()
    q_emb = embedder.encode([query]).tolist()
    parts  = []

    for col in client.list_collections():
        try:
            n = min(top_k, col.count())
            if n == 0:
                continue
            res = col.query(query_embeddings=q_emb, n_results=n)
            for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
                src = meta.get("source", col.name).upper()
                parts.append(f"[{src}]\n{doc[:600]}")
        except Exception as e:
            print(f"  ⚠️  {col.name}: {e}")

    return "\n\n---\n\n".join(parts)


if __name__ == "__main__":
    build_knowledge_base()
