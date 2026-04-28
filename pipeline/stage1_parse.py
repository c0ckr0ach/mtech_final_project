"""
Stage 1: Parse & Normalize
Streams the NDJSON log file in chunks and produces a normalized Parquet file.
"""
import json, os, re
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm.auto import tqdm


# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH         = "/content/data_path/apt29_evals_day1_manual_2020-05-01225525.json"
NORMALIZED_PARQUET = "/content/data/normalized.parquet"
CHUNK_SIZE        = 50_000          # rows kept in memory before flushing
# ──────────────────────────────────────────────────────────────────────────────


def _safe_int(val, default=0):
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _hex_to_int(val):
    try:
        return int(val, 16) if isinstance(val, str) and val.lower().startswith("0x") else 0
    except ValueError:
        return 0


def _parse_dt(s):
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def _process_depth(path: str) -> int:
    return path.count("\\") if path else 0


def _basename(path: str) -> str:
    return path.split("\\")[-1].lower() if path else ""


def normalize_event(ev: dict) -> dict:
    """Flatten one raw Sysmon JSON event into a ML-ready dict."""
    dt   = _parse_dt(ev.get("EventTime", ""))
    img  = ev.get("Image") or ev.get("SourceImage") or ""
    timg = ev.get("TargetImage") or ""
    tobj = ev.get("TargetObject") or ""
    tfn  = ev.get("TargetFilename") or ""
    msg  = ev.get("Message") or ""

    return {
        # ── identifiers ──────────────────────────────────────────────────────
        "record_number"    : _safe_int(ev.get("RecordNumber")),
        "event_time"       : ev.get("EventTime", ""),
        "timestamp"        : dt.isoformat() if dt else "",
        "hour_of_day"      : dt.hour        if dt else -1,
        "day_of_week"      : dt.weekday()   if dt else -1,
        # ── event metadata ───────────────────────────────────────────────────
        "event_id"         : _safe_int(ev.get("EventID")),
        "channel"          : ev.get("Channel", ""),
        "source_name"      : ev.get("SourceName", ""),
        "severity"         : ev.get("Severity", ""),
        "severity_value"   : _safe_int(ev.get("SeverityValue")),
        "hostname"         : ev.get("Hostname", ""),
        # ── identity ─────────────────────────────────────────────────────────
        "account_name"     : ev.get("AccountName", ""),
        "domain"           : ev.get("Domain", ""),
        "user_id"          : ev.get("UserID", ""),
        "is_system"        : 1 if ev.get("AccountName", "").upper() == "SYSTEM" else 0,
        # ── process / image ──────────────────────────────────────────────────
        "image"            : img,
        "image_base"       : _basename(img),
        "process_depth"    : _process_depth(img),
        "process_id"       : _safe_int(ev.get("ProcessId") or ev.get("SourceProcessId")),
        "target_image"     : timg,
        "target_image_base": _basename(timg),
        # ── access / registry / file ─────────────────────────────────────────
        "granted_access"   : _hex_to_int(ev.get("GrantedAccess", "0x0")),
        "target_object"    : tobj,
        "target_filename"  : tfn,
        # ── text ─────────────────────────────────────────────────────────────
        "message"          : msg,
        "message_len"      : len(msg),
        "call_trace"       : ev.get("CallTrace", ""),
        "rule_name"        : ev.get("RuleName", ""),
    }


def parse_stage(data_path: str = DATA_PATH,
                out_path: str  = NORMALIZED_PARQUET,
                chunk_size: int = CHUNK_SIZE) -> pd.DataFrame:
    """
    Streams the NDJSON file line-by-line, normalizes each event,
    and saves a single consolidated Parquet file.
    Returns the final DataFrame.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    chunks, records, errors, total = [], [], 0, 0

    print(f"📂  Streaming: {data_path}")
    with open(data_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in tqdm(fh, desc="Parsing events", unit=" lines"):
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
                records.append(normalize_event(ev))
                total += 1
            except json.JSONDecodeError:
                errors += 1
                continue

            if len(records) >= chunk_size:
                chunks.append(pd.DataFrame(records))
                records = []

    if records:
        chunks.append(pd.DataFrame(records))

    print(f"\n✅  Parsed {total:,} events | JSON errors: {errors:,}")

    df = pd.concat(chunks, ignore_index=True)

    # ── post-parse feature engineering ───────────────────────────────────────
    # One-hot top-15 EventIDs
    top_eids = df["event_id"].value_counts().head(15).index.tolist()
    for eid in top_eids:
        df[f"eid_{eid}"] = (df["event_id"] == eid).astype(np.int8)

    # Channel bucket
    df["channel_bucket"] = df["channel"].str.extract(r"(Sysmon|Security|System|Application)",
                                                       expand=False).fillna("Other")

    df.to_parquet(out_path, index=False)
    print(f"💾  Saved → {out_path}  ({df.shape[0]:,} rows × {df.shape[1]} cols)")
    return df


if __name__ == "__main__":
    df = parse_stage()
    print(df.dtypes)
    print(df.head(3))
