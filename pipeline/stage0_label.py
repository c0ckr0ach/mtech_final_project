"""
Stage 0: Ground Truth Labeling
Reads normalized.parquet (produced by Stage 1) and adds a `ground_truth`
column using a three-tier rule set derived from the MITRE ATT&CK APT29
technique mapping and Sysmon event semantics.

Ground truth construction rationale:
  The OTRF APT29 Mordor dataset is semi-labeled by design.
  Each capture file contains both adversarial tradecraft events AND
  normal background endpoint activity generated during the same timeframe.
  Since no per-row binary label exists, we derive labels from:

    Tier 1 — High-confidence MITRE-mapped rules (EventID + target/process)
    Tier 2 — Known-bad process image names (heuristic)
    Tier 3 — Default  →  benign (0)

References:
  OTRF Threat Hunter Playbook: https://threathunterplaybook.com/
  MITRE ATT&CK APT29 group:    https://attack.mitre.org/groups/G0016/
"""
import os
import pandas as pd

NORMALIZED_PARQUET = "/content/data/normalized.parquet"
LABELED_PARQUET    = "/content/data/labeled.parquet"

# ── Tier 1 ────────────────────────────────────────────────────────────────────
# High-confidence rules: (EventID, field_name, substring_to_match)
# Derived from documented APT29 TTPs in the OTRF dataset and ATT&CK G0016.

TIER1_RULES = [
    # T1003.001 — LSASS Memory Credential Dumping
    (10,  "target_image",  "lsass.exe"),
    # T1547.001 — Registry Run Key Persistence
    (13,  "target_object", "\\CurrentVersion\\Run"),
    (13,  "target_object", "\\CurrentVersion\\RunOnce"),
    # T1059.001 — PowerShell execution chain (office apps spawning shells)
    (1,   "image_base",    "powershell.exe"),
    (1,   "image_base",    "cmd.exe"),
    # T1021.002 — SMB/Windows Admin Shares lateral movement
    (3,   "image_base",    "net.exe"),
    (3,   "image_base",    "net1.exe"),
    # T1055 — Process Injection
    (8,   "target_image",  ""),          # CreateRemoteThread — any target
    # T1543.003 — Windows Service creation
    (13,  "target_object", "\\Services\\"),
    # T1070.001 — Clear Windows Event Logs
    (1,   "image_base",    "wevtutil.exe"),
    # T1053.005 — Scheduled Task creation
    (1,   "image_base",    "schtasks.exe"),
    (11,  "target_filename","\\Tasks\\"),
    # T1027 — Obfuscated file write
    (11,  "target_filename", ".ps1"),
    (11,  "target_filename", ".vbs"),
    (11,  "target_filename", ".hta"),
    # T1218.011 — Signed Binary Proxy (rundll32/regsvr32)
    (1,   "image_base",    "rundll32.exe"),
    (1,   "image_base",    "regsvr32.exe"),
    # T1071.001 — Web Protocol C2
    (3,   "image_base",    "powershell.exe"),
    # T1140 — Deobfuscate via certutil
    (1,   "image_base",    "certutil.exe"),
    # T1036 — Masquerading (processes outside System32)
    (1,   "image_base",    "svchost.exe"),   # confirmed after path check below
]

# ── Tier 2 ────────────────────────────────────────────────────────────────────
# Known offensive tooling image basenames (heuristic).
KNOWN_BAD_IMAGES = {
    "mimikatz.exe", "psexec.exe", "psexecsvc.exe", "cobalt strike",
    "meterpreter", "empire", "covenant", "pwdump", "fgdump",
    "wce.exe", "gsecdump.exe", "lsadump", "procdump.exe",
    "rubeus.exe", "kerbrute.exe", "bloodhound.exe", "sharphound.exe",
    "invoke-mimikatz", "invoke-bloodhound",
}


def _apply_tier1(df: pd.DataFrame) -> pd.Series:
    """
    Vectorised application of Tier-1 rules.
    Returns a boolean Series: True = rule matched.
    """
    mask = pd.Series(False, index=df.index)

    for eid, field, substring in TIER1_RULES:
        eid_match = df["event_id"] == eid
        if not substring:
            # Match purely on EventID (e.g., EventID 8 = CreateRemoteThread)
            mask |= eid_match
        else:
            col = df[field].fillna("").str.lower()
            mask |= eid_match & col.str.contains(substring.lower(), regex=False)

    # Special case: EventID 1 svchost.exe NOT in System32 is suspicious
    # (legitimate svchost lives in C:\Windows\System32\svchost.exe)
    svchost_mask = (
        (df["event_id"] == 1)
        & df["image_base"].str.lower().eq("svchost.exe")
        & ~df["image"].str.lower().str.contains("system32", na=False)
    )
    mask |= svchost_mask

    return mask


def _apply_tier2(df: pd.DataFrame) -> pd.Series:
    """
    Returns a boolean Series: True = image_base is in the known-bad list.
    """
    return df["image_base"].str.lower().isin(KNOWN_BAD_IMAGES)


def label_stage(normalized_path: str = NORMALIZED_PARQUET,
                labeled_path: str   = LABELED_PARQUET) -> pd.DataFrame:
    """
    End-to-end Stage 0 entry point.
    Reads normalized.parquet, assigns ground_truth ∈ {0, 1}, saves labeled.parquet.
    Returns the labeled DataFrame.
    """
    os.makedirs(os.path.dirname(labeled_path), exist_ok=True)

    print("Loading normalized parquet...")
    df = pd.read_parquet(normalized_path)
    print(f"    Shape: {df.shape}")

    # ── Apply rules ───────────────────────────────────────────────────────────
    tier1_mask = _apply_tier1(df)
    tier2_mask = _apply_tier2(df)

    df["ground_truth"] = (tier1_mask | tier2_mask).astype(int)

    # ── Audit / diagnostics ───────────────────────────────────────────────────
    n_malicious = df["ground_truth"].sum()
    n_benign    = len(df) - n_malicious
    prevalence  = n_malicious / len(df) * 100

    print(f"\n  ── Ground Truth Distribution ───────────────────────────")
    print(f"     Benign    (0) : {n_benign:>8,}  ({100 - prevalence:.1f}%)")
    print(f"     Malicious (1) : {n_malicious:>8,}  ({prevalence:.1f}%)")
    print(f"     Total         : {len(df):>8,}")

    # Breakdown by which EventIDs were flagged
    malicious_df = df[df["ground_truth"] == 1]
    eid_breakdown = (
        malicious_df.groupby("event_id")
        .agg(count=("event_id", "size"), image_examples=("image_base", lambda x: list(x.unique())[:3]))
        .sort_values("count", ascending=False)
        .head(12)
    )
    print("\n  Top malicious EventIDs flagged:")
    print(eid_breakdown.to_string())
    print("  ────────────────────────────────────────────────────────\n")

    df.to_parquet(labeled_path, index=False)
    print(f"  Labeled dataset saved → {labeled_path}")

    return df


if __name__ == "__main__":
    label_stage()
