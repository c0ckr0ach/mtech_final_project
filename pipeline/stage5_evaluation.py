"""
Stage 5: Quantitative ML Evaluation
Loads anomalies.parquet (which contains ground_truth labels from Stage 0
and anomaly scores from all three models in Stage 2).

Generates thesis-quality evaluation artefacts:
  ├── figures/roc_curve_comparison.png        — Overlaid ROC curves
  ├── figures/pr_curve_comparison.png         — Overlaid Precision-Recall curves
  ├── figures/confusion_matrices.png          — 3 × confusion matrices at optimal threshold
  └── metrics_report.json                     — All numeric scores

Metrics computed per model
──────────────────────────
  • ROC-AUC   — area under the ROC curve (threshold-free discrimination)
  • PR-AUC    — area under the Precision-Recall curve (better for imbalanced data)
  • F1 @ opt  — F1 at the threshold that maximises it
  • Precision @ opt
  • Recall    @ opt
  • FPR       @ opt  — False Positive Rate (SOC analyst burden)
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for Colab/headless
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, f1_score, precision_score, recall_score,
)

LABELED_PARQUET    = "/content/data/labeled.parquet"
ANOMALIES_PARQUET  = "/content/data/anomalies.parquet"
FIGURES_DIR        = "/content/data/figures"
METRICS_JSON       = "/content/data/metrics_report.json"

# Model display names and their score columns.
# NOTE: IF and OCSVM scores are "lower = more anomalous", so we negate them.
MODELS = {
    "Isolation Forest": {
        "score_col"  : "if_anomaly_score",
        "flag_col"   : "if_is_anomaly",
        "negate"     : True,          # negate so higher = more anomalous for ROC
        "color"      : "#e94560",
        "linestyle"  : "-",
    },
    "One-Class SVM": {
        "score_col"  : "ocsvm_anomaly_score",
        "flag_col"   : "ocsvm_is_anomaly",
        "negate"     : True,
        "color"      : "#f5a623",
        "linestyle"  : "--",
    },
    "Deep Autoencoder": {
        "score_col"  : "ae_anomaly_score",
        "flag_col"   : "ae_is_anomaly",
        "negate"     : False,         # AE: higher reconstruction error = more anomalous
        "color"      : "#00b4d8",
        "linestyle"  : "-.",
    },
}

plt.style.use("dark_background")
FONT = {"family": "DejaVu Sans", "size": 12}
plt.rc("font", **FONT)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _optimal_threshold_metrics(y_true, y_score):
    """
    Find the threshold that maximises F1.
    Returns dict with precision, recall, f1, fpr at that threshold.
    """
    prec, rec, thresholds = precision_recall_curve(y_true, y_score)
    # avoid div-by-zero
    f1s = np.where((prec + rec) == 0, 0, 2 * prec * rec / (prec + rec))
    best_idx = np.argmax(f1s)
    best_thr = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]

    y_pred = (y_score >= best_thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "threshold" : float(best_thr),
        "precision" : float(precision_score(y_true, y_pred, zero_division=0)),
        "recall"    : float(recall_score(y_true, y_pred, zero_division=0)),
        "f1"        : float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr"       : float(fpr),
        "cm"        : confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


# ── Plot 1: ROC Curves ────────────────────────────────────────────────────────

def plot_roc_curves(y_true, model_scores: dict, out_path: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "w--", linewidth=1, label="Random (AUC = 0.50)")

    for name, (score, cfg) in model_scores.items():
        fpr, tpr, _ = roc_curve(y_true, score)
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr,
                color=cfg["color"], linestyle=cfg["linestyle"], linewidth=2.5,
                label=f"{name}  (AUC = {roc_auc:.3f})")

    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=13)
    ax.set_title("ROC Curve Comparison — All Anomaly Detectors", fontsize=15, pad=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── Plot 2: Precision-Recall Curves ──────────────────────────────────────────

def plot_pr_curves(y_true, model_scores: dict, out_path: str):
    baseline = y_true.mean()
    fig, ax  = plt.subplots(figsize=(8, 6))
    ax.axhline(baseline, color="white", linestyle="--", linewidth=1,
               label=f"Random (AP = {baseline:.3f})")

    for name, (score, cfg) in model_scores.items():
        prec, rec, _ = precision_recall_curve(y_true, score)
        ap           = average_precision_score(y_true, score)
        ax.plot(rec, prec,
                color=cfg["color"], linestyle=cfg["linestyle"], linewidth=2.5,
                label=f"{name}  (AP = {ap:.3f})")

    ax.set_xlabel("Recall", fontsize=13)
    ax.set_ylabel("Precision", fontsize=13)
    ax.set_title("Precision-Recall Curve Comparison", fontsize=15, pad=14)
    ax.legend(loc="upper right", fontsize=11)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── Plot 3: Confusion Matrices ────────────────────────────────────────────────

def plot_confusion_matrices(cms: dict, out_path: str):
    n    = len(cms)
    fig  = plt.figure(figsize=(5 * n, 5))
    gs   = gridspec.GridSpec(1, n, figure=fig, wspace=0.4)

    for i, (name, cm) in enumerate(cms.items()):
        ax = fig.add_subplot(gs[i])
        cm_arr = np.array(cm)
        sns.heatmap(cm_arr, annot=True, fmt="d", cmap="YlOrRd", ax=ax,
                    cbar=False, linewidths=0.5,
                    xticklabels=["Benign", "Malicious"],
                    yticklabels=["Benign", "Malicious"])
        ax.set_title(name, fontsize=12, pad=8)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)

    fig.suptitle("Confusion Matrices at Optimal F1 Threshold", fontsize=14, y=1.02)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def evaluation_stage(anomalies_path: str = ANOMALIES_PARQUET,
                     labeled_path: str   = LABELED_PARQUET,
                     figures_dir: str    = FIGURES_DIR,
                     metrics_path: str   = METRICS_JSON) -> dict:
    """
    End-to-end Stage 5 entry point.

    Evaluates on the **full labeled dataset** (196K events) rather than just
    the anomaly subset.  This is the correct evaluation scope because:
      - Ground truth labels exist for ALL 196K events (from Stage 0).
      - The unsupervised models produce continuous scores for ALL events.
      - Evaluating only on the flagged anomaly subset biases recall upward
        and causes ROC-AUC < 0.5 when the subset misses most positives.

    Strategy:
      1. Load labeled.parquet (all events, ground_truth column present).
      2. Load anomalies.parquet (flagged events with model scores).
      3. Left-join model scores back onto labeled.parquet by row index.
         Un-flagged events receive the least-anomalous fill value per model.
      4. Compute ROC-AUC / PR-AUC / F1 over all 196K events.
    """
    os.makedirs(figures_dir, exist_ok=True)

    # ── Load full labeled dataset ──────────────────────────────────────────────
    print("Loading full labeled parquet (all events) …")
    df_full = pd.read_parquet(labeled_path)
    df_full = df_full.reset_index(drop=True)   # ensure clean 0-based index
    print(f"    Shape: {df_full.shape}")

    if "ground_truth" not in df_full.columns:
        raise ValueError(
            "ground_truth column not found in labeled.parquet. "
            "Run stage0_label.py first."
        )

    y_true = df_full["ground_truth"].values
    print(f"    Total positives (full corpus): {y_true.sum():,} / {len(y_true):,}")

    # ── Merge model scores from anomaly subset ─────────────────────────────────
    print("\nLoading anomaly scores from anomalies.parquet …")
    df_anom = pd.read_parquet(anomalies_path).reset_index(drop=True)
    print(f"    Anomaly set shape: {df_anom.shape}")
    print(f"    Positives in anomaly set: {df_anom['ground_truth'].sum():,} / {len(df_anom):,}")

    # Score columns produced by Stage 2
    score_cols = [cfg["score_col"] for cfg in MODELS.values()]
    flag_cols  = [cfg["flag_col"]  for cfg in MODELS.values()]
    all_model_cols = score_cols + flag_cols

    # Try merging by the original DataFrame index that Stage 2 preserved.
    # Stage 2 saves a subset of df_full; if the parquet preserves the original
    # integer index we can use it; otherwise fall back to an inner join on
    # shared non-score columns.
    merge_cols = [c for c in all_model_cols if c in df_anom.columns]
    if "original_index" in df_anom.columns:
        df_scores = df_anom.set_index("original_index")[merge_cols]
        df_merged = df_full.join(df_scores, how="left")
    else:
        # Stage 2 resets the index but keeps all original feature columns.
        # Use timestamp + event_id as a composite key for the join.
        join_keys = ["timestamp", "event_id", "hostname", "process_id"]
        join_keys = [k for k in join_keys if k in df_full.columns and k in df_anom.columns]
        if join_keys:
            df_merged = df_full.merge(
                df_anom[[*join_keys, *merge_cols]].drop_duplicates(subset=join_keys),
                on=join_keys, how="left"
            )
        else:
            print("  WARNING: Cannot join by index or key — falling back to anomaly-subset evaluation.")
            df_merged = df_anom  # degenerate fallback

    # Fill un-flagged events with the most-benign score per model
    for name, cfg in MODELS.items():
        col = cfg["score_col"]
        if col not in df_merged.columns:
            continue
        if cfg["negate"]:
            # IF / OCSVM: lower score = more anomalous → fill missing with max (most benign)
            df_merged[col] = df_merged[col].fillna(df_merged[col].max())
        else:
            # AE: higher score = more anomalous → fill missing with min (most benign)
            df_merged[col] = df_merged[col].fillna(df_merged[col].min())

    y_true_eval = df_merged["ground_truth"].values

    # Build score dict for the full evaluation
    model_scores = {}
    for name, cfg in MODELS.items():
        if cfg["score_col"] not in df_merged.columns:
            print(f"  Skipping {name} — column '{cfg['score_col']}' not found.")
            continue
        raw   = df_merged[cfg["score_col"]].fillna(0).values
        score = -raw if cfg["negate"] else raw
        model_scores[name] = (score, cfg)

    if not model_scores:
        raise ValueError("No model score columns found.")

    # ── Compute all metrics ───────────────────────────────────────────────────
    print("\n── Evaluation Metrics (full 196K corpus) ──────────────────")
    report = {}
    cms    = {}

    for name, (score, cfg) in model_scores.items():
        fpr_curve, tpr_curve, _ = roc_curve(y_true_eval, score)
        roc_auc = auc(fpr_curve, tpr_curve)
        pr_auc  = average_precision_score(y_true_eval, score)
        opt     = _optimal_threshold_metrics(y_true_eval, score)
        cms[name] = opt["cm"]

        report[name] = {
            "roc_auc"   : round(roc_auc, 4),
            "pr_auc"    : round(pr_auc, 4),
            "f1"        : round(opt["f1"], 4),
            "precision" : round(opt["precision"], 4),
            "recall"    : round(opt["recall"], 4),
            "fpr"       : round(opt["fpr"], 4),
            "eval_scope": "full_corpus",
        }

        print(f"\n  {name}")
        print(f"    ROC-AUC   : {roc_auc:.4f}")
        print(f"    PR-AUC    : {pr_auc:.4f}")
        print(f"    F1 (opt)  : {opt['f1']:.4f}")
        print(f"    Precision : {opt['precision']:.4f}")
        print(f"    Recall    : {opt['recall']:.4f}")
        print(f"    FPR (opt) : {opt['fpr']:.4f}")

    print("────────────────────────────────────────────────────────────")

    # ── Generate figures ──────────────────────────────────────────────────────
    print("\n── Generating figures ─────────────────────────────────────")
    plot_roc_curves(y_true_eval, model_scores,
                    os.path.join(figures_dir, "roc_curve_comparison.png"))
    plot_pr_curves(y_true_eval, model_scores,
                   os.path.join(figures_dir, "pr_curve_comparison.png"))
    plot_confusion_matrices(cms,
                            os.path.join(figures_dir, "confusion_matrices.png"))

    # ── Save JSON report ──────────────────────────────────────────────────────
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Metrics saved → {metrics_path}")

    return report


if __name__ == "__main__":
    evaluation_stage()
