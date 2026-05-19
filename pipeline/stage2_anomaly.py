"""
Stage 2: Anomaly Detection — Comparative Multi-Model
Loads labeled.parquet, scores every event with three detectors:

  Model A — Isolation Forest  (existing, ensemble tree-based)
  Model B — One-Class SVM     (kernel-based, supports non-linear boundaries)
  Model C — Deep Autoencoder  (reconstruction-error, PyTorch / CUDA)

Each model adds its own anomaly score column. The union of events flagged
by ANY model is saved to anomalies.parquet.

A UMAP 2-D projection is rendered coloured by Isolation Forest score.

Research rationale:
  A thesis requires comparative evaluation rather than a single algorithm.
  Running all three on the same feature matrix allows a direct comparison
  of Precision / Recall / F1 / ROC-AUC in Stage 5 (stage5_evaluation.py).
"""
import os
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import umap

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

LABELED_PARQUET    = "/content/data/labeled.parquet"
ANOMALIES_PARQUET  = "/content/data/anomalies.parquet"
CONTAMINATION      = 0.05
UMAP_SAMPLE        = 60_000
RANDOM_STATE       = 42

# ── Autoencoder hyperparameters ───────────────────────────────────────────────
AE_EPOCHS          = 30
AE_BATCH_SIZE      = 1024
AE_LR              = 1e-3
AE_HIDDEN_DIMS     = [128, 64, 32]   # encoder layers; decoder mirrors this


# ── Feature matrix ────────────────────────────────────────────────────────────

def build_feature_matrix(df: pd.DataFrame):
    """
    Returns (X_scaled, feature_cols, scaler) for the anomaly detectors.
    Uses numeric + engineered one-hot columns only.
    """
    base_cols = [
        "event_id", "severity_value", "is_system",
        "process_depth", "hour_of_day", "day_of_week",
        "granted_access", "message_len",
    ]
    eid_cols     = [c for c in df.columns if c.startswith("eid_")]
    feature_cols = base_cols + eid_cols

    X      = df[feature_cols].fillna(0).astype(float).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, feature_cols, scaler


# ── Model A: Isolation Forest ─────────────────────────────────────────────────

def run_isolation_forest(df: pd.DataFrame,
                         X_scaled: np.ndarray,
                         contamination: float = CONTAMINATION) -> pd.DataFrame:
    print("  [A] Training Isolation Forest …")
    iso = IsolationForest(
        contamination=contamination,
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    df = df.copy()
    df["if_label"]        = iso.fit_predict(X_scaled)        # -1 = anomaly
    df["if_anomaly_score"] = iso.score_samples(X_scaled)     # lower = worse
    df["if_is_anomaly"]   = (df["if_label"] == -1).astype(int)
    n = df["if_is_anomaly"].sum()
    print(f"      IF anomalies: {n:,}  ({n / len(df) * 100:.2f}%)")
    return df


# ── Model B: One-Class SVM ────────────────────────────────────────────────────

def run_one_class_svm(df: pd.DataFrame,
                      X_scaled: np.ndarray,
                      nu: float = CONTAMINATION) -> pd.DataFrame:
    """
    One-Class SVM with RBF kernel.
    `nu` is an upper bound on the fraction of training errors (≈ contamination).
    Subsamples to 30 k rows for tractability (OCSVM is O(n²) in memory).
    """
    print("  [B] Training One-Class SVM …")
    n_fit = min(30_000, len(X_scaled))
    rng   = np.random.default_rng(RANDOM_STATE)
    idx   = rng.choice(len(X_scaled), n_fit, replace=False)

    ocsvm = OneClassSVM(kernel="rbf", nu=nu, gamma="scale")
    ocsvm.fit(X_scaled[idx])

    df = df.copy()
    df["ocsvm_label"]        = ocsvm.predict(X_scaled)       # -1 = anomaly
    df["ocsvm_anomaly_score"] = ocsvm.score_samples(X_scaled) # lower = worse
    df["ocsvm_is_anomaly"]   = (df["ocsvm_label"] == -1).astype(int)
    n = df["ocsvm_is_anomaly"].sum()
    print(f"      OCSVM anomalies: {n:,}  ({n / len(df) * 100:.2f}%)")
    return df


# ── Model C: Deep Autoencoder ─────────────────────────────────────────────────

class _Autoencoder(nn.Module):
    """Simple symmetric fully-connected autoencoder."""
    def __init__(self, input_dim: int, hidden_dims: list[int]):
        super().__init__()
        # Encoder
        enc_layers = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU()]
            prev = h
        self.encoder = nn.Sequential(*enc_layers)
        # Decoder (mirror)
        dec_layers = []
        for h in reversed(hidden_dims[:-1]):
            dec_layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU()]
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def run_autoencoder(df: pd.DataFrame,
                    X_scaled: np.ndarray,
                    epochs: int      = AE_EPOCHS,
                    batch_size: int  = AE_BATCH_SIZE,
                    lr: float        = AE_LR,
                    hidden_dims: list = AE_HIDDEN_DIMS,
                    contamination: float = CONTAMINATION) -> pd.DataFrame:
    """
    Trains a deep autoencoder on the full dataset (unsupervised).
    Anomaly score = per-sample mean squared reconstruction error.
    Events above the (1 - contamination) quantile are flagged anomalous.
    """
    print("  [C] Training Deep Autoencoder …")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"      Using device: {device}")

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    loader   = DataLoader(TensorDataset(X_tensor, X_tensor),
                          batch_size=batch_size, shuffle=True)

    model     = _Autoencoder(X_scaled.shape[1], hidden_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for x_batch, _ in loader:
            optimizer.zero_grad()
            loss = criterion(model(x_batch), x_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(x_batch)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"      Epoch [{epoch+1:02d}/{epochs}]  loss={total_loss/len(X_scaled):.6f}")

    # ── Compute reconstruction errors ─────────────────────────────────────────
    model.eval()
    all_errors = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i + batch_size]
            recon = model(batch)
            err   = ((recon - batch) ** 2).mean(dim=1).cpu().numpy()
            all_errors.append(err)
    recon_errors = np.concatenate(all_errors)

    threshold = np.quantile(recon_errors, 1 - contamination)

    df = df.copy()
    df["ae_anomaly_score"] = recon_errors                              # higher = worse (inverted from IF)
    df["ae_is_anomaly"]    = (recon_errors >= threshold).astype(int)
    n = df["ae_is_anomaly"].sum()
    print(f"      AE anomalies  : {n:,}  ({n / len(df) * 100:.2f}%)")
    return df


# ── UMAP visualisation ────────────────────────────────────────────────────────

def run_umap(X_scaled: np.ndarray, df: pd.DataFrame,
             sample_size: int = UMAP_SAMPLE):
    """UMAP 2-D projection of a random sample, coloured by IF anomaly score."""
    n   = min(sample_size, len(df))
    idx = np.random.default_rng(RANDOM_STATE).choice(len(df), n, replace=False)

    print(f"  UMAP on {n:,} samples …")
    reducer = umap.UMAP(
        n_components=2, n_neighbors=15, min_dist=0.1,
        metric="euclidean", random_state=RANDOM_STATE, low_memory=True,
        init="random",   # avoid spectral init failures on high-cardinality data
    )
    emb = reducer.fit_transform(X_scaled[idx])

    plot_df = pd.DataFrame({
        "x"              : emb[:, 0],
        "y"              : emb[:, 1],
        "if_is_anomaly"  : df["if_is_anomaly"].iloc[idx].values,
        "if_anomaly_score": df["if_anomaly_score"].iloc[idx].values,
        "event_id"       : df["event_id"].iloc[idx].values,
        "hostname"       : df["hostname"].iloc[idx].values,
        "image_base"     : df["image_base"].iloc[idx].values,
    })

    fig = px.scatter(
        plot_df, x="x", y="y",
        color="if_anomaly_score",
        color_continuous_scale=["#e94560", "#f5a623", "#0f3460", "#16213e"],
        symbol="if_is_anomaly",
        hover_data=["event_id", "hostname", "image_base"],
        title="UMAP — Isolation Forest Anomaly Scores (Comparative Run)",
        template="plotly_dark",
        opacity=0.55,
        height=650,
        labels={"if_anomaly_score": "IF Score (lower = more anomalous)",
                "if_is_anomaly": "IF Anomaly"},
    )
    fig.update_traces(marker_size=3)
    fig.update_layout(
        font_family="Inter, sans-serif",
        title_font_size=18,
        coloraxis_colorbar_title="IF Score",
    )
    return fig


# ── Entry point ───────────────────────────────────────────────────────────────

def anomaly_stage(labeled_path: str    = LABELED_PARQUET,
                  anomalies_path: str  = ANOMALIES_PARQUET,
                  contamination: float = CONTAMINATION) -> pd.DataFrame:
    """
    End-to-end Stage 2 entry point.
    Runs all three anomaly detectors, saves anomalies.parquet, renders UMAP.
    """
    os.makedirs(os.path.dirname(anomalies_path), exist_ok=True)

    print("Loading labeled parquet …")
    df = pd.read_parquet(labeled_path)
    print(f"    Shape: {df.shape}")
    if "ground_truth" in df.columns:
        print(f"    Ground truth positives: {df['ground_truth'].sum():,} "
              f"({df['ground_truth'].mean() * 100:.1f}%)")

    X_scaled, feature_cols, _ = build_feature_matrix(df)

    print("\n── Running anomaly detectors ──────────────────────────────")
    df = run_isolation_forest(df, X_scaled, contamination)
    df = run_one_class_svm(df, X_scaled, nu=contamination)
    df = run_autoencoder(df, X_scaled, contamination=contamination)
    print("────────────────────────────────────────────────────────────")

    # Union anomaly flag: event is anomalous if ANY model flags it
    df["is_anomaly"] = (
        (df["if_is_anomaly"] == 1) |
        (df["ocsvm_is_anomaly"] == 1) |
        (df["ae_is_anomaly"] == 1)
    ).astype(int)

    n_union = df["is_anomaly"].sum()
    print(f"\n  Union anomalies (any model): {n_union:,}  ({n_union / len(df) * 100:.2f}%)")

    # Keep backward-compatible `anomaly_score` column (= IF score)
    df["anomaly_score"] = df["if_anomaly_score"]

    fig = run_umap(X_scaled, df)
    fig.show()

    anomalies_df = df[df["is_anomaly"] == 1].copy()
    anomalies_df.to_parquet(anomalies_path, index=False)
    print(f"\n  Anomalies saved → {anomalies_path}")

    # Summary table
    summary = (
        anomalies_df.groupby("event_id")
        .agg(count=("event_id", "size"),
             avg_if_score=("if_anomaly_score", "mean"),
             avg_ae_score=("ae_anomaly_score", "mean"),
             hosts=("hostname", "nunique"))
        .sort_values("avg_if_score")
        .head(15)
    )
    print("\nTop anomalous EventIDs (all models):\n", summary.to_string())

    return anomalies_df


if __name__ == "__main__":
    anomaly_stage()
