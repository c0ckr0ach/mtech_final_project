"""
Stage 2: Anomaly Detection
Loads normalized.parquet, scores every event with Isolation Forest,
projects to 2-D via UMAP, and saves anomalies.parquet.
"""
import os
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import umap

NORMALIZED_PARQUET  = "/content/data/normalized.parquet"
ANOMALIES_PARQUET   = "/content/data/anomalies.parquet"
CONTAMINATION       = 0.05          # expected anomaly fraction
UMAP_SAMPLE         = 60_000        # rows sent to UMAP (memory guard)
RANDOM_STATE        = 42


def build_feature_matrix(df: pd.DataFrame):
    """
    Returns (X_scaled, feature_cols) for the isolation forest.
    Uses numeric + engineered one-hot columns only.
    """
    base_cols = [
        "event_id", "severity_value", "is_system",
        "process_depth", "hour_of_day", "day_of_week",
        "granted_access", "message_len",
    ]
    eid_cols = [c for c in df.columns if c.startswith("eid_")]
    feature_cols = base_cols + eid_cols

    X = df[feature_cols].fillna(0).astype(float).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, feature_cols, scaler


def run_isolation_forest(df: pd.DataFrame, X_scaled: np.ndarray,
                          contamination: float = CONTAMINATION) -> pd.DataFrame:
    print("Training Isolation Forest...")
    iso = IsolationForest(
        contamination=contamination,
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    df = df.copy()
    df["anomaly_label"] = iso.fit_predict(X_scaled)          # -1 = anomaly
    df["anomaly_score"]  = iso.score_samples(X_scaled)       # lower = more anomalous
    df["is_anomaly"]     = (df["anomaly_label"] == -1).astype(int)

    n = df["is_anomaly"].sum()
    print(f"Anomalies detected: {n:,}  ({n / len(df) * 100:.2f}%)")
    return df


def run_umap(X_scaled: np.ndarray, df: pd.DataFrame,
             sample_size: int = UMAP_SAMPLE) -> "go.Figure":
    """UMAP 2-D projection of a random sample, coloured by anomaly score."""
    n = min(sample_size, len(df))
    idx = np.random.default_rng(RANDOM_STATE).choice(len(df), n, replace=False)

    print(f"UMAP on {n:,} samples...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=RANDOM_STATE,
        low_memory=True,
    )
    emb = reducer.fit_transform(X_scaled[idx])

    plot_df = pd.DataFrame({
        "x": emb[:, 0],
        "y": emb[:, 1],
        "is_anomaly"   : df["is_anomaly"].iloc[idx].values,
        "anomaly_score": df["anomaly_score"].iloc[idx].values,
        "event_id"     : df["event_id"].iloc[idx].values,
        "hostname"     : df["hostname"].iloc[idx].values,
        "image_base"   : df["image_base"].iloc[idx].values,
    })

    fig = px.scatter(
        plot_df, x="x", y="y",
        color="anomaly_score",
        color_continuous_scale=["#e94560", "#f5a623", "#0f3460", "#16213e"],
        symbol="is_anomaly",
        hover_data=["event_id", "hostname", "image_base"],
        title="UMAP — Isolation Forest Anomaly Scores",
        template="plotly_dark",
        opacity=0.55,
        height=650,
        labels={"anomaly_score": "IF Score (lower = more anomalous)",
                "is_anomaly": "Anomaly"},
    )
    fig.update_traces(marker_size=3)
    fig.update_layout(
        font_family="Inter, sans-serif",
        title_font_size=18,
        coloraxis_colorbar_title="Score",
    )
    return fig


def anomaly_stage(normalized_path: str = NORMALIZED_PARQUET,
                  anomalies_path: str   = ANOMALIES_PARQUET,
                  contamination: float  = CONTAMINATION) -> pd.DataFrame:
    """
    End-to-end Stage 2 entry point.
    Returns DataFrame with anomaly columns added.
    """
    os.makedirs(os.path.dirname(anomalies_path), exist_ok=True)

    print("Loading normalized parquet...")
    df = pd.read_parquet(normalized_path)
    print(f"    Shape: {df.shape}")

    X_scaled, feature_cols, _ = build_feature_matrix(df)
    df = run_isolation_forest(df, X_scaled, contamination)

    fig = run_umap(X_scaled, df)
    fig.show()

    anomalies_df = df[df["is_anomaly"] == 1].copy()
    anomalies_df.to_parquet(anomalies_path, index=False)
    print(f"Anomalies saved -> {anomalies_path}")

    # Summary table
    summary = (
        anomalies_df.groupby("event_id")
        .agg(count=("event_id", "size"),
             avg_score=("anomaly_score", "mean"),
             hosts=("hostname", "nunique"))
        .sort_values("avg_score")
        .head(15)
    )
    print("\nTop anomalous EventIDs:\n", summary.to_string())

    return anomalies_df


if __name__ == "__main__":
    anomaly_stage()
