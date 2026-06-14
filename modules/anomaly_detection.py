"""
Anomaly Detection Module
Methods: Isolation Forest, Z-Score, IQR
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass, field


@dataclass
class AnomalyResult:
    method: str = ""
    n_anomalies: int = 0
    anomaly_indices: list = field(default_factory=list)
    anomaly_df: object = None  # pd.DataFrame
    scores: list = field(default_factory=list)
    columns_used: list = field(default_factory=list)
    threshold: float = 0.0
    contamination: float = 0.05
    fig: object = None


def detect_isolation_forest(df: pd.DataFrame, contamination: float = 0.05) -> AnomalyResult:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    numeric = df.select_dtypes(include=np.number).dropna(axis=1)
    if numeric.empty:
        raise ValueError("No numeric columns for Isolation Forest.")

    X = numeric.fillna(numeric.median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
    preds = model.fit_predict(X_scaled)
    scores = model.decision_function(X_scaled)

    anomaly_mask = preds == -1
    anomaly_idx = np.where(anomaly_mask)[0].tolist()

    result = AnomalyResult(
        method="Isolation Forest",
        n_anomalies=int(anomaly_mask.sum()),
        anomaly_indices=anomaly_idx,
        anomaly_df=df.iloc[anomaly_idx].copy() if anomaly_idx else pd.DataFrame(),
        scores=scores.tolist(),
        columns_used=numeric.columns.tolist(),
        contamination=contamination,
    )

    # Visualization (scatter of first 2 PCA components)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    labels = ["Anomaly" if p == -1 else "Normal" for p in preds]
    fig = px.scatter(
        x=components[:, 0], y=components[:, 1],
        color=labels,
        color_discrete_map={"Anomaly": "#ef4444", "Normal": "#4f8ef7"},
        title="Isolation Forest — PCA Visualization",
        labels={"x": "PC1", "y": "PC2"},
        template="plotly_dark",
        opacity=0.7,
    )
    result.fig = fig
    return result


def detect_zscore(df: pd.DataFrame, threshold: float = 3.0) -> AnomalyResult:
    numeric = df.select_dtypes(include=np.number).fillna(0)
    if numeric.empty:
        raise ValueError("No numeric columns.")

    z = np.abs((numeric - numeric.mean()) / numeric.std(ddof=0).replace(0, 1))
    anomaly_mask = (z > threshold).any(axis=1)
    anomaly_idx = np.where(anomaly_mask)[0].tolist()

    result = AnomalyResult(
        method="Z-Score",
        n_anomalies=int(anomaly_mask.sum()),
        anomaly_indices=anomaly_idx,
        anomaly_df=df.iloc[anomaly_idx].copy() if anomaly_idx else pd.DataFrame(),
        scores=(z.max(axis=1)).tolist(),
        columns_used=numeric.columns.tolist(),
        threshold=threshold,
    )

    # Visualization: max z-score per row
    max_z = z.max(axis=1)
    colors = ["#ef4444" if v else "#4f8ef7" for v in anomaly_mask]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=max_z.values, mode="markers",
        marker=dict(color=colors, size=4, opacity=0.6),
        name="Max Z-Score per row",
    ))
    fig.add_hline(y=threshold, line_dash="dash", line_color="#f59e0b",
                  annotation_text=f"Threshold ({threshold})")
    fig.update_layout(
        title="Z-Score Anomaly Detection",
        xaxis_title="Row Index", yaxis_title="Max Z-Score",
        template="plotly_dark",
    )
    result.fig = fig
    return result


def detect_iqr(df: pd.DataFrame, factor: float = 1.5) -> AnomalyResult:
    numeric = df.select_dtypes(include=np.number).fillna(0)
    if numeric.empty:
        raise ValueError("No numeric columns.")

    Q1 = numeric.quantile(0.25)
    Q3 = numeric.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    anomaly_mask = ((numeric < lower) | (numeric > upper)).any(axis=1)
    anomaly_idx = np.where(anomaly_mask)[0].tolist()

    # Score: max of (deviation / IQR range) per row
    deviation = ((numeric - numeric.median()).abs() / IQR.replace(0, 1)).max(axis=1)

    result = AnomalyResult(
        method="IQR",
        n_anomalies=int(anomaly_mask.sum()),
        anomaly_indices=anomaly_idx,
        anomaly_df=df.iloc[anomaly_idx].copy() if anomaly_idx else pd.DataFrame(),
        scores=deviation.tolist(),
        columns_used=numeric.columns.tolist(),
        threshold=factor,
    )

    # Box plots for top numeric columns
    cols_to_plot = numeric.columns[:4].tolist()
    import plotly.express as px
    fig = px.box(
        df[cols_to_plot].fillna(0),
        title="IQR Outlier Detection — Box Plots",
        template="plotly_dark",
        points="outliers",
    )
    result.fig = fig
    return result
