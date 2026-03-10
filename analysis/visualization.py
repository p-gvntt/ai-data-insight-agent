"""
Visualization module for the AI Data Analysis pipeline.
Generates a full visual EDA suite:
  - Distribution histograms with KDE
  - Skewness bar chart
  - Correlation heatmap
  - Pair plot
  - Cluster scatter (PCA projection)
  - Cluster size bar chart
  - Silhouette score by k
  - Anomaly highlight scatter

All functions return matplotlib Figure objects so they can be
rendered in Streamlit (st.pyplot) or saved to disk.
"""

import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Shared style
PALETTE   = "muted"
BG_COLOR  = "#F8F9FA"
GRID_COLOR = "#E0E0E0"
ACCENT    = "#2E75B6"
ANOMALY_COLOR = "#E74C3C"
FONT_SIZE = 11


def _style_ax(ax, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    """Apply consistent styling to an axes object."""
    ax.set_facecolor(BG_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.8, linestyle="--")
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=FONT_SIZE + 1, fontweight="bold", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FONT_SIZE)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FONT_SIZE)
    ax.tick_params(labelsize=FONT_SIZE - 1)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)


def _save_fig(fig: plt.Figure) -> bytes:
    """Serialize figure to PNG bytes for Streamlit or file saving."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()


# Distribution histograms with KDE
def plot_distributions(df: pd.DataFrame) -> plt.Figure:
    """
    One histogram + KDE per numeric column.
    Skewed distributions highlighted with a warning annotation.
    """
    numeric = df.select_dtypes(include="number")
    cols = numeric.columns.tolist()
    if not cols:
        return None

    n_cols = min(3, len(cols))
    n_rows = (len(cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6 * n_cols, 4 * n_rows),
                             facecolor=BG_COLOR)
    axes = np.array(axes).flatten()

    for i, col in enumerate(cols):
        ax = axes[i]
        series = numeric[col].dropna()
        skew = float(series.skew())

        ax.hist(series, bins=30, color=ACCENT, alpha=0.6,
                edgecolor="white", linewidth=0.5, density=True)

        # KDE overlay
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(series)
            x = np.linspace(series.min(), series.max(), 300)
            ax.plot(x, kde(x), color=ACCENT, linewidth=2)
        except Exception:
            pass

        # Mean and median lines
        ax.axvline(series.mean(), color="#E74C3C", linewidth=1.5,
                   linestyle="--", label=f"Mean {series.mean():.2f}")
        ax.axvline(series.median(), color="#2ECC71", linewidth=1.5,
                   linestyle=":", label=f"Median {series.median():.2f}")

        label = f"skew={skew:.2f}"
        if abs(skew) >= 1:
            label += " ⚠ highly skewed"
        ax.legend(fontsize=FONT_SIZE - 2, framealpha=0.7)
        _style_ax(ax, title=col, xlabel=col, ylabel="Density")
        ax.set_title(f"{col}\n{label}", fontsize=FONT_SIZE, fontweight="bold")

    # Hide unused axes
    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Distributions", fontsize=FONT_SIZE + 4,
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# Skewness bar chart
def plot_skewness(df: pd.DataFrame) -> plt.Figure:
    """
    Horizontal bar chart of skewness per numeric column.
    Bars coloured by severity: green (|skew|<0.5), orange (0.5–1), red (>1).
    """
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return None

    skew = numeric.skew().sort_values()
    colors = []
    for v in skew:
        if abs(v) < 0.5:
            colors.append("#2ECC71")
        elif abs(v) < 1.0:
            colors.append("#F39C12")
        else:
            colors.append("#E74C3C")

    fig, ax = plt.subplots(figsize=(8, max(3, len(skew) * 0.5 + 1)),
                           facecolor=BG_COLOR)
    bars = ax.barh(skew.index, skew.values, color=colors,
                   edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="#555555", linewidth=1)
    ax.axvline(-0.5, color="#F39C12", linewidth=1,
               linestyle="--", alpha=0.5, label="|skew| = 0.5")
    ax.axvline(0.5, color="#F39C12", linewidth=1, linestyle="--", alpha=0.5)
    ax.axvline(-1.0, color="#E74C3C", linewidth=1,
               linestyle="--", alpha=0.5, label="|skew| = 1.0")
    ax.axvline(1.0, color="#E74C3C", linewidth=1, linestyle="--", alpha=0.5)

    # Value labels
    for bar, val in zip(bars, skew.values):
        ax.text(val + (0.03 if val >= 0 else -0.03), bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center",
                ha="left" if val >= 0 else "right",
                fontsize=FONT_SIZE - 2)

    legend_patches = [
        mpatches.Patch(color="#2ECC71", label="Symmetric (|skew| < 0.5)"),
        mpatches.Patch(color="#F39C12", label="Moderate (0.5–1.0)"),
        mpatches.Patch(color="#E74C3C", label="High (|skew| > 1.0)"),
    ]
    ax.legend(handles=legend_patches, fontsize=FONT_SIZE - 2,
              loc="lower right", framealpha=0.8)
    _style_ax(ax, title="Skewness by Column", xlabel="Skewness", ylabel="")
    fig.tight_layout()
    return fig


# Correlation heatmap
def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """
    Pearson correlation heatmap with annotated values.
    Only numeric columns with variance > 0 included.
    """
    numeric = df.select_dtypes(include="number")
    numeric = numeric.loc[:, numeric.std() > 0]
    if numeric.shape[1] < 2:
        return None

    corr = numeric.corr(method="pearson")
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(max(6, len(corr) * 1.1),
                                    max(5, len(corr) * 0.9)),
                           facecolor=BG_COLOR)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        linewidths=0.5, linecolor="white",
        annot_kws={"size": FONT_SIZE - 2},
        ax=ax
    )
    ax.set_title("Pearson Correlation Heatmap",
                 fontsize=FONT_SIZE + 2, fontweight="bold", pad=12)
    ax.tick_params(labelsize=FONT_SIZE - 1, rotation=45)
    fig.tight_layout()
    return fig


# Pair plot
def plot_pairplot(df: pd.DataFrame, max_cols: int = 5) -> plt.Figure:
    """
    Seaborn pair plot for numeric columns (capped at max_cols to avoid
    an unreadable grid on wide datasets).
    """
    numeric = df.select_dtypes(include="number").dropna()
    if numeric.shape[1] < 2:
        return None

    # Select most variable columns if too many
    if numeric.shape[1] > max_cols:
        cv = (numeric.std() / numeric.mean().abs()).nlargest(max_cols)
        numeric = numeric[cv.index]

    g = sns.pairplot(numeric, diag_kind="kde", plot_kws={"alpha": 0.4,
                     "color": ACCENT, "s": 15},
                     diag_kws={"color": ACCENT, "fill": True, "alpha": 0.5})
    g.figure.suptitle("Pair Plot", fontsize=FONT_SIZE + 2,
                      fontweight="bold", y=1.01)
    g.figure.set_facecolor(BG_COLOR)
    return g.figure


# Cluster scatter (PCA)
def plot_clusters(df: pd.DataFrame, cluster_labels: list,
                  anomaly_labels: list = None) -> plt.Figure:
    """
    2D PCA projection of clusters. Anomaly rows shown as red X markers
    if anomaly_labels provided.
    """
    numeric = df.select_dtypes(include="number").fillna(
        df.select_dtypes(include="number").median()
    )
    if numeric.shape[1] < 2 or not cluster_labels:
        return None

    X = StandardScaler().fit_transform(numeric)
    coords = PCA(n_components=2, random_state=42).fit_transform(X)

    labels = np.array(cluster_labels)
    is_anomaly = (np.array(anomaly_labels) == -1
                  if anomaly_labels and len(anomaly_labels) == len(df)
                  else np.zeros(len(df), dtype=bool))

    unique_labels = np.unique(labels)
    palette = sns.color_palette(PALETTE, len(unique_labels))

    fig, ax = plt.subplots(figsize=(9, 6), facecolor=BG_COLOR)

    for idx, label in enumerate(unique_labels):
        mask = (labels == label) & ~is_anomaly
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   color=palette[idx], alpha=0.7, s=40,
                   label=f"Cluster {label}", edgecolors="white",
                   linewidths=0.3)

    # Anomalies on top
    if is_anomaly.any():
        ax.scatter(coords[is_anomaly, 0], coords[is_anomaly, 1],
                   color=ANOMALY_COLOR, marker="x", s=80,
                   linewidths=1.5, label="Anomaly", zorder=5)

    _style_ax(ax, title="Cluster Projection (PCA)",
              xlabel="PC 1", ylabel="PC 2")
    ax.legend(fontsize=FONT_SIZE - 1, framealpha=0.8,
              loc="best", markerscale=1.2)
    fig.tight_layout()
    return fig


# Cluster size bar chart
def plot_cluster_sizes(cluster_sizes: dict) -> plt.Figure:
    """Bar chart of cluster sizes with singleton warning."""
    if not cluster_sizes:
        return None

    labels = list(cluster_sizes.keys())
    sizes  = list(cluster_sizes.values())
    colors = [ANOMALY_COLOR if s <= 2 else ACCENT for s in sizes]

    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 0.9), 4),
                           facecolor=BG_COLOR)
    bars = ax.bar(labels, sizes, color=colors, edgecolor="white",
                  linewidth=0.5)

    for bar, size in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                str(size), ha="center", va="bottom",
                fontsize=FONT_SIZE - 1)
        if size <= 2:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() / 2,
                    "⚠", ha="center", va="center",
                    fontsize=14, color="white")

    legend_patches = [
        mpatches.Patch(color=ACCENT, label="Normal cluster"),
        mpatches.Patch(color=ANOMALY_COLOR, label="Singleton / near-singleton (≤ 2)"),
    ]
    ax.legend(handles=legend_patches, fontsize=FONT_SIZE - 1, framealpha=0.8)
    _style_ax(ax, title="Cluster Sizes", xlabel="Cluster", ylabel="Members")
    fig.tight_layout()
    return fig


# Silhouette score by k
def plot_silhouette_scores(scores_by_k: dict, best_k: int) -> plt.Figure:
    """
    Line chart of silhouette score vs k.
    Best k highlighted with a vertical line and annotation.
    """
    if not scores_by_k:
        return None

    ks     = [int(k) for k in scores_by_k.keys()]
    scores = [float(v) for v in scores_by_k.values()]

    fig, ax = plt.subplots(figsize=(7, 4), facecolor=BG_COLOR)
    ax.plot(ks, scores, marker="o", color=ACCENT,
            linewidth=2, markersize=7, markerfacecolor="white",
            markeredgewidth=2)
    ax.axvline(best_k, color=ANOMALY_COLOR, linewidth=1.5,
               linestyle="--", label=f"Best k = {best_k}")
    ax.axhline(0.5, color="#2ECC71", linewidth=1,
               linestyle=":", alpha=0.7, label="Strong threshold (0.5)")
    ax.axhline(0.25, color="#F39C12", linewidth=1,
               linestyle=":", alpha=0.7, label="Weak threshold (0.25)")

    best_score = scores_by_k.get(str(best_k)) or scores_by_k.get(best_k)
    if best_score is not None:
        ax.annotate(f"k={best_k}\n{float(best_score):.3f}",
                    xy=(best_k, float(best_score)),
                    xytext=(best_k + 0.3, float(best_score) + 0.01),
                    fontsize=FONT_SIZE - 1,
                    arrowprops=dict(arrowstyle="->", color="#555"))

    ax.set_xticks(ks)
    ax.legend(fontsize=FONT_SIZE - 1, framealpha=0.8)
    _style_ax(ax, title="Silhouette Score by k",
              xlabel="Number of Clusters (k)",
              ylabel="Silhouette Score")
    fig.tight_layout()
    return fig


# Anomaly highlight scatter
def plot_anomalies(df: pd.DataFrame, anomaly_labels: list,
                   x_col: str = None, y_col: str = None) -> plt.Figure:
    """
    Scatter plot of two numeric columns with anomalies highlighted.
    Auto-selects the two highest-variance columns if x_col/y_col not given.
    """
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2 or not anomaly_labels:
        return None

    if not x_col or not y_col:
        top2 = numeric.var().nlargest(2).index.tolist()
        x_col, y_col = top2[0], top2[1]

    labels = np.array(anomaly_labels)
    normal  = labels == 1
    anomaly = labels == -1

    fig, ax = plt.subplots(figsize=(9, 6), facecolor=BG_COLOR)
    ax.scatter(df.loc[normal,  x_col], df.loc[normal,  y_col],
               color=ACCENT, alpha=0.5, s=30, label="Normal",
               edgecolors="white", linewidths=0.3)
    ax.scatter(df.loc[anomaly, x_col], df.loc[anomaly, y_col],
               color=ANOMALY_COLOR, marker="x", s=100,
               linewidths=2, label=f"Anomaly (n={anomaly.sum()})", zorder=5)

    _style_ax(ax, title="Anomaly Detection",
              xlabel=x_col, ylabel=y_col)
    ax.legend(fontsize=FONT_SIZE, framealpha=0.8)
    fig.tight_layout()
    return fig


# Master function
def generate_all_charts(df: pd.DataFrame,
                        eda: dict,
                        patterns: dict) -> dict:
    """
    Generate charts that mirror exactly what the LLM report shows.
    Uses the same suppression logic as insights.py / report_agent.py:

      - correlation_heatmap  → only if notable_pairs exist (|r| > 0.5)
      - cluster_scatter      → only if cluster quality is moderate or strong
      - cluster_sizes        → only if cluster quality is moderate or strong
      - silhouette_scores    → always (shows WHY quality is weak/strong)
      - anomaly_scatter      → only if anomalies were actually detected
      - distributions        → always
      - skewness             → always
      - pair_plot            → always

    Charts that are suppressed are logged so Streamlit can show
    a "not available" message rather than a blank gap.
    """
    import logging
    log = logging.getLogger(__name__)

    cluster_info   = patterns.get("clusters", {})
    anomaly_info   = patterns.get("anomalies", {})
    cluster_labels = cluster_info.get("cluster_labels", [])
    anomaly_labels = anomaly_info.get("labels", [])
    cluster_sizes  = cluster_info.get("cluster_sizes", {})
    scores_by_k    = cluster_info.get("silhouette_scores_by_k", {})
    best_k         = cluster_info.get("best_k", 2)
    cluster_quality = cluster_info.get("cluster_quality", "weak")   # "weak"|"moderate"|"strong"
    anomaly_count  = anomaly_info.get("anomaly_count", 0)
    notable_pairs  = eda.get("correlations", {}).get("notable_pairs", [])

    # Suppression flags — mirror insights.py logic exactly
    show_correlation = len(notable_pairs) > 0
    show_clusters    = cluster_quality in ("moderate", "strong")
    show_anomalies   = anomaly_count > 0

    if not show_correlation:
        log.info("Correlation heatmap suppressed — no notable pairs (|r| > 0.5) found.")
    if not show_clusters:
        log.info("Cluster charts suppressed — silhouette score too weak to show meaningful segments.")
    if not show_anomalies:
        log.info("Anomaly scatter suppressed — no anomalies detected.")

    # Task list with conditional gates
    tasks = [
        # Always shown
        ("distributions",       True,               lambda: plot_distributions(df)),
        ("skewness",            True,               lambda: plot_skewness(df)),
        ("pair_plot",           True,               lambda: plot_pairplot(df)),
        ("silhouette_scores",   True,               lambda: plot_silhouette_scores(scores_by_k, best_k)),
        # Conditional
        ("correlation_heatmap", show_correlation,   lambda: plot_correlation_heatmap(df)),
        ("cluster_scatter",     show_clusters,      lambda: plot_clusters(df, cluster_labels, anomaly_labels)),
        ("cluster_sizes",       show_clusters,      lambda: plot_cluster_sizes(cluster_sizes)),
        ("anomaly_scatter",     show_anomalies,     lambda: plot_anomalies(df, anomaly_labels)),
    ]

    charts = {}
    suppressed = []

    for name, condition, fn in tasks:
        if not condition:
            suppressed.append(name)
            continue
        try:
            fig = fn()
            if fig is not None:
                charts[name] = fig
        except Exception as e:
            log.warning(f"Chart '{name}' failed: {e}")

    # Attach suppression list so Streamlit can render explanatory messages
    charts["_suppressed"] = suppressed
    return charts


def save_charts(charts: dict, output_dir: str) -> list:
    """
    Save all chart figures to output_dir as PNG files.
    Skips metadata keys like _suppressed that are not Figure objects.
    Returns list of saved file paths.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    saved = []
    for name, fig in charts.items():
        if name.startswith("_") or not hasattr(fig, "savefig"):
            continue
        path = os.path.join(output_dir, f"{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        saved.append(path)
    return saved