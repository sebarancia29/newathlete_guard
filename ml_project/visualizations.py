"""
visualizations.py
-----------------
Generates and saves charts for the Athlete Injury Prediction System:
  1. Feature importance bar charts (Injury model + Severity model)
  2. Confusion matrix heatmaps (Injury + Severity)
  3. Severity distribution pie chart
  4. Fatigue index vs Recovery score scatter plot (colored by injury)
  5. Class distribution bar chart

All figures are saved to the 'plots/' directory.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for all environments)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
PALETTE = "viridis"


def _save(fig, name: str):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {path}")
    return path


# ── 1. Feature Importance ─────────────────────────────────────────────────────

def plot_feature_importance(model, feature_names: list[str],
                             title: str, filename: str):
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, max(4, len(feature_names) * 0.45)))
    colors = plt.cm.viridis(np.linspace(0.3, 0.85, len(importances)))
    bars = ax.barh(importances.index, importances.values, color=colors, edgecolor="white")

    for bar, val in zip(bars, importances.values):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9, color="#333")

    ax.set_xlabel("Feature Importance", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, importances.max() * 1.18)
    fig.tight_layout()
    return _save(fig, filename)


# ── 2. Confusion Matrix Heatmap ───────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, class_names: list[str],
                           title: str, filename: str):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, data, fmt, subtitle in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2%"],
        ["Counts", "Normalized (row %)"]
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, ax=ax,
            annot_kws={"size": 12, "weight": "bold"},
        )
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)
        ax.set_title(f"{title} — {subtitle}", fontsize=11, fontweight="bold")

    fig.tight_layout()
    return _save(fig, filename)


# ── 3. Severity Distribution Pie ─────────────────────────────────────────────

def plot_severity_distribution(df: pd.DataFrame, filename: str = "severity_distribution.png"):
    counts = df["injury_severity"].value_counts()
    colors = ["#4CAF50", "#FF9800", "#F44336"]
    explode = [0.04] * len(counts)

    fig, ax = plt.subplots(figsize=(7, 5))
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=counts.index,
        autopct="%1.1f%%", colors=colors,
        explode=explode, startangle=140,
        wedgeprops=dict(edgecolor="white", linewidth=1.5),
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")
        at.set_color("white")

    ax.set_title("Injury Severity Distribution", fontsize=13, fontweight="bold", pad=15)
    fig.tight_layout()
    return _save(fig, filename)


# ── 4. Scatter: Fatigue Index vs Recovery Score ───────────────────────────────

def plot_fatigue_vs_recovery(df: pd.DataFrame,
                              filename: str = "fatigue_vs_recovery_scatter.png"):
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {0: "#2196F3", 1: "#F44336"}
    labels = {0: "No Injury", 1: "Injury"}

    for val in [0, 1]:
        mask = df["injury_indicator"] == val
        ax.scatter(
            df.loc[mask, "fatigue_index"],
            df.loc[mask, "recovery_score"],
            c=colors[val], label=labels[val],
            alpha=0.65, s=55, edgecolors="white", linewidths=0.4,
        )

    ax.set_xlabel("Fatigue Index", fontsize=12)
    ax.set_ylabel("Recovery Score", fontsize=12)
    ax.set_title("Fatigue Index vs Recovery Score\n(colored by Injury Status)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11, framealpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()
    return _save(fig, filename)


# ── 5. Class Distribution Bar ─────────────────────────────────────────────────

def plot_class_distribution(df: pd.DataFrame,
                             filename: str = "injury_class_distribution.png"):
    counts = df["injury_indicator"].value_counts().sort_index()
    labels = ["No Injury", "Injury"]
    bar_colors = ["#2196F3", "#F44336"]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(labels, counts.values, color=bar_colors,
                  edgecolor="white", width=0.5)

    for bar, val in zip(bars, counts.values):
        pct = val / len(df) * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 4,
                f"{val}\n({pct:.1f}%)", ha="center", va="bottom",
                fontsize=11, fontweight="bold")

    ax.set_ylabel("Number of Athletes", fontsize=11)
    ax.set_title("Injury Class Distribution", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, counts.max() * 1.2)
    fig.tight_layout()
    return _save(fig, filename)


# ── 6. CV Score Comparison Bar Chart ─────────────────────────────────────────

def plot_cv_comparison(cv_results: dict, filename: str = "cv_model_comparison.png"):
    """
    cv_results: dict of {model_name: {metric: (mean, std), ...}}
    Example:
      {
        "Logistic Regression": {"Accuracy": (0.82, 0.03), "F1": (0.65, 0.05)},
        "Random Forest":       {"Accuracy": (0.86, 0.02), "F1": (0.72, 0.04)},
      }
    """
    models  = list(cv_results.keys())
    metrics = list(next(iter(cv_results.values())).keys())
    x       = np.arange(len(metrics))
    width   = 0.35
    palette = ["#2196F3", "#4CAF50", "#FF9800"]

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, model in enumerate(models):
        means = [cv_results[model][m][0] for m in metrics]
        stds  = [cv_results[model][m][1] for m in metrics]
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, label=model,
                      color=palette[i % len(palette)],
                      yerr=stds, capsize=4, edgecolor="white")
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012,
                    f"{mean:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_title("5-Fold CV Model Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    fig.tight_layout()
    return _save(fig, filename)


# ── Main orchestrator ─────────────────────────────────────────────────────────

def generate_all_plots(df: pd.DataFrame,
                        inj_model, sev_model,
                        inj_features: list[str],
                        sev_features: list[str]):
    """Generate and save all visualizations. Returns list of saved paths."""
    print("\n" + "=" * 55)
    print("  GENERATING VISUALIZATIONS")
    print("=" * 55)

    saved = []

    # ── Injury side ───────────────────────────────────────────────────────────
    X = df[inj_features]
    y_inj = df["injury_indicator"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_inj, test_size=0.2, random_state=42, stratify=y_inj
    )
    y_pred_inj = inj_model.predict(X_test)

    saved.append(plot_feature_importance(
        inj_model, inj_features,
        "Feature Importance — Injury Prediction Model",
        "injury_feature_importance.png",
    ))
    saved.append(plot_confusion_matrix(
        y_test, y_pred_inj,
        ["No Injury", "Injury"],
        "Injury Prediction",
        "injury_confusion_matrix.png",
    ))

    # ── Severity side ─────────────────────────────────────────────────────────
    X_sev = df[sev_features]
    y_sev = df["injury_severity"].map({"mild": 0, "moderate": 1, "severe": 2})
    Xs_train, Xs_test, ys_train, ys_test = train_test_split(
        X_sev, y_sev, test_size=0.2, random_state=42, stratify=y_sev
    )
    y_pred_sev = sev_model.predict(Xs_test)

    saved.append(plot_feature_importance(
        sev_model, sev_features,
        "Feature Importance — Injury Severity Model",
        "severity_feature_importance.png",
    ))
    saved.append(plot_confusion_matrix(
        ys_test, y_pred_sev,
        ["Mild", "Moderate", "Severe"],
        "Injury Severity",
        "severity_confusion_matrix.png",
    ))

    # ── Dataset-level charts ──────────────────────────────────────────────────
    saved.append(plot_severity_distribution(df))
    saved.append(plot_fatigue_vs_recovery(df))
    saved.append(plot_class_distribution(df))

    print(f"\n  {len(saved)} plots saved to -> {PLOTS_DIR}")
    return saved


if __name__ == "__main__":
    import joblib
    from generate_datasets import main as gen
    from data_preprocessing import preprocess
    from feature_engineering import engineer_features

    gen()
    df, _ = preprocess()
    df = engineer_features(df)

    inj_bundle = joblib.load(os.path.join(os.path.dirname(__file__), "models", "injury_model.pkl"))
    sev_bundle = joblib.load(os.path.join(os.path.dirname(__file__), "models", "severity_model.pkl"))

    generate_all_plots(
        df,
        inj_bundle["model"],  sev_bundle["model"],
        inj_bundle["features"], sev_bundle["features"],
    )
