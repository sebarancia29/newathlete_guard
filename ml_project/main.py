"""
main.py
-------
End-to-end CLI orchestration for the Athlete Injury Prediction and
Recovery Recommendation System.

Pipeline:
  1. Generate synthetic datasets
  2. Load & preprocess data
  3. Feature engineering
  4. Train injury prediction model (Logistic Regression + Random Forest + SMOTE)
  5. Train injury severity model  (Random Forest + SMOTE)
  6. Generate recovery recommendations
  7. Visualizations
  8. Print summary results table
"""

import os
import sys
import time
import io
import pandas as pd

# Force UTF-8 output on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


BANNER = """
+=========================================================================+
|   Athlete Injury Prediction & Recovery Recommendation System  v2.0      |
|   Machine Learning Pipeline  |  scikit-learn  |  SMOTE  |  Plotly UI   |
+=========================================================================+
"""


def section(title: str):
    print(f"\n{'=' * 65}")
    print(f"  STEP: {title}")
    print(f"{'=' * 65}")


# ── Pipeline steps ─────────────────────────────────────────────────────────────

def step_generate_data():
    section("1 / 8 -- Dataset Generation")
    from generate_datasets import main as gen_main
    gen_main()


def step_preprocess() -> tuple[pd.DataFrame, dict]:
    section("2 / 8 -- Data Preprocessing")
    from data_preprocessing import preprocess
    return preprocess()


def step_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    section("3 / 8 -- Feature Engineering")
    from feature_engineering import engineer_features
    return engineer_features(df)


def step_train_injury_model(df: pd.DataFrame):
    section("4 / 8 -- Injury Prediction Model  [SMOTE + 5-Fold CV]")
    from injury_prediction_model import train_injury_model
    return train_injury_model(df)


def step_train_severity_model(df: pd.DataFrame):
    section("5 / 8 -- Injury Severity Model  [SMOTE + 5-Fold CV]")
    from severity_model import train_severity_model
    return train_severity_model(df)


def step_recommendations(df: pd.DataFrame, n_sample: int = 10) -> pd.DataFrame:
    section("6 / 8 -- Recovery Recommendations")
    from recommendation_engine import generate_all_recommendations, display_recommendations
    df_rec = generate_all_recommendations(df)
    display_recommendations(df_rec, n=n_sample)
    return df_rec


def step_visualizations(df: pd.DataFrame, inj_model, sev_model,
                         inj_features, sev_features):
    section("7 / 8 -- Visualizations")
    try:
        from visualizations import generate_all_plots
        return generate_all_plots(df, inj_model, sev_model, inj_features, sev_features)
    except Exception as e:
        print(f"  [WARN] Visualizations skipped: {e}")


# ── Summary table ──────────────────────────────────────────────────────────────

def print_summary_table(df_rec: pd.DataFrame, n: int = 15):
    section("8 / 8 -- Athlete Summary Table")
    cols = [c for c in [
        "age", "gender", "position",
        "fatigue_index", "risk_score", "recovery_score",
        "injury_indicator", "injury_severity", "rest_days",
    ] if c in df_rec.columns]

    sample = df_rec.head(n)[cols].copy()

    # Decode encoded columns for readability
    if "gender" in sample.columns:
        sample["gender"] = sample["gender"].map({0: "F", 1: "M"})
    if "injury_indicator" in sample.columns:
        sample["injury_indicator"] = sample["injury_indicator"].map({0: "No", 1: "Yes"})

    for col in ["fatigue_index", "risk_score", "recovery_score"]:
        if col in sample.columns:
            sample[col] = sample[col].round(3)

    if HAS_TABULATE:
        print(tabulate(sample, headers="keys", tablefmt="grid", showindex=False))
    else:
        print(sample.to_string(index=False))


def print_recommendation_stats(df_rec: pd.DataFrame):
    print("\n" + "=" * 65)
    print("  RECOMMENDATION STATISTICS")
    print("=" * 65)
    total    = len(df_rec)
    injured  = int(df_rec["injury_indicator"].sum()) if "injury_indicator" in df_rec.columns else 0
    sev_dist = df_rec["injury_severity"].value_counts() if "injury_severity" in df_rec.columns else {}

    print(f"  Total athletes       : {total:,}")
    print(f"  Predicted injured    : {injured:,} ({injured / total * 100:.1f}%)")
    print(f"  Severity breakdown   :")
    if hasattr(sev_dist, "items"):
        for sev, cnt in sev_dist.items():
            bar = "#" * int(cnt / total * 45)
            print(f"    {sev:<12} {cnt:>4} ({cnt / total * 100:5.1f}%)  {bar}")


def verify_saved_models():
    print("\n" + "=" * 65)
    print("  SAVED MODEL VERIFICATION")
    print("=" * 65)
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    for fname in ["injury_model.pkl", "severity_model.pkl"]:
        path = os.path.join(models_dir, fname)
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            print(f"  [OK] {fname:<32} ({size_kb:.1f} KB)")
        else:
            print(f"  [MISSING] {fname}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(BANNER)
    t_start = time.time()

    step_generate_data()

    df, encoders = step_preprocess()
    df = step_feature_engineering(df)

    inj_model, inj_features         = step_train_injury_model(df)
    sev_model, sev_le, sev_features  = step_train_severity_model(df)

    df_rec = step_recommendations(df, n_sample=10)

    step_visualizations(df, inj_model, sev_model, inj_features, sev_features)

    print_summary_table(df_rec, n=15)
    print_recommendation_stats(df_rec)
    verify_saved_models()

    elapsed = time.time() - t_start
    print(f"\n{'=' * 65}")
    print(f"  [DONE] Pipeline complete in {elapsed:.2f} seconds")
    print(f"  Run the Streamlit UI:  streamlit run app.py")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()
