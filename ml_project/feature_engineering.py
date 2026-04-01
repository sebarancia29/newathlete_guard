"""
feature_engineering.py
-----------------------
Creates derived features from the merged dataset:
  - workload_index  : training_hours x match_frequency
  - fatigue_index   : fatigue_score / (recovery_time + 1)
  - recovery_score  : (recovery_time x sleep_duration) / (heart_rate + 1)
  - risk_score      : composite risk (0-1 normalised)
  - injury_severity : rule-derived label — mild / moderate / severe
"""

import pandas as pd
import numpy as np


def add_workload_index(df: pd.DataFrame) -> pd.DataFrame:
    df["workload_index"] = (df["training_hours"] * df["match_frequency"]).round(2)
    return df


def add_fatigue_index(df: pd.DataFrame) -> pd.DataFrame:
    df["fatigue_index"] = (df["fatigue_score"] / (df["recovery_time"] + 1)).round(4)
    return df


def add_recovery_score(df: pd.DataFrame) -> pd.DataFrame:
    df["recovery_score"] = (
        (df["recovery_time"] * df["sleep_duration"]) / (df["heart_rate"] + 1)
    ).round(4)
    return df


def add_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Composite risk score (0–1 range).
    Weights: fatigue_index 35%, workload_index 30%, prev injury 20%, low recovery 15%.
    """
    fi_norm  = np.clip(df["fatigue_index"] / df["fatigue_index"].max(), 0, 1)
    wi_norm  = np.clip(df["workload_index"] / df["workload_index"].max(), 0, 1)
    rs_inv   = 1 - np.clip(df["recovery_score"] / (df["recovery_score"].max() + 1e-9), 0, 1)
    prev_inj = df["previous_injury"].astype(float) if "previous_injury" in df.columns \
               else pd.Series(0.0, index=df.index)

    df["risk_score"] = (
        0.35 * fi_norm +
        0.30 * wi_norm +
        0.20 * prev_inj +
        0.15 * rs_inv
    ).round(4)
    return df


def assign_injury_severity(row: pd.Series) -> str:
    """
    Rule-based severity:
      severe   : risk_score > 0.70 OR (fatigue_index > 1.8 AND recovery_score < 0.40)
      moderate : risk_score > 0.45 OR fatigue_index > 1.2 OR workload_index > 250
      mild     : otherwise
    """
    rs  = row.get("risk_score", 0)
    fi  = row["fatigue_index"]
    rsc = row["recovery_score"]
    wi  = row["workload_index"]

    if rs > 0.70 or (fi > 1.8 and rsc < 0.40):
        return "severe"
    elif rs > 0.45 or fi > 1.2 or rsc < 0.25 or wi > 250:
        return "moderate"
    else:
        return "mild"


def add_injury_severity(df: pd.DataFrame) -> pd.DataFrame:
    df["injury_severity"] = df.apply(assign_injury_severity, axis=1)
    dist = df["injury_severity"].value_counts().to_dict()
    print(f"  [Feature Eng] Severity distribution: {dist}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline."""
    print("\n" + "=" * 50)
    print("  FEATURE ENGINEERING")
    print("=" * 50)

    df = add_workload_index(df)
    print(f"  workload_index  -> min={df['workload_index'].min():.1f}, "
          f"max={df['workload_index'].max():.1f}, mean={df['workload_index'].mean():.1f}")

    df = add_fatigue_index(df)
    print(f"  fatigue_index   -> min={df['fatigue_index'].min():.2f}, "
          f"max={df['fatigue_index'].max():.2f}, mean={df['fatigue_index'].mean():.2f}")

    df = add_recovery_score(df)
    print(f"  recovery_score  -> min={df['recovery_score'].min():.4f}, "
          f"max={df['recovery_score'].max():.4f}, mean={df['recovery_score'].mean():.4f}")

    df = add_risk_score(df)
    print(f"  risk_score      -> min={df['risk_score'].min():.4f}, "
          f"max={df['risk_score'].max():.4f}, mean={df['risk_score'].mean():.4f}")

    df = add_injury_severity(df)

    print(f"\n  Final shape after engineering: {df.shape}")
    return df


if __name__ == "__main__":
    from generate_datasets import main as gen
    from data_preprocessing import preprocess
    gen()
    df, _ = preprocess()
    df = engineer_features(df)
    print(df[["workload_index", "fatigue_index", "recovery_score",
              "risk_score", "injury_severity"]].head(10))
