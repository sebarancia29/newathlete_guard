"""
data_preprocessing.py
---------------------
Loads, cleans, and merges the two datasets:
  - sports_injury_dataset.csv
  - fitbit_fitness_dataset.csv

Returns a unified DataFrame plus a dict of encoders for UI replication.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


# ── Encoding maps (consistent for UI) ──────────────────────────────────────────

INTENSITY_MAP     = {"Low": 0, "Medium": 1, "High": 2}
ACTIVITY_MAP      = {"Sedentary": 0, "Lightly Active": 1, "Active": 2, "Very Active": 3}

GENDER_ENCODER    = LabelEncoder()
GENDER_ENCODER.classes_ = np.array(["Female", "Male"])

POSITION_ENCODER  = LabelEncoder()
POSITION_ENCODER.classes_ = np.array([
    "Basketball Guard", "Cyclist", "Defender", "Forward",
    "Goalkeeper", "Gymnast", "Midfielder", "Sprinter",
    "Swimmer", "Tennis Player"
])


def _load_csvs():
    inj_path    = os.path.join(DATA_DIR, "sports_injury_dataset.csv")
    fitbit_path = os.path.join(DATA_DIR, "fitbit_fitness_dataset.csv")

    injury_df = pd.read_csv(inj_path)
    fitbit_df = pd.read_csv(fitbit_path)
    print(f"  Loaded injury dataset : {injury_df.shape}")
    print(f"  Loaded Fitbit dataset : {fitbit_df.shape}")
    return injury_df, fitbit_df


def _handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    fill_report = {}
    for col in num_cols:
        n = df[col].isna().sum()
        if n > 0:
            df[col] = df[col].fillna(df[col].median())
            fill_report[col] = n

    for col in cat_cols:
        n = df[col].isna().sum()
        if n > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
            fill_report[col] = n

    if fill_report:
        print(f"  Missing-value fills   : {fill_report}")
    else:
        print("  No missing values found.")
    return df


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    # training_intensity
    df["training_intensity"] = df["training_intensity"].map(INTENSITY_MAP).fillna(1).astype(int)

    # activity_level (may be absent when working on injury-only slice)
    if "activity_level" in df.columns:
        df["activity_level"] = df["activity_level"].map(ACTIVITY_MAP).fillna(1).astype(int)

    # gender
    if "gender" in df.columns:
        df["gender"] = df["gender"].apply(
            lambda g: 1 if str(g).strip().lower() == "male" else 0
        )

    # position
    if "position" in df.columns:
        known = list(POSITION_ENCODER.classes_)
        df["position"] = df["position"].apply(
            lambda p: known.index(p) if p in known else 0
        )

    return df


def merge_datasets(injury_df: pd.DataFrame, fitbit_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge by index-based alignment (same synthetic generation seed → same order).
    In a real project this could be joined on athlete_id.
    """
    merged = pd.concat(
        [injury_df.reset_index(drop=True), fitbit_df.reset_index(drop=True)],
        axis=1
    )
    print(f"  Merged shape          : {merged.shape}")
    return merged


def preprocess() -> tuple[pd.DataFrame, dict]:
    print("\n" + "=" * 50)
    print("  DATA PREPROCESSING")
    print("=" * 50)

    injury_df, fitbit_df = _load_csvs()

    # Handle missing before merge
    injury_df = _handle_missing(injury_df)
    fitbit_df = _handle_missing(fitbit_df)

    # Merge
    df = merge_datasets(injury_df, fitbit_df)

    # Drop athlete_id (not a model feature)
    if "athlete_id" in df.columns:
        df.set_index("athlete_id", inplace=True)
        df.index.name = "athlete_id"

    # Encode categoricals
    df = _encode_categoricals(df)

    print(f"\n  Final shape           : {df.shape}")
    print(f"  Columns               : {list(df.columns)}")

    encoders = {
        "training_intensity": INTENSITY_MAP,
        "activity_level":     ACTIVITY_MAP,
        "gender_map":         {"Female": 0, "Male": 1},
        "position_encoder":   POSITION_ENCODER,
    }
    return df, encoders


if __name__ == "__main__":
    from generate_datasets import main as gen
    gen()
    df, enc = preprocess()
    print(df.head())
