"""
severity_model.py
-----------------
Trains and evaluates a Random Forest classifier for injury severity:
  Classes: mild (0) / moderate (1) / severe (2)

Exposes predict_single() for UI inference.
Saves trained model + metadata to models/severity_model.pkl
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from imblearn.over_sampling import SMOTE

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

FEATURE_COLS = [
    "age",
    "gender",
    "position",
    "previous_injury",
    "training_intensity",
    "match_frequency",
    "training_hours",
    "fatigue_score",
    "recovery_time",
    "activity_level",
    "sleep_duration",
    "heart_rate",
    "steps_per_day",
    "calories_burned",
    "hydration_level",
    "workload_index",
    "fatigue_index",
    "recovery_score",
    "risk_score",
    "injury_indicator",
]
TARGET_COL    = "injury_severity"
SEVERITY_ORDER = ["mild", "moderate", "severe"]


def _print_separator(title: str):
    print(f"\n{'─' * 55}")
    print(f"  {title}")
    print(f"{'─' * 55}")


def train_severity_model(df: pd.DataFrame) -> tuple[RandomForestClassifier, LabelEncoder, list[str]]:
    """
    Trains a Random Forest on injury_severity.
    Returns (model, label_encoder, feature_columns).
    """
    _print_separator("INJURY SEVERITY MODEL")

    le = LabelEncoder()
    le.classes_ = np.array(SEVERITY_ORDER)
    y = df[TARGET_COL].map({"mild": 0, "moderate": 1, "severe": 2})

    available_features = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available_features].copy()

    print(f"  Features used         : {len(available_features)}")
    print(f"  Class distribution    : {dict(y.value_counts().sort_index())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"  Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

    # ── SMOTE ─────────────────────────────────────────────────────────────────
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"  After SMOTE           : {dict(pd.Series(y_train_res).value_counts().sort_index())}")

    # ── Cross-validation ───────────────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=15, min_samples_split=4,
        random_state=42, n_jobs=-1
    )
    cv_scores = cross_val_score(rf, X_train_res, y_train_res, cv=cv, scoring="accuracy")
    print(f"  5-Fold CV Accuracy    : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    rf.fit(X_train_res, y_train_res)
    preds = rf.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    print(f"\n  Test Accuracy         : {acc:.4f}  ({acc * 100:.2f}%)")

    print("\n  Classification Report:")
    for line in classification_report(y_test, preds, target_names=SEVERITY_ORDER).splitlines():
        print(f"    {line}")

    print("  Confusion Matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(y_test, preds)
    header = "            " + "  ".join(f"{s:>10}" for s in SEVERITY_ORDER)
    print(header)
    for i, row_label in enumerate(SEVERITY_ORDER):
        row_str = "  ".join(f"{v:>10}" for v in cm[i])
        print(f"  {row_label:<12}{row_str}")

    print("\n  Top 8 Feature Importances:")
    importances = pd.Series(rf.feature_importances_, index=available_features).nlargest(8)
    for feat, imp in importances.items():
        bar = "#" * int(imp * 60)
        print(f"    {feat:<25} {imp:.4f}  {bar}")

    # ── Save ───────────────────────────────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "severity_model.pkl")
    joblib.dump({
        "model":       rf,
        "encoder":     le,
        "features":    available_features,
        "accuracy":    acc,
        "importances": importances.to_dict(),
    }, model_path)
    print(f"\n  [OK] Severity model saved -> {model_path}")

    return rf, le, available_features


def predict_single(input_dict: dict, model: RandomForestClassifier,
                   features: list[str]) -> tuple[str, list[float]]:
    """
    Predict injury severity for a single athlete.

    Returns
    -------
    (severity_label, class_probabilities)
      severity_label       : 'mild' | 'moderate' | 'severe'
      class_probabilities  : [prob_mild, prob_moderate, prob_severe]
    """
    row   = pd.DataFrame([{f: input_dict.get(f, 0) for f in features}])
    idx   = int(model.predict(row)[0])
    proba = model.predict_proba(row)[0].tolist()
    label = SEVERITY_ORDER[idx]
    return label, proba


if __name__ == "__main__":
    from generate_datasets import main as gen
    from data_preprocessing import preprocess
    from feature_engineering import engineer_features
    gen()
    df, _ = preprocess()
    df = engineer_features(df)
    train_severity_model(df)
