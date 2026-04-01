"""
injury_prediction_model.py
--------------------------
Trains and evaluates:
  1. Logistic Regression  (baseline)
  2. Random Forest        (primary — saved as injury_model.pkl)

Uses SMOTE to balance classes and Stratified K-Fold for robust evaluation.
Exposes predict_single() helper for UI inference.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
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
]
TARGET_COL = "injury_indicator"


def _print_separator(title: str):
    print(f"\n{'─' * 55}")
    print(f"  {title}")
    print(f"{'─' * 55}")


def train_injury_model(df: pd.DataFrame) -> tuple[RandomForestClassifier, list[str]]:
    """
    Trains Logistic Regression (baseline) + Random Forest (primary).
    Returns (best_model, feature_columns).
    """
    _print_separator("INJURY PREDICTION MODEL")

    available_features = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available_features].copy()
    y = df[TARGET_COL].copy()

    print(f"  Features used : {len(available_features)}")
    print(f"  Class balance : {dict(y.value_counts())}")

    # ── Train / test split ─────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # ── SMOTE on training set ──────────────────────────────────────────────────
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"  After SMOTE   : {dict(pd.Series(y_train_res).value_counts())}")

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Logistic Regression (baseline)
    # ─────────────────────────────────────────────────────────────────────────
    _print_separator("Logistic Regression — Baseline")
    lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    lr.fit(X_train_res, y_train_res)
    lr_preds = lr.predict(X_test)
    lr_acc   = accuracy_score(y_test, lr_preds)
    lr_auc   = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
    print(f"  Accuracy : {lr_acc:.4f}  ({lr_acc * 100:.2f}%)")
    print(f"  ROC-AUC  : {lr_auc:.4f}")
    print("\n  Classification Report:")
    for line in classification_report(y_test, lr_preds, target_names=["No Injury", "Injury"]).splitlines():
        print(f"    {line}")

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Random Forest (primary)
    # ─────────────────────────────────────────────────────────────────────────
    _print_separator("Random Forest — Primary Model")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=15, min_samples_split=4,
        random_state=42, n_jobs=-1
    )
    cv_scores = cross_val_score(rf, X_train_res, y_train_res, cv=cv, scoring="accuracy")
    print(f"  5-Fold CV Accuracy : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    rf.fit(X_train_res, y_train_res)
    rf_preds = rf.predict(X_test)
    rf_acc   = accuracy_score(y_test, rf_preds)
    rf_auc   = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    print(f"  Test Accuracy : {rf_acc:.4f}  ({rf_acc * 100:.2f}%)")
    print(f"  ROC-AUC       : {rf_auc:.4f}")

    print("\n  Classification Report:")
    for line in classification_report(y_test, rf_preds, target_names=["No Injury", "Injury"]).splitlines():
        print(f"    {line}")

    print("\n  Confusion Matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(y_test, rf_preds)
    print(f"              No Injury   Injury")
    for i, label in enumerate(["No Injury", "Injury "]):
        row_str = "  ".join(f"{v:>10}" for v in cm[i])
        print(f"  {label:<12}{row_str}")

    print("\n  Top 8 Feature Importances:")
    importances = pd.Series(rf.feature_importances_, index=available_features).nlargest(8)
    for feat, imp in importances.items():
        bar = "#" * int(imp * 60)
        print(f"    {feat:<25} {imp:.4f}  {bar}")

    # ── Save ───────────────────────────────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "injury_model.pkl")
    joblib.dump({
        "model":    rf,
        "features": available_features,
        "accuracy": rf_acc,
        "auc":      rf_auc,
        "importances": importances.to_dict(),
    }, model_path)
    print(f"\n  [OK] Injury model saved -> {model_path}")

    return rf, available_features


def predict_single(input_dict: dict, model: RandomForestClassifier,
                   features: list[str]) -> tuple[int, float]:
    """
    Predict injury risk for a single athlete.

    Parameters
    ----------
    input_dict : dict with feature values (keys must match feature names)
    model      : trained RandomForestClassifier
    features   : list of feature column names (in training order)

    Returns
    -------
    (prediction, probability)
      prediction = 0 (no injury) or 1 (injury)
      probability = probability of injury (class 1)
    """
    row = pd.DataFrame([{f: input_dict.get(f, 0) for f in features}])
    pred  = int(model.predict(row)[0])
    proba = float(model.predict_proba(row)[0][1])
    return pred, proba


if __name__ == "__main__":
    from generate_datasets import main as gen
    from data_preprocessing import preprocess
    from feature_engineering import engineer_features
    gen()
    df, _ = preprocess()
    df = engineer_features(df)
    train_injury_model(df)
