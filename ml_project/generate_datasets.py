"""
generate_datasets.py
--------------------
Generates two synthetic CSV datasets for the Athlete Injury Prediction System:
  1. sports_injury_dataset.csv  - core athlete performance & injury data
  2. fitbit_fitness_dataset.csv - physiological & lifestyle data (Fitbit-style)
"""

import numpy as np
import pandas as pd
import os

RANDOM_SEED = 42
N_ATHLETES = 1000

np.random.seed(RANDOM_SEED)

POSITIONS = ["Forward", "Midfielder", "Defender", "Goalkeeper", "Sprinter",
             "Swimmer", "Cyclist", "Gymnast", "Basketball Guard", "Tennis Player"]

GENDERS = ["Male", "Female"]


def generate_sports_injury_dataset(n: int) -> pd.DataFrame:
    """Generate core sports injury dataset with rich fields."""
    athlete_ids = [f"ATH{str(i).zfill(4)}" for i in range(1, n + 1)]

    age                 = np.random.randint(16, 40, size=n)
    gender              = np.random.choice(GENDERS, size=n, p=[0.65, 0.35])
    position            = np.random.choice(POSITIONS, size=n)
    previous_injury     = np.random.choice([0, 1], size=n, p=[0.55, 0.45])

    training_intensity  = np.random.choice(["Low", "Medium", "High"], size=n, p=[0.30, 0.45, 0.25])
    match_frequency     = np.random.randint(1, 15, size=n)          # matches/month
    training_hours      = np.random.uniform(5, 40, size=n).round(1) # hours/week
    fatigue_score       = np.random.uniform(1, 10, size=n).round(2) # 1–10 scale
    recovery_time       = np.random.uniform(0.5, 7, size=n).round(2)# days

    # Injury indicator — probabilistic, influenced by all major risk factors
    injury_prob = (
        0.12
        + 0.06 * (training_intensity == "High").astype(float)
        + 0.05 * (fatigue_score / 10)
        + 0.04 * (training_hours / 40)
        - 0.05 * (recovery_time / 7)
        + 0.07 * previous_injury.astype(float)
        + 0.015 * np.clip((age - 16) / 24, 0, 1)   # older athletes slightly higher risk
    )
    injury_prob    = np.clip(injury_prob, 0.05, 0.95)
    injury_indicator = (np.random.rand(n) < injury_prob).astype(int)

    df = pd.DataFrame({
        "athlete_id":        athlete_ids,
        "age":               age,
        "gender":            gender,
        "position":          position,
        "previous_injury":   previous_injury,
        "training_intensity": training_intensity,
        "match_frequency":   match_frequency,
        "training_hours":    training_hours,
        "fatigue_score":     fatigue_score,
        "recovery_time":     recovery_time,
        "injury_indicator":  injury_indicator,
    })

    # Introduce a small % of missing values (~3%) to simulate real-world data
    for col in ["fatigue_score", "recovery_time", "training_hours"]:
        mask = np.random.rand(n) < 0.03
        df.loc[mask, col] = np.nan

    return df


def generate_fitbit_dataset(n: int) -> pd.DataFrame:
    """Generate Fitbit-style physiological & lifestyle dataset."""
    sleep_duration = np.random.normal(loc=7.0, scale=1.2, size=n).round(1)
    sleep_duration = np.clip(sleep_duration, 3, 11)

    heart_rate     = np.random.normal(loc=72, scale=12, size=n).round(0).astype(int)
    heart_rate     = np.clip(heart_rate, 50, 130)

    activity_level = np.random.choice(
        ["Sedentary", "Lightly Active", "Active", "Very Active"],
        size=n,
        p=[0.15, 0.30, 0.35, 0.20],
    )
    steps_per_day  = np.random.randint(2000, 20000, size=n)
    calories_burned = np.random.randint(1500, 4500, size=n)
    hydration_level = np.random.uniform(1.0, 4.0, size=n).round(1)  # litres/day

    df = pd.DataFrame({
        "sleep_duration":  sleep_duration,
        "heart_rate":      heart_rate,
        "activity_level":  activity_level,
        "steps_per_day":   steps_per_day,
        "calories_burned": calories_burned,
        "hydration_level": hydration_level,
    })

    # Introduce a small % of missing values (~2%)
    for col in ["sleep_duration", "heart_rate", "hydration_level"]:
        mask = np.random.rand(n) < 0.02
        df.loc[mask, col] = np.nan

    return df


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(out_dir, exist_ok=True)

    print("[1/2] Generating sports injury dataset...")
    injury_df = generate_sports_injury_dataset(N_ATHLETES)
    path1 = os.path.join(out_dir, "sports_injury_dataset.csv")
    injury_df.to_csv(path1, index=False)
    print(f"      Saved -> {path1}  ({len(injury_df)} rows, {injury_df.shape[1]} cols)")

    print("[2/2] Generating Fitbit fitness dataset...")
    fitbit_df = generate_fitbit_dataset(N_ATHLETES)
    path2 = os.path.join(out_dir, "fitbit_fitness_dataset.csv")
    fitbit_df.to_csv(path2, index=False)
    print(f"      Saved -> {path2}  ({len(fitbit_df)} rows, {fitbit_df.shape[1]} cols)")

    print("\nDataset generation complete.")
    return injury_df, fitbit_df


if __name__ == "__main__":
    main()
