"""
recommendation_engine.py
------------------------
Rule-based recovery recommendation system.
Produces personalized suggestions based on:
  - Predicted injury risk (injury_indicator)
  - Predicted severity category (injury_severity)
  - Physiological signals (heart_rate, fatigue_index, risk_score)
  - Athlete position (position-specific advice)
"""

from __future__ import annotations
import pandas as pd


# ── Base severity rules ────────────────────────────────────────────────────────

SEVERITY_RULES: dict[str, dict] = {
    "severe": {
        "rest_days": "5–7 days",
        "avoid": [
            "High-intensity training",
            "Contact sports",
            "Heavy weightlifting",
            "Long-distance running",
            "Competitive matches",
        ],
        "tips": [
            "Seek immediate physiotherapy assessment.",
            "Apply RICE protocol (Rest, Ice, Compression, Elevation).",
            "Consume anti-inflammatory foods (turmeric, omega-3 rich fish, berries).",
            "Sleep at least 9 hours per night — tissue repair peaks during deep sleep.",
            "Monitor pain levels and consult a sports physician before resuming activity.",
            "Ensure adequate protein intake (1.8–2.4 g/kg body weight) for muscle repair.",
        ],
    },
    "moderate": {
        "rest_days": "2–3 days",
        "avoid": [
            "Explosive movements (sprints, plyometrics)",
            "Heavy resistance training",
            "Competitive play or matches",
        ],
        "tips": [
            "Perform light stretching and foam rolling daily.",
            "Stay hydrated — aim for at least 3 L of water per day.",
            "Use compression garments on affected areas during recovery.",
            "Prioritise 7–8 hours of quality sleep.",
            "Gradually resume training at 40–60% intensity after rest period.",
            "Track HRV (heart rate variability) to gauge readiness.",
        ],
    },
    "mild": {
        "rest_days": "1 day",
        "avoid": [
            "Overloading the affected muscle group",
            "Skipping warm-up routines",
        ],
        "tips": [
            "Light aerobic activity (walking, cycling, swimming) is encouraged.",
            "Gently stretch affected areas after a 5-minute warm-up.",
            "Monitor fatigue score over the next 48 hours.",
            "Ensure adequate protein intake (1.6–2.2 g/kg body weight).",
            "Plan a scheduled full rest day within the next 3–4 days.",
        ],
    },
}


# ── Supplemental / threshold rules ────────────────────────────────────────────

SUPPLEMENTAL_RULES = {
    "high_heart_rate": {
        "threshold": 100,
        "tips": [
            "Elevated heart rate — increase fluid intake immediately.",
            "Practice diaphragmatic (belly) breathing to activate parasympathetic recovery.",
            "Avoid caffeine and stimulants for 24 hours.",
        ],
    },
    "high_fatigue": {
        "threshold": 1.5,
        "tips": [
            "High fatigue index — reduce training volume by 30–40% this week.",
            "Consider a 20-minute nap in the early afternoon (before 3 pm).",
            "Review your sleep environment: darkness, cool temperature, reduced noise.",
        ],
    },
    "low_recovery": {
        "threshold": 0.20,
        "tips": [
            "Low recovery score — prioritise sleep above all other interventions.",
            "Schedule a deload week with only mobility and light aerobic work.",
        ],
    },
    "high_risk_score": {
        "threshold": 0.65,
        "tips": [
            "Composite risk score is high — consider requesting a fitness test from your team physician.",
            "Reduce match load for the next 2 weeks where possible.",
        ],
    },
    "no_injury": {
        "tips": [
            "No injury risk detected — maintain current training load.",
            "Continue monitoring fatigue and recovery metrics weekly.",
            "Include 1 full rest day per week as a preventive measure.",
            "Keep sleep duration above 7 hours to sustain performance.",
        ],
    },
}


# ── Position-specific advice ───────────────────────────────────────────────────

POSITION_TIPS: dict[str, list[str]] = {
    "Forward": [
        "Focus on lower-limb flexibility — hamstring and hip-flexor stretching.",
        "Explosive sprint drills should pause during injury recovery.",
    ],
    "Midfielder": [
        "Prioritise cardiovascular endurance recovery over strength work.",
        "Monitor cumulative match minutes — high-volume midfielders are fatigue-prone.",
    ],
    "Defender": [
        "Tackle drills and contact practice should cease until cleared by physio.",
        "Concentrate on core stability exercises during recovery.",
    ],
    "Goalkeeper": [
        "Avoid diving and explosive lateral movements until fully recovered.",
        "Focus on upper-body mobility and reaction drills at low intensity.",
    ],
    "Sprinter": [
        "Hamstring and glute activation is critical before any return-to-running.",
        "Avoid maximal-speed efforts; limit to 70% pace during recovery.",
    ],
    "Swimmer": [
        "Shoulder mobility exercises (rotator cuff) are recommended daily.",
        "Pool jogging or kick-board drills can maintain fitness without arm load.",
    ],
    "Cyclist": [
        "Use a stationary bike at low resistance to maintain aerobic base.",
        "Monitor knee-tracking mechanics before returning to full cycling loads.",
    ],
    "Gymnast": [
        "Avoid weight-bearing tricks until balance and proprioception are confirmed.",
        "Focus on flexibility and cognitive/confidence recovery.",
    ],
    "Basketball Guard": [
        "Ankle-stability exercises are critical — use resistance bands.",
        "Practice ball-handling and shooting stationary before lateral movement.",
    ],
    "Tennis Player": [
        "Avoid serving and overhead strokes until shoulder is fully mobile.",
        "Focus on footwork agility at 50% pace before return to baseline rallying.",
    ],
}

# Position index → name map (matches encoding in data_preprocessing.py)
_POSITION_NAMES = [
    "Basketball Guard", "Cyclist", "Defender", "Forward",
    "Goalkeeper", "Gymnast", "Midfielder", "Sprinter",
    "Swimmer", "Tennis Player"
]


# ── Core recommendation function ───────────────────────────────────────────────

def generate_recommendation(row: pd.Series) -> dict:
    """
    Generate a full recovery recommendation for a single athlete row.
    """
    injury_flag  = int(row.get("injury_indicator", 0))
    severity     = str(row.get("injury_severity", "mild")).lower()
    heart_rate   = float(row.get("heart_rate", 70))
    fatigue_idx  = float(row.get("fatigue_index", 0.0))
    recovery_sc  = float(row.get("recovery_score", 0.5))
    risk_sc      = float(row.get("risk_score", 0.0))
    position_enc = int(row.get("position", -1))

    rec: dict = {}

    if injury_flag == 0:
        rec["status"]    = "Low Injury Risk"
        rec["risk_level"] = "low"
        rec["rest_days"] = "0 (active recovery recommended)"
        rec["avoid"]     = []
        rec["tips"]      = list(SUPPLEMENTAL_RULES["no_injury"]["tips"])
    else:
        if severity == "severe":
            rec["risk_level"] = "high"
        elif severity == "moderate":
            rec["risk_level"] = "medium"
        else:
            rec["risk_level"] = "low"

        rec["status"]    = f"Injury Risk — Severity: {severity.capitalize()}"
        base = SEVERITY_RULES.get(severity, SEVERITY_RULES["mild"])
        rec["rest_days"] = base["rest_days"]
        rec["avoid"]     = list(base["avoid"])
        rec["tips"]      = list(base["tips"])

    # Supplemental thresholds
    if heart_rate > SUPPLEMENTAL_RULES["high_heart_rate"]["threshold"]:
        rec["tips"].extend(SUPPLEMENTAL_RULES["high_heart_rate"]["tips"])

    if fatigue_idx > SUPPLEMENTAL_RULES["high_fatigue"]["threshold"]:
        rec["tips"].extend(SUPPLEMENTAL_RULES["high_fatigue"]["tips"])

    if recovery_sc < SUPPLEMENTAL_RULES["low_recovery"]["threshold"]:
        rec["tips"].extend(SUPPLEMENTAL_RULES["low_recovery"]["tips"])

    if risk_sc > SUPPLEMENTAL_RULES["high_risk_score"]["threshold"]:
        rec["tips"].extend(SUPPLEMENTAL_RULES["high_risk_score"]["tips"])

    # Position-specific advice
    if 0 <= position_enc < len(_POSITION_NAMES):
        pos_name = _POSITION_NAMES[position_enc]
        if pos_name in POSITION_TIPS:
            rec["tips"].extend(POSITION_TIPS[pos_name])
        rec["position_name"] = pos_name
    else:
        rec["position_name"] = "N/A"

    return rec


def recommend_for_input(input_dict: dict) -> dict:
    """
    Generate a recommendation directly from a raw input dict (for UI use).
    Accepts the same keys as generate_recommendation's row.
    """
    return generate_recommendation(pd.Series(input_dict))


def generate_all_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """Apply recommendation generation to every row."""
    results = df.apply(generate_recommendation, axis=1)
    rec_df  = pd.DataFrame(list(results))
    return pd.concat([df.reset_index(drop=True), rec_df.reset_index(drop=True)], axis=1)


def display_recommendations(df: pd.DataFrame, n: int = 10):
    """Print recommendations for the first n athletes."""
    print("\n" + "=" * 60)
    print("  RECOVERY RECOMMENDATIONS")
    print("=" * 60)

    id_col = "athlete_id" if "athlete_id" in df.columns else None
    sample = df.head(n)

    for i, (_, row) in enumerate(sample.iterrows(), start=1):
        aid = row.get(id_col) if id_col else f"Athlete #{i}"
        print(f"\n[{i}] {aid}")
        print(f"    Status    : {row['status']}")
        print(f"    Rest Days : {row['rest_days']}")
        if row["avoid"]:
            print(f"    Avoid     : {'; '.join(row['avoid'][:2])}")
        print(f"    Tips      :")
        for tip in row["tips"][:3]:
            print(f"      * {tip}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    from generate_datasets import main as gen
    from data_preprocessing import preprocess
    from feature_engineering import engineer_features

    gen()
    df, _ = preprocess()
    df = engineer_features(df)
    df_rec = generate_all_recommendations(df)
    display_recommendations(df_rec, n=5)
