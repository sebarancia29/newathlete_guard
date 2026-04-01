"""
google_fit.py
-------------
Google Fit API integration for the Athlete Injury Prediction System.

Two operating modes
-------------------
LIVE MODE  — requires credentials.json from a Google Cloud project with
             Fitness API enabled. Performs OAuth2 and fetches real-time
             heart rate, steps, calories, and sleep data.

SIMULATION — when credentials.json is absent (default for demos), samples
             a realistic row from the local Fitbit dataset, conditioned on
             the athlete's entered fatigue score for coherence.

Exposed API
-----------
  is_credentials_available() -> bool
  get_oauth_url(state)        -> str   (LIVE only)
  exchange_code(code, state)  -> dict  (LIVE only) — stores creds in session
  fetch_all_live(creds)       -> dict
  simulate_from_fitbit(fatigue_score, seed=None) -> dict
  describe_source(live: bool) -> str   — human-readable badge text
"""

from __future__ import annotations

import json
import os
import random
import time
from typing import Optional

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT            = os.path.dirname(__file__)
_CREDENTIALS_PATH = os.path.join(_ROOT, "credentials.json")
_FITBIT_PATH     = os.path.join(_ROOT, "data", "fitbit_fitness_dataset.csv")

# ── Google Fit data-type identifiers ──────────────────────────────────────────
_DT_HEART_RATE = "com.google.heart_rate.bpm"
_DT_STEPS      = "com.google.step_count.delta"
_DT_CALORIES   = "com.google.calories.expended"
_DT_SLEEP      = "com.google.sleep.segment"
_DT_ACTIVITY   = "com.google.activity.segment"

# OAuth2 scopes
_SCOPES = [
    "https://www.googleapis.com/auth/fitness.activity.read",
    "https://www.googleapis.com/auth/fitness.body.read",
    "https://www.googleapis.com/auth/fitness.sleep.read",
]

# activity_level encoding (matches app.py ACTIVITY_MAP)
_ACTIVITY_ENCODE = {"Sedentary": 0, "Lightly Active": 1, "Active": 2, "Very Active": 3}


# ══════════════════════════════════════════════════════════════════════════════
# ── Credential helpers ────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def is_credentials_available() -> bool:
    """Return True if a credentials.json file exists in the project root."""
    return os.path.isfile(_CREDENTIALS_PATH)


def _load_client_config() -> dict:
    with open(_CREDENTIALS_PATH, "r") as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# ── LIVE MODE — OAuth2 flow ───────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def get_oauth_url(state: str = "athleteguard") -> str:
    """
    Build the Google OAuth2 authorisation URL.
    The user must be redirected to this URL in their browser.

    Returns the URL string. Raises RuntimeError if credentials.json missing.
    """
    try:
        from google_auth_oauthlib.flow import Flow
    except ImportError as exc:
        raise RuntimeError(
            "google-auth-oauthlib is not installed. "
            "Run: pip install google-auth-oauthlib"
        ) from exc

    if not is_credentials_available():
        raise RuntimeError("credentials.json not found in project root.")

    flow = Flow.from_client_secrets_file(
        _CREDENTIALS_PATH,
        scopes=_SCOPES,
        redirect_uri="http://localhost:8501",
    )
    auth_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        state=state,
        prompt="consent",
    )
    return auth_url


def exchange_code(code: str, state: str = "athleteguard") -> dict:
    """
    Exchange the authorisation code returned by Google for credentials.
    Returns a serialisable credentials dict suitable for storage in
    st.session_state or a JSON file.

    Raises RuntimeError on failure.
    """
    try:
        from google_auth_oauthlib.flow import Flow
    except ImportError as exc:
        raise RuntimeError("google-auth-oauthlib is not installed.") from exc

    flow = Flow.from_client_secrets_file(
        _CREDENTIALS_PATH,
        scopes=_SCOPES,
        redirect_uri="http://localhost:8501",
        state=state,
    )
    flow.fetch_token(code=code)
    creds = flow.credentials
    return {
        "token":         creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri":     creds.token_uri,
        "client_id":     creds.client_id,
        "client_secret": creds.client_secret,
        "scopes":        list(creds.scopes),
    }


def _build_service(creds_dict: dict):
    """Build a Google Fit API service object from a stored credentials dict."""
    try:
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise RuntimeError(
            "google-api-python-client is not installed. "
            "Run: pip install google-api-python-client"
        ) from exc

    creds = Credentials(
        token=creds_dict["token"],
        refresh_token=creds_dict.get("refresh_token"),
        token_uri=creds_dict.get("token_uri", "https://oauth2.googleapis.com/token"),
        client_id=creds_dict["client_id"],
        client_secret=creds_dict["client_secret"],
        scopes=creds_dict["scopes"],
    )
    return build("fitness", "v1", credentials=creds, cache_discovery=False)


def _nanoseconds_window(hours: int = 24) -> tuple[int, int]:
    """Return (start_ns, end_ns) for a trailing window in nanoseconds."""
    end_ms   = int(time.time() * 1000)
    start_ms = end_ms - hours * 3_600_000
    return start_ms * 1_000_000, end_ms * 1_000_000


def _aggregate(service, data_type: str, bucket_by_time_ms: int = 86_400_000) -> list:
    """Call the Fitness aggregate endpoint and return buckets."""
    start_ns, end_ns = _nanoseconds_window(hours=24)
    body = {
        "aggregateBy": [{"dataTypeName": data_type}],
        "bucketByTime": {"durationMillis": bucket_by_time_ms},
        "startTimeMillis": start_ns // 1_000_000,
        "endTimeMillis":   end_ns   // 1_000_000,
    }
    return service.users().dataset().aggregate(userId="me", body=body).execute().get("bucket", [])


def _bucket_mean(buckets: list, value_key: str = "fpVal") -> Optional[float]:
    vals = []
    for b in buckets:
        for ds in b.get("dataset", []):
            for pt in ds.get("point", []):
                for v in pt.get("value", []):
                    if value_key in v:
                        vals.append(v[value_key])
    return float(np.mean(vals)) if vals else None


def _bucket_sum(buckets: list, value_key: str = "intVal") -> Optional[float]:
    total = 0.0
    found = False
    for b in buckets:
        for ds in b.get("dataset", []):
            for pt in ds.get("point", []):
                for v in pt.get("value", []):
                    if value_key in v:
                        total += v[value_key]
                        found  = True
    return total if found else None


def fetch_all_live(creds_dict: dict) -> dict:
    """
    Fetch today's health metrics from Google Fit.

    Returns
    -------
    dict with keys: heart_rate, steps_per_day, calories_burned,
                    sleep_duration, activity_level (encoded 0-3)

    Missing values fall back to sensible defaults.
    """
    service = _build_service(creds_dict)

    # ── Heart Rate ─────────────────────────────────────────────────────────────
    hr_buckets = _aggregate(service, _DT_HEART_RATE)
    heart_rate = _bucket_mean(hr_buckets, "fpVal") or 72.0

    # ── Steps ──────────────────────────────────────────────────────────────────
    step_buckets = _aggregate(service, _DT_STEPS)
    steps = _bucket_sum(step_buckets, "intVal") or 8000

    # ── Calories ───────────────────────────────────────────────────────────────
    cal_buckets = _aggregate(service, _DT_CALORIES)
    calories = _bucket_sum(cal_buckets, "fpVal") or 2400

    # ── Sleep (previous night, up to 12h window) ───────────────────────────────
    sleep_duration = _fetch_sleep_duration(service)

    # ── Activity level inferred from steps ────────────────────────────────────
    activity_level = _steps_to_activity(int(steps))

    return {
        "heart_rate":     round(float(heart_rate), 1),
        "steps_per_day":  int(steps),
        "calories_burned": int(calories),
        "sleep_duration": round(float(sleep_duration), 1),
        "activity_level": activity_level,
    }


def _fetch_sleep_duration(service) -> float:
    """Estimate last night's sleep duration in hours from sleep segments."""
    try:
        end_ns   = int(time.time() * 1_000_000_000)
        start_ns = end_ns - 12 * 3_600 * 1_000_000_000   # last 12 h
        result   = service.users().sessions().list(
            userId="me",
            startTime=_iso(start_ns // 1_000_000_000 - 3600),
            endTime=_iso(end_ns   // 1_000_000_000),
            activityType=72,   # SLEEP
        ).execute()
        sessions = result.get("session", [])
        total_ms = sum(
            int(s["endTimeMillis"]) - int(s["startTimeMillis"])
            for s in sessions
        )
        return round(total_ms / 3_600_000, 1) if total_ms else 7.0
    except Exception:
        return 7.0


def _iso(epoch_seconds: int) -> str:
    import datetime
    return datetime.datetime.utcfromtimestamp(epoch_seconds).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _steps_to_activity(steps: int) -> int:
    """Map daily step count to encoded activity level."""
    if steps < 4000:
        return 0   # Sedentary
    elif steps < 8000:
        return 1   # Lightly Active
    elif steps < 12000:
        return 2   # Active
    else:
        return 3   # Very Active


# ══════════════════════════════════════════════════════════════════════════════
# ── SIMULATION MODE ───────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

_fitbit_cache: Optional[pd.DataFrame] = None


def _load_fitbit() -> pd.DataFrame:
    global _fitbit_cache
    if _fitbit_cache is None:
        _fitbit_cache = pd.read_csv(_FITBIT_PATH)
    return _fitbit_cache


def simulate_from_fitbit(fatigue_score: float, seed: Optional[int] = None) -> dict:
    """
    Sample a realistic physiology row from the Fitbit dataset.

    Rows are filtered by fatigue-correlated activity level so the simulated
    data is coherent with the entered fatigue score:
      fatigue 1–3  -> Very Active / Active
      fatigue 4–6  -> Active / Lightly Active
      fatigue 7–10 -> Lightly Active / Sedentary

    Parameters
    ----------
    fatigue_score : float  (1–10)
    seed          : int | None  — for reproducibility (None = random)

    Returns
    -------
    dict with keys: heart_rate, steps_per_day, calories_burned,
                    sleep_duration, activity_level (int 0–3), activity_label (str)
    """
    df = _load_fitbit().copy()

    if fatigue_score <= 3:
        preferred = ["Very Active", "Active"]
    elif fatigue_score <= 6:
        preferred = ["Active", "Lightly Active"]
    else:
        preferred = ["Lightly Active", "Sedentary"]

    subset = df[df["activity_level"].isin(preferred)]
    if subset.empty:
        subset = df   # fallback: use full dataset

    rng = random.Random(seed)
    row = subset.iloc[rng.randint(0, len(subset) - 1)]

    act_label = str(row["activity_level"])
    act_enc   = _ACTIVITY_ENCODE.get(act_label, 2)

    return {
        "heart_rate":     round(float(row["heart_rate"]), 1),
        "steps_per_day":  int(row["steps_per_day"]),
        "calories_burned": int(row["calories_burned"]),
        "sleep_duration": round(float(row["sleep_duration"]), 1),
        "activity_level": act_enc,
        "activity_label": act_label,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ── Utility ───────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def describe_source(live: bool) -> str:
    return "🟢 Google Fit (Live)" if live else "🔵 Simulated Wearable (Fitbit Dataset)"


# ══════════════════════════════════════════════════════════════════════════════
# ── Quick self-test (python google_fit.py) ───────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("credentials.json found:", is_credentials_available())
    print("\nSimulation test (fatigue=7):")
    result = simulate_from_fitbit(fatigue_score=7, seed=42)
    for k, v in result.items():
        print(f"  {k:<20}: {v}")
