"""
app.py
------
Streamlit UI for the Athlete Injury Prediction & Recovery Recommendation System.
4 pages:
  1. Home         — system overview & live dataset stats
  2. Predict      — interactive form -> injury risk + severity + recommendations
  3. Data Explorer— training-data visualisations
  4. Model Performance — accuracy, confusion matrix, feature importance
"""

import os, sys, joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Ensure the ml_project directory is on the import path ─────────────────────
ROOT = os.path.dirname(__file__)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AthleteGuard AI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom dark sports-themed CSS ─────────────────────────────────────────────
st.markdown("""
<style>
 @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

 /* ── Base ──────────────────────────────────────────────────── */
 html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
 .stApp { background: #0d1117; color: #e6edf3; }

 /* Force ALL app text readable */
 .stApp p, .stApp span, .stApp div,
 .stApp li, .stApp label, .stApp small { color: #e6edf3 !important; }

 /* ── Sidebar ────────────────────────────────────────────────── */
 section[data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #30363d; }
 section[data-testid="stSidebar"] * { color: #e6edf3 !important; }

 /* ── Headers ────────────────────────────────────────────────── */
 h1 { color: #58a6ff !important; font-weight: 800; }
 h2 { color: #79c0ff !important; font-weight: 700; }
 h3 { color: #a5d6ff !important; font-weight: 600; }
 h4, h5, h6 { color: #c9d1d9 !important; font-weight: 600; }

 /* ── Widget labels (small text above each input) ─────────────── */
 label,
 [data-testid="stWidgetLabel"],
 [data-testid="stWidgetLabel"] p,
 [data-testid="stWidgetLabel"] span {
 color: #c9d1d9 !important;
 font-size: 0.88rem !important;
 font-weight: 500 !important;
 }

 /* ── Captions / helper text ──────────────────────────────────── */
 .stCaption, [data-testid="stCaptionContainer"],
 [data-testid="stCaptionContainer"] p { color: #8b949e !important; }

 /* ── Text input & textarea ───────────────────────────────────── */
 .stTextInput input, .stTextArea textarea {
 background: #21262d !important;
 border: 1px solid #30363d !important;
 border-radius: 8px !important;
 color: #e6edf3 !important;
 }
 .stTextInput input::placeholder,
 .stTextArea textarea::placeholder { color: #6e7681 !important; }

 /* ── Number input ────────────────────────────────────────────── */
 .stNumberInput input {
 background: #21262d !important;
 border: 1px solid #30363d !important;
 border-radius: 8px !important;
 color: #e6edf3 !important;
 }
 .stNumberInput button {
 background: #30363d !important;
 color: #e6edf3 !important;
 border: 1px solid #484f58 !important;
 }
 .stNumberInput button:hover { background: #388bfd !important; }

 /* ── Selectbox / dropdowns ───────────────────────────────────── */
 .stSelectbox > div > div,
 .stMultiSelect > div > div {
 background: #21262d !important;
 border: 1px solid #30363d !important;
 border-radius: 8px !important;
 color: #e6edf3 !important;
 }
 .stSelectbox > div > div > div,
 .stSelectbox [data-baseweb="select"] span,
 .stSelectbox [data-baseweb="select"] div,
 .stMultiSelect [data-baseweb="select"] span,
 .stMultiSelect [data-baseweb="select"] div { color: #e6edf3 !important; }
 [data-baseweb="popover"], [data-baseweb="menu"],
 [role="listbox"], [role="option"] {
 background: #21262d !important;
 color: #e6edf3 !important;
 }
 [role="option"]:hover { background: #2d333b !important; }

 /* ── Sliders ─────────────────────────────────────────────────── */
 .stSlider > div > div { background: #21262d; border-radius: 8px; }
 .stSlider [data-testid="stTickBarMin"],
 .stSlider [data-testid="stTickBarMax"],
 .stSlider span { color: #c9d1d9 !important; }

 /* ── Radio & Checkbox ─────────────────────────────────────────── */
 .stRadio > div > label > div > p { color: #e6edf3 !important; }
 .stCheckbox > label > div > p { color: #e6edf3 !important; }

 /* ── Metric cards ─────────────────────────────────────────────── */
 div[data-testid="metric-container"] {
 background: #161b22; border: 1px solid #30363d;
 border-radius: 12px; padding: 16px;
 }
 div[data-testid="metric-container"] label { color: #8b949e !important; }
 div[data-testid="metric-container"] [data-testid="metric-value"] { color: #e6edf3 !important; }
 div[data-testid="metric-container"] div[data-testid="metric-delta-icon-up"] { color: #56d364 !important; }

 /* ── Alert banners ────────────────────────────────────────────── */
 [data-testid="stAlert"] p { color: #e6edf3 !important; }

 /* ── Dataframe ────────────────────────────────────────────────── */
 [data-testid="stDataFrame"] * { color: #e6edf3 !important; }

 /* ── Expanders ────────────────────────────────────────────────── */
 .streamlit-expanderHeader { background: #161b22 !important; color: #79c0ff !important; }
 .streamlit-expanderContent { background: #0d1117 !important; }

 /* ── Divider ──────────────────────────────────────────────────── */
 hr { border-color: #30363d; }

 /* ── Risk badges ──────────────────────────────────────────────── */
 .risk-badge {
 display: inline-block; padding: 8px 20px; border-radius: 20px;
 font-weight: 700; font-size: 1.1rem; letter-spacing: 1px; margin-bottom: 8px;
 }
 .risk-high { background: #3d1d1d; color: #ff7b72; border: 2px solid #ff7b72; }
 .risk-medium { background: #2d2a1d; color: #e3b341; border: 2px solid #e3b341; }
 .risk-low { background: #1d2d1d; color: #56d364; border: 2px solid #56d364; }

 /* ── Result cards ─────────────────────────────────────────────── */
 .result-card {
 background: #161b22; border: 1px solid #30363d;
 border-radius: 16px; padding: 24px; margin: 12px 0;
 }
 .tip-item {
 background: #21262d; border-left: 3px solid #58a6ff;
 border-radius: 6px; padding: 10px 14px; margin: 6px 0;
 font-size: 0.9rem; color: #c9d1d9 !important;
 }
 .avoid-item {
 background: #21262d; border-left: 3px solid #ff7b72;
 border-radius: 6px; padding: 8px 14px; margin: 5px 0;
 font-size: 0.88rem; color: #ffa198 !important;
 }

 /* ── Buttons ─────────────────────────────────────────────────── */
 .stButton > button,
 .stFormSubmitButton > button,
 .stDownloadButton > button {
 background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
 color: #ffffff !important;
 font-weight: 700 !important;
 font-size: 0.95rem !important;
 border: none !important;
 border-radius: 10px !important;
 padding: 10px 28px !important;
 transition: opacity 0.2s !important;
 }
 .stButton > button:hover,
 .stFormSubmitButton > button:hover,
 .stDownloadButton > button:hover { opacity: 0.85 !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants / helpers ────────────────────────────────────────────────────────
# ── Positions: professional athletes + student-athletes ───────────────────────
POSITIONS = [
    # Professional / Club Sports
    "Basketball Guard", "Cyclist", "Defender", "Forward",
    "Goalkeeper", "Gymnast", "Midfielder", "Sprinter",
    "Swimmer", "Tennis Player",
    # Student-Athlete Categories
    "Student - Basketball", "Student - Swimming",
    "Student - Track & Field", "Student - Football / Soccer",
    "Student - Tennis", "Student - Gymnastics",
    "Student - Cycling", "Student - Volleyball",
    "Student - Badminton", "Student - Cricket",
    "Student - General Fitness",
]

# ── Sport-specific recommended fatigue scores ──────────────────────────────────
# These defaults represent the TYPICAL training load for each position/sport.
# Users can always adjust the slider; this just gives a sensible starting point.
POSITION_FATIGUE_DEFAULTS = {
    "Basketball Guard":          6.0,
    "Cyclist":                   7.0,
    "Defender":                  6.0,
    "Forward":                   6.5,
    "Goalkeeper":                5.0,
    "Gymnast":                   8.0,
    "Midfielder":                7.0,
    "Sprinter":                  7.5,
    "Swimmer":                   7.0,
    "Tennis Player":             5.5,
    "Student - Basketball":      5.0,
    "Student - Swimming":        6.0,
    "Student - Track & Field":   6.0,
    "Student - Football / Soccer": 5.5,
    "Student - Tennis":          4.0,
    "Student - Gymnastics":      7.0,
    "Student - Cycling":         5.5,
    "Student - Volleyball":      5.0,
    "Student - Badminton":       4.5,
    "Student - Cricket":         4.5,
    "Student - General Fitness": 4.0,
}

INTENSITY_MAP  = {"Low": 0, "Medium": 1, "High": 2}
ACTIVITY_MAP   = {"Sedentary": 0, "Lightly Active": 1, "Active": 2, "Very Active": 3}
SEVERITY_ORDER = ["mild", "moderate", "severe"]
SEVERITY_COLOR = {"mild": "#56d364", "moderate": "#e3b341", "severe": "#ff7b72"}

# ── Athlete records path ───────────────────────────────────────────────────────
RECORDS_PATH = os.path.join(ROOT, "data", "athlete_records.csv")


@st.cache_resource(show_spinner="Training models — first run only…")
def load_or_train_models():
    """Run the full pipeline once & cache results."""
    from generate_datasets import main as gen_data
    from data_preprocessing import preprocess
    from feature_engineering import engineer_features
    from injury_prediction_model import train_injury_model
    from severity_model import train_severity_model
    from recommendation_engine import generate_all_recommendations

    gen_data()
    df, encoders          = preprocess()
    df                    = engineer_features(df)
    inj_model, inj_feats  = train_injury_model(df)
    sev_model, sev_le, sev_feats = train_severity_model(df)
    df_rec                = generate_all_recommendations(df)

    return df_rec, inj_model, inj_feats, sev_model, sev_le, sev_feats


@st.cache_data(show_spinner=False)
def get_model_meta():
    """Load saved pkl files for performance metrics (no retraining)."""
    models_dir = os.path.join(ROOT, "models")
    inj_path   = os.path.join(models_dir, "injury_model.pkl")
    sev_path   = os.path.join(models_dir, "severity_model.pkl")
    inj_meta   = joblib.load(inj_path) if os.path.exists(inj_path) else {}
    sev_meta   = joblib.load(sev_path) if os.path.exists(sev_path) else {}
    return inj_meta, sev_meta


# ── Athlete record persistence ────────────────────────────────────────────────
def save_athlete_record(name: str, gender: str, position: str,
                        age: int, training_hours: float, match_freq: int,
                        training_int: str, fatigue_score: float,
                        previous_inj: str, sleep_dur: float,
                        recovery_time: float, hydration: float,
                        heart_rate: int, steps_per_day: int,
                        calories: int, activity_lv: int,
                        input_dict: dict, inj_prob: float,
                        risk_level: str, sev_label: str, rec: dict) -> None:
    """Append one prediction record to athlete_records.csv."""
    import datetime
    row = {
        "timestamp":         datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "name":              name or "—",
        "age":               age,
        "gender":            gender,
        "position":          position,
        "training_hours":    training_hours,
        "match_frequency":   match_freq,
        "training_intensity": training_int,
        "fatigue_score":     fatigue_score,
        "previous_injury":   previous_inj,
        "sleep_duration":    sleep_dur,
        "recovery_time":     recovery_time,
        "hydration_level":   hydration,
        "heart_rate":        heart_rate,
        "steps_per_day":     steps_per_day,
        "calories_burned":   calories,
        "activity_level":    activity_lv,
        "workload_index":    round(input_dict.get("workload_index", 0), 4),
        "fatigue_index":     round(input_dict.get("fatigue_index",  0), 4),
        "recovery_score":    round(input_dict.get("recovery_score", 0), 4),
        "risk_score":        round(input_dict.get("risk_score",     0), 4),
        "injury_probability": round(inj_prob * 100, 1),
        "risk_level":        risk_level,
        "injury_severity":   sev_label,
        "rest_days":         rec.get("rest_days", ""),
        "tips":              " | ".join(rec.get("tips",  [])),
        "avoid":             " | ".join(rec.get("avoid", [])),
    }
    new_df = pd.DataFrame([row])
    if os.path.exists(RECORDS_PATH):
        new_df.to_csv(RECORDS_PATH, mode="a", header=False, index=False)
    else:
        new_df.to_csv(RECORDS_PATH, index=False)


# ── Sidebar ───────────────────────────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.markdown("## AthleteGuard AI")
        st.markdown("*Injury Prediction & Recovery*")
        st.markdown("---")
        role = st.session_state.get("role", "athlete")
        if role == "coach":
            pages = ["Live Dashboard", "Predict", "Home", "Athlete Records", "Data Explorer", "Model Performance"]
        else:
            pages = ["Live Dashboard", "Predict", "My Records"]

        page = st.radio("Navigate", pages, label_visibility="collapsed")
        st.markdown("---")
        st.markdown("**Models**")
        models_dir = os.path.join(ROOT, "models")
        for fname, label in [("injury_model.pkl", "Injury Model"), ("severity_model.pkl", "Severity Model")]:
            path = os.path.join(models_dir, fname)
            if os.path.exists(path):
                st.success(f"{label}")
            else:
                st.warning(f"{label} (training pending)")
        st.markdown("---")
        st.caption("© 2025 AthleteGuard AI | ML Powered")
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state["logged_in"] = False
            st.session_state.pop("username", None)
            st.rerun()
            
    return page


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ═══════════════════════════════════════════════════════════════════════════════
def page_home(df: pd.DataFrame):
    st.markdown("# AthleteGuard AI")
    st.markdown("### Athlete Injury Prediction & Recovery Recommendation System")
    st.markdown("""
 > Powered by **Machine Learning** (Random Forest + SMOTE) and a **rule-based recommendation engine**,
 > this system helps athletes and coaches predict injury risk, assess severity,
 > and receive personalised recovery guidance.
 """)
    st.markdown("---")

    # ── KPI cards ──────────────────────────────────────────────────────────────
    total      = len(df)
    n_injured  = int(df["injury_indicator"].sum())
    pct_inj    = n_injured / total * 100
    sev_counts = df["injury_severity"].value_counts()
    n_severe   = int(sev_counts.get("severe", 0))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Athletes",    f"{total:,}")
    c2.metric("Injury Cases",      f"{n_injured:,}", f"{pct_inj:.1f}% of cohort")
    c3.metric("Severe Cases",      f"{n_severe:,}",  f"{n_severe/total*100:.1f}% of cohort")
    c4.metric("Avg Fatigue Index", f"{df['fatigue_index'].mean():.3f}")

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Injury Risk Distribution")
        pie_df = df["injury_indicator"].map({0: "No Injury", 1: "Injury"}).value_counts().reset_index()
        pie_df.columns = ["Status", "Count"]
        fig = px.pie(pie_df, values="Count", names="Status",
                     color="Status",
                     color_discrete_map={"No Injury": "#56d364", "Injury": "#ff7b72"},
                     hole=0.5)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#e6edf3", legend=dict(font=dict(color="#e6edf3")),
                          margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("#### Severity Breakdown")
        sev_df = df["injury_severity"].value_counts().reindex(SEVERITY_ORDER).fillna(0).reset_index()
        sev_df.columns = ["Severity", "Count"]
        colors = [SEVERITY_COLOR[s] for s in sev_df["Severity"]]
        fig2 = px.bar(sev_df, x="Severity", y="Count", color="Severity",
                      color_discrete_sequence=colors, text="Count")
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#e6edf3", showlegend=False,
                           margin=dict(l=0, r=0, t=10, b=0),
                           xaxis=dict(gridcolor="#30363d"), yaxis=dict(gridcolor="#30363d"))
        fig2.update_traces(textposition="outside")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Fatigue Index vs Recovery Score")
    sample = df.sample(min(400, len(df)), random_state=42)
    fig3 = px.scatter(
        sample,
        x="fatigue_index", y="recovery_score",
        color="injury_severity",
        color_discrete_map=SEVERITY_COLOR,
        opacity=0.7, size_max=6,
        labels={"fatigue_index": "Fatigue Index", "recovery_score": "Recovery Score"},
    )
    fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font_color="#e6edf3", legend_title_text="Severity",
                       legend=dict(font=dict(color="#e6edf3")),
                       margin=dict(l=0, r=0, t=10, b=0),
                       xaxis=dict(gridcolor="#30363d"), yaxis=dict(gridcolor="#30363d"))
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    st.markdown("#### System Pipeline")
    st.markdown("""
 ```
 Raw Data (Sports + Fitbit)
 │
 ▼
 Data Preprocessing ──► Encoding, Imputation, Merging
 │
 ▼
 Feature Engineering ──► workload_index, fatigue_index, recovery_score, risk_score
 │
 ▼
 ML Models ──► Injury Prediction (RF + SMOTE)
 Severity Classification (RF + SMOTE)
 │
 ▼
 Recommendation Engine ► Rule-based + Position-specific advice
 │
 ▼
 AthleteGuard UI ──► Interactive prediction & personalised recovery plan
 ```
 """)




# =============================================================================
# PAGE 2 -- PREDICT  (with Google Fit / Wearable integration)
# =============================================================================
def _handle_google_fit_oauth():
    """
    Handle OAuth2 callback and Connect button for Google Fit.
    Stores wearable data + source in st.session_state.
    Must be called BEFORE the prediction form so redirects work.

    Two-phase approach to survive Streamlit reruns:
      Phase 1 — URL has ?code=: save code+state to session_state, clear URL, rerun.
      Phase 2 — URL is clean, _pending_oauth_code set: exchange the code exactly once.
    """
    from google_fit import (
        is_credentials_available, get_oauth_url, exchange_code,
        fetch_all_live, simulate_from_fitbit, describe_source,
    )

    # ── Phase 1: capture the code from the URL before any rerun sees it again ──
    params = st.query_params
    if "code" in params and "gfit_creds" not in st.session_state:
        code  = params["code"]
        state = params.get("state", "athleteguard")
        # Store in session_state so Phase 2 can use it after the URL is cleared
        st.session_state.setdefault("_pending_oauth_code",  code)
        st.session_state.setdefault("_pending_oauth_state", state)
        # Clear the URL immediately — this prevents every subsequent rerun
        # from re-entering Phase 1 and trying to re-exchange the same code.
        st.query_params.clear()
        st.rerun()
        return

    # ── Phase 2: exchange the saved code (runs once, URL is already clean) ─────
    if (
        "_pending_oauth_code" in st.session_state
        and "gfit_creds" not in st.session_state
        and not st.session_state.get("_oauth_exchanging", False)
    ):
        st.session_state["_oauth_exchanging"] = True   # prevent duplicate calls
        code  = st.session_state.pop("_pending_oauth_code")
        state = st.session_state.pop("_pending_oauth_state", "athleteguard")
        try:
            creds = exchange_code(code, state)
            data  = fetch_all_live(creds)
            st.session_state["gfit_creds"]      = creds
            st.session_state["wearable_data"]   = data
            st.session_state["wearable_live"]   = True
            st.session_state["wearable_source"] = describe_source(True)
            st.success("Google Fit connected — data auto-filled below.")
        except Exception as e:
            msg = str(e)
            if "invalid_grant" in msg:
                st.error(
                    "Authentication code expired (this can happen if the "
                    "page reloaded too slowly). Please click **Connect Google Fit** again."
                )
            else:
                st.error(f"Google Fit auth error: {e}")
        finally:
            st.session_state["_oauth_exchanging"] = False
        return

    # -- Wearable panel UI -----------------------------------------------------
    st.markdown("---")
    st.markdown("### Wearable Data Source")

    cols = st.columns([1, 1, 1])

    # Button 1: Connect Google Fit (live)
    with cols[0]:
        if is_credentials_available():
            if st.button("Connect Google Fit", use_container_width=True,
                         help="Authenticate with your Google account to fetch live data"):
                try:
                    url = get_oauth_url()
                    # Use JS redirect — more reliable than meta refresh inside
                    # Streamlit's single-page React app.
                    st.markdown(
                        f'<script>window.location.href = "{url}";</script>',
                        unsafe_allow_html=True,
                    )
                    st.info("Redirecting to Google... if nothing happens, "
                            f"[click here]({url}).")
                except Exception as e:
                    st.error(f"Could not build OAuth URL: {e}")
        else:
            st.markdown(
                '<div style="background:#21262d;border:1px solid #30363d;border-radius:8px;'
                'padding:10px;font-size:0.82rem;color:#8b949e">'
                '<b>credentials.json</b> not found.<br>'
                'See <a href="#" style="color:#58a6ff">GOOGLE_FIT_SETUP.md</a> to enable live mode.'
                '</div>',
                unsafe_allow_html=True,
            )

    # Button 2: Simulate wearable data
    with cols[1]:
        if st.button("Use Simulated Wearable Data", use_container_width=True,
                     help="Sample realistic values from Fitbit dataset based on your fatigue score"):
            # Use fatigue score from session state if available, else default 5
            fatigue = st.session_state.get("_last_fatigue", 5.0)
            from google_fit import simulate_from_fitbit, describe_source
            data = simulate_from_fitbit(fatigue_score=fatigue)
            st.session_state["wearable_data"]   = data
            st.session_state["wearable_live"]   = False
            st.session_state["wearable_source"] = describe_source(False)
            st.rerun()

    # Button 3: Clear / manual
    with cols[2]:
        if st.button("Use Manual Entry", use_container_width=True,
                     help="Enter physiological values manually"):
            for k in ["wearable_data", "wearable_live", "wearable_source", "gfit_creds"]:
                st.session_state.pop(k, None)
            st.rerun()

    # -- Status badge ----------------------------------------------------------
    src = st.session_state.get("wearable_source")
    if src:
        badge_color = "#1d2d1d" if "Live" in src else "#1d1d2d"
        border_color = "#56d364" if "Live" in src else "#58a6ff"
        st.markdown(
            f'<div style="background:{badge_color};border:1px solid {border_color};'
            f'border-radius:8px;padding:8px 14px;margin:6px 0;font-size:0.9rem">'
            f'{src}</div>',
            unsafe_allow_html=True,
        )

    # -- Show fetched values with override expander ----------------------------
    w = st.session_state.get("wearable_data")
    if w:
        act_labels = ["Sedentary", "Lightly Active", "Active", "Very Active"]
        act_label  = w.get("activity_label", act_labels[w.get("activity_level", 2)])
        st.markdown(
            f'<div style="background:#21262d;border:1px solid #30363d;border-radius:10px;'
            f'padding:12px 18px;margin:8px 0;font-size:0.88rem;line-height:1.8">'
            f'<b>\u2764\ufe0f Heart Rate</b>: {w["heart_rate"]} BPM &nbsp;|&nbsp; '
            f'<b>\U0001f463 Steps/Day</b>: {w["steps_per_day"]:,} &nbsp;|&nbsp; '
            f'<b>\U0001f525 Calories</b>: {w["calories_burned"]:,} kcal &nbsp;|&nbsp; '
            f'<b>\U0001f4a4 Sleep</b>: {w["sleep_duration"]} hrs &nbsp;|&nbsp; '
            f'<b>\U0001f3af Activity</b>: {act_label}'
            f'</div>',
            unsafe_allow_html=True,
        )

        with st.expander("\u270f\ufe0f Override wearable values manually"):
            oc1, oc2, oc3 = st.columns(3)
            w["heart_rate"]     = oc1.number_input("Heart Rate (BPM)", 50, 130,
                                                    int(w["heart_rate"]), key="ov_hr")
            w["steps_per_day"]  = oc2.number_input("Steps/Day",  1000, 25000,
                                                    int(w["steps_per_day"]), step=500, key="ov_steps")
            w["calories_burned"]= oc3.number_input("Calories/Day", 1200, 5000,
                                                    int(w["calories_burned"]), step=100, key="ov_cal")
            oc4, oc5 = st.columns(2)
            w["sleep_duration"] = oc4.slider("Sleep Hours", 3.0, 11.0,
                                             float(w["sleep_duration"]), 0.5, key="ov_sleep")
            ov_act = oc5.selectbox("Activity Level",
                                   ["Sedentary", "Lightly Active", "Active", "Very Active"],
                                   index=w.get("activity_level", 2), key="ov_act")
            w["activity_level"] = ACTIVITY_MAP[ov_act]
            st.session_state["wearable_data"] = w

    st.markdown("---")


def page_predict(inj_model, inj_feats, sev_model, sev_feats):
    from injury_prediction_model import predict_single as inj_predict
    from severity_model import predict_single as sev_predict
    from recommendation_engine import recommend_for_input
    from feature_engineering import (
        add_workload_index, add_fatigue_index, add_recovery_score, add_risk_score
    )

    st.markdown("# Predict Injury Risk")
    st.markdown("Fill in the athlete's details and click **Predict Injury Risk** for an instant assessment.")

    # -- Wearable panel (outside form) -----------------------------------------
    _handle_google_fit_oauth()

    # -- Defaults from wearable or manual ----------------------------------------
    w = st.session_state.get("wearable_data", {})
    default_hr    = int(w.get("heart_rate",     72))
    default_steps = int(w.get("steps_per_day",  8000))
    default_cal   = int(w.get("calories_burned", 2400))
    default_sleep = float(w.get("sleep_duration", 7.0))
    default_act   = w.get("activity_level", 2)
    act_labels    = ["Sedentary", "Lightly Active", "Active", "Very Active"]

    with st.form("prediction_form"):
        # Optional athlete name
        athlete_name = st.text_input("Athlete Name (optional)",
                                     placeholder="e.g. John Smith",
                                     help="Used to identify the record on the Athlete Records page")

        st.markdown("### Basic Information")
        r1c1, r1c2, r1c3 = st.columns(3)
        age      = r1c1.number_input("Age (years)", 14, 45, 24)
        gender   = r1c2.selectbox("Gender", ["Male", "Female"])
        position = r1c3.selectbox("Sport / Position", POSITIONS, index=3)

        st.markdown("### Performance")
        r2c1, r2c2, r2c3 = st.columns(3)
        training_hours = r2c1.number_input("Training Hours/Week", 1.0, 50.0, 15.0, step=0.5)
        match_freq     = r2c2.number_input("Matches / Sessions per Month", 1, 20, 4)
        training_int   = r2c3.selectbox("Training Intensity", ["Low", "Medium", "High"], index=1)

        st.markdown("### Physical Condition")
        # Sport-specific fatigue default
        _fatigue_default = float(POSITION_FATIGUE_DEFAULTS.get(position, 5.0))
        _fatigue_guide   = {
            1: "Very fresh — minimal training",
            2: "Light activity",
            3: "Moderate activity",
            4: "Noticeable tiredness",
            5: "Moderate fatigue (typical training day)",
            6: "High training load",
            7: "Very high load — near peak",
            8: "Heavy fatigue — competition week",
            9: "Extreme fatigue — overtraining risk",
            10: "Total exhaustion",
        }
        r3c1, r3c2 = st.columns(2)
        fatigue_score = r3c1.slider(
            f"Fatigue Score (1–10)  ·  recommended for {position}: **{_fatigue_default:.0f}**",
            1.0, 10.0, _fatigue_default, 0.5,
            help="How tired / fatigued does the athlete feel right now?\n\n"
                 + "\n".join(f"**{k}** — {v}" for k, v in _fatigue_guide.items()),
        )
        r3c1.caption(f"ℹ {_fatigue_guide[max(1, min(10, int(round(fatigue_score))))]}")
        previous_inj  = r3c2.selectbox("Previous Injury?", ["No", "Yes"])

        st.markdown("### Recovery & Lifestyle")
        r4c1, r4c2, r4c3 = st.columns(3)
        sleep_dur     = r4c1.slider("Sleep Hours/Night", 3.0, 11.0, default_sleep, 0.5)
        recovery_time = r4c2.slider("Recovery Days/Week", 0.5, 7.0, 2.0, 0.5)
        hydration     = r4c3.slider("Hydration (L/day)",  1.0, 4.0, 2.5, 0.1)

        # -- Physiological -- shown only if NOT auto-filled by wearable ----------
        if not w:
            st.markdown("### Physiological (Manual)")
            r5c1, r5c2, r5c3 = st.columns(3)
            heart_rate    = r5c1.number_input("Heart Rate (BPM)", 50, 130, default_hr)
            steps_per_day = r5c2.number_input("Steps/Day", 1000, 25000, default_steps, step=500)
            calories      = r5c3.number_input("Calories Burned/Day", 1200, 5000, default_cal, step=100)
            activity_lv   = ACTIVITY_MAP[st.selectbox(
                "Activity Level", act_labels, index=default_act
            )]
        else:
            # Values already set by wearable panel; read from session state
            heart_rate    = default_hr
            steps_per_day = default_steps
            calories      = default_cal
            activity_lv   = default_act

        submitted = st.form_submit_button("Predict Injury Risk", use_container_width=True)

    # Store last fatigue so the simulation button can use it next click
    st.session_state["_last_fatigue"] = fatigue_score

    if not submitted:
        st.info("Complete the form above and click **Predict Injury Risk** to see results.")
        return

    # -- Re-read wearable values (may have been overridden in expander) ---------
    w = st.session_state.get("wearable_data", {})
    if w:
        heart_rate    = int(w.get("heart_rate",     heart_rate))
        steps_per_day = int(w.get("steps_per_day",  steps_per_day))
        calories      = int(w.get("calories_burned", calories))
        sleep_dur     = float(w.get("sleep_duration", sleep_dur))
        activity_lv   = int(w.get("activity_level",  activity_lv))

    raw = pd.DataFrame([{
        "age":               age,
        "gender":            1 if gender == "Male" else 0,
        "position":          POSITIONS.index(position),
        "previous_injury":   1 if previous_inj == "Yes" else 0,
        "training_intensity": INTENSITY_MAP[training_int],
        "match_frequency":   match_freq,
        "training_hours":    training_hours,
        "fatigue_score":     fatigue_score,
        "recovery_time":     recovery_time,
        "activity_level":    activity_lv,
        "sleep_duration":    sleep_dur,
        "heart_rate":        heart_rate,
        "steps_per_day":     steps_per_day,
        "calories_burned":   calories,
        "hydration_level":   hydration,
    }])

    raw = add_workload_index(raw)
    raw = add_fatigue_index(raw)
    raw = add_recovery_score(raw)
    raw = add_risk_score(raw)

    input_dict = raw.iloc[0].to_dict()

    inj_pred, inj_prob   = inj_predict(input_dict, inj_model, inj_feats)
    sev_label, sev_proba = sev_predict(input_dict, sev_model, sev_feats)

    input_dict["injury_indicator"] = inj_pred
    input_dict["injury_severity"]  = sev_label
    rec = recommend_for_input(input_dict)

    risk_level = rec.get("risk_level", "low")
    risk_text  = {"low": "LOW RISK", "medium": "MEDIUM RISK", "high": "HIGH RISK"}[risk_level]
    risk_class = f"risk-{risk_level}"

    # ── Persist the record ────────────────────────────────────────────────────
    save_athlete_record(
        name=athlete_name, gender=gender, position=position,
        age=age, training_hours=training_hours, match_freq=match_freq,
        training_int=training_int, fatigue_score=fatigue_score,
        previous_inj=previous_inj, sleep_dur=sleep_dur,
        recovery_time=recovery_time, hydration=hydration,
        heart_rate=heart_rate, steps_per_day=steps_per_day,
        calories=calories, activity_lv=activity_lv,
        input_dict=input_dict, inj_prob=inj_prob,
        risk_level=risk_level, sev_label=sev_label, rec=rec,
    )

    st.markdown("---")
    st.markdown("## Assessment Results")

    rc1, rc2, rc3 = st.columns([1.2, 1.2, 1.6])

    with rc1:
        st.markdown("#### Injury Risk")
        st.markdown(f'<span class="risk-badge {risk_class}">{risk_text}</span>',
                    unsafe_allow_html=True)
        st.markdown(f"**Probability:** `{inj_prob * 100:.1f}%`")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(inj_prob * 100, 1),
            number={"suffix": "%", "font": {"color": "#e6edf3", "size": 28}},
            gauge={
                "axis":  {"range": [0, 100], "tickcolor": "#8b949e"},
                "bar":   {"color": ("#ff7b72" if risk_level == "high" else
                                    "#e3b341" if risk_level == "medium" else "#56d364")},
                "bgcolor": "#21262d",
                "steps": [
                    {"range": [0, 40],  "color": "#0d1117"},
                    {"range": [40, 70], "color": "#0d1117"},
                    {"range": [70, 100],"color": "#0d1117"},
                ],
                "threshold": {"line": {"color": "white", "width": 3},
                              "thickness": 0.85, "value": inj_prob * 100},
            },
        ))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e6edf3",
                          height=220, margin=dict(l=20, r=20, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with rc2:
        st.markdown("#### Injury Severity")
        sev_color = SEVERITY_COLOR.get(sev_label, "#8b949e")
        st.markdown(
            f'<span class="risk-badge" style="background:#21262d;color:{sev_color};'
            f'border:2px solid {sev_color}">{sev_label.upper()}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(f"**Rest Recommended:** `{rec['rest_days']}`")

        sev_fig = go.Figure(go.Bar(
            x=SEVERITY_ORDER,
            y=[p * 100 for p in sev_proba],
            marker_color=[SEVERITY_COLOR[s] for s in SEVERITY_ORDER],
            text=[f"{p*100:.1f}%" for p in sev_proba],
            textposition="outside",
        ))
        sev_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e6edf3", showlegend=False, height=220,
            margin=dict(l=10, r=10, t=10, b=30),
            xaxis=dict(gridcolor="#30363d"), yaxis=dict(gridcolor="#30363d", range=[0, 115]),
        )
        st.plotly_chart(sev_fig, use_container_width=True)

    with rc3:
        st.markdown("#### Computed Risk Factors")
        src_label = st.session_state.get("wearable_source", "\u26aa Manual Entry")
        st.markdown(f"**Data Source:** {src_label}")
        st.markdown(f"""
        | Metric | Value |
        |--------|-------|
        | Workload Index  | `{input_dict['workload_index']:.2f}` |
        | Fatigue Index   | `{input_dict['fatigue_index']:.4f}` |
        | Recovery Score  | `{input_dict['recovery_score']:.4f}` |
        | Composite Risk  | `{input_dict['risk_score']:.4f}` |
        | Heart Rate      | `{heart_rate} BPM` |
        | Steps/Day       | `{steps_per_day:,}` |
        | Calories        | `{calories:,} kcal` |
        | Sleep           | `{sleep_dur} hrs` |
        | Previous Injury | `{'Yes' if input_dict['previous_injury'] else 'No'}` |
        """)

    st.markdown("---")
    st.markdown("## Personalised Recovery Plan")
    pos_name = rec.get("position_name", position)
    st.markdown(f"*Recommendations for **{pos_name}** \u00b7 {gender} \u00b7 Age {age}*")

    col_av, col_tips = st.columns([1, 2])

    with col_av:
        if rec["avoid"]:
            st.markdown("### Activities to Avoid")
            for item in rec["avoid"]:
                st.markdown(f'<div class="avoid-item">{item}</div>', unsafe_allow_html=True)

    with col_tips:
        st.markdown("### Recovery Tips")
        for tip in rec["tips"]:
            st.markdown(f'<div class="tip-item">{tip}</div>', unsafe_allow_html=True)



# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ATHLETE RECORDS
# ═══════════════════════════════════════════════════════════════════════════════
def page_records():
    st.markdown("# Athlete Records")
    st.markdown("All predictions submitted via the Predict page, stored persistently.")
    st.markdown("---")

    if not os.path.exists(RECORDS_PATH):
        st.info("No records yet. Go to ** Predict**, fill in an athlete's details, "
                "and click **Predict Injury Risk** — the entry will appear here.")
        return

    df_r = pd.read_csv(RECORDS_PATH)
    if st.session_state.get("role") == "athlete":
        df_r = df_r[df_r["name"].str.lower() == st.session_state.get("username", "")]
        
    if df_r.empty:
        st.info("No records found.")
        return

    # ── KPI strip ─────────────────────────────────────────────────────────────
    total     = len(df_r)
    high_risk = int((df_r["risk_level"] == "high").sum())
    avg_prob  = df_r["injury_probability"].mean()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Records",       f"{total}")
    k2.metric("High-Risk Athletes",   f"{high_risk}",
              delta=f"{high_risk/total*100:.0f}% of total",
              delta_color="inverse")
    k3.metric("Avg Injury Probability", f"{avg_prob:.1f}%")
    k4.metric("Positions Covered",   f"{df_r['position'].nunique()}")

    st.markdown("---")

    # ── Filters ───────────────────────────────────────────────────────────────
    fc1, fc2, fc3 = st.columns(3)
    risk_filter = fc1.multiselect("Filter by Risk Level",
                                  ["low", "medium", "high"],
                                  default=["low", "medium", "high"])
    pos_opts    = ["All"] + sorted(df_r["position"].unique().tolist())
    pos_filter  = fc2.selectbox("Filter by Position", pos_opts)
    sev_filter  = fc3.multiselect("Filter by Severity",
                                  ["mild", "moderate", "severe"],
                                  default=["mild", "moderate", "severe"])

    mask = (df_r["risk_level"].isin(risk_filter)) & \
           (df_r["injury_severity"].isin(sev_filter))
    if pos_filter != "All":
        mask &= df_r["position"] == pos_filter
    df_filtered = df_r[mask].reset_index(drop=True)

    st.markdown(f"**{len(df_filtered)}** record(s) matching filters.")

    # ── Risk distribution chart ────────────────────────────────────────────────
    ch1, ch2 = st.columns(2)
    with ch1:
        st.markdown("#### Risk Level Distribution")
        rc = df_filtered["risk_level"].value_counts().reset_index()
        rc.columns = ["Risk", "Count"]
        color_map = {"low": "#56d364", "medium": "#e3b341", "high": "#ff7b72"}
        fig_pie = px.pie(rc, values="Count", names="Risk",
                         color="Risk", color_discrete_map=color_map, hole=0.45)
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                              font_color="#e6edf3",
                              legend=dict(font=dict(color="#e6edf3")),
                              margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_pie, use_container_width=True)

    with ch2:
        st.markdown("#### Injury Probability by Position")
        if len(df_filtered) > 0:
            pos_agg = (df_filtered.groupby("position")["injury_probability"]
                       .mean().reset_index()
                       .sort_values("injury_probability", ascending=True))
            fig_bar = px.bar(pos_agg, x="injury_probability", y="position",
                             orientation="h",
                             color="injury_probability",
                             color_continuous_scale=["#56d364", "#e3b341", "#ff7b72"],
                             text=pos_agg["injury_probability"].apply(lambda v: f"{v:.1f}%"))
            fig_bar.update_traces(textposition="outside")
            fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)",
                                  font_color="#e6edf3", showlegend=False,
                                  margin=dict(l=0, r=60, t=10, b=0),
                                  xaxis=dict(gridcolor="#30363d"),
                                  yaxis=dict(gridcolor="#30363d"))
            st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # ── Summary table ─────────────────────────────────────────────────────────
    st.markdown("#### All Records")
    display_cols = ["timestamp", "name", "age", "gender", "position",
                    "fatigue_score", "injury_probability", "risk_level",
                    "injury_severity", "rest_days"]
    display_cols = [c for c in display_cols if c in df_filtered.columns]

    def _color_risk(val):
        colors = {"high": "color:#ff7b72;font-weight:700",
                  "medium": "color:#e3b341;font-weight:700",
                  "low": "color:#56d364;font-weight:700"}
        return colors.get(str(val).lower(), "")

    styled = (df_filtered[display_cols]
              .style
              .map(_color_risk, subset=["risk_level"])
              .format({"injury_probability": "{:.1f}%",
                       "fatigue_score": "{:.1f}"}))
    st.dataframe(styled, use_container_width=True, height=320)

    # ── Download button ────────────────────────────────────────────────────────
    csv_bytes = df_filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ Download Records as CSV",
        data=csv_bytes,
        file_name="athlete_records.csv",
        mime="text/csv",
    )

    st.markdown("---")

    # ── Individual record cards ────────────────────────────────────────────────
    st.markdown("#### Detailed Record View")
    for idx, row in df_filtered.iterrows():
        rl = row["risk_level"]
        icon = {"high": "", "medium": "", "low": ""}.get(rl, "")
        label = (f"{icon} #{idx+1} · "
                 f"{row.get('name','—')} · "
                 f"{row.get('position','?')} · "
                 f"Age {row.get('age','?')} · "
                 f"{row.get('injury_probability','?')}% risk · "
                 f"{row.get('timestamp','')[:16]}")
        with st.expander(label):
            d1, d2, d3 = st.columns(3)
            d1.metric("Injury Probability", f"{row['injury_probability']:.1f}%")
            d2.metric("Severity",           row["injury_severity"].capitalize())
            d3.metric("Rest Days",          row["rest_days"])

            st.markdown(f"""
            | Field | Value |
            |-------|-------|
            | Training Hours/wk | `{row['training_hours']}` |
            | Training Intensity | `{row['training_intensity']}` |
            | Fatigue Score | `{row['fatigue_score']}` |
            | Sleep Hours | `{row['sleep_duration']} hrs` |
            | Heart Rate | `{row['heart_rate']} BPM` |
            | Steps/Day | `{int(row['steps_per_day']):,}` |
            | Calories | `{int(row['calories_burned']):,} kcal` |
            | Previous Injury | `{row['previous_injury']}` |
            | Workload Index | `{row['workload_index']:.4f}` |
            | Fatigue Index | `{row['fatigue_index']:.4f}` |
            | Recovery Score | `{row['recovery_score']:.4f}` |
            """)

            tips  = [t.strip() for t in str(row.get("tips",  "")).split("|") if t.strip()]
            avoid = [a.strip() for a in str(row.get("avoid", "")).split("|") if a.strip()]

            if tips:
                st.markdown("** Recovery Tips**")
                for t in tips:
                    st.markdown(f'<div class="tip-item"> {t}</div>',
                                unsafe_allow_html=True)
            if avoid:
                st.markdown("** Activities to Avoid**")
                for a in avoid:
                    st.markdown(f'<div class="avoid-item"> {a}</div>',
                                unsafe_allow_html=True)

            # Delete button
            if st.button(f"Delete this record", key=f"del_{idx}"):
                df_r = df_r.drop(index=idx).reset_index(drop=True)
                df_r.to_csv(RECORDS_PATH, index=False)
                st.success("Record deleted.")
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — DATA EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
def page_data_explorer(df: pd.DataFrame):
    st.markdown("# Data Explorer")
    st.markdown("Visualise the training dataset to understand distributions and relationships.")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Training Hours Distribution")
        fig = px.histogram(df, x="training_hours", nbins=30, color_discrete_sequence=["#58a6ff"])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#e6edf3", margin=dict(l=0, r=0, t=10, b=0),
                          xaxis=dict(gridcolor="#30363d"), yaxis=dict(gridcolor="#30363d"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Fatigue Score Distribution")
        fig2 = px.histogram(df, x="fatigue_score", nbins=25, color_discrete_sequence=["#e3b341"])
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#e6edf3", margin=dict(l=0, r=0, t=10, b=0),
                           xaxis=dict(gridcolor="#30363d"), yaxis=dict(gridcolor="#30363d"))
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### Injury Rate by Training Intensity")
        int_map_rev = {0: "Low", 1: "Medium", 2: "High"}
        tmp = df.copy()
        tmp["intensity_label"] = tmp["training_intensity"].map(int_map_rev)
        agg = tmp.groupby("intensity_label")["injury_indicator"].mean().reset_index()
        agg.columns = ["Intensity", "Injury Rate"]
        agg["Intensity"] = pd.Categorical(agg["Intensity"], ["Low", "Medium", "High"])
        agg = agg.sort_values("Intensity")
        fig3 = px.bar(agg, x="Intensity", y="Injury Rate",
                      color="Injury Rate", color_continuous_scale=["#56d364", "#e3b341", "#ff7b72"],
                      text=agg["Injury Rate"].apply(lambda x: f"{x*100:.1f}%"))
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#e6edf3", showlegend=False, margin=dict(l=0, r=0, t=10, b=0),
                           xaxis=dict(gridcolor="#30363d"), yaxis=dict(gridcolor="#30363d"))
        fig3.update_traces(textposition="outside")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("#### Sleep Duration vs Injury Risk")
        fig4 = px.box(df, x=df["injury_indicator"].map({0: "No Injury", 1: "Injury"}),
                      y="sleep_duration",
                      color=df["injury_indicator"].map({0: "No Injury", 1: "Injury"}),
                      color_discrete_map={"No Injury": "#56d364", "Injury": "#ff7b72"})
        fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#e6edf3", showlegend=False, margin=dict(l=0, r=0, t=10, b=0),
                           xaxis=dict(gridcolor="#30363d"), yaxis=dict(gridcolor="#30363d"),
                           xaxis_title="", yaxis_title="Sleep Duration (hrs)")
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("#### Correlation Heatmap — Numerical Features")
    num_cols = ["training_hours", "fatigue_score", "recovery_time", "sleep_duration",
                "heart_rate", "steps_per_day", "workload_index", "fatigue_index",
                "recovery_score", "risk_score", "injury_indicator"]
    corr = df[[c for c in num_cols if c in df.columns]].corr()
    fig5 = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                     text_auto=".2f", aspect="auto")
    fig5.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e6edf3",
                       margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig5, use_container_width=True)

    with st.expander("Raw Dataset Preview (first 50 rows)"):
        display_cols = [c for c in ["training_hours", "fatigue_score", "recovery_time",
                                     "sleep_duration", "heart_rate", "workload_index",
                                     "fatigue_index", "recovery_score", "risk_score",
                                     "injury_indicator", "injury_severity"] if c in df.columns]
        st.dataframe(df[display_cols].head(50), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
def page_model_performance():
    st.markdown("# Model Performance")
    st.markdown("Evaluation metrics for the trained Random Forest models.")
    st.markdown("---")

    inj_meta, sev_meta = get_model_meta()

    if not inj_meta:
        st.warning("Models not yet trained. Visit **Home** page to trigger training.")
        return

    col1, col2 = st.columns(2)
    col1.metric("Injury Model Accuracy", f"{inj_meta.get('accuracy', 0)*100:.2f}%")
    col1.metric("Injury Model ROC-AUC",  f"{inj_meta.get('auc', 0):.4f}")
    col2.metric("Severity Model Accuracy", f"{sev_meta.get('accuracy', 0)*100:.2f}%")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    # Feature importance — Injury model
    with col_a:
        st.markdown("#### Injury Model — Feature Importances")
        imp = inj_meta.get("importances", {})
        if imp:
            imp_df = pd.DataFrame(list(imp.items()), columns=["Feature", "Importance"])
            imp_df = imp_df.sort_values("Importance", ascending=True)
            fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                         color="Importance", color_continuous_scale="blues", text="Importance")
            fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font_color="#e6edf3", showlegend=False, height=420,
                              margin=dict(l=0, r=40, t=10, b=0),
                              xaxis=dict(gridcolor="#30363d"), yaxis=dict(gridcolor="#30363d"))
            st.plotly_chart(fig, use_container_width=True)

    # Feature importance — Severity model
    with col_b:
        st.markdown("#### Severity Model — Feature Importances")
        imp2 = sev_meta.get("importances", {})
        if imp2:
            imp_df2 = pd.DataFrame(list(imp2.items()), columns=["Feature", "Importance"])
            imp_df2 = imp_df2.sort_values("Importance", ascending=True)
            fig2 = px.bar(imp_df2, x="Importance", y="Feature", orientation="h",
                          color="Importance", color_continuous_scale="reds", text="Importance")
            fig2.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#e6edf3", showlegend=False, height=420,
                               margin=dict(l=0, r=40, t=10, b=0),
                               xaxis=dict(gridcolor="#30363d"), yaxis=dict(gridcolor="#30363d"))
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Model Architecture Summary")
    st.markdown("""
 | Component | Algorithm | Technique | Features |
 |-----------|-----------|-----------|----------|
 | Injury Prediction | Random Forest (300 trees) | SMOTE + 5-fold CV | 19 features |
 | Severity Classification | Random Forest (300 trees) | SMOTE + 5-fold CV | 20 features |
 | Baseline Comparison | Logistic Regression | SMOTE | 19 features |
 | Recommendation | Rule-based Engine | Threshold rules + position logic | — |
 """)
    st.markdown("""
 **Evaluation Metrics Used**
 - **Accuracy** — Overall correct predictions
 - **ROC-AUC** — Discrimination ability across thresholds (injury model)
 - **Precision / Recall / F1** — Per-class performance (severity model)
 - **Confusion Matrix** — Visual breakdown of true vs predicted labels
 """)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — LIVE DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
import time
import random

def page_live_dashboard(inj_model, inj_feats):
    from injury_prediction_model import predict_single as inj_predict
    
    st.markdown("# 🚀 Live Performance Dashboard")
    st.markdown("Real-time monitoring of athlete telemetry, risk scores, and performance trends.")
    st.markdown("---")
    
    # ── Role Selection ─────────────
    role = st.session_state.get("role", "athlete")
    if role == "coach":
        athlete_opts = ["alex", "jordan", "casey", "riley", "taylor", "sam"]
        selected_athlete = st.selectbox("🎯 Select Athlete to Monitor", athlete_opts, index=0)
    else:
        selected_athlete = st.session_state.get("username", "alex")
        st.markdown(f"**Welcome back, {selected_athlete.title()}!** Here is your personal dashboard.")

    # Load from CSV
    try:
        df_hist = pd.read_csv(RECORDS_PATH)
        df_hist = df_hist[df_hist["name"].str.lower() == selected_athlete.lower()].tail(7)
        if df_hist.empty:
            raise ValueError
        days = df_hist["timestamp"].apply(lambda x: x.split(" ")[0][-5:]).tolist()  # MM-DD
        workload = df_hist["workload_index"].tolist()
        fatigue = df_hist["fatigue_score"].tolist()
        sleep = df_hist["sleep_duration"].tolist()
        recovery = df_hist["recovery_score"].tolist()
    except:
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        workload = [5.2, 5.8, 6.4, 4.5, 7.1, 7.8, 3.5]
        fatigue = [3.0, 3.5, 4.2, 3.8, 5.0, 6.5, 2.5]
        sleep = [7.5, 7.0, 6.5, 8.0, 7.2, 6.0, 8.5]
        recovery = [80, 75, 68, 85, 78, 60, 90]

    st.markdown("### 📊 Weekly Performance & Recovery Trends")
    st.markdown("Aggregated historical data over the last few days.")
    trend_col1, trend_col2 = st.columns(2)
    
    with trend_col1:
        df_trends = pd.DataFrame({"Day": days, "Workload": workload, "Fatigue": fatigue})
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=df_trends["Day"], y=df_trends["Workload"], mode='lines+markers', name='Workload', line=dict(color='#58a6ff')))
        fig_trend.add_trace(go.Bar(x=df_trends["Day"], y=df_trends["Fatigue"], name='Fatigue', marker_color='#e3b341', opacity=0.6))
        fig_trend.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e6edf3", margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=250
        )
        st.plotly_chart(fig_trend, use_container_width=True)
        
    with trend_col2:
        df_sleep = pd.DataFrame({"Day": days, "Sleep (hrs)": sleep, "Recovery Score": recovery})
        
        fig_sleep = make_subplots(specs=[[{"secondary_y": True}]])
        fig_sleep.add_trace(go.Bar(x=df_sleep["Day"], y=df_sleep["Sleep (hrs)"], name='Sleep (hrs)', marker_color='#8b949e'), secondary_y=False)
        fig_sleep.add_trace(go.Scatter(x=df_sleep["Day"], y=df_sleep["Recovery Score"], mode='lines+markers', name='Recovery Score', line=dict(color='#56d364')), secondary_y=True)
        fig_sleep.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e6edf3", margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=250
        )
        fig_sleep.update_yaxes(title_text="Sleep", secondary_y=False)
        fig_sleep.update_yaxes(title_text="Recovery", secondary_y=True, range=[0, 100])
        st.plotly_chart(fig_sleep, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 🔴 REAL-TIME SYSTEMS")
    
    # ── Live Telemetry Simulator ─────────────
    # We initialize session state for live values if not exists
    if "live_hr" not in st.session_state:
        st.session_state.live_hr = 70
        st.session_state.live_fatigue = 4.0
        st.session_state.live_workload = 5.0
        st.session_state.sim_active = False

    def toggle_sim():
        st.session_state.sim_active = not st.session_state.sim_active

    st.button("Toggle WebSocket Stream Simulator", on_click=toggle_sim)
    
    # Placeholders for live updates
    metric_cols = st.columns(4)
    m_hr = metric_cols[0].empty()
    m_fat = metric_cols[1].empty()
    m_work = metric_cols[2].empty()
    m_risk = metric_cols[3].empty()
    
    alert_placeholder = st.empty()
    
    # Base athlete dict to feed into model
    input_dict = {
        "age": 24, "gender": 1, "position": 3, "previous_injury": 0,
        "training_intensity": 2, "match_frequency": 4, "training_hours": 15.0,
        "recovery_time": 2.0, "activity_level": 2, "sleep_duration": 7.0,
        "steps_per_day": 8000, "calories_burned": 2500, "hydration_level": 2.5,
        "recovery_score": 0.5, "risk_score": 0.0, "workload_index": 0.0, "fatigue_index": 0.0
    }
    
    if st.session_state.sim_active:
        # Streamlit while loop for real-time simulation
        # It runs continuously updating placeholders
        for _ in range(50):
            if not st.session_state.sim_active:
                break
                
            # Random walk for telemetry
            st.session_state.live_hr += random.randint(-3, 4)
            st.session_state.live_hr = max(60, min(180, st.session_state.live_hr))
            
            st.session_state.live_fatigue += random.uniform(-0.1, 0.2)
            st.session_state.live_fatigue = max(1.0, min(10.0, st.session_state.live_fatigue))
            
            st.session_state.live_workload += random.uniform(-0.2, 0.3)
            st.session_state.live_workload = max(1.0, min(15.0, st.session_state.live_workload))
            
            input_dict["heart_rate"] = st.session_state.live_hr
            input_dict["fatigue_score"] = st.session_state.live_fatigue
            input_dict["workload_index"] = st.session_state.live_workload
            
            # Predict
            pred, proba = inj_predict(input_dict, inj_model, inj_feats)
            risk_pct = proba * 100
            
            # Risk Level
            if risk_pct > 70:
                risk_color = "#ff7b72"
                risk_label = "HIGH RISK"
            elif risk_pct > 40:
                risk_color = "#e3b341"
                risk_label = "MEDIUM RISK"
            else:
                risk_color = "#56d364"
                risk_label = "LOW RISK"
                
            # Update metrics
            m_hr.metric("Live Heart Rate", f"{st.session_state.live_hr} BPM")
            m_fat.metric("Live Fatigue", f"{st.session_state.live_fatigue:.1f}/10")
            m_work.metric("Live Workload", f"{st.session_state.live_workload:.1f}")
            
            # Custom styled metric for Risk
            m_risk.markdown(f'''
                <div data-testid="metric-container" style="border-color:{risk_color};">
                <label>AI Injury Risk</label>
                <div data-testid="metric-value" style="color:{risk_color}!important;">{risk_pct:.1f}%</div>
                <div style="font-size:0.8rem;color:{risk_color};font-weight:bold;">{risk_label}</div>
                </div>
            ''', unsafe_allow_html=True)
            
            # Alert Logic
            if risk_pct > 70:
                alert_placeholder.error("🚨 **EMERGENCY TRIGGERED (Push Alert):** Injury risk critical. **SMART RECOMMENDATION: Stop training immediately!**")
            elif risk_pct > 40:
                alert_placeholder.warning("⚠️ **SMART RECOMMENDATION:** Reduce intensity. Heart rate & fatigue indicate elevated risk.")
            else:
                alert_placeholder.success("✅ Athlete operating in safe optimal zone.", icon="ℹ️")
                
            time.sleep(1.5)
            # st.rerun() at the end to keep it smooth if we used st.rerun instead of placeholders, 
            # but placeholders are better to avoid flickering in the page.
    else:
        # Display static state when not running
        m_hr.metric("Live Heart Rate", f"{st.session_state.live_hr} BPM")
        m_fat.metric("Live Fatigue", f"{st.session_state.live_fatigue:.1f}/10")
        m_work.metric("Live Workload", f"{st.session_state.live_workload:.1f}")
        m_risk.metric("AI Injury Risk", "-- %")
        alert_placeholder.info("Click 'Toggle WebSocket Stream Simulator' to begin live tracking.")

# ═══════════════════════════════════════════════════════════════════════════════
# LOGIN PORTAL
# ═══════════════════════════════════════════════════════════════════════════════
def page_login():
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: #58a6ff;'>AthleteGuard Portal</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #c9d1d9;'>Please sign in to access live dashboards and wearables.</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown('<div style="background:#161b22; padding:30px; border-radius:12px; border:1px solid #30363d;">', unsafe_allow_html=True)
        with st.form("login_form"):
            login_role = st.radio("Login As:", ["Athlete", "Coach"], horizontal=True)
            username = st.text_input("Username / Athlete ID", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Sign In", use_container_width=True)
            
            if submitted:
                if login_role == "Coach":
                    if username.strip().lower() == "coach":
                        st.session_state["logged_in"] = True
                        st.session_state["role"] = "coach"
                        st.session_state["username"] = "coach"
                        st.rerun()
                    else:
                        st.error("Invalid coach credentials. Use username 'coach'.")
                else:
                    if username.strip().lower() in ["alex", "jordan", "casey", "riley", "taylor", "sam"]:
                        st.session_state["logged_in"] = True
                        st.session_state["role"] = "athlete"
                        st.session_state["username"] = username.strip().lower()
                        st.rerun()
                    else:
                        st.error("Invalid athlete. Demo accounts: 'alex', 'jordan', 'casey', 'riley', 'taylor', 'sam'.")
        st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    # Gate the entire app
    if not st.session_state.get("logged_in", False):
        page_login()
        return

    page = sidebar()

    if page in ["Athlete Records", "My Records"]:
        # Records page does not need the ML models loaded
        page_records()
        return

    df, inj_model, inj_feats, sev_model, sev_le, sev_feats = load_or_train_models()

    if page == "Home":
        page_home(df)
    elif page == "Live Dashboard":
        page_live_dashboard(inj_model, inj_feats)
    elif page == "Predict":
        page_predict(inj_model, inj_feats, sev_model, sev_feats)
    elif page == "Data Explorer":
        page_data_explorer(df)
    elif page == "Model Performance":
        page_model_performance()


if __name__ == "__main__":
    main()
