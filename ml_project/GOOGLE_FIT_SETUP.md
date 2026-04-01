# Google Fit Setup Guide — AthleteGuard AI

This guide enables **live Google Fit data** in the Predict page. Without it, the app uses simulated Fitbit data (which works just as well for demonstration).

> **Note:** As of May 2024, Google Fit API is deprecated for *new* registrations.  
> If you already have an existing Google Cloud project with Fitness API enabled, follow the steps below. Otherwise, the simulated mode is your best option.

---

## Step 1 — Create or Open a Google Cloud Project

1. Go to [console.cloud.google.com](https://console.cloud.google.com/)
2. Click **Select a project** → **New Project** → give it a name (e.g., `AthleteGuardAI`)
3. Click **Create**

---

## Step 2 — Enable the Fitness API

1. In the left menu go to **APIs & Services** → **Library**
2. Search for **Fitness API**
3. Click **Enable**

---

## Step 3 — Configure OAuth Consent Screen

1. Go to **APIs & Services** → **OAuth consent screen**
2. Select **External** → **Create**
3. Fill in: App name (`AthleteGuard AI`), Support email, Developer contact
4. Click **Save and Continue** through all steps
5. Under **Scopes**, add:
   - `https://www.googleapis.com/auth/fitness.activity.read`
   - `https://www.googleapis.com/auth/fitness.body.read`
   - `https://www.googleapis.com/auth/fitness.sleep.read`
6. Add your Google account email as a **Test user**

---

## Step 4 — Create OAuth 2.0 Credentials

1. Go to **APIs & Services** → **Credentials**
2. Click **+ Create Credentials** → **OAuth client ID**
3. Application type: **Web application**
4. Name: `AthleteGuardAI-Local`
5. Under **Authorized redirect URIs**, add:
   ```
   http://localhost:8501
   ```
6. Click **Create**
7. Click **Download JSON** → save as `credentials.json`

---

## Step 5 — Place Credentials in Project

```
ml_project/
├── credentials.json   ← place here
├── app.py
├── google_fit.py
└── ...
```

---

## Step 6 — Install Dependencies

```powershell
cd c:\Users\sebar\OneDrive\Documents\claimsense\ml_project
pip install -r requirements.txt
```

---

## Step 7 — Run the App

```powershell
python -m streamlit run app.py
```

1. Navigate to the **🔍 Predict** page
2. Click **🔗 Connect Google Fit**
3. Complete the Google login in the browser
4. You'll be redirected back — the app auto-fills Heart Rate, Steps, Calories, Sleep, and Activity Level from your real data

---

## Without credentials.json (Simulation Mode)

If you skip this setup, click **📱 Use Simulated Wearable Data** on the Predict page. The app will sample realistic physiological values from the Fitbit dataset, scaled to your entered fatigue score — perfect for demonstration purposes.
