import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_history():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    records_path = os.path.join(data_dir, "athlete_records.csv")

    athletes = [
        {"name": "alex", "age": 24, "gender": "Male", "position": "Basketball Guard", "risk_trend": "increasing", "injury_severity": "severe"},
        {"name": "jordan", "age": 22, "gender": "Female", "position": "Sprinter", "risk_trend": "increasing", "injury_severity": "mild"},
        {"name": "casey", "age": 25, "gender": "Male", "position": "Defender", "risk_trend": "stable", "injury_severity": "none"},
        {"name": "riley", "age": 21, "gender": "Female", "position": "Swimmer", "risk_trend": "stable", "injury_severity": "none"},
        {"name": "taylor", "age": 23, "gender": "Male", "position": "Forward", "risk_trend": "decreasing", "injury_severity": "none"},
        {"name": "sam", "age": 26, "gender": "Female", "position": "Tennis Player", "risk_trend": "increasing", "injury_severity": "severe"},
    ]

    records = []
    base_date = datetime.now() - timedelta(days=5)

    for athlete in athletes:
        name = athlete["name"]
        trend = athlete["risk_trend"]
        sev = athlete["injury_severity"]
        
        # Base stats
        workload = 5.0
        fatigue = 4.0
        recovery = 80.0
        sleep = 8.0
        risk_pct = 10.0

        for day in range(1, 6):
            timestamp_str = (base_date + timedelta(days=day)).strftime("%Y-%m-%d %H:%M:%S")
            
            # Evolve stats based on trend
            if trend == "increasing":
                workload += np.random.uniform(0.5, 1.5)
                fatigue += np.random.uniform(0.5, 1.2)
                recovery -= np.random.uniform(5, 12)
                sleep -= np.random.uniform(0.2, 0.8)
                risk_pct += np.random.uniform(10, 20)
            elif trend == "decreasing":
                workload -= np.random.uniform(0.2, 0.8)
                fatigue -= np.random.uniform(0.2, 0.8)
                recovery += np.random.uniform(2, 6)
                sleep += np.random.uniform(0.1, 0.5)
                risk_pct -= np.random.uniform(2, 5)
            else:
                workload += np.random.uniform(-0.5, 0.5)
                fatigue += np.random.uniform(-0.5, 0.5)
                recovery += np.random.uniform(-5, 5)
                sleep += np.random.uniform(-0.5, 0.5)
                risk_pct += np.random.uniform(-5, 5)

            # Cap values
            workload = max(1.0, min(15.0, workload))
            fatigue = max(1.0, min(10.0, fatigue))
            recovery = max(0.0, min(100.0, recovery))
            sleep = max(4.0, min(12.0, sleep))
            risk_pct = max(0.0, min(99.0, risk_pct))

            # Define severity and label based on risk and athlete profile
            if day == 5 and sev != "none": # Max severity on day 5
                risk_level = "high"
                risk_pct = max(80.0, risk_pct)
                injury_severity = sev
                rest = "14 days" if sev == "severe" else "3 days"
            else:
                if risk_pct > 70:
                    risk_level = "high"
                    injury_severity = sev if day > 3 and sev != "none" else "mild"
                    rest = "7 days"
                elif risk_pct > 40:
                    risk_level = "medium"
                    injury_severity = "mild"
                    rest = "2 days"
                else:
                    risk_level = "low"
                    injury_severity = "none"
                    rest = "0 days"

            row = {
                "timestamp": timestamp_str,
                "name": name,
                "age": athlete["age"],
                "gender": athlete["gender"],
                "position": athlete["position"],
                "training_hours": round(workload * 2, 1),
                "match_frequency": 4,
                "training_intensity": "High" if workload > 8 else "Medium",
                "fatigue_score": round(fatigue, 1),
                "previous_injury": "Yes" if sev != "none" else "No",
                "sleep_duration": round(sleep, 1),
                "recovery_time": 2.0,
                "hydration_level": 2.5,
                "heart_rate": int(60 + fatigue * 5),
                "steps_per_day": int(8000 + workload * 500),
                "calories_burned": int(2000 + workload * 100),
                "activity_level": 2,
                "workload_index": round(workload, 4),
                "fatigue_index": round(fatigue / 10.0, 4),
                "recovery_score": round(recovery, 4),
                "risk_score": round(risk_pct / 100.0, 4),
                "injury_probability": round(risk_pct, 1),
                "risk_level": risk_level,
                "injury_severity": injury_severity,
                "rest_days": rest,
                "tips": "Stay hydrated | Monitor fatigue",
                "avoid": "High intensity sprints"
            }
            records.append(row)

    df_new = pd.DataFrame(records)
    
    # Overwrite the existing records csv completely for demo
    df_new.to_csv(records_path, index=False)
    print(f"Generated {len(df_new)} historical records at {records_path}")

if __name__ == "__main__":
    generate_history()
