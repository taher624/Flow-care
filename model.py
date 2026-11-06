"""
Flow Care - AI layer
====================

Two kinds of AI, both tested honestly:

A) HOURLY - for each department, every hour:
   1. FORECAST  how full it will be 8 hours from now
   2. WARN      early when it is about to become full
   3. DETECT    steps running slower than normal for that hour of the week

B) DAILY SURGE PLANNING (Hajj, Ramadan, Eid) - 7 days ahead:
   forecasts ED visits, heat-illness cases and peak bed use for each coming day,
   so the hospital can plan beds and staff before the surge arrives.

Testing:
  - Learns from 2023-2025. Tested on 2026 up to today, which it has never seen,
    including Ramadan 2026 and Hajj 2026.
  - Every forecast is compared with a simple guess:
      hourly: "in 8 hours it will be as full as now"
      daily:  "next week's day will be the same as this week's"
  - To check that the AI really learned Hajj and Ramadan, the same models are
    also trained WITHOUT the Hajj/Ramadan/Eid calendar and compared.

ALL DATA IS SIMULATED. No real patient data is used.
Run:  python model.py      (after python simulate.py)
"""

import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

HORIZON = 8
TEST_START = "2026-01-01"
TEST_END = "2026-11-07"      # "today": results only use hours that have already happened
CAPACITY = {"ed": 40, "med": 120, "surg": 52, "icu": 16, "mat": 46, "ld": 10}
ARAFAH = [datetime(2023, 6, 27), datetime(2024, 6, 15), datetime(2025, 6, 5),
          datetime(2026, 5, 26), datetime(2027, 5, 15)]
RAMADAN_START = [datetime(2023, 3, 23), datetime(2024, 3, 11), datetime(2025, 3, 1),
                 datetime(2026, 2, 18), datetime(2027, 2, 8)]
SEASONAL = ["ramadan", "ramadan_day", "hajj_season", "hajj_peak", "eid", "days_to_arafah"]

# Hourly forecasts: name -> column
HOURLY_TARGETS = {
    "ed_total": "ed_total", "ed_boarders": "ed_boarders", "med": "med_occupied",
    "surg": "surg_occupied", "icu": "icu_occupied", "mat": "mat_occupied", "ld": "ld_occupied",
}
# Early warnings: name -> (column, threshold, description)
WARNINGS = {
    "med": ("med_occupied", 117, "Internal Medicine wards reach 97% (117 of 120 beds)"),
    "surg": ("surg_occupied", 50, "Surgical wards reach 96% (50 of 52 beds)"),
    "icu": ("icu_occupied", 15, "ICU reaches 15 of 16 beds"),
    "mat": ("mat_occupied", 44, "Maternity ward reaches 44 of 46 beds"),
    "ed": ("ed_boarders", 8, "8 or more admitted patients waiting in the ED"),
}


def calendar(ts):
    f = pd.DataFrame(index=ts.index)
    f["hour"] = ts.dt.hour
    f["weekday"] = ts.dt.weekday
    f["weekend"] = ts.dt.weekday.isin([4, 5]).astype(int)
    doy = ts.dt.dayofyear
    f["season_sin"] = np.sin(2 * np.pi * doy / 365)
    f["season_cos"] = np.cos(2 * np.pi * doy / 365)
    f["year"] = ts.dt.year - 2023
    # days until (or since, negative) this year's Day of Arafah
    arafah = ts.dt.year.map({a.year: a for a in ARAFAH})
    f["days_to_arafah"] = ((arafah - ts).dt.total_seconds() / 86400).clip(-60, 60)
    rs = ts.dt.year.map({r.year: r for r in RAMADAN_START})
    rd = (ts - rs).dt.total_seconds() / 86400
    f["ramadan_day"] = np.where((rd >= 0) & (rd < 30), rd, -1)
    return f


def hourly_features(df):
    f = calendar(df.timestamp)
    for c in ["ramadan", "hajj_season", "hajj_peak", "eid"]:
        f[c] = df[c]
    for c in ["med_occupied", "surg_occupied", "icu_occupied", "mat_occupied", "ld_occupied",
              "ld_blocked", "ob_triage_waiting", "ed_in_treatment", "ed_boarders",
              "boarders_med", "boarders_surg", "boarders_icu", "ed_longest_board_h",
              "med_ready_to_go", "surg_ready_to_go", "mat_ready_to_go",
              "icu_blocked_stepdown", "scan_queue"]:
        f[c] = df[c]
    for w in (3, 6, 24):
        f[f"ed_arrivals_{w}h"] = df.ed_arrivals.rolling(w, min_periods=1).sum()
    f["heat_24h"] = df.ed_heat_arrivals.rolling(24, min_periods=1).sum()
    f["births_6h"] = df.births.rolling(6, min_periods=1).sum()
    f["disch_6h"] = (df.discharges_med + df.discharges_surg).rolling(6, min_periods=1).sum()
    for c in ["med_occupied", "surg_occupied", "icu_occupied", "mat_occupied", "ed_boarders",
              "ed_total", "ld_occupied"]:
        f[f"{c}_chg4"] = df[c].diff(4)
        f[f"{c}_yday"] = df[c].shift(24 - HORIZON)
    hrs_to_8 = (8 - f.hour) % 24
    f["op_list_soon"] = ((hrs_to_8 > 0) & (hrs_to_8 <= HORIZON)).astype(int)
    return f


def episodes_metrics(full, alert, test_idx):
    """How many 'becomes full' episodes in the test year were warned >= 2h ahead."""
    onsets = [i for i in test_idx if i >= 6 and full[i] and not full[i - 6:i].any()]
    warned, leads = 0, []
    for o in onsets:
        hits = [i for i in range(max(0, o - HORIZON), o - 1) if alert[i] and not full[i]]
        if hits:
            warned += 1
            leads.append(o - hits[0])
    alerts = [i for i in test_idx if alert[i] and not full[i] and i + 10 < len(full)]
    correct = sum(full[i + 1:i + 11].any() for i in alerts)
    return {"episodes": len(onsets), "warned": warned,
            "pct_warned": round(100 * warned / max(len(onsets), 1), 1),
            "median_lead_h": float(np.median(leads)) if leads else None,
            "pct_alerts_correct": round(100 * correct / max(len(alerts), 1), 1)}


def fit_change_model(X, y_now, y_future, mask):
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0)
    m.fit(X[mask], (y_future - y_now)[mask])
    return m


def main():
    df = pd.read_csv("data/hospital_hourly.csv", parse_dates=["timestamp"])
    X = hourly_features(df)
    X_noseason = X.drop(columns=SEASONAL)
    ok = np.arange(len(df)) < len(df) - HORIZON
    train = (df.timestamp < TEST_START).values & ok
    test = ((df.timestamp >= TEST_START) & (df.timestamp < TEST_END)).values & ok
    periods = {
        "Normal days": (df.ramadan + df.hajj_season + df.eid == 0).values,
        "Ramadan": (df.ramadan == 1).values,
        "Hajj season": (df.hajj_season == 1).values,
        "Hajj days & Eid al-Adha": ((df.hajj_peak == 1) | ((df.eid == 1) & (X.days_to_arafah.abs() <= 6))).values,
    }

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    out = pd.DataFrame({"timestamp": df.timestamp})
    metrics = {"train": "2023-2025", "test": "2026 so far (never seen in training)",
               "horizon_hours": HORIZON, "forecasts": {}, "warnings": {}, "seasonal_check": {}}

    # ---- A1. hourly forecasts --------------------------------------------
    for name, col in HOURLY_TARGETS.items():
        y_now = df[col].values.astype(float)
        y_fut = df[col].shift(-HORIZON).values
        m = fit_change_model(X, y_now, y_fut, train)
        joblib.dump(m, f"models/forecast_{name}.joblib")
        upper = CAPACITY.get(name) if name not in ("ed_total", "ed_boarders") else None
        p = np.clip(y_now + m.predict(X), 0, upper)
        out[f"{name}_f"] = np.round(p, 1)
        ai = mean_absolute_error(y_fut[test], p[test])
        base = mean_absolute_error(y_fut[test], y_now[test])
        metrics["forecasts"][name] = {"ai_error": round(ai, 2), "simple_guess_error": round(base, 2),
                                      "improvement_pct": round(100 * (1 - ai / base), 1)}
        # same model without Hajj / Ramadan / Eid calendar -> did the AI learn the seasons?
        if name in ("ed_total", "ed_boarders", "med", "icu"):
            m2 = fit_change_model(X_noseason, y_now, y_fut, train)
            p2 = np.clip(y_now + m2.predict(X_noseason), 0, upper)
            metrics["seasonal_check"][name] = {
                per: {"with_calendar": round(mean_absolute_error(y_fut[test & msk], p[test & msk]), 2),
                      "without_calendar": round(mean_absolute_error(y_fut[test & msk], p2[test & msk]), 2),
                      "simple_guess": round(mean_absolute_error(y_fut[test & msk], y_now[test & msk]), 2)}
                for per, msk in periods.items()}

    # ---- A2. early warnings ------------------------------------------------
    test_idx = np.where(test)[0]
    for name, (col, thr, desc) in WARNINGS.items():
        full = (df[col] >= thr).values
        y = (df[col].shift(-HORIZON) >= thr).astype(int).values
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)
        clf.fit(X[train], y[train])
        joblib.dump(clf, f"models/warning_{name}.joblib")
        prob = clf.predict_proba(X)[:, 1]
        out[f"{name}_p"] = np.round(prob, 3)
        metrics["warnings"][name] = {"definition": desc, **episodes_metrics(full, prob >= 0.5, test_idx)}

    # ---- A3. normal levels for bottleneck detection ------------------------
    how = df.timestamp.dt.weekday * 24 + df.timestamp.dt.hour
    trn = df.timestamp < TEST_START
    normal_mask = trn & (df.ramadan + df.hajj_season + df.eid == 0)
    for c in ["scan_queue", "med_ready_to_go", "surg_ready_to_go", "mat_ready_to_go", "ed_arrivals"]:
        typical = df[normal_mask].groupby(how[normal_mask])[c].mean()
        out[f"{c}_normal"] = how.map(typical).round(1).values
    out.to_csv("results/hourly_predictions.csv", index=False)

    # ---- B. daily surge planning, 7 days ahead -----------------------------
    day = df.set_index("timestamp").resample("D").agg(
        ed_visits=("ed_arrivals", "sum"), heat_cases=("ed_heat_arrivals", "sum"),
        icu_peak=("icu_occupied", "max"),
        boarders_peak=("ed_boarders", "max"), transfers=("transfers_out", "sum"),
        scan_peak=("scan_queue", "max"),
        ramadan=("ramadan", "max"), hajj_season=("hajj_season", "max"),
        hajj_peak=("hajj_peak", "max"), eid=("eid", "max")).reset_index()
    D = calendar(day.timestamp).drop(columns=["hour"])
    for c in ["ramadan", "hajj_season", "hajj_peak", "eid"]:
        D[c] = day[c]
    daily_targets = ["ed_visits", "heat_cases", "icu_peak", "boarders_peak"]
    for c in daily_targets:                       # only information available 7 days before
        D[f"{c}_lag7"] = day[c].shift(7)
        D[f"{c}_avg7"] = day[c].shift(7).rolling(7, min_periods=1).mean()
    D_noseason = D.drop(columns=SEASONAL)
    dtrain = (day.timestamp < TEST_START) & D["ed_visits_lag7"].notna()
    dtest = (day.timestamp >= TEST_START) & (day.timestamp < TEST_END)
    dper = {
        "Ramadan": day.ramadan == 1,
        "Hajj season": day.hajj_season == 1,
        "Hajj days & Eid al-Adha": (day.hajj_peak == 1) | ((day.eid == 1) & (D.days_to_arafah.abs() <= 6)),
        "Normal days": day.ramadan + day.hajj_season + day.eid == 0,
    }
    daily_out = pd.DataFrame({"date": day.timestamp.dt.strftime("%Y-%m-%d")})
    metrics["daily_7day"] = {}
    for c in daily_targets:
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, min_samples_leaf=10, random_state=0)
        m.fit(D[dtrain], day[c][dtrain])
        joblib.dump(m, f"models/daily_{c}.joblib")
        m2 = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, min_samples_leaf=10, random_state=0)
        m2.fit(D_noseason[dtrain], day[c][dtrain])
        p, p2 = m.predict(D), m2.predict(D_noseason)
        daily_out[c] = day[c]
        daily_out[f"{c}_f"] = np.round(p, 1)
        daily_out[f"{c}_f_nocal"] = np.round(p2, 1)
        daily_out[f"{c}_lastweek"] = day[c].shift(7)
        res = {}
        for per, msk in dper.items():
            mk = dtest & msk
            if mk.sum() == 0:
                continue
            res[per] = {"days": int(mk.sum()),
                        "ai_error": round(mean_absolute_error(day[c][mk], p[mk]), 1),
                        "ai_without_calendar_error": round(mean_absolute_error(day[c][mk], p2[mk]), 1),
                        "last_week_guess_error": round(mean_absolute_error(day[c][mk], day[c].shift(7)[mk]), 1),
                        "average_actual": round(float(day[c][mk].mean()), 1)}
        metrics["daily_7day"][c] = res
    # typical normal-day level for each weekday (from training years, outside seasons)
    nm = (day.timestamp < TEST_START) & dper["Normal days"]
    for c in daily_targets:
        typ = day[nm].groupby(day.timestamp.dt.weekday)[c].mean()
        daily_out[f"{c}_normal"] = day.timestamp.dt.weekday.map(typ).round(1)
    for c in ["ramadan", "hajj_season", "hajj_peak", "eid", "transfers", "scan_peak"]:
        daily_out[c] = day[c]
    daily_out.to_csv("results/daily_predictions.csv", index=False)

    with open("results/metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
