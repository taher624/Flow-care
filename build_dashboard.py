"""
Flow Care - build the app
=========================

Packs the live hospital feed (the simulated hospital from two days ago until the
end of 2027), the AI's hourly forecasts and the 7-day surge forecasts into one
self-contained web page. The page follows the real clock (Riyadh time): it only
ever shows hours up to "now" plus the AI's forecast for the next 8 hours.

    dashboard/index.html

Pages (one link): Home · Internal Medicine · Surgery · ED · OB/GYN · ICU ·
                  Log a patient (shared with the team) · Hajj & Ramadan

Run:  python build_dashboard.py      (after simulate.py and model.py)
"""

import json
import os

import numpy as np
import pandas as pd

FEED_START = "2026-11-05"        # two days before today, for the 24-hour charts

h = pd.read_csv("data/hospital_hourly.csv", parse_dates=["timestamp"])
p = pd.read_csv("results/hourly_predictions.csv", parse_dates=["timestamp"])
daily = pd.read_csv("results/daily_predictions.csv")
metrics = json.load(open("results/metrics.json"))

d = h.merge(p, on="timestamp")
d = d[d.timestamp >= FEED_START].reset_index(drop=True)

HOURLY = {
    "edTotal": "ed_total", "edWait": "ed_boarders", "edLongest": "ed_longest_board_h",
    "edArr": "ed_arrivals", "edArrN": "ed_arrivals_normal", "edHeat": "ed_heat_arrivals",
    "edTotalF": "ed_total_f", "edWaitF": "ed_boarders_f", "edP": "ed_p",
    "transfers": "transfers_out",
    "med": "med_occupied", "medF": "med_f", "medP": "med_p", "medReady": "med_ready_to_go",
    "medReadyN": "med_ready_to_go_normal", "bMed": "boarders_med", "medDisch": "discharges_med",
    "surgDisch": "discharges_surg", "matDisch": "discharges_mat",
    "surg": "surg_occupied", "surgF": "surg_f", "surgP": "surg_p", "surgReady": "surg_ready_to_go",
    "surgReadyN": "surg_ready_to_go_normal", "bSurg": "boarders_surg",
    "surgDone": "surgeries_done", "surgCancel": "surgeries_cancelled",
    "icu": "icu_occupied", "icuF": "icu_f", "icuBlocked": "icu_blocked_stepdown", "bIcu": "boarders_icu",
    "mat": "mat_occupied", "matF": "mat_f", "matP": "mat_p", "matReady": "mat_ready_to_go",
    "matReadyN": "mat_ready_to_go_normal", "ld": "ld_occupied", "ldF": "ld_f",
    "ldBlocked": "ld_blocked", "obTriage": "ob_triage_waiting", "births": "births",
    "csDone": "cs_planned_done", "csPost": "cs_postponed",
    "scanQ": "scan_queue", "scanQN": "scan_queue_normal",
    "ramadan": "ramadan", "hajjS": "hajj_season", "hajjP": "hajj_peak", "eid": "eid",
}


def compact(series):
    out = []
    for v in series.fillna(0).tolist():
        v = float(v)
        out.append(int(v) if v.is_integer() else round(v, 2))
    return out


data = {"start": d.timestamp.iloc[0].strftime("%Y-%m-%dT%H:00"), "n": len(d),
        "metrics": metrics}
for k, c in HOURLY.items():
    data[k] = compact(d[c])

# daily 7-day-ahead surge forecasts, 2026-2027
day = daily[daily.date >= "2026-01-01"].reset_index(drop=True)
data["daily"] = {"date": day.date.tolist()}
for c in ["ed_visits", "heat_cases", "icu_peak", "boarders_peak"]:
    for suf in ["", "_f", "_f_nocal", "_normal", "_lastweek"]:
        data["daily"][c + suf] = compact(day[c + suf])
for c in ["ramadan", "hajj_season", "hajj_peak", "eid", "transfers", "scan_peak"]:
    data["daily"][c] = compact(day[c])

# when do ED patients arrive? Ramadan vs normal days (2023-2025)
y25 = h[h.timestamp < "2026-01-01"]
normal = y25[(y25.ramadan + y25.hajj_season + y25.eid) == 0]
ram = y25[y25.ramadan == 1]
data["arrivalProfile"] = {
    "normal": [round(v, 2) for v in normal.groupby(normal.timestamp.dt.hour).ed_arrivals.mean()],
    "ramadan": [round(v, 2) for v in ram.groupby(ram.timestamp.dt.hour).ed_arrivals.mean()],
}

template = open("dashboard_template.html", encoding="utf-8").read()
html = template.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
os.makedirs("dashboard", exist_ok=True)
open("dashboard/index.html", "w", encoding="utf-8").write(html)
print(f"Wrote dashboard/index.html ({len(html) / 1024:.0f} KB, {len(d):,} hours)")
