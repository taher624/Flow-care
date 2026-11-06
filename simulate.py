"""
Flow Care - Hospital patient-flow simulation
============================================

Simulates five years (2023-2027) of hour-by-hour patient flow through a
mid-sized hospital in Jeddah, Saudi Arabia. ALL DATA IS SIMULATED.
No real patient data is used.

Departments
  ED                Emergency Department: assessment, then home or admission.
                    Admitted patients wait in the ED until a ward bed is free ("boarding").
  Internal Medicine Medical wards.
  Surgery           Surgical wards + the planned operating list (cancelled on the
                    day if no bed is free).
  OB/GYN            Labour & delivery rooms + maternity ward, planned caesareans.
  ICU               Intensive care; patients step down to a ward when stable and
                    stay stuck in ICU if no ward bed is free.
  Imaging           Shared scanners; patients cannot go home until scans are done.

Saudi patterns built in
  Weekend (Friday/Saturday): fewer scans, slower discharges, no planned lists.
  Ramadan: fewer ED visits during the fast and more after iftar, shorter working
           hours (discharges and scans squeezed into fewer hours), smaller
           operating lists.
  Hajj season (Jeddah): more ED visits for weeks; around the Hajj days a sharp
           rise in heat illness, with more ICU admissions.
  Eid holidays: no planned lists, slow discharges, busy ED.

Output: data/hospital_hourly.csv  (one row per hour)

The app shows the hours that match the real clock "now", so the hospital
looks live; the AI never sees hours after the present when it is tested.
Run:    python simulate.py
"""

import os
from collections import deque
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

SEED = 7
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# Hospital set-up
# ---------------------------------------------------------------------------
CAPACITY = {"ED": 40, "MED": 120, "SURG": 52, "ICU": 16, "LD": 10, "MAT": 46}
TRANSFER_AFTER_H = 24   # admitted patients still waiting after 24 h are transferred to another hospital

WARMUP_DAYS = 30
FIRST_DAY = datetime(2023, 1, 1)
LAST_DAY = datetime(2027, 12, 31)

# Approximate Islamic calendar dates (for simulation only)
RAMADAN = [(datetime(2023, 3, 23), datetime(2023, 4, 21)),
           (datetime(2024, 3, 11), datetime(2024, 4, 10)),
           (datetime(2025, 3, 1), datetime(2025, 3, 30)),
           (datetime(2026, 2, 18), datetime(2026, 3, 20)),
           (datetime(2027, 2, 8), datetime(2027, 3, 10))]
EID_FITR = [(datetime(2023, 4, 21), datetime(2023, 4, 25)),
            (datetime(2024, 4, 10), datetime(2024, 4, 14)),
            (datetime(2025, 3, 30), datetime(2025, 4, 3)),
            (datetime(2026, 3, 20), datetime(2026, 3, 24)),
            (datetime(2027, 3, 10), datetime(2027, 3, 14))]
ARAFAH = [datetime(2023, 6, 27), datetime(2024, 6, 15), datetime(2025, 6, 5),
          datetime(2026, 5, 26), datetime(2027, 5, 15)]


def hajj_season(ts):   # pilgrims arriving through Jeddah until they leave
    return any(a - timedelta(days=30) <= ts < a + timedelta(days=12) for a in ARAFAH)


def hajj_peak(ts):     # the Hajj days themselves: heat, crowds, Mina
    return any(a - timedelta(days=2) <= ts < a + timedelta(days=4) for a in ARAFAH)


def eid_adha(ts):
    return any(a + timedelta(days=1) <= ts < a + timedelta(days=5) for a in ARAFAH)


def ramadan(ts):
    return any(s <= ts < e for s, e in RAMADAN)


def eid(ts):
    return eid_adha(ts) or any(s <= ts < e for s, e in EID_FITR)


def is_weekend(ts):
    return ts.weekday() in (4, 5)  # Friday, Saturday


ED_HOURLY_PROFILE = np.array([
    4.0, 3.2, 2.6, 2.2, 2.0, 2.2, 3.0, 4.6, 6.6, 8.6, 9.6, 10.0,
    9.8, 9.4, 9.2, 9.2, 9.4, 9.8, 10.2, 10.0, 9.0, 7.6, 6.2, 5.0])
RAMADAN_PROFILE = np.array([      # quiet during the fast, busy after iftar
    7.0, 6.0, 4.8, 3.6, 2.4, 1.8, 1.8, 2.4, 3.6, 4.8, 5.6, 6.0,
    6.0, 6.0, 6.0, 6.0, 6.0, 6.4, 7.4, 10.4, 12.4, 12.6, 11.4, 9.2])
BIRTH_PROFILE = np.array([        # slightly more labour at night / early morning
    1.15, 1.2, 1.2, 1.2, 1.15, 1.1, 1.05, 1.0, 0.95, 0.9, 0.9, 0.9,
    0.9, 0.9, 0.9, 0.9, 0.9, 0.95, 0.95, 1.0, 1.0, 1.05, 1.1, 1.1])


def ed_arrival_rate(ts):
    rate = (RAMADAN_PROFILE if ramadan(ts) else ED_HOURLY_PROFILE)[ts.hour]
    if is_weekend(ts):
        rate *= 1.08
    if hajj_season(ts):
        rate *= 1.15
    if hajj_peak(ts):
        rate *= 1.30
    if eid(ts):
        rate *= 1.15
    rate *= 1 + 0.07 * np.cos(2 * np.pi * (ts.timetuple().tm_yday - 15) / 365)
    rate *= 1 + 0.03 * (ts.year - 2023)          # slow growth in demand
    return rate


def heat_share(ts):
    """Share of ED arrivals with heat-related illness."""
    if hajj_peak(ts):
        return 0.18
    if hajj_season(ts):
        return 0.06
    return 0.02 if ts.month in (6, 7, 8) else 0.0


def imaging_capacity(ts):
    """Inpatient scan slots this hour (scanners are shared with ED and clinics)."""
    h = ts.hour
    if eid(ts) or is_weekend(ts):
        lam = 0.6 if 8 <= h < 16 else 0.2
    elif ramadan(ts):
        lam = 1.4 if 10 <= h < 16 else (0.7 if 21 <= h or h < 1 else 0.3)
    else:
        lam = 1.5 if 8 <= h < 20 else 0.3
    return int(rng.poisson(lam))


def discharge_probability(ts):
    """Chance a medically-ready patient actually leaves this hour."""
    h = ts.hour
    if ramadan(ts):     # shorter working day
        p = 0.24 if 10 <= h < 15 else (0.07 if 15 <= h < 17 or 21 <= h < 23 else 0.01)
    else:
        p = 0.22 if 10 <= h < 17 else (0.08 if 8 <= h < 10 or 17 <= h < 20 else 0.01)
    if is_weekend(ts):
        p *= 0.45
    if eid(ts):
        p *= 0.6
    return p


def electives_today(ts):
    if is_weekend(ts) or eid(ts):
        return 0
    if ramadan(ts):
        return 8
    if hajj_season(ts):
        return 9
    return 11


def los(unit):
    median_days = {"MED": 3.0, "SURG": 2.4, "ICU": 2.3, "MAT_NVD": 1.4, "MAT_CS": 2.8}[unit]
    return float(np.clip(rng.lognormal(np.log(median_days * 24), 0.5), 8, 24 * 30))


class Patient:
    __slots__ = ("unit", "ready_t", "needs_scan", "scan_req_t", "scan_done_t", "after_icu")

    def __init__(self, unit, t, ready_t=None, scan_prob=0.0):
        self.unit = unit
        self.ready_t = t + los(unit) if ready_t is None else ready_t
        self.needs_scan = rng.random() < scan_prob
        self.scan_req_t = t + rng.uniform(1, 14) if self.needs_scan else None
        self.scan_done_t = None
        self.after_icu = None


SCAN_PROB = {"MED": 0.45, "SURG": 0.30}


def simulate():
    start = FIRST_DAY - timedelta(days=WARMUP_DAYS)
    hours = int((LAST_DAY + timedelta(days=1) - start).total_seconds() // 3600)
    beds = {u: [] for u in ("MED", "SURG", "ICU", "MAT")}
    ld_rooms = []               # [finish_time, is_caesarean]
    ld_blocked = []             # delivered, waiting for a maternity bed
    ob_triage = deque()         # women in labour waiting for a delivery room
    ed_treating = []
    boarders = deque()
    scan_queue = deque()
    rows = []

    for h in range(hours):
        ts = start + timedelta(hours=h)
        t = float(h)

        # ---- ED arrivals ---------------------------------------------------
        n_arr = rng.poisson(ed_arrival_rate(ts))
        n_heat = rng.binomial(n_arr, heat_share(ts))
        icu_share = 0.10 + (0.06 if hajj_peak(ts) else 0)
        for j in range(n_arr):
            heat = j < n_heat
            admitted = rng.random() < (0.35 if heat else 0.17)
            treat = rng.gamma(4, 1.1) if admitted else rng.gamma(3, 1.1)
            target = None
            if admitted:
                if heat:
                    target = "ICU" if rng.random() < 0.25 else "MED"
                else:
                    r = rng.random()
                    target = "ICU" if r < icu_share else ("SURG" if r < icu_share + 0.16 else "MED")
            ed_treating.append((t + treat, target))
        keep = []
        for finish, target in ed_treating:
            if finish <= t:
                if target:
                    boarders.append((t, target))
            else:
                keep.append((finish, target))
        ed_treating = keep

        # ---- Imaging ----------------------------------------------------------
        cap = imaging_capacity(ts)
        served, waits = 0, []
        while scan_queue and served < cap:
            p = scan_queue.popleft()
            p.scan_done_t = t
            waits.append(t - p.scan_req_t)
            p.ready_t = max(p.ready_t, t + 6)
            served += 1
        for unit in ("MED", "SURG"):
            for p in beds[unit]:
                if p.needs_scan and p.scan_done_t is None and p.scan_req_t <= t and p not in scan_queue:
                    scan_queue.append(p)
        for p in scan_queue:
            p.ready_t = max(p.ready_t, t + 1)

        # ---- Ward discharges ---------------------------------------------------
        pdis = discharge_probability(ts)
        disch = {"MED": 0, "SURG": 0, "MAT": 0}
        ready = {}
        for unit in ("MED", "SURG", "MAT"):
            kept, r = [], 0
            for p in beds[unit]:
                if p.ready_t <= t:
                    if rng.random() < pdis:
                        disch[unit] += 1
                        continue
                    r += 1
                kept.append(p)
            beds[unit] = kept
            ready[unit] = r

        # ---- ICU step-down -----------------------------------------------------
        kept = []
        for p in beds["ICU"]:
            if p.ready_t <= t:
                dest = p.after_icu or "MED"
                if len(beds[dest]) < CAPACITY[dest]:
                    beds[dest].append(Patient(dest, t, ready_t=t + los(dest) * 0.6))
                    continue
            kept.append(p)
        beds["ICU"] = kept
        icu_blocked = sum(1 for p in beds["ICU"] if p.ready_t <= t)

        # ---- Planned operating list 08:00 --------------------------------------
        surg_done = surg_cancel = 0
        if ts.hour == 8:
            for _ in range(electives_today(ts)):
                needs_icu = rng.random() < 0.12
                unit = "ICU" if needs_icu else "SURG"
                if len(beds[unit]) < CAPACITY[unit]:
                    if needs_icu:
                        p = Patient("ICU", t, ready_t=t + rng.uniform(18, 40))
                        p.after_icu = "SURG"
                    else:
                        p = Patient("SURG", t, scan_prob=SCAN_PROB["SURG"])
                    beds[unit].append(p)
                    surg_done += 1
                else:
                    surg_cancel += 1

        # ---- OB/GYN ------------------------------------------------------------
        births = 0
        n_labour = rng.poisson(0.50 * BIRTH_PROFILE[ts.hour] * (1 + 0.02 * (ts.year - 2023)))
        for _ in range(n_labour):
            ob_triage.append(t)
        # delivery rooms finish -> maternity ward (or stay blocked in the room)
        still = []
        for finish, cs in ld_rooms:
            if finish > t:
                still.append((finish, cs))
                continue
            births += 1
            if rng.random() < 0.005 and len(beds["ICU"]) < CAPACITY["ICU"]:
                beds["ICU"].append(Patient("ICU", t))   # rare maternal ICU admission
            else:
                ld_blocked.append(cs)
        ld_rooms = still
        waiting_room = []
        for cs in ld_blocked:
            if len(beds["MAT"]) < CAPACITY["MAT"]:
                beds["MAT"].append(Patient("MAT_CS" if cs else "MAT_NVD", t))
            else:
                waiting_room.append(cs)
        ld_blocked = waiting_room
        # rooms in use = labouring + delivered but no ward bed yet
        while ob_triage and len(ld_rooms) + len(ld_blocked) < CAPACITY["LD"]:
            ob_triage.popleft()
            ld_rooms.append((t + rng.gamma(5, 1.6), rng.random() < 0.18))
        cs_done = cs_postponed = 0
        if ts.hour == 9 and not is_weekend(ts) and not eid(ts):
            for _ in range(3):
                if len(beds["MAT"]) < CAPACITY["MAT"]:
                    beds["MAT"].append(Patient("MAT_CS", t))
                    cs_done += 1
                    births += 1
                else:
                    cs_postponed += 1

        # ---- ED patients get free ward beds (first come, first served) --------
        waiting = deque()
        transfers = 0
        while boarders:
            dt_, unit = boarders.popleft()
            if t - dt_ >= TRANSFER_AFTER_H:
                transfers += 1
            elif len(beds[unit]) < CAPACITY[unit]:
                beds[unit].append(Patient(unit, t, scan_prob=SCAN_PROB.get(unit, 0)))
            else:
                waiting.append((dt_, unit))
        boarders = waiting
        board_h = [t - d for d, _ in boarders]

        if ts < FIRST_DAY:
            continue
        rows.append({
            "timestamp": ts,
            "ramadan": int(ramadan(ts)), "hajj_season": int(hajj_season(ts)),
            "hajj_peak": int(hajj_peak(ts)), "eid": int(eid(ts)),
            "ed_arrivals": n_arr, "ed_heat_arrivals": n_heat,
            "ed_in_treatment": len(ed_treating), "ed_boarders": len(boarders),
            "ed_total": len(ed_treating) + len(boarders),
            "ed_longest_board_h": round(max(board_h), 1) if board_h else 0.0,
            "med_occupied": len(beds["MED"]), "surg_occupied": len(beds["SURG"]),
            "icu_occupied": len(beds["ICU"]), "mat_occupied": len(beds["MAT"]),
            "ld_occupied": len(ld_rooms) + len(ld_blocked), "ld_blocked": len(ld_blocked),
            "ob_triage_waiting": len(ob_triage),
            "med_ready_to_go": ready["MED"], "surg_ready_to_go": ready["SURG"],
            "mat_ready_to_go": ready["MAT"], "icu_blocked_stepdown": icu_blocked,
            "boarders_med": sum(1 for _, u in boarders if u == "MED"),
            "boarders_surg": sum(1 for _, u in boarders if u == "SURG"),
            "boarders_icu": sum(1 for _, u in boarders if u == "ICU"),
            "scan_queue": len(scan_queue), "scans_done": served,
            "discharges_med": disch["MED"], "discharges_surg": disch["SURG"],
            "discharges_mat": disch["MAT"],
            "transfers_out": transfers,
            "surgeries_done": surg_done, "surgeries_cancelled": surg_cancel,
            "births": births, "cs_planned_done": cs_done, "cs_postponed": cs_postponed,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = simulate()
    df.to_csv("data/hospital_hourly.csv", index=False)
    print(f"Simulated {len(df):,} hours ({df.timestamp.min():%Y-%m-%d} to "
          f"{df.timestamp.max():%Y-%m-%d}) -> data/hospital_hourly.csv")
