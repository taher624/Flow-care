# Flow Care

**Hospital-wide patient flow, with AI that warns before departments fill up, including during Hajj and Ramadan.**
Team Trimed · Medathon 2026 (Saudi Youth Healthcare Forum · Wadi Jeddah · King Abdulaziz University)

> **Prototype.** Everything here runs on a **simulated hospital**. No real patient data is used. Not for clinical use.

## The idea

A delay in one department spreads through the whole hospital: a slow scan holds up a
discharge, the bed stays full, an ED patient waits for that bed, and a planned operation
is cancelled. Flow Care connects every department into one view so teams can see,
predict and act on patient flow together, and plan ahead for Hajj and Ramadan surges.

## The app (one page, `dashboard/index.html`)

| Page | What it shows |
|---|---|
| **Home** | The whole hospital: each department's status and 8-hour forecast, the patient journey, and hospital-wide actions. Tap a department to open it. |
| **Internal Medicine** | Medical wards (120 beds): forecast, early warning, patients cleared to go home, scan backlog, ED patients waiting for a medical bed. |
| **Surgery** | Surgical wards (52 beds) and the daily operating list: forecast, cancellations for lack of beds. |
| **Emergency (ED)** | Patients in the ED, admitted patients waiting for a bed, arrivals vs normal, heat illness, transfers out. |
| **OB/GYN** | Maternity ward (46 beds) and delivery rooms (10): forecast, women waiting for a room, postponed planned caesareans. |
| **ICU** | ICU beds (16), patients waiting in the ED for ICU, patients ready to step down with no ward bed. |
| **Log a patient** | Log an arrival or admission in two taps; discharge, mark a scan done, or remove. Shared live with the team. |
| **Hajj & Ramadan** | The next 7 days with a daily surge plan, a countdown and preview for Ramadan 2027 and Hajj 2027, and the AI's test results on Ramadan and Hajj 2026. |

The app runs **live on Riyadh time**: it shows the hospital as it is right now and the AI's forecast for the
next 8 hours, updating every hour. **Log a patient** (two taps, no name or ID number) adds a patient to the
chosen department; the department's numbers, forecast and warnings update straight away. Logged patients are
saved in the page's shared database, so everyone with the link sees the same list.

## Files

| File | What it does |
|---|---|
| `simulate.py` | Simulates 2023–2027 of hourly patient flow in a mid-sized Jeddah hospital: ED, Internal Medicine, Surgery and operating lists, OB/GYN, ICU, imaging, discharges. Includes the Saudi weekend, Ramadan (shift of ED visits to the evening, shorter working hours), the Hajj season and Hajj days (more ED visits, heat illness, more ICU admissions) and Eid holidays. |
| `model.py` | The AI. Learns from 2023–2025 and is tested on 2026 up to today. |
| `build_dashboard.py` | Builds the one-page app from the results. |
| `dashboard_template.html` | The app design. |
| `results/metrics.json` | All test results. |

## Run it

```bash
pip install -r requirements.txt
python simulate.py          # simulated hospital, 2023-2027
python model.py             # train on 2023-2025, test on 2026 so far
python build_dashboard.py   # build dashboard/index.html
```

## How the AI works

- **Hourly forecasts (8 hours ahead)** for each department, using gradient-boosted decision trees
  (scikit-learn). Inputs: time of day, weekday, Islamic-calendar period (Ramadan day, days to the Day of
  Arafah, Eid), current patients in every area, patients waiting in the ED and for which ward, patients
  cleared to go home, scan backlog, recent arrivals, births, discharges and heat-illness cases.
- **Early warnings:** a second model per department estimates the chance it will be full within 8 hours.
- **Bottlenecks:** each step is compared with its normal level for that hour of the week.
- **Daily surge planning (7 days ahead):** forecasts ED visits, heat-illness cases, peak ED waiting and
  peak ICU use, using only information available a week before.
- **Recommendations:** clear rules turn forecasts into actions for specific teams. Staff approve or
  dismiss every recommendation. Flow Care never makes clinical decisions.

## Results (2026 so far, never seen in training)

Compared with a simple guess ("in 8 hours it will be as full as now"): Internal Medicine ±2.5 beds vs ±6.6,
surgical ±1.7 vs ±3.5, maternity ±1.5 vs ±3.4, patients in the ED ±4.9 vs ±17.6. ICU is not better yet (±1.2 vs ±1.2).

Early warnings at least 2 hours ahead: Internal Medicine 91% of 67 episodes, ED waiting 85% of 34,
maternity 81% of 111, surgery 71% of 92. ICU warnings did not work (12%), so the ICU page uses clear rules.

Hajj 2026 (Hajj days and Eid al-Adha), ED visits forecast 7 days ahead: ±26 per day with the Islamic calendar,
±91 without it, ±66 for "same as last week". Heat illness: ±4 vs ±36 vs ±33. Ramadan 2026: the calendar
improved the hour-by-hour ED forecast (±4.8 vs ±6.7).

## Limitations

- All data is simulated. Next step: test with real, anonymised hospital data.
- ICU forecasts are not yet better than a simple guess; real planned-surgery lists would help.
- The AI learned each season from only three earlier years. Islamic dates for 2027 are expected dates and depend on the moon sighting.
- Logged patients are added on top of the AI's forecast; they do not yet change the model's own inputs.
- Not yet used by real staff; effects on waiting times and outcomes are unmeasured.
- Must be checked against Saudi PDPL and SFDA guidance on AI software before any real use.
