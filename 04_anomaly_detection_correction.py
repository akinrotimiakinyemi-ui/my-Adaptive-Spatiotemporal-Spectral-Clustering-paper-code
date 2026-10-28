"""
SCRIPT 04 — Two-Stage Anomaly Detection and Correction
=======================================================
Detects and corrects two documented anomaly types in the cumulative
case matrix:

  Stage 1 — Isolated spike detection (n=4 found)
    A cumulative count that jumps sharply on one date and reverts
    on the next. Cannot reflect a real cumulative trajectory.
    Fix: linear interpolation between the two flanking values.

  Stage 2 — Monotonicity enforcement (n=204 found)
    A cumulative count that drops below its own running maximum.
    Consistent with periodic case reclassification.
    Fix: running-maximum (isotonic) filter.

Every correction is logged to anomaly_audit_log.csv with the raw
and corrected values. No silent changes.

Input:  data/processed/cum_confirmed_matrix.csv
Output: data/processed/cum_confirmed_matrix_clean.csv
        data/processed/anomaly_audit_log.csv
"""

import numpy as np
import pandas as pd

IN_PATH   = "data/processed/cum_confirmed_matrix.csv"
OUT_CLEAN = "data/processed/cum_confirmed_matrix_clean.csv"
AUDIT_LOG = "data/processed/anomaly_audit_log.csv"

SPIKE_RATIO = 3.0   # value >= 3x both neighbours → isolated spike

matrix = pd.read_csv(IN_PATH, index_col=0)
matrix.columns = pd.to_datetime(matrix.columns)
matrix = matrix.apply(pd.to_numeric, errors="coerce")

audit_records = []
matrix_despiked = matrix.copy()

# ── Stage 1: isolated spike detection ───────────────────────────────────────
for unit in matrix.index:
    valid = matrix.loc[unit].dropna()
    if len(valid) < 3:
        continue
    vals, dates = valid.values, valid.index
    for i in range(1, len(vals) - 1):
        prev_v, cur_v, next_v = vals[i - 1], vals[i], vals[i + 1]
        if prev_v <= 0 or next_v <= 0:
            continue
        if cur_v >= SPIKE_RATIO * prev_v and cur_v >= SPIKE_RATIO * next_v:
            corrected = (prev_v + next_v) / 2
            audit_records.append({
                "admin_unit": unit, "date": dates[i],
                "raw_value": cur_v, "corrected_value": corrected,
                "anomaly_type": "isolated_spike"
            })
            matrix_despiked.loc[unit, dates[i]] = corrected

print(f"Stage 1 — isolated spikes detected and corrected: "
      f"{sum(1 for r in audit_records if r['anomaly_type'] == 'isolated_spike')}")

# ── Stage 2: monotonicity enforcement ───────────────────────────────────────
clean_matrix = matrix_despiked.copy()
for unit in matrix_despiked.index:
    valid = matrix_despiked.loc[unit].dropna()
    if len(valid) < 2:
        continue
    dates = valid.index
    running_max = -np.inf
    corrected = valid.copy()
    for d in dates:
        v = valid[d]
        if v < running_max:
            audit_records.append({
                "admin_unit": unit, "date": d,
                "raw_value": v, "corrected_value": running_max,
                "anomaly_type": "non_monotonic_drop"
            })
            corrected[d] = running_max
        else:
            running_max = v
    clean_matrix.loc[unit, dates] = corrected.values

n_drops = sum(1 for r in audit_records if r["anomaly_type"] == "non_monotonic_drop")
print(f"Stage 2 — non-monotonic drops corrected: {n_drops}")

audit_df = pd.DataFrame(audit_records)
audit_df.to_csv(AUDIT_LOG, index=False)
clean_matrix.to_csv(OUT_CLEAN)

print(f"\nTotal anomalies corrected: {len(audit_df)}")
print(f"Audit log → {AUDIT_LOG}")
print(f"Clean matrix → {OUT_CLEAN}")

# Spot-check the two previously problematic districts
for unit in ["Sierra Leone - Western area rural", "Liberia - Maryland"]:
    if unit in clean_matrix.index:
        peak = clean_matrix.loc[unit].max(skipna=True)
        print(f"  {unit} post-clean peak: {peak:.0f}")
