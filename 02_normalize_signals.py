"""
SCRIPT 02 — Normalize Variable Names to Canonical Signals
==========================================================
Maps the 130+ inconsistent free-text indicator names across Sierra Leone,
Liberia, and Guinea down to four canonical signals:

    cum_confirmed_cases   — cumulative confirmed case count
    cum_confirmed_deaths  — cumulative deaths (confirmed)
    new_confirmed_cases   — new confirmed cases that reporting day
    new_confirmed_deaths  — new confirmed deaths that reporting day

Unmapped variables (composites, sex-disaggregated, etc.) are logged to
data/processed/unmapped_variables_log.csv for full auditability.

Input:  data/processed/west_africa_district_panel_long.csv
Output: data/processed/west_africa_canonical_signals.csv
        data/processed/unmapped_variables_log.csv
"""

import pandas as pd
import re

IN_PATH  = "data/processed/west_africa_district_panel_long.csv"
OUT_PATH = "data/processed/west_africa_canonical_signals.csv"
LOG_PATH = "data/processed/unmapped_variables_log.csv"

SL_MAP = {
    "cum_confirmed":  "cum_confirmed_cases",
    "death_confirmed": "cum_confirmed_deaths",
    "new_confirmed":  "new_confirmed_cases",
}


def map_liberia(var):
    v = re.sub(r"\s+", " ", var.strip().lower())
    if v in ("total case/s (confirmed)", "total confirmed cases"):
        return "cum_confirmed_cases"
    if v in ("new case/s (confirmed)",):
        return "new_confirmed_cases"
    if v in ("total death/s in confirmed cases",):
        return "cum_confirmed_deaths"
    return None


def map_guinea(var):
    v = re.sub(r"\s+", " ", var.strip().lower())
    if v in ("total cases of confirmed",):
        return "cum_confirmed_cases"
    if v in ("new cases of confirmed",):
        return "new_confirmed_cases"
    if v in ("total deaths of confirmed",):
        return "cum_confirmed_deaths"
    if v in ("new deaths registered today (confirmed)",):
        return "new_confirmed_deaths"
    return None


if __name__ == "__main__":
    df = pd.read_csv(IN_PATH, parse_dates=["date"])
    canonical, unmapped = [], []

    for _, row in df.iterrows():
        country, var = row["country"], row["variable"]
        canon = None
        if country == "Sierra Leone":
            canon = SL_MAP.get(var)
        elif country == "Liberia":
            canon = map_liberia(var)
        elif country == "Guinea":
            canon = map_guinea(var)

        if canon:
            canonical.append({"date": row["date"], "country": country,
                               "admin_unit": row["admin_unit"],
                               "signal": canon, "value": row["value"]})
        else:
            unmapped.append({"country": country, "variable": var})

    canon_df = pd.DataFrame(canonical).drop_duplicates()
    canon_df.to_csv(OUT_PATH, index=False)

    unmapped_df = pd.DataFrame(unmapped).drop_duplicates()
    unmapped_df.to_csv(LOG_PATH, index=False)

    print(f"Canonical records: {len(canon_df)} → {OUT_PATH}")
    print(f"Unmapped (excluded): {len(unmapped_df)} unique variable names → {LOG_PATH}")
    print("\nCoverage by signal × country:")
    print(canon_df.groupby(["country", "signal"]).size().unstack(fill_value=0))
