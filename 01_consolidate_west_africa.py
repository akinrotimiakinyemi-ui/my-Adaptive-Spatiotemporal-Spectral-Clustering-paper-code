"""
SCRIPT 01 — Consolidate West Africa Daily Situation Report CSVs
================================================================
Merges 243 daily district/county/prefecture-level Ebola situation
report snapshots for Sierra Leone, Liberia, and Guinea into a single
unified long-format panel:
    date | country | admin_unit | variable | value

Source: cmrivers/ebola (GitHub)
    - sl_data/      → 113 Sierra Leone district CSVs
    - liberia_data/ → 108 Liberia county CSVs
    - guinea_data/  → 22  Guinea prefecture CSVs

Output:
    data/processed/west_africa_district_panel_long.csv
"""

import pandas as pd
import glob
import os

RAW_DIR = "data/raw/west_africa_2014"
OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

SL_DISTRICTS = [
    "Kailahun", "Kenema", "Kono", "Kambia", "Koinadugu", "Bombali",
    "Tonkolili", "Port Loko", "Pujehun", "Bo", "Moyamba", "Bonthe",
    "Western area urban", "Western area rural"
]
LIBERIA_COUNTIES = [
    "Bomi County", "Bong County", "Gbarpolu County", "Grand Bassa",
    "Grand Cape Mount", "Grand Gedeh", "Lofa County", "Margibi County",
    "Maryland County", "Montserrado County", "Nimba County",
    "River Gee County", "RiverCess County", "Sinoe County"
]
GUINEA_PREFECTURES = [
    "Conakry", "Gueckedou", "Macenta", "Dabola", "Kissidougou",
    "Dinguiraye", "Telimele", "Boffa", "Kouroussa", "Siguiri", "Pita",
    "Mzerekore", "Yomou", "Dubreka", "Forecariah", "Kerouane", "Coyah",
    "Dalaba", "Beyla", "Kindia", "Lola"
]


def consolidate_sierra_leone():
    files = sorted(glob.glob(f"{RAW_DIR}/sl_data/*.csv"))
    records = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if "date" not in df.columns or "variable" not in df.columns:
                continue
            for _, row in df.iterrows():
                for d in SL_DISTRICTS:
                    if d in df.columns and pd.notna(row.get(d)):
                        records.append({"date": row["date"], "country": "Sierra Leone",
                                        "admin_unit": d, "variable": row["variable"],
                                        "value": row[d]})
        except Exception:
            continue
    out = pd.DataFrame(records)
    print(f"[Sierra Leone] files={len(files)}, records={len(out)}")
    return out


def consolidate_liberia():
    files = sorted(glob.glob(f"{RAW_DIR}/liberia_data/*.csv"))
    records = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if "Date" not in df.columns or "Variable" not in df.columns:
                continue
            for _, row in df.iterrows():
                for c in LIBERIA_COUNTIES:
                    if c in df.columns and pd.notna(row.get(c)):
                        records.append({"date": row["Date"], "country": "Liberia",
                                        "admin_unit": c.replace(" County", ""),
                                        "variable": row["Variable"], "value": row[c]})
        except Exception:
            continue
    out = pd.DataFrame(records)
    print(f"[Liberia] files={len(files)}, records={len(out)}")
    return out


def consolidate_guinea():
    files = sorted(glob.glob(f"{RAW_DIR}/guinea_data/*.csv"))
    records = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if "Date" not in df.columns or "Description" not in df.columns:
                continue
            for _, row in df.iterrows():
                for p in GUINEA_PREFECTURES:
                    if p in df.columns and pd.notna(row.get(p)):
                        records.append({"date": row["Date"], "country": "Guinea",
                                        "admin_unit": p, "variable": row["Description"],
                                        "value": row[p]})
        except Exception:
            continue
    out = pd.DataFrame(records)
    print(f"[Guinea] files={len(files)}, records={len(out)}")
    return out


if __name__ == "__main__":
    sl = consolidate_sierra_leone()
    lb = consolidate_liberia()
    gn = consolidate_guinea()

    for df_, name in [(sl, "Sierra Leone"), (lb, "Liberia"), (gn, "Guinea")]:
        if len(df_) == 0:
            continue
        df_["date"] = pd.to_datetime(df_["date"], errors="coerce", format="mixed")

    panel = pd.concat([sl, lb, gn], ignore_index=True)
    panel = panel.dropna(subset=["date"])

    out_path = f"{OUT_DIR}/west_africa_district_panel_long.csv"
    panel.to_csv(out_path, index=False)
    print(f"\nFinal panel: {len(panel)} records → {out_path}")
    print(f"Date range: {panel['date'].min()} to {panel['date'].max()}")
    print(f"Countries: {panel['country'].unique().tolist()}")
    print(f"Admin units: {panel['admin_unit'].nunique()}")
