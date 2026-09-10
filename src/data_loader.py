"""
data_loader.py
--------------
Loads and cleans CDC Community Profile Report CSV snapshots.

Functions:
    normalize_col_name(c) -- standardize column names across schema versions
    clean_fips(x)         -- parse FIPS to zero-padded 5-digit string
    load_all_csvs(folder) -- load all CSVs into one panel DataFrame
"""

import os
import glob
import numpy as np
import pandas as pd


def normalize_col_name(c: str) -> str:
    """Lowercase, strip and remove special characters from a column name."""
    c = str(c).strip().lower()
    c = c.replace("%", "pct")
    c = c.replace("-", "_").replace(" ", "_").replace("/", "_")
    while "__" in c:
        c = c.replace("__", "_")
    return c.strip("_")


def clean_fips(x) -> str:
    """Parse a FIPS value to a zero-padded 5-digit string, or np.nan."""
    if pd.isna(x):
        return np.nan
    s = "".join(ch for ch in str(x) if ch.isdigit())
    if s == "":
        return np.nan
    return s.zfill(5)[-5:]


_RENAME_MAP = {
    "cases_per_100k_last_7_days":                   "casesper100klast7days",
    "cases_last_7_days":                             "caseslast7days",
    "total_cases":                                   "totalcases",
    "deaths_last_7_days":                            "deathslast7days",
    "deaths_per_100k_last_7_days":                   "deathsper100klast7days",
    "total_deaths":                                  "totaldeaths",
    "test_positivity_rate_last_7_days":              "testpositivityratelast7days",
    "confirmed_covid_hosp_last_7_days":              "confirmedcovidhosplast7days",
    "confirmed_covid_hosp_per_100_beds_last_7_days": "confirmedcovidhospper100bedslast7days",
    "suspected_covid_hosp_last_7_days":              "suspectedcovidhosplast7days",
    "suspected_covid_hosp_per_100_beds_last_7_days": "suspectedcovidhospper100bedslast7days",
    "pct_inpatient_beds_used_avg_last_7_days":       "pctinpatientbedsusedavglast7days",
    "pct_inpatient_beds_used_covid_avg_last_7_days": "pctinpatientbedsusedcovidavglast7days",
    "pct_icu_beds_used_avg_last_7_days":             "pcticubedsusedavglast7days",
    "pct_icu_beds_used_covid_avg_last_7_days":       "pcticubedsusedcovidavglast7days",
}


def load_all_csvs(data_folder: str) -> pd.DataFrame:
    """
    Load all CDC CSV snapshots into a single panel DataFrame.

    Returns DataFrame sorted and deduplicated by (date, fips),
    with standardized column names across all schema versions.
    """
    files = sorted(glob.glob(os.path.join(data_folder, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {data_folder}")

    dfs = []
    for fp in files:
        df = pd.read_csv(fp, dtype={"fips": str}, low_memory=False)
        df.columns = [normalize_col_name(c) for c in df.columns]
        df = df.rename(columns={k: v for k, v in _RENAME_MAP.items()
                                 if k in df.columns})
        if "fips" not in df.columns:
            continue

        df["fips"] = df["fips"].apply(clean_fips)
        df = df[df["fips"].notna()].copy()

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        else:
            stem = os.path.basename(fp).replace(".csv", "").replace("_", "-")
            df["date"] = pd.to_datetime(stem, errors="coerce")

        if "county" not in df.columns:
            df["county"] = None
        if "state" not in df.columns:
            df["state"] = None

        dfs.append(df)

    panel = pd.concat(dfs, ignore_index=True)
    panel = panel[panel["date"].notna()].copy()
    panel["state"]     = panel["state"].astype(str).str.upper().str.strip()
    panel["county"]    = panel["county"].astype(str).str.strip()
    panel["statefips"] = panel["fips"].str[:2]
    panel = panel[
        ~panel["county"].str.contains("unallocated", case=False, na=False)
    ].copy()
    panel = (
        panel
        .sort_values(["date", "fips"])
        .drop_duplicates(subset=["date", "fips"], keep="last")
        .reset_index(drop=True)
    )
    return panel
