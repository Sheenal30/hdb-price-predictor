"""
Phase 1.2 – Clean the raw HDB resale data
----------------------------------------
Input : data/raw/hdb_resale_raw.csv   (~900 k rows)
Output: data/processed/clean_hdb.csv  (ready for EDA / modelling)
"""

if __name__ == "__main__":

    # ─── 1. Imports ──────────────────────────────────────────────────
    import pandas as pd
    from pathlib import Path

    # ─── 2. Paths (work from anywhere) ───────────────────────────────
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    RAW_PATH  = PROJECT_ROOT / "data/raw/hdb_resale_raw.csv"
    OUT_DIR   = PROJECT_ROOT / "data/processed"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE  = OUT_DIR / "clean_hdb.csv"

    # ─── 3. Load ─────────────────────────────────────────────────────
    df = pd.read_csv(RAW_PATH)
    print(f"Loaded {len(df):,} rows · {df.shape[1]} columns")

    # ─── 4. Basic cleaning ───────────────────────────────────────────
    df.columns = df.columns.str.strip()          # trim spaces
    df["sale_date"]  = pd.to_datetime(df["month"], format="%Y-%m")
    df["sale_year"]  = df["sale_date"].dt.year
    df["sale_month"] = df["sale_date"].dt.month
    df = df.drop_duplicates()

    num_cols = ["floor_area_sqm", "resale_price", "lease_commence_date"]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    # ─── 5. Feature engineering ──────────────────────────────────────
    df["price_per_sqm"] = df["resale_price"] / df["floor_area_sqm"]

    # remaining_lease text → numeric years
    df["lease_remaining_years"] = (
        df["remaining_lease"].str.extract(r"(\d+)").astype(float)
    )
    df["flat_age"] = df["sale_year"] - df["lease_commence_date"]
    df.loc[df["flat_age"] < 0, "flat_age"] = None

    df["lease_remaining_years"] = (
        df["lease_remaining_years"]
          .fillna(99 - df["flat_age"])
          .clip(lower=0, upper=99)
    )

    df = df[df["resale_price"] >= 20_000]        # drop obvious typos

    # ─── 6. One-hot dummies ──────────────────────────────────────────
    dummies = pd.get_dummies(df[["town", "flat_type"]], drop_first=True)
    df = pd.concat([df, dummies], axis=1)

    # ─── 7. Column housekeeping ─────────────────────────────────────
    # 7-a  drop original sparse text col
    df = df.drop(columns=["remaining_lease"])

    # 7-b  snake_case *everything* (handles new dummy cols)
    df.columns = (
        df.columns
          .str.lower()
          .str.replace(r"[ /\-]", "_", regex=True)
          .str.replace(r"__+", "_", regex=True)   # collapse doubles
    )

    # 7-c  merge any duplicate cols created by renaming (e.g. multi_generation)
    def merge_dupes(frame: pd.DataFrame) -> pd.DataFrame:
        merged = {}
        for col in frame.columns.unique():
            subset = frame.filter(regex=f"^{col}$")
            if subset.shape[1] == 1:
                merged[col] = subset.iloc[:, 0]
            elif subset.dtypes.eq(bool).all():
                merged[col] = subset.any(axis=1)          # logical OR
            else:
                merged[col] = subset.iloc[:, 0]           # keep first
        return pd.DataFrame(merged)

    df = merge_dupes(df)

    # ─── 8. Save ─────────────────────────────────────────────────────
    df.to_csv(OUT_FILE, index=False)
    print(f"✅  Saved cleaned file → {OUT_FILE}\nFinal shape: {df.shape}")