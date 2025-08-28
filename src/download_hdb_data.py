"""
HDB Resale Data Downloader

Downloading every resale record (1990-present) from data.gov.sg.
Writing one CSV to <repo>/data/raw/hdb_resale_raw.csv.
Paths auto-resolve, allowing script to be run from anywhere.
"""

# 1. Imports
import time
import requests
import pandas as pd
from pathlib import Path

# 2. Configuration
RESOURCE_IDS = [
    "adbbddd3-30e2-445f-a123-29bee150a6fe",  # 1990-1999
    "8c00bf08-9124-479e-aeca-7cc411d884c4",  # 2000-2012
    "83b2fc37-ce8c-4df4-968b-370fd818138b",  # 2012-2014
    "1b702208-44bf-4829-b620-4615ee19b57c",  # 2015-2016
    "f1765b54-a209-4718-8d38-a39237f502b3",  # 2017-present
]

API_URL   = "https://data.gov.sg/api/action/datastore_search"
PAGE_SIZE = 10_000            # starting small to dodge 413 errors

# 3. Helper: downloading one resource_id page-by-page
def download_resource(rid, page_size=PAGE_SIZE):
    """Fetching all rows for a single resource_id, shrinking page_size if 413."""
    print(f"⇣  Downloading {rid}")
    offset, frames = 0, []

    while True:
        try:
            resp = requests.get(
                API_URL,
                params={"resource_id": rid, "limit": page_size, "offset": offset},
                timeout=30,
            )
            resp.raise_for_status()

        except requests.exceptions.HTTPError as err:
            # 413 Payload Too Large error, halving page_size and retrying
            if resp.status_code == 413 and page_size > 1_000:
                page_size //= 2
                print(f"   413 hit, retrying with page_size={page_size}")
                time.sleep(1)
                continue
            raise err  # any other HTTP error stops the script

        rows = resp.json()["result"]["records"]
        if not rows:                      # empty list means no more data
            break

        frames.append(pd.DataFrame(rows))
        offset += page_size
        if offset % 50_000 == 0:
            print(f"   …{offset:,} rows so far")

    return pd.concat(frames, ignore_index=True)

# 4. Main: only runs when called directly
if __name__ == "__main__":
    # Locating the repo root (one level above src/)
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    RAW_DIR  = PROJECT_ROOT / "data/raw"
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Downloading all five slices
    all_dfs = [download_resource(rid) for rid in RESOURCE_IDS]
    full_df = pd.concat(all_dfs, ignore_index=True)

    # Saving to <repo>/data/raw/hdb_resale_raw.csv
    out_path = RAW_DIR / "hdb_resale_raw.csv"
    full_df.to_csv(out_path, index=False)

    print(f"\n Saved {len(full_df):,} rows → {out_path.resolve()}")