import pandas as pd
from pathlib import Path


CURRENT_FILE = Path(
    "data/labels/malicious.csv"
)

EXTERNAL_FILE = Path(
    "data/labels/external_dodgy_wallets.csv"
)


# ============================================================
# LOAD
# ============================================================

current = pd.read_csv(CURRENT_FILE)
external = pd.read_csv(EXTERNAL_FILE)

print(f"Current malicious : {len(current):,}")
print(f"External malicious: {len(external):,}")


# ============================================================
# NORMALIZE
# ============================================================

for df in [current, external]:

    df["address"] = (
        df["address"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["type"] = (
        df["type"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["label"] = 1


# Keep only the standard columns
current = current[
    ["address", "type", "label"]
]

external = external[
    ["address", "type", "label"]
]


# ============================================================
# CHECK OVERLAP
# ============================================================

overlap = (
    set(current["address"])
    &
    set(external["address"])
)

print(f"Overlap: {len(overlap):,}")


# ============================================================
# MERGE
# ============================================================

merged = pd.concat(
    [current, external],
    ignore_index=True,
)


# Safety: one address should appear only once
merged = merged.drop_duplicates(
    subset=["address"],
    keep="first",
)


# ============================================================
# SAVE
# ============================================================

merged.to_csv(
    CURRENT_FILE,
    index=False,
)


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 60)
print("MERGE COMPLETE")
print("=" * 60)

print(f"Final malicious wallets: {len(merged):,}")

print("\nTYPE DISTRIBUTION:")
print(
    merged["type"]
    .value_counts()
    .to_string()
)

print(f"\nSaved to: {CURRENT_FILE}")