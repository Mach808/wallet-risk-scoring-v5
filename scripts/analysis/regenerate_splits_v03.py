from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = (
    ROOT
    / "data"
    / "processed"
    / "combined_wallet_features_v03.csv"
)

SPLIT_DIR = (
    ROOT
    / "data"
    / "splits"
)

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING v0.3 LABELED DATASET")
print("=" * 70)

df = pd.read_csv(
    FEATURE_FILE
)

df["address"] = (
    df["address"]
    .astype(str)
    .str.strip()
    .str.lower()
)

print(
    f"Wallets: {len(df)}"
)

print(
    f"Unique addresses: "
    f"{df['address'].nunique()}"
)

print()
print("Label distribution:")
print(
    df["label"]
    .value_counts()
    .sort_index()
)


# ============================================================
# VALIDATION
# ============================================================

if len(df) != 810:
    raise RuntimeError(
        f"Expected 810 wallets, found {len(df)}"
    )

if df["address"].duplicated().any():
    raise RuntimeError(
        "Duplicate addresses detected."
    )

if set(df["label"].unique()) != {0, 1}:
    raise RuntimeError(
        "Expected labels 0 and 1 only."
    )


# ============================================================
# 70 / 15 / 15 STRATIFIED SPLIT
# ============================================================

print()
print("=" * 70)
print("CREATING 70 / 15 / 15 SPLIT")
print("=" * 70)

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=RANDOM_STATE,
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=RANDOM_STATE,
)


# ============================================================
# DISPLAY SPLITS
# ============================================================

print(
    f"Train      : {len(train_df)}"
)

print(
    f"Validation : {len(val_df)}"
)

print(
    f"Test       : {len(test_df)}"
)

print()
print("Train labels:")
print(
    train_df["label"]
    .value_counts()
    .sort_index()
)

print()
print("Validation labels:")
print(
    val_df["label"]
    .value_counts()
    .sort_index()
)

print()
print("Test labels:")
print(
    test_df["label"]
    .value_counts()
    .sort_index()
)


# ============================================================
# SAVE
# ============================================================

SPLIT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

train_df[
    ["address"]
].to_csv(
    SPLIT_DIR
    / "train_addresses.csv",
    index=False,
)

val_df[
    ["address"]
].to_csv(
    SPLIT_DIR
    / "val_addresses.csv",
    index=False,
)

test_df[
    ["address"]
].to_csv(
    SPLIT_DIR
    / "test_addresses.csv",
    index=False,
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print()
print("=" * 70)
print("SPLIT VALIDATION")
print("=" * 70)

train_addresses = set(
    train_df["address"]
)

val_addresses = set(
    val_df["address"]
)

test_addresses = set(
    test_df["address"]
)

all_split_addresses = (
    train_addresses
    | val_addresses
    | test_addresses
)

print(
    f"Train      : {len(train_addresses)}"
)

print(
    f"Validation : {len(val_addresses)}"
)

print(
    f"Test       : {len(test_addresses)}"
)

print(
    f"Combined   : {len(all_split_addresses)}"
)

print(
    f"Original   : {len(df)}"
)


# No overlap

if train_addresses & val_addresses:
    raise RuntimeError(
        "Train/validation overlap detected."
    )

if train_addresses & test_addresses:
    raise RuntimeError(
        "Train/test overlap detected."
    )

if val_addresses & test_addresses:
    raise RuntimeError(
        "Validation/test overlap detected."
    )


# Every wallet must appear

if all_split_addresses != set(
    df["address"]
):

    missing = (
        set(df["address"])
        - all_split_addresses
    )

    extra = (
        all_split_addresses
        - set(df["address"])
    )

    raise RuntimeError(
        "Split universe mismatch.\n"
        f"Missing: {missing}\n"
        f"Extra: {extra}"
    )


print()
print("Duplicate addresses : 0")
print("Split overlap       : 0")
print("Missing wallets     : 0")
print("✓ SPLIT VALIDATION PASSED")

print()
print("Saved:")
print(
    SPLIT_DIR
    / "train_addresses.csv"
)
print(
    SPLIT_DIR
    / "val_addresses.csv"
)
print(
    SPLIT_DIR
    / "test_addresses.csv"
)