from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering import (
    compute_features,
    FEATURE_COLUMNS,
)


# ============================================================
# FILES
# ============================================================

TRANSACTIONS_FILE = Path(
    "data/raw/transactions.csv"
)

MALICIOUS_FILE = Path(
    "data/labels/malicious.csv"
)

BENIGN_FILE = Path(
    "data/labels/benign.csv"
)

OUTPUT_FILE = Path(
    "data/processed/wallet_features.csv"
)


# ============================================================
# LOAD LABELS
# ============================================================

print("=" * 70)
print("LOADING LABELS")
print("=" * 70)

malicious = pd.read_csv(
    MALICIOUS_FILE
)

benign = pd.read_csv(
    BENIGN_FILE
)

print(
    f"Malicious before rugpull removal: "
    f"{len(malicious)}"
)


# ============================================================
# REMOVE RUGPULLS
# ============================================================

# Rugpull dataset contains an uncertain mixture
# of contract addresses and EOAs, so exclude it
# from the MVP.

malicious = malicious[
    malicious["type"]
    .astype(str)
    .str.strip()
    .str.lower()
    != "rugpull"
].copy()


print(
    f"Malicious after rugpull removal : "
    f"{len(malicious)}"
)


# ============================================================
# COMBINE LABELS
# ============================================================

labels = pd.concat(
    [
        malicious,
        benign,
    ],
    ignore_index=True,
)


labels["address"] = (
    labels["address"]
    .astype(str)
    .str.strip()
    .str.lower()
)


labels = labels.drop_duplicates(
    subset=["address"]
)


print(
    f"Unique labeled wallets          : "
    f"{len(labels)}"
)


# ============================================================
# LOAD TRANSACTIONS
# ============================================================

print()
print("=" * 70)
print("LOADING TRANSACTIONS")
print("=" * 70)

tx = pd.read_csv(
    TRANSACTIONS_FILE
)

print(
    f"Transactions: {len(tx):,}"
)


# ============================================================
# COMPUTE FEATURES
# ============================================================

print()
print("=" * 70)
print("BUILDING FEATURES — MVP v0.2")
print("=" * 70)

print(
    "Computing reusable transaction features..."
)


all_features = compute_features(
    tx
)


print(
    f"All wallets with transactions: "
    f"{len(all_features)}"
)


# ============================================================
# KEEP ONLY LABELED WALLETS
# ============================================================

labeled_addresses = set(
    labels["address"]
)


df = all_features[
    all_features["address"]
    .isin(labeled_addresses)
].copy()


print(
    f"Labeled wallets with transactions: "
    f"{len(df)}"
)


# ============================================================
# ADD LABEL INFORMATION
# ============================================================

label_lookup = (
    labels[
        [
            "address",
            "type",
            "label",
        ]
    ]
    .drop_duplicates(
        subset=["address"]
    )
)


df = df.merge(
    label_lookup,
    on="address",
    how="inner",
)


# ============================================================
# ORDER COLUMNS
# ============================================================

df = df[
    [
        "address",
        "type",
        "label",
        *FEATURE_COLUMNS,
    ]
]


# ============================================================
# VALIDATION
# ============================================================

print()
print("=" * 70)
print("FEATURE DATASET — MVP v0.2")
print("=" * 70)

print(
    f"Wallets  : {len(df)}"
)

print(
    f"Features : {len(FEATURE_COLUMNS)}"
)


# ------------------------------------------------------------
# Replace accidental infinities
# ------------------------------------------------------------

df[FEATURE_COLUMNS] = (
    df[FEATURE_COLUMNS]
    .replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )
)


print()
print("NaN values:")

nan_counts = (
    df[FEATURE_COLUMNS]
    .isna()
    .sum()
)

problem_nan = nan_counts[
    nan_counts > 0
]


if len(problem_nan):

    print(problem_nan)

else:

    print("None")


# ------------------------------------------------------------
# Fill unexpected NaNs
# ------------------------------------------------------------

df[FEATURE_COLUMNS] = (
    df[FEATURE_COLUMNS]
    .fillna(0.0)
)


numeric = (
    df[FEATURE_COLUMNS]
    .to_numpy(
        dtype=float
    )
)


print()

print(
    "Infinite values:",
    np.isinf(numeric).sum(),
)


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print()
print("LABEL DISTRIBUTION")
print("-" * 70)

print(
    df["label"]
    .value_counts()
    .sort_index()
)


# ============================================================
# MALICIOUS TYPES
# ============================================================

print()
print("MALICIOUS TYPES")
print("-" * 70)

print(
    df[
        df["label"] == 1
    ]["type"]
    .value_counts()
)


# ============================================================
# NEW FEATURE SUMMARY
# ============================================================

NEW_FEATURES = [

    "in_out_tx_ratio",

    "net_eth_flow",

    "median_tx_value",

    "std_tx_value",

    "distinct_active_days",

    "tx_frequency",

    "incoming_value_ratio",

    "zero_value_tx_ratio",

    "external_tx_ratio",

    "counterparty_reuse_ratio",

    "avg_time_between_tx",

    "median_time_between_tx",

    "max_time_between_tx",

    "burstiness",
]


print()
print("NEW FEATURE SUMMARY")
print("-" * 70)

print(
    df[
        NEW_FEATURES
    ]
    .describe()
    .T[
        [
            "min",
            "mean",
            "50%",
            "max",
        ]
    ]
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


df.to_csv(
    OUTPUT_FILE,
    index=False,
)


print()
print("=" * 70)

print(
    f"Saved {len(df)} wallets "
    f"with {len(FEATURE_COLUMNS)} features"
)

print(
    f"to: {OUTPUT_FILE}"
)

print("=" * 70)