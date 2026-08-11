from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

ETH_FEATURE_FILE = Path(
    "data/processed/wallet_features.csv"
)

ERC20_FEATURE_FILE = Path(
    "data/processed/erc20_wallet_features.csv"
)

ZERO_TX_FILE = Path(
    "data/raw/zero_transaction_wallets.csv"
)

OUTPUT_FILE = Path(
    "data/processed/combined_wallet_features_v03.csv"
)


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

ETH_FEATURES = [
    "total_tx_count",
    "incoming_tx_count",
    "outgoing_tx_count",
    "total_eth_received",
    "total_eth_sent",
    "avg_tx_value",
    "max_tx_value",
    "unique_senders",
    "unique_receivers",
    "unique_counterparties",
    "activity_span_days",
    "internal_tx_ratio",
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


ERC20_FEATURES = [
    "erc20_tx_count",
    "erc20_in_count",
    "erc20_out_count",
    "erc20_unique_tokens",
    "erc20_unique_senders",
    "erc20_unique_receivers",
    "erc20_unique_counterparties",
    "erc20_in_out_ratio",
    "erc20_activity_span_days",
    "erc20_tx_frequency",
    "erc20_zero_value_ratio",
    "erc20_counterparty_reuse_ratio",
]


ALL_FEATURES = (
    ETH_FEATURES
    + ERC20_FEATURES
)


# ============================================================
# HELPERS
# ============================================================

def normalize_addresses(df, column="address"):

    df[column] = (
        df[column]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return df


def check_duplicates(df, name):

    duplicates = df[
        "address"
    ].duplicated().sum()

    print(
        f"{name} duplicate addresses: "
        f"{duplicates}"
    )

    if duplicates > 0:

        raise RuntimeError(
            f"{name} contains duplicate addresses."
        )


# ============================================================
# LOAD ETH FEATURES
# ============================================================

print("=" * 70)
print("LOADING ETH FEATURES")
print("=" * 70)

eth = pd.read_csv(
    ETH_FEATURE_FILE
)

eth = normalize_addresses(
    eth
)

check_duplicates(
    eth,
    "ETH feature dataset"
)

print(
    f"ETH wallets    : {len(eth)}"
)

print(
    f"ETH features   : {len(ETH_FEATURES)}"
)


# ============================================================
# LOAD ERC-20 FEATURES
# ============================================================

print()
print("=" * 70)
print("LOADING ERC-20 FEATURES")
print("=" * 70)

erc20 = pd.read_csv(
    ERC20_FEATURE_FILE
)

erc20 = normalize_addresses(
    erc20
)

check_duplicates(
    erc20,
    "ERC-20 feature dataset"
)

print(
    f"ERC-20 wallets : {len(erc20)}"
)

print(
    f"ERC-20 features: {len(ERC20_FEATURES)}"
)


# ============================================================
# LOAD ZERO-TRANSACTION WALLETS
# ============================================================

print()
print("=" * 70)
print("LOADING ZERO-TRANSACTION WALLETS")
print("=" * 70)

zero_tx = pd.read_csv(
    ZERO_TX_FILE
)

# Try to identify the address column robustly.
if "address" in zero_tx.columns:

    zero_addresses = (
        zero_tx["address"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

elif "wallet_address" in zero_tx.columns:

    zero_addresses = (
        zero_tx["wallet_address"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

else:

    # If the file contains a single column,
    # treat that column as addresses.

    if len(zero_tx.columns) == 1:

        zero_addresses = (
            zero_tx.iloc[:, 0]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    else:

        raise RuntimeError(
            "Could not identify address column "
            "in zero_transaction_wallets.csv"
        )


zero_addresses = set(
    zero_addresses
)

print(
    f"Zero-transaction wallets: "
    f"{len(zero_addresses)}"
)


# ============================================================
# CHECK REQUIRED ETH COLUMNS
# ============================================================

missing_eth = [
    col
    for col in ETH_FEATURES
    if col not in eth.columns
]

if missing_eth:

    raise RuntimeError(
        "Missing ETH features: "
        + ", ".join(missing_eth)
    )


# ============================================================
# CHECK REQUIRED ERC-20 COLUMNS
# ============================================================

missing_erc20 = [
    col
    for col in ERC20_FEATURES
    if col not in erc20.columns
]

if missing_erc20:

    raise RuntimeError(
        "Missing ERC-20 features: "
        + ", ".join(missing_erc20)
    )


# ============================================================
# BUILD CLEAN LABEL UNIVERSE
# ============================================================

print()
print("=" * 70)
print("BUILDING CLEAN LABEL UNIVERSE")
print("=" * 70)

# ERC-20 dataset contains the cleaned label universe:
# 834 unique wallets after rugpull removal.

labels = erc20[
    [
        "address",
        "label",
        "type",
    ]
].copy()

labels = normalize_addresses(
    labels
)

check_duplicates(
    labels,
    "Label universe"
)

print(
    f"Clean labeled wallets: "
    f"{len(labels)}"
)

print()
print(
    labels[
        "label"
    ].value_counts()
    .sort_index()
)


# ============================================================
# REMOVE ZERO-ACTIVITY WALLETS
# ============================================================

print()
print("=" * 70)
print("REMOVING ZERO-TRANSACTION WALLETS")
print("=" * 70)

before = len(labels)

labels = labels[
    ~labels["address"].isin(
        zero_addresses
    )
].copy()

removed = (
    before
    - len(labels)
)

print(
    f"Before : {before}"
)

print(
    f"Removed: {removed}"
)

print(
    f"Active : {len(labels)}"
)


# ============================================================
# MERGE ETH FEATURES
# ============================================================

print()
print("=" * 70)
print("MERGING ETH FEATURES")
print("=" * 70)

combined = labels.merge(
    eth[
        [
            "address"
        ]
        + ETH_FEATURES
    ],
    on="address",
    how="left",
)

eth_matches = (
    combined[
        ETH_FEATURES[0]
    ].notna().sum()
)

print(
    f"ETH feature matches: "
    f"{eth_matches}/{len(combined)}"
)


# ============================================================
# MERGE ERC-20 FEATURES
# ============================================================

print()
print("=" * 70)
print("MERGING ERC-20 FEATURES")
print("=" * 70)

combined = combined.merge(
    erc20[
        [
            "address"
        ]
        + ERC20_FEATURES
    ],
    on="address",
    how="left",
    suffixes=(
        "",
        "_erc20",
    ),
)


# ============================================================
# HANDLE MISSING MODALITY
# ============================================================

# A missing ETH or ERC-20 feature vector means that
# the wallet has no observed activity in that modality.
# These are represented as zeros.

combined[
    ETH_FEATURES
] = combined[
    ETH_FEATURES
].fillna(0)

combined[
    ERC20_FEATURES
] = combined[
    ERC20_FEATURES
].fillna(0)


# ============================================================
# FINAL COLUMN ORDER
# ============================================================

combined = combined[
    [
        "address",
        "label",
        "type",
    ]
    + ALL_FEATURES
]


# ============================================================
# VALIDATION
# ============================================================

print()
print("=" * 70)
print("COMBINED DATASET VALIDATION")
print("=" * 70)


# ------------------------------------------------------------
# Wallet count
# ------------------------------------------------------------

print(
    f"Wallets : {len(combined)}"
)

if len(combined) != 812:

    raise RuntimeError(
        f"Expected 812 active wallets, "
        f"got {len(combined)}"
    )


# ------------------------------------------------------------
# Feature count
# ------------------------------------------------------------

print(
    f"Features: {len(ALL_FEATURES)}"
)

if len(ALL_FEATURES) != 38:

    raise RuntimeError(
        "Expected exactly 38 features."
    )


# ------------------------------------------------------------
# Duplicate addresses
# ------------------------------------------------------------

check_duplicates(
    combined,
    "Combined dataset"
)


# ------------------------------------------------------------
# NaN
# ------------------------------------------------------------

nan_count = (
    combined[
        ALL_FEATURES
    ]
    .isna()
    .sum()
    .sum()
)

print(
    f"NaN values: {nan_count}"
)

if nan_count > 0:

    raise RuntimeError(
        "NaN values detected."
    )


# ------------------------------------------------------------
# Infinity
# ------------------------------------------------------------

numeric = combined[
    ALL_FEATURES
].select_dtypes(
    include=np.number
)

infinite_count = np.isinf(
    numeric.to_numpy()
).sum()

print(
    f"Infinite values: "
    f"{infinite_count}"
)

if infinite_count > 0:

    raise RuntimeError(
        "Infinite values detected."
    )


# ------------------------------------------------------------
# Label distribution
# ------------------------------------------------------------

print()
print("LABEL DISTRIBUTION")

print(
    combined[
        "label"
    ].value_counts()
    .sort_index()
)


# ------------------------------------------------------------
# Activity coverage
# ------------------------------------------------------------

eth_active = (
    combined["total_tx_count"] > 0
)

erc20_active = (
    combined["erc20_tx_count"] > 0
)

# Keep wallets with observable activity
# in at least one transaction modality.
combined = combined[
    eth_active | erc20_active
].copy()

# Recalculate masks after filtering
eth_active = (
    combined["total_tx_count"] > 0
)

erc20_active = (
    combined["erc20_tx_count"] > 0
)

print()
print("ACTIVITY COVERAGE")

print(
    f"ETH active       : "
    f"{eth_active.sum()}"
)

print(
    f"ERC-20 active    : "
    f"{erc20_active.sum()}"
)

print(
    f"Both active      : "
    f"{(eth_active & erc20_active).sum()}"
)

print(
    f"ETH only         : "
    f"{(eth_active & ~erc20_active).sum()}"
)

print(
    f"ERC-20 only      : "
    f"{(~eth_active & erc20_active).sum()}"
)

print(
    f"Neither          : "
    f"{(~eth_active & ~erc20_active).sum()}"
)

inactive = combined[
    (combined["total_tx_count"] <= 0)
    &
    (combined["erc20_tx_count"] <= 0)
].copy()

print()
print("=" * 70)
print("INACTIVE WALLETS AFTER ACTIVITY FILTER")
print("=" * 70)

print(
    f"Count: {len(inactive)}"
)

if len(inactive) > 0:
    print(
        inactive[
            [
                "address",
                "label",
                "type",
                "total_tx_count",
                "erc20_tx_count",
            ]
        ].to_string(index=False)
    )

activity_union = (
    (combined["total_tx_count"] > 0)
    | (combined["erc20_tx_count"] > 0)
).sum()

print(f"Activity union: {activity_union}")
print(f"Total wallets : {len(combined)}")

if activity_union != len(combined):
    raise RuntimeError(
        "Activity union does not equal wallet count."
    )

# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

combined.to_csv(
    OUTPUT_FILE,
    index=False,
)

print()
print("=" * 70)
print("COMBINED FEATURE DATASET SAVED")
print("=" * 70)

print(
    OUTPUT_FILE
)