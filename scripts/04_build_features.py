from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# FILES
# ============================================================

TRANSACTIONS_FILE = Path("data/raw/transactions.csv")
MALICIOUS_FILE = Path("data/labels/malicious.csv")
BENIGN_FILE = Path("data/labels/benign.csv")

OUTPUT_FILE = Path("data/processed/wallet_features.csv")


# ============================================================
# LOAD LABELS
# ============================================================

print("Loading labels...")

malicious = pd.read_csv(MALICIOUS_FILE)
benign = pd.read_csv(BENIGN_FILE)

print(
    f"Malicious before rugpull removal: "
    f"{len(malicious)}"
)

# REMOVE RUGPULL FIRST
malicious = malicious[
    malicious["type"]
    .astype(str)
    .str.strip()
    .str.lower()
    != "rugpull"
].copy()

print(
    f"Malicious after rugpull removal: "
    f"{len(malicious)}"
)

# ONLY NOW combine the datasets
labels = pd.concat(
    [malicious, benign],
    ignore_index=True
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
    f"Unique labeled wallets: "
    f"{len(labels)}"
)

# ONLY NOW create label_map
label_map = (
    labels
    .set_index("address")[["type", "label"]]
    .to_dict("index")
)


# ============================================================
# REMOVE RUGPULL ADDRESSES
# ============================================================

print(f"Malicious before rugpull removal: {len(malicious)}")

malicious = malicious[
    malicious["type"].str.lower() != "rugpull"
].copy()

print(f"Malicious after rugpull removal: {len(malicious)}")


# ============================================================
# LOAD TRANSACTIONS
# ============================================================

print("Loading transactions...")

tx = pd.read_csv(TRANSACTIONS_FILE)

print(f"Transactions: {len(tx):,}")

for col in [
    "wallet_address",
    "from_address",
    "to_address"
]:
    tx[col] = (
        tx[col]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )


# ============================================================
# CLEAN VALUE
# ============================================================

# Alchemy asset-transfer `value` is already represented
# as the transferred asset amount. For external/internal
# ETH transfers, do NOT divide it by 1e18 again.

tx["value"] = pd.to_numeric(
    tx["value"],
    errors="coerce"
)

invalid_values = tx["value"].isna().sum()

if invalid_values:
    print(
        f"WARNING: {invalid_values} transactions "
        "have invalid value; replacing with 0."
    )

tx["value"] = (
    tx["value"]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0.0)
)

# We are only using external + internal in MVP v0.1.
unexpected_categories = set(
    tx["category"].dropna().unique()
) - {"external", "internal"}

if unexpected_categories:
    print(
        "WARNING: unexpected categories:",
        unexpected_categories
    )


# ============================================================
# CLEAN TIMESTAMP
# ============================================================

tx["timestamp"] = pd.to_datetime(
    tx["timestamp"],
    errors="coerce",
    utc=True
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

print()
print("Building wallet features...")

features = []

wallet_groups = tx.groupby(
    "wallet_address",
    sort=False
)

for i, (wallet, group) in enumerate(
    wallet_groups,
    start=1
):

    # Ignore transactions attached to addresses outside
    # our labeled dataset.
    if wallet not in label_map:
        continue

    incoming = group[
        group["direction"] == "incoming"
    ]

    outgoing = group[
        group["direction"] == "outgoing"
    ]

    # --------------------------------------------------------
    # Transaction counts
    # --------------------------------------------------------

    total_tx_count = len(group)
    incoming_tx_count = len(incoming)
    outgoing_tx_count = len(outgoing)

    # --------------------------------------------------------
    # ETH flow
    # --------------------------------------------------------

    total_eth_received = incoming["value"].sum()
    total_eth_sent = outgoing["value"].sum()

    avg_tx_value = group["value"].mean()
    max_tx_value = group["value"].max()

    # --------------------------------------------------------
    # Counterparties
    # --------------------------------------------------------

    senders = set(
        incoming["from_address"]
        .replace("", np.nan)
        .dropna()
    )

    receivers = set(
        outgoing["to_address"]
        .replace("", np.nan)
        .dropna()
    )

    # Avoid counting the wallet itself as a counterparty.
    senders.discard(wallet)
    receivers.discard(wallet)

    unique_senders = len(senders)
    unique_receivers = len(receivers)

    unique_counterparties = len(
        senders | receivers
    )

    # --------------------------------------------------------
    # Active days
    # --------------------------------------------------------

    valid_times = (
        group["timestamp"]
        .dropna()
        .sort_values()
    )

    if len(valid_times) >= 2:

        delta = (
            valid_times.iloc[-1]
            - valid_times.iloc[0]
        )

        # +1 means activity occurring within a single
        # calendar day is represented as one active-day span.
        active_days = delta.days + 1

    elif len(valid_times) == 1:

        active_days = 1

    else:

        active_days = 0

    # --------------------------------------------------------
    # Internal transaction ratio
    # --------------------------------------------------------

    internal_tx_count = (
        group["category"]
        .eq("internal")
        .sum()
    )

    internal_tx_ratio = (
        internal_tx_count / total_tx_count
        if total_tx_count > 0
        else 0
    )

    # --------------------------------------------------------
    # LABEL
    # --------------------------------------------------------

    wallet_info = label_map[wallet]

    features.append({

        "address":
            wallet,

        "type":
            wallet_info["type"],

        "label":
            wallet_info["label"],

        "total_tx_count":
            total_tx_count,

        "incoming_tx_count":
            incoming_tx_count,

        "outgoing_tx_count":
            outgoing_tx_count,

        "total_eth_received":
            total_eth_received,

        "total_eth_sent":
            total_eth_sent,

        "avg_tx_value":
            avg_tx_value,

        "max_tx_value":
            max_tx_value,

        "unique_senders":
            unique_senders,

        "unique_receivers":
            unique_receivers,

        "unique_counterparties":
            unique_counterparties,

        "active_days":
            active_days,

        "internal_tx_ratio":
            internal_tx_ratio
    })

    if i % 100 == 0:

        print(
            f"Processed {i} wallet groups..."
        )


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(features)

print()
print("=" * 60)
print("FEATURE DATASET")
print("=" * 60)

print(f"Wallets: {len(df)}")


# ============================================================
# VALIDATION
# ============================================================

feature_columns = [

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

    "active_days",
    "internal_tx_ratio"
]


print()
print("Checking NaN values...")

nan_counts = (
    df[feature_columns]
    .isna()
    .sum()
)

print(nan_counts)


print()
print("Checking infinite values...")

numeric = df[
    feature_columns
].to_numpy(dtype=float)

inf_count = np.isinf(
    numeric
).sum()

print(
    f"Infinite values: {inf_count}"
)


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print()
print("LABEL DISTRIBUTION")
print("-" * 60)

print(
    df["label"]
    .value_counts()
    .sort_index()
)


print()
print("MALICIOUS TYPES")
print("-" * 60)

print(
    df[
        df["label"] == 1
    ]["type"]
    .value_counts()
)


# ============================================================
# FEATURE SUMMARY
# ============================================================

print()
print("FEATURE SUMMARY")
print("-" * 60)

print(
    df[feature_columns]
    .describe()
    .T[
        [
            "min",
            "mean",
            "50%",
            "max"
        ]
    ]
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("=" * 60)

print(
    f"Saved features to:"
)

print(
    OUTPUT_FILE
)

print("=" * 60)