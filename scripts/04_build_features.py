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

print("=" * 70)
print("LOADING LABELS")
print("=" * 70)

malicious = pd.read_csv(MALICIOUS_FILE)
benign = pd.read_csv(BENIGN_FILE)

print(f"Malicious before rugpull removal: {len(malicious)}")

# Rugpull dataset contains an uncertain mixture of
# contract addresses and EOAs, so exclude it from MVP.
malicious = malicious[
    malicious["type"]
    .astype(str)
    .str.strip()
    .str.lower()
    != "rugpull"
].copy()

print(f"Malicious after rugpull removal : {len(malicious)}")

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

print(f"Unique labeled wallets          : {len(labels)}")

label_map = (
    labels
    .set_index("address")[["type", "label"]]
    .to_dict("index")
)


# ============================================================
# LOAD TRANSACTIONS
# ============================================================

print()
print("=" * 70)
print("LOADING TRANSACTIONS")
print("=" * 70)

tx = pd.read_csv(TRANSACTIONS_FILE)

print(f"Transactions: {len(tx):,}")

for col in [
    "wallet_address",
    "from_address",
    "to_address",
]:

    tx[col] = (
        tx[col]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )


# ============================================================
# CLEAN VALUES
# ============================================================

tx["value"] = pd.to_numeric(
    tx["value"],
    errors="coerce"
)

invalid_values = tx["value"].isna().sum()

if invalid_values:

    print(
        f"WARNING: {invalid_values} invalid values. "
        "Replacing with 0."
    )

tx["value"] = (
    tx["value"]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .fillna(0.0)
)


# ============================================================
# CLEAN TIMESTAMPS
# ============================================================

tx["timestamp"] = pd.to_datetime(
    tx["timestamp"],
    errors="coerce",
    utc=True
)


# ============================================================
# BUILD FEATURES
# ============================================================

print()
print("=" * 70)
print("BUILDING FEATURES — MVP v0.2")
print("=" * 70)

features = []

wallet_groups = tx.groupby(
    "wallet_address",
    sort=False
)


for i, (wallet, group) in enumerate(
    wallet_groups,
    start=1
):

    if wallet not in label_map:
        continue

    incoming = group[
        group["direction"] == "incoming"
    ]

    outgoing = group[
        group["direction"] == "outgoing"
    ]

    # ========================================================
    # 1–3. TRANSACTION COUNTS
    # ========================================================

    total_tx_count = len(group)

    incoming_tx_count = len(
        incoming
    )

    outgoing_tx_count = len(
        outgoing
    )


    # ========================================================
    # 4–7. VALUE FEATURES
    # ========================================================

    total_eth_received = float(
        incoming["value"].sum()
    )

    total_eth_sent = float(
        outgoing["value"].sum()
    )

    avg_tx_value = float(
        group["value"].mean()
    )

    max_tx_value = float(
        group["value"].max()
    )


    # ========================================================
    # 8–10. COUNTERPARTIES
    # ========================================================

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

    senders.discard(wallet)
    receivers.discard(wallet)

    unique_senders = len(
        senders
    )

    unique_receivers = len(
        receivers
    )

    unique_counterparties = len(
        senders | receivers
    )


    # ========================================================
    # TIMESTAMP PREPARATION
    # ========================================================

    valid_times = (
        group["timestamp"]
        .dropna()
        .sort_values()
    )


    # ========================================================
    # 11. ACTIVITY SPAN
    # ========================================================

    if len(valid_times) >= 2:

        activity_span_days = (
            valid_times.iloc[-1]
            - valid_times.iloc[0]
        ).total_seconds() / 86400

        # Minimum one day for our frequency calculation.
        activity_span_days = max(
            activity_span_days,
            1.0
        )

    elif len(valid_times) == 1:

        activity_span_days = 1.0

    else:

        activity_span_days = 0.0


    # ========================================================
    # 12. INTERNAL TX RATIO
    # ========================================================

    internal_tx_count = int(
        group["category"]
        .eq("internal")
        .sum()
    )

    internal_tx_ratio = (
        internal_tx_count
        / total_tx_count
        if total_tx_count > 0
        else 0.0
    )


    # ========================================================
    # NEW v0.2 FEATURES
    # ========================================================


    # ========================================================
    # 13. IN / OUT TX RATIO
    # ========================================================

    in_out_tx_ratio = (
        incoming_tx_count
        / (outgoing_tx_count + 1)
    )


    # ========================================================
    # 14. NET ETH FLOW
    #
    # Positive → net receiver
    # Negative → net sender
    # ========================================================

    net_eth_flow = (
        total_eth_received
        - total_eth_sent
    )


    # ========================================================
    # 15. MEDIAN TX VALUE
    # ========================================================

    median_tx_value = float(
        group["value"].median()
    )


    # ========================================================
    # 16. TX VALUE STANDARD DEVIATION
    # ========================================================

    std_tx_value = float(
        group["value"].std(ddof=0)
    )

    if np.isnan(std_tx_value):
        std_tx_value = 0.0


    # ========================================================
    # 17. DISTINCT ACTIVE DAYS
    #
    # Unlike activity_span_days, this counts actual calendar
    # days on which activity occurred.
    # ========================================================

    if len(valid_times) > 0:

        distinct_active_days = (
            valid_times
            .dt.date
            .nunique()
        )

    else:

        distinct_active_days = 0


    # ========================================================
    # 18. TX FREQUENCY
    #
    # Transactions per day across observed activity span.
    # ========================================================

    tx_frequency = (
        total_tx_count
        / activity_span_days
        if activity_span_days > 0
        else 0.0
    )


    # ========================================================
    # 19. INCOMING VALUE RATIO
    # ========================================================

    total_flow = (
        total_eth_received
        + total_eth_sent
    )

    incoming_value_ratio = (
        total_eth_received
        / total_flow
        if total_flow > 0
        else 0.0
    )


    # ========================================================
    # 20. ZERO VALUE TX RATIO
    # ========================================================

    zero_value_count = int(
        group["value"]
        .eq(0)
        .sum()
    )

    zero_value_tx_ratio = (
        zero_value_count
        / total_tx_count
        if total_tx_count > 0
        else 0.0
    )


    # ========================================================
    # 21. EXTERNAL TX RATIO
    # ========================================================

    external_tx_count = int(
        group["category"]
        .eq("external")
        .sum()
    )

    external_tx_ratio = (
        external_tx_count
        / total_tx_count
        if total_tx_count > 0
        else 0.0
    )


    # ========================================================
    # 22. COUNTERPARTY REUSE RATIO
    #
    # Higher value means repeated interaction with the same
    # counterparties.
    # ========================================================

    counterparty_reuse_ratio = (
        total_tx_count
        / unique_counterparties
        if unique_counterparties > 0
        else 0.0
    )


    # ========================================================
    # TEMPORAL FEATURES
    #
    # Calculate time between consecutive transactions.
    # Values are stored in HOURS.
    # ========================================================

    if len(valid_times) >= 2:

        timestamps_seconds = (
            valid_times
            .astype("int64")
            .to_numpy()
            / 1e9
        )

        time_diffs_seconds = np.diff(
            timestamps_seconds
        )

        time_diffs_hours = (
            time_diffs_seconds
            / 3600
        )

    else:

        time_diffs_hours = np.array(
            [],
            dtype=float
        )


    # ========================================================
    # 23. AVG TIME BETWEEN TX
    # ========================================================

    if len(time_diffs_hours) > 0:

        avg_time_between_tx = float(
            np.mean(
                time_diffs_hours
            )
        )

    else:

        avg_time_between_tx = 0.0


    # ========================================================
    # 24. MEDIAN TIME BETWEEN TX
    # ========================================================

    if len(time_diffs_hours) > 0:

        median_time_between_tx = float(
            np.median(
                time_diffs_hours
            )
        )

    else:

        median_time_between_tx = 0.0


    # ========================================================
    # 25. MAX TIME BETWEEN TX
    # ========================================================

    if len(time_diffs_hours) > 0:

        max_time_between_tx = float(
            np.max(
                time_diffs_hours
            )
        )

    else:

        max_time_between_tx = 0.0


    # ========================================================
    # 26. BURSTINESS
    #
    # B = (std - mean) / (std + mean)
    #
    # Approximately:
    #
    # -1 → highly regular
    #  0 → random-ish
    # +1 → highly bursty
    # ========================================================

    if len(time_diffs_hours) > 1:

        mean_delta = float(
            np.mean(
                time_diffs_hours
            )
        )

        std_delta = float(
            np.std(
                time_diffs_hours
            )
        )

        denominator = (
            std_delta
            + mean_delta
        )

        burstiness = (
            (std_delta - mean_delta)
            / denominator
            if denominator > 0
            else 0.0
        )

    else:

        burstiness = 0.0


    # ========================================================
    # LABEL
    # ========================================================

    wallet_info = label_map[
        wallet
    ]


    # ========================================================
    # ADD ROW
    # ========================================================

    features.append({

        "address":
            wallet,

        "type":
            wallet_info["type"],

        "label":
            wallet_info["label"],


        # v0.1
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

        "activity_span_days":
            activity_span_days,

        "internal_tx_ratio":
            internal_tx_ratio,


        # v0.2
        "in_out_tx_ratio":
            in_out_tx_ratio,

        "net_eth_flow":
            net_eth_flow,

        "median_tx_value":
            median_tx_value,

        "std_tx_value":
            std_tx_value,

        "distinct_active_days":
            distinct_active_days,

        "tx_frequency":
            tx_frequency,

        "incoming_value_ratio":
            incoming_value_ratio,

        "zero_value_tx_ratio":
            zero_value_tx_ratio,

        "external_tx_ratio":
            external_tx_ratio,

        "counterparty_reuse_ratio":
            counterparty_reuse_ratio,

        "avg_time_between_tx":
            avg_time_between_tx,

        "median_time_between_tx":
            median_time_between_tx,

        "max_time_between_tx":
            max_time_between_tx,

        "burstiness":
            burstiness,
    })


    if i % 100 == 0:

        print(
            f"Processed {i} wallet groups..."
        )


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(
    features
)


FEATURE_COLUMNS = [

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
        [np.inf, -np.inf],
        np.nan
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


# Fill any unexpected NaNs safely.

df[FEATURE_COLUMNS] = (
    df[FEATURE_COLUMNS]
    .fillna(0.0)
)


numeric = df[
    FEATURE_COLUMNS
].to_numpy(
    dtype=float
)

print()

print(
    "Infinite values:",
    np.isinf(numeric).sum()
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
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
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