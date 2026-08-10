import numpy as np
import pandas as pd


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

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
# CLEAN TRANSACTIONS
# ============================================================

def clean_transactions(tx: pd.DataFrame) -> pd.DataFrame:
    """
    Clean transaction data before feature computation.

    Expected columns:
        wallet_address
        from_address
        to_address
        direction
        category
        value
        timestamp
    """

    tx = tx.copy()

    # --------------------------------------------------------
    # Normalize addresses
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Clean transaction values
    # --------------------------------------------------------

    tx["value"] = pd.to_numeric(
        tx["value"],
        errors="coerce",
    )

    tx["value"] = (
        tx["value"]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
    )

    # --------------------------------------------------------
    # Clean timestamps
    # --------------------------------------------------------

    tx["timestamp"] = pd.to_datetime(
        tx["timestamp"],
        errors="coerce",
        utc=True,
    )

    return tx


# ============================================================
# COMPUTE FEATURES FOR ONE WALLET
# ============================================================

def compute_wallet_features(
    group: pd.DataFrame,
) -> dict:
    """
    Compute the exact MVP v0.2 feature set
    for one wallet's transactions.

    The wallet address is taken from
    group["wallet_address"].
    """

    if group.empty:
        raise ValueError(
            "Cannot compute features from an empty transaction group."
        )

    wallet = str(
        group["wallet_address"].iloc[0]
    ).strip().lower()

    # ========================================================
    # INCOMING / OUTGOING
    # ========================================================

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

        # Minimum one day for frequency calculation.
        activity_span_days = max(
            activity_span_days,
            1.0,
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
    # 13. IN / OUT TX RATIO
    # ========================================================

    in_out_tx_ratio = (
        incoming_tx_count
        / (outgoing_tx_count + 1)
    )

    # ========================================================
    # 14. NET ETH FLOW
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
    # Time differences are measured in HOURS.
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
            dtype=float,
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
    # RETURN
    # ========================================================

    return {

        "address":
            wallet,

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
    }


# ============================================================
# COMPUTE FEATURES FOR MULTIPLE WALLETS
# ============================================================

def compute_features(
    tx: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute MVP v0.2 features for every wallet
    represented in the transaction dataframe.

    Returns one row per wallet.
    """

    tx = clean_transactions(tx)

    features = []

    wallet_groups = tx.groupby(
        "wallet_address",
        sort=False,
    )

    for wallet, group in wallet_groups:

        features.append(
            compute_wallet_features(
                group
            )
        )

    result = pd.DataFrame(
        features
    )

    # Ensure exact feature order.
    if not result.empty:

        result = result[
            [
                "address",
                *FEATURE_COLUMNS,
            ]
        ]

    return result