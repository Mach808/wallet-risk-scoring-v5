from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

TOKEN_TX_FILE = Path(
    "data/raw/token_transactions.csv"
)

MALICIOUS_FILE = Path(
    "data/labels/malicious.csv"
)

BENIGN_FILE = Path(
    "data/labels/benign.csv"
)

OUTPUT_FILE = Path(
    "data/processed/erc20_wallet_features.csv"
)


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


# ============================================================
# LOAD LABELLED WALLETS
# ============================================================

def load_labeled_wallets():

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

    # ========================================================
    # REMOVE RUGPULL ADDRESSES
    #
    # Same rule used by the frozen ETH feature pipeline.
    # ========================================================

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

    # ========================================================
    # ADD LABELS
    # ========================================================

    malicious["label"] = 1
    benign["label"] = 0

    # ========================================================
    # COMBINE
    # ========================================================

    labels = pd.concat(
        [
            malicious,
            benign,
        ],
        ignore_index=True,
    )

    # ========================================================
    # NORMALIZE ADDRESSES
    # ========================================================

    labels["address"] = (
        labels["address"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # ========================================================
    # REMOVE DUPLICATE ADDRESSES
    #
    # This matches the ETH feature pipeline.
    # ========================================================

    labels = labels.drop_duplicates(
        subset=["address"],
        keep="first",
    )

    print(
        f"Unique labeled wallets          : "
        f"{len(labels)}"
    )

    # Keep type because we will need it
    # for later malicious-type analysis.

    labels = labels[
        [
            "address",
            "type",
            "label",
        ]
    ]

    return labels


# ============================================================
# LOAD ERC-20 TRANSACTIONS
# ============================================================

def load_transactions():

    print(
        "Loading ERC-20 transactions..."
    )

    df = pd.read_csv(
        TOKEN_TX_FILE
    )

    print(
        f"Transactions: {len(df):,}"
    )

    # Normalize addresses
    address_columns = [
        "wallet_address",
        "from_address",
        "to_address",
        "token_contract",
    ]

    for column in address_columns:

        if column in df.columns:

            df[column] = (
                df[column]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.lower()
            )

    # Parse timestamps
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=True,
    )

    return df


# ============================================================
# COMPUTE FEATURES FOR ONE WALLET
# ============================================================

def compute_wallet_features(
    wallet,
    group,
):

    tx_count = len(group)

    incoming = group[
        group["direction"] == "incoming"
    ]

    outgoing = group[
        group["direction"] == "outgoing"
    ]

    in_count = len(
        incoming
    )

    out_count = len(
        outgoing
    )


    # --------------------------------------------------------
    # Unique tokens
    # --------------------------------------------------------

    unique_tokens = (
        group[
            group["token_contract"] != ""
        ]["token_contract"]
        .nunique()
    )


    # --------------------------------------------------------
    # Unique senders / receivers
    # --------------------------------------------------------

    unique_senders = (
        group[
            group["from_address"] != ""
        ]["from_address"]
        .nunique()
    )

    unique_receivers = (
        group[
            group["to_address"] != ""
        ]["to_address"]
        .nunique()
    )


    # --------------------------------------------------------
    # Counterparties
    # --------------------------------------------------------

    counterparties = set(
        group["from_address"]
    ) | set(
        group["to_address"]
    )

    counterparties.discard("")

    unique_counterparties = len(
        counterparties
    )


    # --------------------------------------------------------
    # Incoming / outgoing ratio
    # --------------------------------------------------------

    if out_count > 0:

        in_out_ratio = (
            in_count / out_count
        )

    elif in_count > 0:

        in_out_ratio = float(
            in_count
        )

    else:

        in_out_ratio = 0.0


    # --------------------------------------------------------
    # Activity span
    # --------------------------------------------------------

    valid_times = group[
        "timestamp"
    ].dropna().sort_values()

    if len(valid_times) >= 2:

        activity_span_days = (
            (
                valid_times.iloc[-1]
                - valid_times.iloc[0]
            ).total_seconds()
            / 86400.0
        )

    else:

        activity_span_days = 0.0


    # --------------------------------------------------------
    # Transaction frequency
    # --------------------------------------------------------

    if activity_span_days > 0:

        tx_frequency = (
            tx_count
            / activity_span_days
        )

    else:

        # A wallet with transactions
        # but only one timestamp has
        # effectively one observation
        # period. Keep this finite.
        tx_frequency = float(
            tx_count
        )


    # --------------------------------------------------------
    # Zero-value transfers
    # --------------------------------------------------------

    numeric_values = pd.to_numeric(
        group["value"],
        errors="coerce",
    ).fillna(0.0)

    zero_value_count = int(
        (numeric_values == 0).sum()
    )

    zero_value_ratio = (
        zero_value_count / tx_count
        if tx_count > 0
        else 0.0
    )


    # --------------------------------------------------------
    # Counterparty reuse
    #
    # Total transactions divided by
    # unique counterparties.
    # --------------------------------------------------------

    if unique_counterparties > 0:

        counterparty_reuse_ratio = (
            tx_count
            / unique_counterparties
        )

    else:

        counterparty_reuse_ratio = 0.0


    return {

        "address":
            wallet,

        "erc20_tx_count":
            tx_count,

        "erc20_in_count":
            in_count,

        "erc20_out_count":
            out_count,

        "erc20_unique_tokens":
            unique_tokens,

        "erc20_unique_senders":
            unique_senders,

        "erc20_unique_receivers":
            unique_receivers,

        "erc20_unique_counterparties":
            unique_counterparties,

        "erc20_in_out_ratio":
            in_out_ratio,

        "erc20_activity_span_days":
            activity_span_days,

        "erc20_tx_frequency":
            tx_frequency,

        "erc20_zero_value_ratio":
            zero_value_ratio,

        "erc20_counterparty_reuse_ratio":
            counterparty_reuse_ratio,
    }


# ============================================================
# BUILD FEATURE DATASET
# ============================================================

def build_features():

    labels = load_labeled_wallets()

    transactions = load_transactions()

    print()
    print("=" * 70)
    print("BUILDING ERC-20 WALLET FEATURES")
    print("=" * 70)

    # --------------------------------------------------------
    # Group transactions by wallet
    # --------------------------------------------------------

    grouped = {
        wallet: group
        for wallet, group
        in transactions.groupby(
            "wallet_address"
        )
    }

    print(
        f"Unique transaction wallets: "
        f"{len(grouped)}"
    )

    # --------------------------------------------------------
    # Compute features ONLY for labelled wallets
    # --------------------------------------------------------

    rows = []

    for wallet in labels["address"]:

        if wallet in grouped:

            features = compute_wallet_features(
                wallet,
                grouped[wallet],
            )

        else:

            # ------------------------------------------------
            # Wallet has ZERO ERC-20 activity.
            # Explicitly create zero features.
            # ------------------------------------------------

            features = {

                "address":
                    wallet,

                "erc20_tx_count":
                    0,

                "erc20_in_count":
                    0,

                "erc20_out_count":
                    0,

                "erc20_unique_tokens":
                    0,

                "erc20_unique_senders":
                    0,

                "erc20_unique_receivers":
                    0,

                "erc20_unique_counterparties":
                    0,

                "erc20_in_out_ratio":
                    0.0,

                "erc20_activity_span_days":
                    0.0,

                "erc20_tx_frequency":
                    0.0,

                "erc20_zero_value_ratio":
                    0.0,

                "erc20_counterparty_reuse_ratio":
                    0.0,
            }

        rows.append(
            features
        )

    features_df = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # Add labels
    # --------------------------------------------------------

    features_df = features_df.merge(
        labels,
        on="address",
        how="left",
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    print()
    print("=" * 70)
    print("ERC-20 FEATURE DATASET")
    print("=" * 70)

    print(
        f"Wallets  : {len(features_df)}"
    )

    print(
        f"Features : {len(ERC20_FEATURES)}"
    )


    # --------------------------------------------------------
    # NaN
    # --------------------------------------------------------

    print()
    print("NaN values:")

    nan_counts = (
        features_df[
            ERC20_FEATURES
        ]
        .isna()
        .sum()
    )

    print(
        nan_counts[
            nan_counts > 0
        ]
        if nan_counts.sum() > 0
        else "None"
    )


    # --------------------------------------------------------
    # Infinite
    # --------------------------------------------------------

    numeric = features_df[
        ERC20_FEATURES
    ].select_dtypes(
        include=np.number
    )

    infinite_count = np.isinf(
        numeric.to_numpy()
    ).sum()

    print()
    print(
        f"Infinite values: "
        f"{infinite_count}"
    )


    # --------------------------------------------------------
    # Label distribution
    # --------------------------------------------------------

    print()
    print("LABEL DISTRIBUTION")
    print("-" * 70)

    print(
        features_df[
            "label"
        ].value_counts()
        .sort_index()
    )


    # --------------------------------------------------------
    # ERC-20 activity coverage
    # --------------------------------------------------------

    active_wallets = int(
        (
            features_df[
                "erc20_tx_count"
            ] > 0
        ).sum()
    )

    zero_wallets = (
        len(features_df)
        - active_wallets
    )

    print()
    print("ERC-20 COVERAGE")
    print("-" * 70)

    print(
        f"Wallets with ERC-20 : "
        f"{active_wallets}"
    )

    print(
        f"Wallets with zero   : "
        f"{zero_wallets}"
    )


    # --------------------------------------------------------
    # Feature summary
    # --------------------------------------------------------

    print()
    print("FEATURE SUMMARY")
    print("-" * 70)

    print(
        features_df[
            ERC20_FEATURES
        ].describe()
        .T[
            [
                "min",
                "mean",
                "50%",
                "max",
            ]
        ]
        .to_string()
    )


    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    features_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 70)
    print("SAVED")
    print("=" * 70)

    print(
        OUTPUT_FILE
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    build_features()