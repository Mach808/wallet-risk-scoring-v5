# import pandas as pd

# from scripts.feature_engineering.feature_engineering import (
#     clean_transactions as clean_eth_transactions,
#     compute_wallet_features as compute_eth_features,
# )

# from scripts.feature_engineering.erc20_features import (
#     compute_wallet_features as compute_erc20_features,
# )


# ERC20_FEATURES = [
#     "erc20_tx_count",
#     "erc20_in_count",
#     "erc20_out_count",
#     "erc20_unique_tokens",
#     "erc20_unique_senders",
#     "erc20_unique_receivers",
#     "erc20_unique_counterparties",
#     "erc20_in_out_ratio",
#     "erc20_activity_span_days",
#     "erc20_tx_frequency",
#     "erc20_zero_value_ratio",
#     "erc20_counterparty_reuse_ratio",
# ]


# def clean_live_erc20(tx: pd.DataFrame) -> pd.DataFrame:

#     tx = tx.copy()

#     address_columns = [
#         "wallet_address",
#         "from_address",
#         "to_address",
#         "token_contract",
#     ]

#     for column in address_columns:

#         if column in tx.columns:

#             tx[column] = (
#                 tx[column]
#                 .fillna("")
#                 .astype(str)
#                 .str.strip()
#                 .str.lower()
#             )

#     tx["timestamp"] = pd.to_datetime(
#         tx["timestamp"],
#         errors="coerce",
#         utc=True,
#     )

#     return tx


# def build_live_features(
#     address,
#     eth_rows,
#     erc20_rows,
# ):

#     address = (
#         str(address)
#         .strip()
#         .lower()
#     )

#     # ========================================================
#     # ETH
#     # ========================================================

#     eth_df = pd.DataFrame(
#         eth_rows
#     )

#     if eth_df.empty:
#         raise ValueError(
#             "No ETH transactions found."
#         )

#     eth_df["wallet_address"] = address

#     eth_df = clean_eth_transactions(
#         eth_df
#     )

#     eth_features = compute_eth_features(
#         eth_df
#     )

#     # ========================================================
#     # ERC-20
#     # ========================================================

#     erc20_df = pd.DataFrame(
#         erc20_rows
#     )

#     if erc20_df.empty:

#         erc20_features = {
#             feature: 0.0
#             for feature in ERC20_FEATURES
#         }

#     else:

#         erc20_df["wallet_address"] = address

#         erc20_df = clean_live_erc20(
#             erc20_df
#         )

#         erc20_features = (
#             compute_erc20_features(
#                 address,
#                 erc20_df,
#             )
#         )

#     # ========================================================
#     # COMBINE
#     # ========================================================

#     features = {
#         **eth_features,
#         **erc20_features,
#     }

#     return features
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# IMPORT EXISTING FEATURE FUNCTIONS
# ============================================================

from scripts.feature_engineering.feature_engineering import (
    clean_transactions,
    compute_wallet_features as compute_eth_features,
)

from scripts.feature_engineering.erc20_features import (
    compute_wallet_features as compute_erc20_features,
)


# ============================================================
# FEATURE ORDER
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
# EMPTY ERC-20 FEATURES
# ============================================================

def empty_erc20_features():
    """
    Return zero-valued ERC-20 features for a wallet
    with no ERC-20 transactions.
    """

    return {
        "erc20_tx_count": 0.0,
        "erc20_in_count": 0.0,
        "erc20_out_count": 0.0,
        "erc20_unique_tokens": 0.0,
        "erc20_unique_senders": 0.0,
        "erc20_unique_receivers": 0.0,
        "erc20_unique_counterparties": 0.0,
        "erc20_in_out_ratio": 0.0,
        "erc20_activity_span_days": 0.0,
        "erc20_tx_frequency": 0.0,
        "erc20_zero_value_ratio": 0.0,
        "erc20_counterparty_reuse_ratio": 0.0,
    }

def empty_eth_features():
    """
    Return zero-valued ETH features for a wallet
    with no ETH transactions.
    """

    return {
        "total_tx_count": 0.0,
        "incoming_tx_count": 0.0,
        "outgoing_tx_count": 0.0,
        "total_eth_received": 0.0,
        "total_eth_sent": 0.0,
        "avg_tx_value": 0.0,
        "max_tx_value": 0.0,
        "unique_senders": 0.0,
        "unique_receivers": 0.0,
        "unique_counterparties": 0.0,
        "activity_span_days": 0.0,
        "internal_tx_ratio": 0.0,
        "in_out_tx_ratio": 0.0,
        "net_eth_flow": 0.0,
        "median_tx_value": 0.0,
        "std_tx_value": 0.0,
        "distinct_active_days": 0.0,
        "tx_frequency": 0.0,
        "incoming_value_ratio": 0.0,
        "zero_value_tx_ratio": 0.0,
        "external_tx_ratio": 0.0,
        "counterparty_reuse_ratio": 0.0,
        "avg_time_between_tx": 0.0,
        "median_time_between_tx": 0.0,
        "max_time_between_tx": 0.0,
        "burstiness": 0.0,
    }

# ============================================================
# COMPUTE FEATURES FOR ONE WALLET
# ============================================================

def compute_live_features(
    address,
    eth_transactions,
    erc20_transactions,
):
    """
    Compute the same 38 features used by the
    offline training pipeline.
    """

    address = (
        str(address)
        .strip()
        .lower()
    )

    # ========================================================
    # ETH
    # ========================================================

    if eth_transactions:

        eth_df = pd.DataFrame(
            eth_transactions
        )

        eth_df = clean_transactions(
            eth_df
        )

        eth_features = compute_eth_features(
            eth_df
        )

    else:

        eth_features = empty_eth_features()

    # ========================================================
    # ERC-20
    # ========================================================

    if erc20_transactions:

        erc20_df = pd.DataFrame(
            erc20_transactions
        )

        # Normalize columns expected by
        # erc20_features.compute_wallet_features
        #
        # The live token fetcher already produces
        # these fields, but normalize them here
        # for safety.

        for column in [
            "wallet_address",
            "from_address",
            "to_address",
            "token_contract",
        ]:

            if column in erc20_df.columns:

                erc20_df[column] = (
                    erc20_df[column]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .str.lower()
                )

        if "timestamp" in erc20_df.columns:

            erc20_df["timestamp"] = pd.to_datetime(
                erc20_df["timestamp"],
                errors="coerce",
                utc=True,
            )

        erc20_features = compute_erc20_features(
            address,
            erc20_df,
        )

    else:

        erc20_features = (
            empty_erc20_features()
        )

    # ========================================================
    # COMBINE
    # ========================================================

    features = {
        **eth_features,
        **erc20_features,
    }

    # The address is an identifier, not a model feature.
    features.pop(
        "address",
        None,
    )

    # ========================================================
    # VALIDATE
    # ========================================================

    missing = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in features
    ]

    if missing:

        raise RuntimeError(
            f"Missing features for "
            f"{address}: {missing}"
        )

    # Keep ONLY the 38 model features
    features = {
        feature: features[feature]
        for feature in FEATURE_COLUMNS
    }

    # --------------------------------------------------------
    # Clean numerical values
    # --------------------------------------------------------

    for feature in FEATURE_COLUMNS:

        value = features[feature]

        try:

            value = float(value)

        except (
            TypeError,
            ValueError,
        ):

            raise RuntimeError(
                f"Non-numeric value for "
                f"{feature}: {value}"
            )

        if not np.isfinite(value):

            raise RuntimeError(
                f"Non-finite value for "
                f"{feature}: {value}"
            )

        features[feature] = value

    return features


# ============================================================
# BUILD FEATURE MATRIX
# ============================================================

def build_live_feature_matrix(
    node_addresses,
    wallet_data_cache,
):
    """
    Build N x 38 feature matrix.

    wallet_data_cache contains transaction data
    for wallets that have already been fetched.
    """

    rows = []

    missing_wallets = []

    for index, address in enumerate(
        node_addresses
    ):

        address = address.lower()

        # ----------------------------------------------------
        # Fetch missing wallet data
        # ----------------------------------------------------

        if address not in wallet_data_cache:

            missing_wallets.append(
                address
            )

            continue

        data = wallet_data_cache[
            address
        ]

        # ----------------------------------------------------
        # Compute features
        # ----------------------------------------------------

        features = compute_live_features(
            address,
            data.get("eth", []),
            data.get("erc20", []),
        )

        rows.append(
            {
                "node_id": index,
                "address": address,
                **features,
            }
        )

    # --------------------------------------------------------
    # Check missing wallets
    # --------------------------------------------------------

    if missing_wallets:

        print()
        print("=" * 70)
        print("MISSING FEATURE DATA")
        print("=" * 70)

        print(
            f"Missing wallets: "
            f"{len(missing_wallets)}"
        )

        for address in missing_wallets:

            print(
                f"  {address}"
            )

        raise RuntimeError(
            "Transaction data is missing "
            "for some graph nodes."
        )

    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    feature_df = pd.DataFrame(
        rows
    )

    feature_df = feature_df.sort_values(
        "node_id"
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Validate dimensions
    # --------------------------------------------------------

    expected_columns = [
        "node_id",
        "address",
        *FEATURE_COLUMNS,
    ]

    if list(
        feature_df.columns
    ) != expected_columns:

        raise RuntimeError(
            "Feature column order mismatch."
        )

    if len(feature_df) != len(
        node_addresses
    ):

        raise RuntimeError(
            "Feature row count does not "
            "match node count."
        )

    # --------------------------------------------------------
    # Final checks
    # --------------------------------------------------------

    matrix = feature_df[
        FEATURE_COLUMNS
    ].to_numpy(
        dtype=np.float32
    )

    if not np.isfinite(
        matrix
    ).all():

        raise RuntimeError(
            "Feature matrix contains "
            "NaN or infinite values."
        )

    print()
    print("=" * 70)
    print("LIVE FEATURE MATRIX")
    print("=" * 70)

    print(
        f"Nodes    : "
        f"{len(feature_df):,}"
    )

    print(
        f"Features : "
        f"{len(FEATURE_COLUMNS)}"
    )

    print(
        f"Shape    : "
        f"{matrix.shape}"
    )

    print(
        f"NaN      : "
        f"{np.isnan(matrix).sum()}"
    )

    print(
        f"Inf      : "
        f"{np.isinf(matrix).sum()}"
    )

    return (
        feature_df,
        matrix,
    )