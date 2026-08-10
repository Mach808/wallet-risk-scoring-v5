from pathlib import Path
import pickle

import pandas as pd

from feature_engineering import (
    compute_wallet_features,
    clean_transactions,
)
from transaction_fetcher import fetch_wallet


# ============================================================
# CONFIG
# ============================================================

MODEL_FILE = Path(
    "models/wallet_risk_mvp_v02.pkl"
)

THRESHOLD = 0.50


# ============================================================
# MODEL FEATURES
# ============================================================

FEATURES = [
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
# LOAD MODEL
# ============================================================

print("=" * 70)
print("LIVE WALLET RISK PREDICTION")
print("=" * 70)

print()
print("Loading Random Forest model...")

with open(
    MODEL_FILE,
    "rb",
) as f:

    model = pickle.load(f)

print("Model loaded.")


# ============================================================
# GET ADDRESS
# ============================================================

print()

address = input(
    "Enter Ethereum wallet address: "
).strip().lower()


# ============================================================
# BASIC VALIDATION
# ============================================================

if not address:

    raise SystemExit(
        "No wallet address provided."
    )


if not address.startswith("0x"):

    raise SystemExit(
        "Invalid Ethereum address."
    )


if len(address) != 42:

    raise SystemExit(
        "Invalid Ethereum address length."
    )


# ============================================================
# FETCH TRANSACTIONS
# ============================================================

print()
print("=" * 70)
print("FETCHING TRANSACTIONS")
print("=" * 70)

print()
print(
    "Fetching external + internal "
    "transactions from Alchemy..."
)

transactions = fetch_wallet(
    address
)


print()
print(
    f"Transactions fetched: "
    f"{len(transactions)}"
)


# ============================================================
# NO TRANSACTIONS
# ============================================================

if not transactions:

    print()
    print(
        "No transactions found for this wallet."
    )

    print(
        "Risk scoring cannot be performed "
        "without transaction history."
    )

    raise SystemExit(0)


# ============================================================
# TRANSACTION DATAFRAME
# ============================================================

tx = pd.DataFrame(
    transactions
)


# ============================================================
# TRANSACTION SUMMARY
# ============================================================

external_count = int(
    (
        tx["category"]
        == "external"
    ).sum()
)

internal_count = int(
    (
        tx["category"]
        == "internal"
    ).sum()
)

incoming_count = int(
    (
        tx["direction"]
        == "incoming"
    ).sum()
)

outgoing_count = int(
    (
        tx["direction"]
        == "outgoing"
    ).sum()
)


print()
print(
    f"  External transactions : "
    f"{external_count}"
)

print(
    f"  Internal transactions : "
    f"{internal_count}"
)

print(
    f"  Incoming transactions : "
    f"{incoming_count}"
)

print(
    f"  Outgoing transactions : "
    f"{outgoing_count}"
)


# ============================================================
# COMPUTE FEATURES
# ============================================================

print()
print("=" * 70)
print("COMPUTING WALLET FEATURES")
print("=" * 70)

tx = clean_transactions(
    tx
)

features = compute_wallet_features(
    tx
)


# ============================================================
# CREATE MODEL INPUT
# ============================================================

feature_df = pd.DataFrame(
    [features]
)


# Make sure the exact feature order
# expected by the model is used.

X = feature_df[
    FEATURES
]


# ============================================================
# CHECK FOR INVALID VALUES
# ============================================================

if X.isna().any().any():

    raise RuntimeError(
        "NaN values detected in wallet features."
    )


if not X.map(
    lambda x: pd.notna(x)
    and x != float("inf")
    and x != float("-inf")
).all().all():

    raise RuntimeError(
        "Invalid infinite values detected "
        "in wallet features."
    )


# ============================================================
# PREDICT
# ============================================================

print()
print("=" * 70)
print("RUNNING RISK MODEL")
print("=" * 70)

risk_score = float(
    model.predict_proba(X)[0][1]
)


prediction = int(
    risk_score >= THRESHOLD
)


# ============================================================
# RESULT
# ============================================================

print()
print("=" * 70)
print("RISK ASSESSMENT")
print("=" * 70)

print()

print(
    f"Wallet      : {address}"
)

print(
    f"Risk Score  : {risk_score:.4f}"
)

print(
    f"Threshold   : {THRESHOLD:.2f}"
)

print(
    f"Decision    : "
    f"{'MALICIOUS' if prediction == 1 else 'BENIGN'}"
)


# ============================================================
# WALLET ACTIVITY
# ============================================================

print()
print("=" * 70)
print("WALLET ACTIVITY")
print("=" * 70)

print()

print(
    f"Transactions       : "
    f"{features['total_tx_count']}"
)

print(
    f"Incoming           : "
    f"{features['incoming_tx_count']}"
)

print(
    f"Outgoing           : "
    f"{features['outgoing_tx_count']}"
)

print(
    f"ETH received       : "
    f"{features['total_eth_received']:.6f}"
)

print(
    f"ETH sent           : "
    f"{features['total_eth_sent']:.6f}"
)

print(
    f"Unique senders     : "
    f"{features['unique_senders']}"
)

print(
    f"Unique receivers   : "
    f"{features['unique_receivers']}"
)

print(
    f"Unique counterparties : "
    f"{features['unique_counterparties']}"
)

print(
    f"Active days        : "
    f"{features['distinct_active_days']}"
)


# ============================================================
# BEHAVIORAL SIGNALS
# ============================================================

print()
print("=" * 70)
print("BEHAVIORAL SIGNALS")
print("=" * 70)

print()

print(
    f"Incoming value ratio  : "
    f"{features['incoming_value_ratio']:.4f}"
)

print(
    f"Counterparty reuse    : "
    f"{features['counterparty_reuse_ratio']:.4f}"
)

print(
    f"Transaction frequency : "
    f"{features['tx_frequency']:.4f}"
)

print(
    f"Internal tx ratio     : "
    f"{features['internal_tx_ratio']:.4f}"
)

print(
    f"External tx ratio     : "
    f"{features['external_tx_ratio']:.4f}"
)

print(
    f"Burstiness            : "
    f"{features['burstiness']:.4f}"
)


# ============================================================
# DONE
# ============================================================

print()
print("=" * 70)
print("LIVE PREDICTION COMPLETE")
print("=" * 70)