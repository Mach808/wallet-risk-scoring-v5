from pathlib import Path
import pickle

import pandas as pd
import streamlit as st

from scripts.feature_engineering import (
    compute_wallet_features,
    clean_transactions,
)

from scripts.data_collection.transaction_fetcher import (
    fetch_wallet,
)


# ============================================================
# CONFIG
# ============================================================

MODEL_FILE = Path(
    "models/wallet_risk_mvp_v02.pkl"
)

THRESHOLD = 0.50


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
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Ethereum Wallet Risk Scoring",
    page_icon="🔎",
    layout="wide",
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    with open(
        MODEL_FILE,
        "rb",
    ) as f:

        return pickle.load(f)


model = load_model()


# ============================================================
# PAGE HEADER
# ============================================================

st.title(
    "🔎 Ethereum Wallet Risk Scoring"
)

st.write(
    "MVP v0.2 — Behavioral Risk Analysis"
)

st.caption(
    "Random Forest model trained on Ethereum wallet "
    "transaction behavior."
)


# ============================================================
# WALLET INPUT
# ============================================================

st.subheader(
    "Analyze a Wallet"
)

address = st.text_input(
    "Ethereum wallet address",
    placeholder="0x...",
)


analyze = st.button(
    "Analyze Wallet",
    type="primary",
    use_container_width=True,
)


# ============================================================
# ANALYSIS
# ============================================================

if analyze:

    # --------------------------------------------------------
    # Validate address
    # --------------------------------------------------------

    address = (
        address
        .strip()
        .lower()
    )

    if not address:

        st.error(
            "Please enter an Ethereum wallet address."
        )

        st.stop()

    if not address.startswith("0x"):

        st.error(
            "Invalid Ethereum address."
        )

        st.stop()

    if len(address) != 42:

        st.error(
            "Ethereum addresses must contain 42 characters."
        )

        st.stop()


    # --------------------------------------------------------
    # Fetch transactions
    # --------------------------------------------------------

    with st.spinner(
        "Fetching Ethereum transaction history..."
    ):

        try:

            transactions = fetch_wallet(
                address
            )

        except Exception as e:

            st.error(
                f"Failed to fetch transactions: {e}"
            )

            st.stop()


    if not transactions:

        st.warning(
            "No transactions were found for this wallet."
        )

        st.info(
            "The MVP requires transaction history "
            "to calculate behavioral features."
        )

        st.stop()


    # --------------------------------------------------------
    # Create transaction dataframe
    # --------------------------------------------------------

    tx = pd.DataFrame(
        transactions
    )


    # --------------------------------------------------------
    # Transaction statistics
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Compute features
    # --------------------------------------------------------

    with st.spinner(
        "Computing wallet behavior..."
    ):

        try:

            tx = clean_transactions(
                tx
            )

            features = (
                compute_wallet_features(
                    tx
                )
            )

        except Exception as e:

            st.error(
                f"Feature computation failed: {e}"
            )

            st.stop()


    # --------------------------------------------------------
    # Model input
    # --------------------------------------------------------

    feature_df = pd.DataFrame(
        [features]
    )

    X = feature_df[
        FEATURES
    ]


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    risk_score = float(
        model.predict_proba(X)[0][1]
    )

    is_malicious = (
        risk_score >= THRESHOLD
    )


    # ========================================================
    # RESULT
    # ========================================================

    st.divider()

    st.subheader(
        "Risk Assessment"
    )

    col1, col2, col3 = st.columns(
        3
    )


    with col1:

        st.metric(
            "Risk Score",
            f"{risk_score:.2f}",
        )


    with col2:

        st.metric(
            "Decision Threshold",
            f"{THRESHOLD:.2f}",
        )


    with col3:

        if is_malicious:

            st.error(
                "⚠️ MALICIOUS"
            )

        else:

            st.success(
                "✓ BENIGN"
            )


    # --------------------------------------------------------
    # Risk bar
    # --------------------------------------------------------

    st.progress(
        min(
            max(
                risk_score,
                0.0,
            ),
            1.0,
        )
    )


    st.caption(
        "The risk score is the Random Forest's "
        "malicious-class prediction score. "
        "It is not a calibrated probability of illicit activity."
    )


    # ========================================================
    # TRANSACTION ACTIVITY
    # ========================================================

    st.divider()

    st.subheader(
        "Transaction Activity"
    )

    c1, c2, c3, c4 = st.columns(
        4
    )

    with c1:

        st.metric(
            "Transactions",
            f"{features['total_tx_count']}",
        )

    with c2:

        st.metric(
            "Incoming",
            f"{features['incoming_tx_count']}",
        )

    with c3:

        st.metric(
            "Outgoing",
            f"{features['outgoing_tx_count']}",
        )

    with c4:

        st.metric(
            "Counterparties",
            f"{features['unique_counterparties']}",
        )


    # ========================================================
    # VALUE ACTIVITY
    # ========================================================

    st.subheader(
        "ETH Activity"
    )

    c1, c2, c3 = st.columns(
        3
    )

    with c1:

        st.metric(
            "ETH Received",
            f"{features['total_eth_received']:.4f}",
        )

    with c2:

        st.metric(
            "ETH Sent",
            f"{features['total_eth_sent']:.4f}",
        )

    with c3:

        st.metric(
            "Net ETH Flow",
            f"{features['net_eth_flow']:.4f}",
        )


    # ========================================================
    # TRANSACTION BREAKDOWN
    # ========================================================

    st.subheader(
        "Transaction Breakdown"
    )

    c1, c2, c3, c4 = st.columns(
        4
    )

    with c1:

        st.metric(
            "External",
            external_count,
        )

    with c2:

        st.metric(
            "Internal",
            internal_count,
        )

    with c3:

        st.metric(
            "Unique Senders",
            features["unique_senders"],
        )

    with c4:

        st.metric(
            "Unique Receivers",
            features["unique_receivers"],
        )


    # ========================================================
    # BEHAVIORAL SIGNALS
    # ========================================================

    st.divider()

    st.subheader(
        "Behavioral Signals"
    )

    c1, c2, c3 = st.columns(
        3
    )

    with c1:

        st.metric(
            "Incoming Value Ratio",
            f"{features['incoming_value_ratio']:.3f}",
        )

        st.metric(
            "Counterparty Reuse",
            f"{features['counterparty_reuse_ratio']:.3f}",
        )


    with c2:

        st.metric(
            "Transaction Frequency",
            f"{features['tx_frequency']:.3f}",
        )

        st.metric(
            "Active Days",
            f"{features['distinct_active_days']}",
        )


    with c3:

        st.metric(
            "Internal Tx Ratio",
            f"{features['internal_tx_ratio']:.3f}",
        )

        st.metric(
            "Burstiness",
            f"{features['burstiness']:.3f}",
        )


    # ========================================================
    # WALLET ADDRESS
    # ========================================================

    st.divider()

    st.subheader(
        "Wallet"
    )

    st.code(
        address,
        language=None,
    )


    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.divider()

    st.caption(
        "⚠️ MVP research prototype. This system estimates "
        "behavioral risk from transaction features and "
        "should not be treated as definitive evidence "
        "that a wallet is illicit."
    )