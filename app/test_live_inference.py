import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv


# ============================================================
# PATH
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# FILES
# ============================================================

GRAPH_FILE = (
    ROOT
    / "data"
    / "processed"
    / "graph"
    / "graph_v03.pt"
)

FEATURE_FILE = (
    ROOT
    / "data"
    / "processed"
    / "combined_wallet_features_v03.csv"
)

MODEL_FILE = (
    ROOT
    / "models"
    / "graphsage"
    / "v03_original"
    / "model.pt"
)

SCALER_FILE = (
    ROOT
    / "models"
    / "graphsage"
    / "v03_original"
    / "scaler.pkl"
)


# ============================================================
# TARGET WALLET
# ============================================================

ADDRESS = (
    "0x9db3c9f81846b1057666f4f6e10f3e9426874f0e"
    .lower()
)


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


COUNT_FEATURES = [
    "total_tx_count",
    "incoming_tx_count",
    "outgoing_tx_count",
    "unique_senders",
    "unique_receivers",
    "unique_counterparties",
    "distinct_active_days",
    "erc20_tx_count",
    "erc20_in_count",
    "erc20_out_count",
    "erc20_unique_tokens",
    "erc20_unique_senders",
    "erc20_unique_receivers",
    "erc20_unique_counterparties",
]


VALUE_FEATURES = [
    "total_eth_received",
    "total_eth_sent",
    "avg_tx_value",
    "max_tx_value",
    "median_tx_value",
    "std_tx_value",
]


TEMPORAL_FEATURES = [
    "activity_span_days",
    "tx_frequency",
    "avg_time_between_tx",
    "median_time_between_tx",
    "max_time_between_tx",
    "erc20_activity_span_days",
    "erc20_tx_frequency",
]


LOG_RATIO_FEATURES = [
    "in_out_tx_ratio",
    "counterparty_reuse_ratio",
    "erc20_in_out_ratio",
    "erc20_counterparty_reuse_ratio",
]


SIGNED_LOG_FEATURES = [
    "net_eth_flow",
]


# ============================================================
# MODEL
# ============================================================

class GraphSAGE(nn.Module):

    def __init__(
        self,
        in_channels,
        hidden_channels,
        dropout,
    ):
        super().__init__()

        self.conv1 = SAGEConv(
            in_channels,
            hidden_channels,
        )

        self.conv2 = SAGEConv(
            hidden_channels,
            hidden_channels,
        )

        self.classifier = nn.Linear(
            hidden_channels,
            1,
        )

        self.dropout = dropout

    def forward(
        self,
        x,
        edge_index,
    ):

        x = self.conv1(
            x,
            edge_index,
        )

        x = F.relu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        x = self.conv2(
            x,
            edge_index,
        )

        x = F.relu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        return self.classifier(
            x
        ).squeeze(-1)


# ============================================================
# LOAD
# ============================================================

device = torch.device("cpu")

print(f"Device: {device}")

data = torch.load(
    GRAPH_FILE,
    weights_only=False,
)

features = pd.read_csv(
    FEATURE_FILE
)

features["address"] = (
    features["address"]
    .astype(str)
    .str.strip()
    .str.lower()
)

node_addresses = (
    features["address"]
    .iloc[data.node_id.numpy()]
    .values
)

matches = np.where(
    node_addresses == ADDRESS
)[0]

if len(matches) != 1:
    raise RuntimeError(
        f"Expected exactly one graph node "
        f"for {ADDRESS}, found {len(matches)}"
    )

node_idx = int(matches[0])

print(
    f"Target node : {node_idx}"
)


# ============================================================
# LOAD MODEL
# ============================================================

checkpoint = torch.load(
    MODEL_FILE,
    map_location=device,
    weights_only=False,
)

model = GraphSAGE(
    checkpoint["in_channels"],
    checkpoint["hidden_channels"],
    checkpoint["dropout"],
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)
model.eval()

scaler = joblib.load(
    SCALER_FILE
)


# ============================================================
# IMPORT LIVE PIPELINE
# ============================================================

from app.live_fetch import (
    fetch_wallet_data,
)

from app.live_features import (
    build_live_features,
)


# ============================================================
# FETCH LIVE DATA
# ============================================================

print()
print("=" * 70)
print("FETCHING LIVE WALLET")
print("=" * 70)

live_data = fetch_wallet_data(
    ADDRESS
)

print(
    f"ETH transactions   : "
    f"{len(live_data['eth'])}"
)

print(
    f"ERC20 transactions : "
    f"{len(live_data['erc20'])}"
)


# ============================================================
# BUILD LIVE FEATURES
# ============================================================

live_features = build_live_features(
    ADDRESS,
    live_data["eth"],
    live_data["erc20"],
)


# ============================================================
# VERIFY FEATURE ORDER
# ============================================================

missing = set(
    FEATURE_COLUMNS
) - set(
    live_features
)

if missing:
    raise RuntimeError(
        f"Missing live features: {missing}"
    )


live_vector = np.array(
    [
        live_features[feature]
        for feature in FEATURE_COLUMNS
    ],
    dtype=np.float64,
)


# ============================================================
# LOAD STORED FEATURES
# ============================================================

stored_vector = (
    features
    .loc[
        features["address"] == ADDRESS,
        FEATURE_COLUMNS,
    ]
    .iloc[0]
    .to_numpy(
        dtype=np.float64
    )
)


# ============================================================
# COMPARE RAW FEATURES
# ============================================================

print()
print("=" * 70)
print("RAW FEATURE COMPARISON")
print("=" * 70)

max_difference = np.max(
    np.abs(
        live_vector
        - stored_vector
    )
)

print(
    f"Maximum raw feature difference: "
    f"{max_difference:.10e}"
)


# ============================================================
# TRANSFORM FUNCTION
# ============================================================

def transform_features(
    vector,
):

    df = pd.DataFrame(
        [vector],
        columns=FEATURE_COLUMNS,
    )

    for feature in (
        COUNT_FEATURES
        + VALUE_FEATURES
        + TEMPORAL_FEATURES
        + LOG_RATIO_FEATURES
    ):

        df[feature] = np.log1p(
            df[feature]
        )

    df["net_eth_flow"] = (
        np.sign(
            df["net_eth_flow"]
        )
        * np.log1p(
            np.abs(
                df["net_eth_flow"]
            )
        )
    )

    return df[
        FEATURE_COLUMNS
    ].values


# ============================================================
# TRANSFORM
# ============================================================

live_transformed = transform_features(
    live_vector
)

stored_transformed = transform_features(
    stored_vector
)


# ============================================================
# SCALE
# ============================================================

live_scaled = scaler.transform(
    live_transformed
).astype(np.float32)

stored_scaled = scaler.transform(
    stored_transformed
).astype(np.float32)


max_scaled_difference = np.max(
    np.abs(
        live_scaled
        - stored_scaled
    )
)

print(
    f"Maximum transformed/scaled "
    f"difference: "
    f"{max_scaled_difference:.10e}"
)


# ============================================================
# CREATE TWO GRAPH COPIES
# ============================================================

stored_graph = data.clone()
live_graph = data.clone()

stored_graph.x = (
    data.x
    .clone()
    .float()
)

live_graph.x = (
    data.x
    .clone()
    .float()
)


# Replace only target wallet feature row
stored_graph.x[node_idx] = torch.tensor(
    stored_scaled[0],
    dtype=torch.float32,
)

live_graph.x[node_idx] = torch.tensor(
    live_scaled[0],
    dtype=torch.float32,
)


# ============================================================
# RUN MODEL
# ============================================================

with torch.no_grad():

    stored_logits = model(
        stored_graph.x,
        stored_graph.edge_index,
    )

    live_logits = model(
        live_graph.x,
        live_graph.edge_index,
    )

stored_probability = torch.sigmoid(
    stored_logits[node_idx]
).item()

live_probability = torch.sigmoid(
    live_logits[node_idx]
).item()


# ============================================================
# RESULT
# ============================================================

print()
print("=" * 70)
print("LIVE VS STORED MODEL INFERENCE")
print("=" * 70)

print(
    f"Stored probability : "
    f"{stored_probability:.8f}"
)

print(
    f"Live probability   : "
    f"{live_probability:.8f}"
)

print(
    f"Difference         : "
    f"{abs(stored_probability - live_probability):.8e}"
)

threshold = checkpoint.get(
    "threshold",
    0.50,
)

print()
print(
    f"Stored prediction : "
    f"{'MALICIOUS' if stored_probability >= threshold else 'BENIGN'}"
)

print(
    f"Live prediction   : "
    f"{'MALICIOUS' if live_probability >= threshold else 'BENIGN'}"
)