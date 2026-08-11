from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


ROOT = Path(__file__).resolve().parents[1]

GRAPH_FILE = ROOT / "data" / "processed" / "graph" / "graph_v03.pt"

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

BOUNDED_FEATURES = [
    "internal_tx_ratio",
    "incoming_value_ratio",
    "zero_value_tx_ratio",
    "external_tx_ratio",
    "burstiness",
    "erc20_zero_value_ratio",
]


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

    def forward(self, x, edge_index):

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

        return self.classifier(x).squeeze(-1)


class WalletRiskModel:

    def __init__(self):

        self.device = torch.device("cpu")

        # Load graph
        self.data = torch.load(
            GRAPH_FILE,
            weights_only=False,
            map_location="cpu",
        )

        # Load original feature ordering
        self.features = pd.read_csv(
            FEATURE_FILE
        )

        self.features["address"] = (
            self.features["address"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        if len(self.features) != self.data.num_nodes:
            raise RuntimeError(
                "Feature/graph node count mismatch."
            )

        # Address -> node ID
        self.address_to_node = {
            address: index
            for index, address in enumerate(
                self.features["address"]
            )
        }

        # Load scaler
        self.scaler = joblib.load(
            SCALER_FILE
        )

        # Load model
        checkpoint = torch.load(
            MODEL_FILE,
            weights_only=False,
            map_location="cpu",
        )

        self.model = GraphSAGE(
            checkpoint["in_channels"],
            checkpoint["hidden_channels"],
            checkpoint["dropout"],
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()

        self.threshold = checkpoint.get(
            "threshold",
            0.50,
        )

        # Transform all graph features
        self.x_scaled = self._transform_features()

        self.data.x = torch.tensor(
            self.x_scaled,
            dtype=torch.float32,
        )

    def _transform_features(self):

        x_df = pd.DataFrame(
            self.data.x.cpu().numpy(),
            columns=FEATURE_COLUMNS,
        )

        for feature in (
            COUNT_FEATURES
            + VALUE_FEATURES
            + TEMPORAL_FEATURES
            + LOG_RATIO_FEATURES
        ):

            x_df[feature] = np.log1p(
                x_df[feature]
            )

        x_df["net_eth_flow"] = (
            np.sign(x_df["net_eth_flow"])
            * np.log1p(
                np.abs(
                    x_df["net_eth_flow"]
                )
            )
        )

        return self.scaler.transform(
            x_df.values
        ).astype(np.float32)

    def predict(self, address):

        address = (
            address
            .strip()
            .lower()
        )

        if address not in self.address_to_node:
            return None

        node_id = self.address_to_node[address]

        x = self.data.x
        edge_index = self.data.edge_index

        with torch.no_grad():

            logits = self.model(
                x,
                edge_index,
            )

            probability = torch.sigmoid(
                logits[node_id]
            ).item()

        prediction = int(
            probability >= self.threshold
        )

        wallet = self.features.iloc[node_id]

        return {
            "address": address,
            "node_id": node_id,
            "risk_score": probability,
            "prediction": prediction,
            "label": int(wallet["label"]),
            "type": wallet["type"],
        }