import sys
from pathlib import Path
from collections import Counter, defaultdict
from torch_geometric.data import Data
import torch
import numpy as np
from live_features import (
    FEATURE_COLUMNS
)


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from app.live_fetch import fetch_wallet_data


# ============================================================
# CONFIG
# ============================================================

TOP_K_FIRST_HOP = 20
TOP_K_SECOND_HOP = 20


# ============================================================
# ADDRESS NORMALIZATION
# ============================================================

def normalize(address):
    return (
        str(address)
        .strip()
        .lower()
    )


# ============================================================
# BUILD INTERACTION COUNTS
# ============================================================

def get_interactions(
    wallet,
    wallet_data,
):
    """
    Count ETH + ERC-20 interactions between the
    wallet and each counterparty.
    """

    wallet = normalize(wallet)

    counts = Counter()

    # --------------------------------------------------------
    # ETH
    # --------------------------------------------------------

    for tx in wallet_data.get("eth", []):

        sender = normalize(
            tx.get("from_address", "")
        )

        receiver = normalize(
            tx.get("to_address", "")
        )

        if sender == wallet:

            if receiver and receiver != wallet:
                counts[receiver] += 1

        elif receiver == wallet:

            if sender and sender != wallet:
                counts[sender] += 1

    # --------------------------------------------------------
    # ERC-20
    # --------------------------------------------------------

    for tx in wallet_data.get("erc20", []):

        sender = normalize(
            tx.get("from_address", "")
        )

        receiver = normalize(
            tx.get("to_address", "")
        )

        if sender == wallet:

            if receiver and receiver != wallet:
                counts[receiver] += 1

        elif receiver == wallet:

            if sender and sender != wallet:
                counts[sender] += 1

    return counts


# ============================================================
# TOP K NEIGHBORS
# ============================================================

def top_neighbors(
    counts,
    k,
):
    """
    Return the top-k neighbors by interaction count.
    """

    return [
        address
        for address, count
        in counts.most_common(k)
    ]


# ============================================================
# BUILD 2-HOP NODE UNIVERSE
# ============================================================

def build_2hop_node_universe(
    target_address,
):

    target_address = normalize(
        target_address
    )

    # --------------------------------------------------------
    # NODE HOP INFORMATION
    # --------------------------------------------------------

    hop = {
        target_address: 0
    }

    # Cache transaction data so we never fetch
    # the same wallet twice.
    wallet_data_cache = {}

    # --------------------------------------------------------
    # FETCH TARGET
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FETCHING TARGET WALLET")
    print("=" * 70)

    target_data = fetch_wallet_data(
        target_address
    )

    wallet_data_cache[
        target_address
    ] = target_data

    print(
        f"ETH transactions   : "
        f"{len(target_data['eth'])}"
    )

    print(
        f"ERC20 transactions : "
        f"{len(target_data['erc20'])}"
    )

    # --------------------------------------------------------
    # DISCOVER 1-HOP
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DISCOVERING 1-HOP TOP-20")
    print("=" * 70)

    target_counts = get_interactions(
        target_address,
        target_data,
    )

    first_hop = top_neighbors(
        target_counts,
        TOP_K_FIRST_HOP,
    )

    first_hop = [
        address
        for address in first_hop
        if address != target_address
    ]

    for address in first_hop:
        hop[address] = 1

    print(
        f"1-hop nodes: "
        f"{len(first_hop)}"
    )

    # --------------------------------------------------------
    # DISCOVER SECOND-HOP CANDIDATES
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DISCOVERING 2-HOP CANDIDATES")
    print("=" * 70)

    # Global counter:
    #
    # candidate address -> total number of
    # interactions observed through first-hop wallets
    #
    # This lets us rank all second-hop candidates
    # globally instead of selecting 20 per first-hop wallet.
    second_hop_scores = Counter()

    for index, wallet in enumerate(
        first_hop,
        start=1,
    ):

        print(
            f"[{index}/{len(first_hop)}] "
            f"Fetching {wallet}"
        )

        try:

            wallet_data = fetch_wallet_data(
                wallet
            )

            wallet_data_cache[
                wallet
            ] = wallet_data

            counts = get_interactions(
                wallet,
                wallet_data,
            )

            for candidate, count in counts.items():

                # Never add:
                # - target
                # - existing 1-hop nodes
                #
                if candidate == target_address:
                    continue

                if candidate in hop:
                    continue

                second_hop_scores[
                    candidate
                ] += count

        except Exception as e:

            print(
                f"Failed to fetch "
                f"{wallet}: {e}"
            )

    # --------------------------------------------------------
    # GLOBAL TOP-20 SECOND-HOP
    # --------------------------------------------------------

    second_hop = [
        address
        for address, score
        in second_hop_scores.most_common(
            TOP_K_SECOND_HOP
        )
    ]

    for address in second_hop:
        hop[address] = 2

    print()
    print("=" * 70)
    print("TOP-20 GLOBAL 2-HOP")
    print("=" * 70)

    print(
        f"2-hop candidates : "
        f"{len(second_hop_scores)}"
    )

    print(
        f"2-hop selected   : "
        f"{len(second_hop)}"
    )

    # --------------------------------------------------------
    # FINAL NODE ORDER
    # --------------------------------------------------------

    node_addresses = list(
        hop.keys()
    )

    # Permanent ordering:
    #
    # hop 0
    # hop 1
    # hop 2
    #
    # address provides deterministic ordering
    # within each hop.
    node_addresses.sort(
        key=lambda address: (
            hop[address],
            address,
        )
    )

    node_to_id = {
        address: index
        for index, address
        in enumerate(node_addresses)
    }

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL LIVE NODE UNIVERSE")
    print("=" * 70)

    print(
        f"Target : 1"
    )

    print(
        f"1-hop  : {len(first_hop)}"
    )

    print(
        f"2-hop  : {len(second_hop)}"
    )

    print(
        f"Total  : {len(node_addresses)}"
    )

    return (
        node_addresses,
        node_to_id,
        hop,
        wallet_data_cache,
    )


# ============================================================
# BUILD DIRECTED EDGES
# ============================================================

def build_edges(
    node_addresses,
    node_to_id,
    wallet_data_cache,
):
    """
    Build directed ETH + ERC-20 edges.

    Multiple transactions between the same pair
    are combined into one edge whose weight is
    the interaction count.
    """

    edge_counts = defaultdict(int)

    node_set = set(
        node_addresses
    )

    # --------------------------------------------------------
    # PROCESS FETCHED WALLETS
    # --------------------------------------------------------

    for wallet, data in wallet_data_cache.items():

        # ----------------------------------------------------
        # ETH
        # ----------------------------------------------------

        for tx in data.get("eth", []):

            sender = normalize(
                tx.get("from_address", "")
            )

            receiver = normalize(
                tx.get("to_address", "")
            )

            if (
                sender in node_set
                and receiver in node_set
                and sender != receiver
            ):

                edge_counts[
                    (
                        node_to_id[sender],
                        node_to_id[receiver],
                    )
                ] += 1

        # ----------------------------------------------------
        # ERC-20
        # ----------------------------------------------------

        for tx in data.get("erc20", []):

            sender = normalize(
                tx.get("from_address", "")
            )

            receiver = normalize(
                tx.get("to_address", "")
            )

            if (
                sender in node_set
                and receiver in node_set
                and sender != receiver
            ):

                edge_counts[
                    (
                        node_to_id[sender],
                        node_to_id[receiver],
                    )
                ] += 1

    # --------------------------------------------------------
    # CONVERT TO TENSORS
    # --------------------------------------------------------

    if edge_counts:

        edges = list(
            edge_counts.keys()
        )

        edge_index = torch.tensor(
            edges,
            dtype=torch.long,
        ).t().contiguous()

        edge_weight = torch.tensor(
            [
                edge_counts[edge]
                for edge in edges
            ],
            dtype=torch.float32,
        )

    else:

        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long,
        )

        edge_weight = torch.empty(
            (0,),
            dtype=torch.float32,
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("LIVE GRAPH")
    print("=" * 70)

    print(
        f"Nodes       : "
        f"{len(node_addresses):,}"
    )

    print(
        f"Edges       : "
        f"{edge_index.shape[1]:,}"
    )

    print(
        f"Edge weights: "
        f"{len(edge_weight):,}"
    )

    return (
        edge_index,
        edge_weight,
    )

def fetch_final_2hop_data(
    node_addresses,
    hop,
    wallet_data_cache,
):
    """
    Fetch transaction data for final 2-hop
    nodes only.

    No further graph expansion is performed.
    """

    second_hop = [
        address
        for address in node_addresses
        if hop[address] == 2
    ]

    print()
    print("=" * 70)
    print("FETCHING FINAL 2-HOP FEATURE DATA")
    print("=" * 70)

    print(
        f"2-hop wallets to fetch: "
        f"{len(second_hop)}"
    )

    for index, address in enumerate(
        second_hop,
        start=1,
    ):

        print(
            f"[{index}/{len(second_hop)}] "
            f"Fetching {address}"
        )

        if address in wallet_data_cache:
            continue

        try:

            wallet_data_cache[
                address
            ] = fetch_wallet_data(
                address
            )

        except Exception as e:

            raise RuntimeError(
                f"Failed to fetch "
                f"2-hop wallet "
                f"{address}: {e}"
            )

    print(
        f"Wallet data available: "
        f"{len(wallet_data_cache)}"
    )

    return wallet_data_cache


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    ADDRESS = (
        "0xec55e632a4496599b2b7090796747264f8e9e1fb"
    )

    (
        node_addresses,
        node_to_id,
        hop,
        wallet_data_cache,
    ) = build_2hop_node_universe(
        ADDRESS
    )

    edge_index, edge_weight = build_edges(
        node_addresses,
        node_to_id,
        wallet_data_cache,
    )
    wallet_data_cache = fetch_final_2hop_data(
    node_addresses,
    hop,
    wallet_data_cache,
)

from app.live_features import (
    build_live_feature_matrix,
)

feature_df, feature_matrix = (
    build_live_feature_matrix(
        node_addresses,
        wallet_data_cache,
    )
)



# ============================================================
# GRAPH SAGE INFERENCE
# ============================================================

import joblib
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


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
# FEATURE TRANSFORMATION
# ============================================================

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


def transform_live_features(feature_df):

    transformed = feature_df[
        FEATURE_COLUMNS
    ].copy()

    for feature in (
        COUNT_FEATURES
        + VALUE_FEATURES
        + TEMPORAL_FEATURES
        + LOG_RATIO_FEATURES
    ):

        transformed[feature] = np.log1p(
            transformed[feature]
        )

    transformed["net_eth_flow"] = (
        np.sign(
            transformed["net_eth_flow"]
        )
        * np.log1p(
            np.abs(
                transformed["net_eth_flow"]
            )
        )
    )

    return transformed


# ============================================================
# LOAD SCALER
# ============================================================

scaler = joblib.load(
    SCALER_FILE
)

transformed_df = transform_live_features(
    feature_df
)

x_scaled = scaler.transform(
    transformed_df
).astype(np.float32)

print()
print("=" * 70)
print("LIVE FEATURE TRANSFORMATION")
print("=" * 70)

print(
    f"Nodes             : {x_scaled.shape[0]}"
)

print(
    f"Features          : {x_scaled.shape[1]}"
)

print(
    f"Transformed max   : "
    f"{np.max(np.abs(x_scaled)):.4f}"
)

print(
    f"Transformed mean  : "
    f"{np.mean(np.abs(x_scaled)):.4f}"
)

print(
    f"Transformed std   : "
    f"{np.std(x_scaled):.4f}"
)


# ============================================================
# BUILD PYTORCH GEOMETRIC DATA
# ============================================================

live_data = Data(
    x=torch.tensor(
        x_scaled,
        dtype=torch.float32,
    ),
    edge_index=edge_index,
    edge_weight=edge_weight,
)


# ============================================================
# GRAPHSAGE MODEL
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
# LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    MODEL_FILE,
    map_location="cpu",
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

model.eval()


# ============================================================
# TARGET NODE
# ============================================================

target_address = (
    ADDRESS.lower()
)

target_node = node_to_id[
    target_address
]


# ============================================================
# INFERENCE
# ============================================================

with torch.no_grad():

    logits = model(
        live_data.x,
        live_data.edge_index,
    )

    probability = torch.sigmoid(
        logits[target_node]
    ).item()


# ============================================================
# RESULT
# ============================================================

threshold = checkpoint.get(
    "threshold",
    0.50,
)

prediction = (
    "MALICIOUS"
    if probability >= threshold
    else "BENIGN"
)

print()
print("=" * 70)
print("LIVE GRAPHSAGE INFERENCE")
print("=" * 70)

print(
    f"Target address : "
    f"{target_address}"
)

print(
    f"Target node    : "
    f"{target_node}"
)

print(
    f"Graph nodes    : "
    f"{live_data.num_nodes:,}"
)

print(
    f"Graph edges    : "
    f"{live_data.num_edges:,}"
)

print(
    f"Risk probability: "
    f"{probability:.6f}"
)

print(
    f"Threshold       : "
    f"{threshold:.4f}"
)

print(
    f"Prediction      : "
    f"{prediction}"
)