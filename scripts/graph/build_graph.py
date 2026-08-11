from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data


# ============================================================
# CONFIG
# ============================================================

FEATURE_FILE = Path(
    "data/processed/combined_wallet_features_v03.csv"
)

ETH_TX_FILE = Path(
    "data/raw/transactions.csv"
)

ERC20_TX_FILE = Path(
    "data/raw/token_transactions.csv"
)

OUTPUT_FILE = Path(
    "data/processed/graph_v03.pt"
)


# ============================================================
# FEATURES
# ============================================================

ETH_FEATURES = [
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

ALL_FEATURES = (
    ETH_FEATURES
    + ERC20_FEATURES
)


# ============================================================
# HELPERS
# ============================================================

def normalize_addresses(
    df,
    columns,
):
    for column in columns:
        if column in df.columns:
            df[column] = (
                df[column]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.lower()
            )

    return df


def find_column(
    df,
    candidates,
    description,
):
    for column in candidates:
        if column in df.columns:
            return column

    raise RuntimeError(
        f"Could not find {description}. "
        f"Expected one of: {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


# ============================================================
# LOAD FEATURE DATASET
# ============================================================

print("=" * 70)
print("LOADING v0.3 NODE FEATURES")
print("=" * 70)

features = pd.read_csv(
    FEATURE_FILE
)

features["address"] = (
    features["address"]
    .astype(str)
    .str.strip()
    .str.lower()
)

print(
    f"Wallets : {len(features)}"
)

print(
    f"Features: {len(ALL_FEATURES)}"
)


# ============================================================
# VALIDATE FEATURE DATASET
# ============================================================

if len(features) != 810:
    raise RuntimeError(
        f"Expected 810 wallets, "
        f"found {len(features)}"
    )

if len(ALL_FEATURES) != 38:
    raise RuntimeError(
        "Expected 38 node features."
    )

if features["address"].duplicated().any():
    raise RuntimeError(
        "Duplicate wallet addresses detected."
    )

missing_features = [
    column
    for column in ALL_FEATURES
    if column not in features.columns
]

if missing_features:
    raise RuntimeError(
        "Missing features: "
        + ", ".join(missing_features)
    )

if features[ALL_FEATURES].isna().any().any():
    raise RuntimeError(
        "NaN values detected in node features."
    )


# ============================================================
# CREATE NODE MAPPING
# ============================================================

print()
print("=" * 70)
print("CREATING NODE MAPPING")
print("=" * 70)

# IMPORTANT:
# This ordering becomes the permanent mapping between:
#
# node index <-> wallet address <-> feature row <-> label

wallet_to_node = {
    address: index
    for index, address
    in enumerate(
        features["address"]
    )
}

print(
    f"Node mapping created: "
    f"{len(wallet_to_node)} wallets"
)


# ============================================================
# LOAD ETH TRANSACTIONS
# ============================================================

print()
print("=" * 70)
print("LOADING ETH TRANSACTIONS")
print("=" * 70)

eth_tx = pd.read_csv(
    ETH_TX_FILE
)

print(
    f"ETH transaction rows: "
    f"{len(eth_tx):,}"
)


eth_from_col = find_column(
    eth_tx,
    [
        "from_address",
        "from",
        "sender",
    ],
    "ETH sender column",
)

eth_to_col = find_column(
    eth_tx,
    [
        "to_address",
        "to",
        "receiver",
    ],
    "ETH receiver column",
)

eth_tx = normalize_addresses(
    eth_tx,
    [
        eth_from_col,
        eth_to_col,
    ],
)


# ============================================================
# LOAD ERC-20 TRANSACTIONS
# ============================================================

print()
print("=" * 70)
print("LOADING ERC-20 TRANSACTIONS")
print("=" * 70)

erc20_tx = pd.read_csv(
    ERC20_TX_FILE
)

print(
    f"ERC-20 transaction rows: "
    f"{len(erc20_tx):,}"
)


erc20_from_col = find_column(
    erc20_tx,
    [
        "from_address",
        "from",
        "sender",
    ],
    "ERC-20 sender column",
)

erc20_to_col = find_column(
    erc20_tx,
    [
        "to_address",
        "to",
        "receiver",
    ],
    "ERC-20 receiver column",
)

erc20_tx = normalize_addresses(
    erc20_tx,
    [
        erc20_from_col,
        erc20_to_col,
    ],
)


# ============================================================
# BUILD ETH EDGES
# ============================================================

print()
print("=" * 70)
print("BUILDING ETH EDGES")
print("=" * 70)

eth_edges = {}

eth_used = 0
eth_unknown = 0
eth_self_loops = 0

for row in eth_tx.itertuples(
    index=False
):

    sender = getattr(
        row,
        eth_from_col,
    )

    receiver = getattr(
        row,
        eth_to_col,
    )

    if (
        sender not in wallet_to_node
        or receiver not in wallet_to_node
    ):

        eth_unknown += 1
        continue

    source = wallet_to_node[
        sender
    ]

    target = wallet_to_node[
        receiver
    ]

    if source == target:

        eth_self_loops += 1
        continue

    key = (
        source,
        target,
    )

    eth_edges[key] = (
        eth_edges.get(
            key,
            0,
        )
        + 1
    )

    eth_used += 1


print(
    f"ETH edges from transactions : "
    f"{eth_used:,}"
)

print(
    f"Unique ETH edges            : "
    f"{len(eth_edges):,}"
)

print(
    f"Unknown endpoint rows       : "
    f"{eth_unknown:,}"
)

print(
    f"Self-loop rows              : "
    f"{eth_self_loops:,}"
)


# ============================================================
# BUILD ERC-20 EDGES
# ============================================================

print()
print("=" * 70)
print("BUILDING ERC-20 EDGES")
print("=" * 70)

erc20_edges = {}

erc20_used = 0
erc20_unknown = 0
erc20_self_loops = 0

for row in erc20_tx.itertuples(
    index=False
):

    sender = getattr(
        row,
        erc20_from_col,
    )

    receiver = getattr(
        row,
        erc20_to_col,
    )

    if (
        sender not in wallet_to_node
        or receiver not in wallet_to_node
    ):

        erc20_unknown += 1
        continue

    source = wallet_to_node[
        sender
    ]

    target = wallet_to_node[
        receiver
    ]

    if source == target:

        erc20_self_loops += 1
        continue

    key = (
        source,
        target,
    )

    erc20_edges[key] = (
        erc20_edges.get(
            key,
            0,
        )
        + 1
    )

    erc20_used += 1


print(
    f"ERC-20 edges from transactions : "
    f"{erc20_used:,}"
)

print(
    f"Unique ERC-20 edges            : "
    f"{len(erc20_edges):,}"
)

print(
    f"Unknown endpoint rows          : "
    f"{erc20_unknown:,}"
)

print(
    f"Self-loop rows                 : "
    f"{erc20_self_loops:,}"
)


# ============================================================
# COMBINE EDGES
# ============================================================

print()
print("=" * 70)
print("COMBINING ETH + ERC-20 EDGES")
print("=" * 70)

combined_edges = {}

for edge, count in eth_edges.items():

    combined_edges[edge] = (
        combined_edges.get(
            edge,
            0,
        )
        + count
    )

for edge, count in erc20_edges.items():

    combined_edges[edge] = (
        combined_edges.get(
            edge,
            0,
        )
        + count
    )


print(
    f"Unique combined edges: "
    f"{len(combined_edges):,}"
)


# ============================================================
# CREATE PYTORCH GEOMETRIC EDGE INDEX
# ============================================================

edge_list = list(
    combined_edges.keys()
)

if len(edge_list) == 0:

    raise RuntimeError(
        "No valid graph edges were created."
    )


edge_index = torch.tensor(
    edge_list,
    dtype=torch.long,
).t().contiguous()


# ============================================================
# EDGE WEIGHTS
# ============================================================

edge_weights = torch.tensor(
    [
        combined_edges[edge]
        for edge in edge_list
    ],
    dtype=torch.float,
)


# ============================================================
# NODE FEATURES
# ============================================================

x = torch.tensor(
    features[
        ALL_FEATURES
    ].to_numpy(
        dtype="float32"
    ),
    dtype=torch.float,
)


# ============================================================
# LABELS
# ============================================================

y = torch.tensor(
    features[
        "label"
    ].to_numpy(),
    dtype=torch.long,
)


# ============================================================
# NODE ID
# ============================================================

node_id = torch.arange(
    len(features),
    dtype=torch.long,
)


# ============================================================
# BUILD DATA OBJECT
# ============================================================

data = Data(
    x=x,
    edge_index=edge_index,
    edge_weight=edge_weights,
    y=y,
    node_id=node_id,
)


# ============================================================
# GRAPH VALIDATION
# ============================================================

print()
print("=" * 70)
print("GRAPH VALIDATION")
print("=" * 70)

print(
    f"Nodes           : "
    f"{data.num_nodes}"
)

print(
    f"Node features   : "
    f"{data.num_node_features}"
)

print(
    f"Edges           : "
    f"{data.num_edges}"
)

print(
    f"Labels          : "
    f"{data.y.shape[0]}"
)

print(
    f"Edge weights    : "
    f"{data.edge_weight.shape[0]}"
)


if data.num_nodes != 810:

    raise RuntimeError(
        "Graph node count does not match "
        "the v0.3 feature dataset."
    )


if data.num_node_features != 38:

    raise RuntimeError(
        "Graph does not contain exactly "
        "38 node features."
    )


if data.y.shape[0] != 810:

    raise RuntimeError(
        "Label count does not match node count."
    )


if data.edge_weight.shape[0] != data.num_edges:

    raise RuntimeError(
        "Edge weight count does not match "
        "edge count."
    )


if torch.isnan(data.x).any():

    raise RuntimeError(
        "NaN detected in graph node features."
    )


if torch.isinf(data.x).any():

    raise RuntimeError(
        "Infinite value detected in graph node features."
    )


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

torch.save(
    data,
    OUTPUT_FILE,
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("GRAPH v0.3 CREATED")
print("=" * 70)

print(
    f"Nodes           : {data.num_nodes}"
)

print(
    f"Features/node   : {data.num_node_features}"
)

print(
    f"Unique edges    : {data.num_edges:,}"
)

print(
    f"ETH edge types  : {len(eth_edges):,}"
)

print(
    f"ERC20 edge types: {len(erc20_edges):,}"
)

print()
print(
    "Saved:"
)

print(
    OUTPUT_FILE
)