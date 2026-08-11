from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = (
    ROOT
    / "data"
    / "processed"
    / "graph"
    / "2hop_bounded_features.csv"
)

NODE_FILE = (
    ROOT
    / "data"
    / "processed"
    / "graph"
    / "2hop_bounded_node_universe.csv"
)

ETH_TX_FILE = (
    ROOT
    / "data"
    / "raw"
    / "transactions.csv"
)

ERC20_TX_FILE = (
    ROOT
    / "data"
    / "raw"
    / "token_transactions.csv"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "graph"
    / "graph_v03_2hop.pt"
)


# ============================================================
# HELPERS
# ============================================================

def normalize(series):
    return (
        series
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )


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
        f"Expected one of {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


# ============================================================
# LOAD FEATURES
# ============================================================

print("=" * 70)
print("LOADING 2-HOP FEATURES")
print("=" * 70)

features = pd.read_csv(
    FEATURE_FILE
)

features["address"] = normalize(
    features["address"]
)

print(
    f"Nodes    : {len(features):,}"
)

print(
    f"Features : "
    f"{len(features.columns) - 5}"
)

if len(features) != 8630:
    raise RuntimeError(
        f"Expected 8,630 nodes, "
        f"found {len(features)}"
    )


# ============================================================
# LOAD NODE UNIVERSE
# ============================================================

print()
print("=" * 70)
print("LOADING NODE UNIVERSE")
print("=" * 70)

nodes = pd.read_csv(
    NODE_FILE
)

nodes["address"] = normalize(
    nodes["address"]
)

if len(nodes) != 8630:
    raise RuntimeError(
        "Node universe does not contain "
        "8,630 nodes."
    )

if set(features["address"]) != set(
    nodes["address"]
):

    raise RuntimeError(
        "Feature dataset and node universe "
        "contain different wallets."
    )

print(
    "Node universe alignment: PASSED"
)


# ============================================================
# CREATE NODE MAPPING
# ============================================================

print()
print("=" * 70)
print("CREATING NODE MAPPING")
print("=" * 70)

# The feature file's ordering becomes the
# permanent node ordering.

wallet_to_node = {
    address: index
    for index, address
    in enumerate(
        features["address"]
    )
}

print(
    f"Node mapping: "
    f"{len(wallet_to_node):,}"
)


# ============================================================
# LABEL / MASK INFORMATION
# ============================================================

labels = features[
    "label"
].to_numpy()

y = torch.tensor(
    labels,
    dtype=torch.long,
)

# Only original labeled wallets are supervised.

labeled_mask = (
    features["label"] != -1
)

train_mask = torch.zeros(
    len(features),
    dtype=torch.bool,
)

val_mask = torch.zeros(
    len(features),
    dtype=torch.bool,
)

test_mask = torch.zeros(
    len(features),
    dtype=torch.bool,
)


# ============================================================
# LOAD SPLITS
# ============================================================

TRAIN_FILE = (
    ROOT
    / "data"
    / "splits"
    / "train_addresses.csv"
)

VAL_FILE = (
    ROOT
    / "data"
    / "splits"
    / "val_addresses.csv"
)

TEST_FILE = (
    ROOT
    / "data"
    / "splits"
    / "test_addresses.csv"
)


def load_split(
    path,
):
    df = pd.read_csv(
        path
    )

    address_column = find_column(
        df,
        [
            "address",
            "wallet_address",
        ],
        "split address column",
    )

    return set(
        normalize(
            df[address_column]
        )
    )


train_addresses = load_split(
    TRAIN_FILE
)

val_addresses = load_split(
    VAL_FILE
)

test_addresses = load_split(
    TEST_FILE
)


# ------------------------------------------------------------
# Apply masks
# ------------------------------------------------------------

for address in train_addresses:

    if address not in wallet_to_node:
        continue

    train_mask[
        wallet_to_node[address]
    ] = True


for address in val_addresses:

    if address not in wallet_to_node:
        continue

    val_mask[
        wallet_to_node[address]
    ] = True


for address in test_addresses:

    if address not in wallet_to_node:
        continue

    test_mask[
        wallet_to_node[address]
    ] = True


# ============================================================
# MASK VALIDATION
# ============================================================

print()
print("=" * 70)
print("SUPERVISED MASK VALIDATION")
print("=" * 70)

print(
    f"Labeled nodes : "
    f"{labeled_mask.sum():,}"
)

print(
    f"Train nodes   : "
    f"{train_mask.sum():,}"
)

print(
    f"Validation    : "
    f"{val_mask.sum():,}"
)

print(
    f"Test nodes    : "
    f"{test_mask.sum():,}"
)

print(
    f"Unlabeled     : "
    f"{(~labeled_mask).sum():,}"
)


# No unlabeled node may accidentally
# enter a supervised split.

if (
    train_mask
    & ~torch.tensor(
        labeled_mask.to_numpy()
        if hasattr(
            labeled_mask,
            "to_numpy",
        )
        else labeled_mask
    )
).any():

    raise RuntimeError(
        "Unlabeled node found in training mask."
    )


# Convert masks to tensors safely.

labeled_mask_tensor = torch.tensor(
    labeled_mask.to_numpy()
    if hasattr(
        labeled_mask,
        "to_numpy",
    )
    else labeled_mask,
    dtype=torch.bool,
)


if (
    train_mask
    & ~labeled_mask_tensor
).any():

    raise RuntimeError(
        "Training mask contains unlabeled nodes."
    )

if (
    val_mask
    & ~labeled_mask_tensor
).any():

    raise RuntimeError(
        "Validation mask contains unlabeled nodes."
    )

if (
    test_mask
    & ~labeled_mask_tensor
).any():

    raise RuntimeError(
        "Test mask contains unlabeled nodes."
    )


# ============================================================
# LOAD ETH TRANSACTIONS
# ============================================================

print()
print("=" * 70)
print("LOADING ETH TRANSACTIONS")
print("=" * 70)

eth = pd.read_csv(
    ETH_TX_FILE
)

eth_from = find_column(
    eth,
    [
        "from_address",
        "from",
        "sender",
    ],
    "ETH sender column",
)

eth_to = find_column(
    eth,
    [
        "to_address",
        "to",
        "receiver",
    ],
    "ETH receiver column",
)

eth[eth_from] = normalize(
    eth[eth_from]
)

eth[eth_to] = normalize(
    eth[eth_to]
)

print(
    f"ETH rows: "
    f"{len(eth):,}"
)


# ============================================================
# LOAD ERC-20 TRANSACTIONS
# ============================================================

print()
print("=" * 70)
print("LOADING ERC-20 TRANSACTIONS")
print("=" * 70)

erc20 = pd.read_csv(
    ERC20_TX_FILE
)

erc20_from = find_column(
    erc20,
    [
        "from_address",
        "from",
        "sender",
    ],
    "ERC-20 sender column",
)

erc20_to = find_column(
    erc20,
    [
        "to_address",
        "to",
        "receiver",
    ],
    "ERC-20 receiver column",
)

erc20[erc20_from] = normalize(
    erc20[erc20_from]
)

erc20[erc20_to] = normalize(
    erc20[erc20_to]
)

print(
    f"ERC-20 rows: "
    f"{len(erc20):,}"
)


# ============================================================
# BUILD DIRECTED EDGES
# ============================================================

print()
print("=" * 70)
print("BUILDING DIRECTED EDGES")
print("=" * 70)

eth_edges = {}

eth_used = 0
eth_unknown = 0
eth_self = 0


for sender, receiver in zip(
    eth[eth_from],
    eth[eth_to],
):

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

        eth_self += 1
        continue

    edge = (
        source,
        target,
    )

    eth_edges[edge] = (
        eth_edges.get(
            edge,
            0,
        )
        + 1
    )

    eth_used += 1


erc20_edges = {}

erc20_used = 0
erc20_unknown = 0
erc20_self = 0


for sender, receiver in zip(
    erc20[erc20_from],
    erc20[erc20_to],
):

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

        erc20_self += 1
        continue

    edge = (
        source,
        target,
    )

    erc20_edges[edge] = (
        erc20_edges.get(
            edge,
            0,
        )
        + 1
    )

    erc20_used += 1


print(
    f"ETH transaction edges used : "
    f"{eth_used:,}"
)

print(
    f"Unique ETH edges            : "
    f"{len(eth_edges):,}"
)

print(
    f"ETH unknown endpoints       : "
    f"{eth_unknown:,}"
)

print(
    f"ETH self-loops              : "
    f"{eth_self:,}"
)

print()

print(
    f"ERC-20 transaction edges used : "
    f"{erc20_used:,}"
)

print(
    f"Unique ERC-20 edges            : "
    f"{len(erc20_edges):,}"
)

print(
    f"ERC-20 unknown endpoints       : "
    f"{erc20_unknown:,}"
)

print(
    f"ERC-20 self-loops              : "
    f"{erc20_self:,}"
)


# ============================================================
# COMBINE EDGES
# ============================================================

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


print()
print(
    f"Unique combined directed edges: "
    f"{len(combined_edges):,}"
)


# ============================================================
# EDGE INDEX
# ============================================================

edge_list = list(
    combined_edges.keys()
)

if not edge_list:
    raise RuntimeError(
        "No graph edges were created."
    )

edge_index = torch.tensor(
    edge_list,
    dtype=torch.long,
).t().contiguous()

edge_weight = torch.tensor(
    [
        combined_edges[e]
        for e in edge_list
    ],
    dtype=torch.float32,
)


# ============================================================
# NODE FEATURES
# ============================================================

# First five columns are:
# node_id, address, label, hop, node_type

feature_columns = [
    column
    for column in features.columns
    if column
    not in [
        "node_id",
        "address",
        "label",
        "hop",
        "node_type",
    ]
]

if len(feature_columns) != 38:

    raise RuntimeError(
        f"Expected 38 feature columns, "
        f"found {len(feature_columns)}"
    )

x = torch.tensor(
    features[
        feature_columns
    ].to_numpy(
        dtype="float32"
    ),
    dtype=torch.float32,
)


# ============================================================
# GRAPH DATA
# ============================================================

data = Data(
    x=x,
    edge_index=edge_index,
    edge_weight=edge_weight,
    y=y,
    train_mask=train_mask,
    val_mask=val_mask,
    test_mask=test_mask,
    labeled_mask=labeled_mask_tensor,
)


# ============================================================
# VALIDATION
# ============================================================

print()
print("=" * 70)
print("GRAPH VALIDATION")
print("=" * 70)

print(
    f"Nodes             : "
    f"{data.num_nodes:,}"
)

print(
    f"Features/node     : "
    f"{data.num_node_features}"
)

print(
    f"Edges             : "
    f"{data.num_edges:,}"
)

print(
    f"Edge weights      : "
    f"{data.edge_weight.shape[0]:,}"
)

print(
    f"Labeled nodes     : "
    f"{data.labeled_mask.sum():,}"
)

print(
    f"Train nodes       : "
    f"{data.train_mask.sum():,}"
)

print(
    f"Validation nodes  : "
    f"{data.val_mask.sum():,}"
)

print(
    f"Test nodes        : "
    f"{data.test_mask.sum():,}"
)


# ------------------------------------------------------------
# Hard assertions
# ------------------------------------------------------------

if data.num_nodes != 8630:

    raise RuntimeError(
        "Incorrect node count."
    )

if data.num_node_features != 38:

    raise RuntimeError(
        "Incorrect feature count."
    )

if data.y.shape[0] != 8630:

    raise RuntimeError(
        "Label count does not match nodes."
    )

if data.edge_weight.shape[0] != data.num_edges:

    raise RuntimeError(
        "Edge weights do not match edges."
    )

if data.train_mask.sum() == 0:

    raise RuntimeError(
        "Training mask is empty."
    )

if data.val_mask.sum() == 0:

    raise RuntimeError(
        "Validation mask is empty."
    )

if data.test_mask.sum() == 0:

    raise RuntimeError(
        "Test mask is empty."
    )

if (
    data.train_mask
    & ~data.labeled_mask
).any():

    raise RuntimeError(
        "Unlabeled nodes found in training mask."
    )

if (
    data.val_mask
    & ~data.labeled_mask
).any():

    raise RuntimeError(
        "Unlabeled nodes found in validation mask."
    )

if (
    data.test_mask
    & ~data.labeled_mask
).any():

    raise RuntimeError(
        "Unlabeled nodes found in test mask."
    )

if torch.isnan(data.x).any():

    raise RuntimeError(
        "NaN values found in node features."
    )

if torch.isinf(data.x).any():

    raise RuntimeError(
        "Infinite values found in node features."
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
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("2-HOP GRAPH CREATED")
print("=" * 70)

print(
    f"Nodes             : "
    f"{data.num_nodes:,}"
)

print(
    f"Features/node     : "
    f"{data.num_node_features}"
)

print(
    f"Unique edges      : "
    f"{data.num_edges:,}"
)

print(
    f"Supervised nodes  : "
    f"{data.labeled_mask.sum():,}"
)

print(
    f"Unlabeled context : "
    f"{(~data.labeled_mask).sum():,}"
)

print()
print(
    "Saved:"
)

print(
    OUTPUT_FILE
)