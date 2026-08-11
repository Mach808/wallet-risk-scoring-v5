from pathlib import Path

import pandas as pd


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
    "data/processed/graph/2hop_node_universe.csv"
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


def find_column(df, candidates, description):

    for column in candidates:
        if column in df.columns:
            return column

    raise RuntimeError(
        f"Could not find {description}. "
        f"Expected one of {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


# ============================================================
# LOAD ORIGINAL LABELED WALLETS
# ============================================================

print("=" * 70)
print("LOADING LABELED WALLETS")
print("=" * 70)

features = pd.read_csv(
    FEATURE_FILE
)

features["address"] = normalize(
    features["address"]
)

labeled_wallets = set(
    features["address"]
)

print(
    f"Labeled wallets: "
    f"{len(labeled_wallets)}"
)

if len(labeled_wallets) != 810:
    raise RuntimeError(
        f"Expected 810 labeled wallets, "
        f"found {len(labeled_wallets)}"
    )


# ============================================================
# LOAD TRANSACTIONS
# ============================================================

print()
print("=" * 70)
print("LOADING TRANSACTIONS")
print("=" * 70)

eth = pd.read_csv(
    ETH_TX_FILE
)

erc20 = pd.read_csv(
    ERC20_TX_FILE
)

print(
    f"ETH transactions   : {len(eth):,}"
)

print(
    f"ERC-20 transactions: {len(erc20):,}"
)


# ============================================================
# IDENTIFY ADDRESS COLUMNS
# ============================================================

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


eth[eth_from] = normalize(
    eth[eth_from]
)

eth[eth_to] = normalize(
    eth[eth_to]
)

erc20[erc20_from] = normalize(
    erc20[erc20_from]
)

erc20[erc20_to] = normalize(
    erc20[erc20_to]
)


# ============================================================
# BUILD DIRECTED ADJACENCY
# ============================================================

print()
print("=" * 70)
print("BUILDING TRANSACTION ADJACENCY")
print("=" * 70)

# adjacency[A] = wallets directly connected to A

adjacency = {}


def add_edges(df, from_col, to_col):

    for sender, receiver in zip(
        df[from_col],
        df[to_col],
    ):

        if not sender or not receiver:
            continue

        if sender == receiver:
            continue

        adjacency.setdefault(
            sender,
            set(),
        ).add(receiver)

        # Treat the transaction graph as
        # undirected for neighborhood discovery.
        adjacency.setdefault(
            receiver,
            set(),
        ).add(sender)


add_edges(
    eth,
    eth_from,
    eth_to,
)

add_edges(
    erc20,
    erc20_from,
    erc20_to,
)

print(
    f"Wallets appearing in adjacency: "
    f"{len(adjacency):,}"
)


# ============================================================
# 1-HOP
# ============================================================

print()
print("=" * 70)
print("DISCOVERING 1-HOP NEIGHBORS")
print("=" * 70)

hop1 = set()

for wallet in labeled_wallets:

    hop1.update(
        adjacency.get(
            wallet,
            set(),
        )
    )

# Never classify original labeled wallets as
# newly discovered nodes.

hop1 -= labeled_wallets


print(
    f"Original labeled wallets : "
    f"{len(labeled_wallets):,}"
)

print(
    f"1-hop new wallets        : "
    f"{len(hop1):,}"
)


# ============================================================
# 2-HOP
# ============================================================

print()
print("=" * 70)
print("DISCOVERING 2-HOP NEIGHBORS")
print("=" * 70)

hop2 = set()

for wallet in hop1:

    hop2.update(
        adjacency.get(
            wallet,
            set(),
        )
    )

# Remove original labeled wallets
# and already discovered 1-hop wallets.

hop2 -= labeled_wallets
hop2 -= hop1


print(
    f"2-hop new wallets        : "
    f"{len(hop2):,}"
)


# ============================================================
# OVERLAP ANALYSIS
# ============================================================

print()
print("=" * 70)
print("ETH / ERC-20 NEIGHBORHOOD OVERLAP")
print("=" * 70)


eth_adjacency = {}


def add_eth_edges():

    for sender, receiver in zip(
        eth[eth_from],
        eth[eth_to],
    ):

        if not sender or not receiver:
            continue

        if sender == receiver:
            continue

        eth_adjacency.setdefault(
            sender,
            set(),
        ).add(receiver)

        eth_adjacency.setdefault(
            receiver,
            set(),
        ).add(sender)


erc20_adjacency = {}


def add_erc20_edges():

    for sender, receiver in zip(
        erc20[erc20_from],
        erc20[erc20_to],
    ):

        if not sender or not receiver:
            continue

        if sender == receiver:
            continue

        erc20_adjacency.setdefault(
            sender,
            set(),
        ).add(receiver)

        erc20_adjacency.setdefault(
            receiver,
            set(),
        ).add(sender)


add_eth_edges()
add_erc20_edges()


def discover_neighbors(
    seeds,
    graph,
):

    result = set()

    for wallet in seeds:

        result.update(
            graph.get(
                wallet,
                set(),
            )
        )

    return result


eth_hop1 = (
    discover_neighbors(
        labeled_wallets,
        eth_adjacency,
    )
    - labeled_wallets
)

erc20_hop1 = (
    discover_neighbors(
        labeled_wallets,
        erc20_adjacency,
    )
    - labeled_wallets
)


print(
    f"ETH 1-hop new wallets   : "
    f"{len(eth_hop1):,}"
)

print(
    f"ERC-20 1-hop new wallets: "
    f"{len(erc20_hop1):,}"
)

print(
    f"1-hop overlap            : "
    f"{len(eth_hop1 & erc20_hop1):,}"
)


# ============================================================
# FINAL NODE UNIVERSE
# ============================================================

all_nodes = (
    labeled_wallets
    | hop1
    | hop2
)

print()
print("=" * 70)
print("FINAL 2-HOP NODE UNIVERSE")
print("=" * 70)

print(
    f"Labeled nodes : "
    f"{len(labeled_wallets):,}"
)

print(
    f"1-hop nodes   : "
    f"{len(hop1):,}"
)

print(
    f"2-hop nodes   : "
    f"{len(hop2):,}"
)

print(
    f"Total nodes   : "
    f"{len(all_nodes):,}"
)


# ============================================================
# CREATE NODE TABLE
# ============================================================

rows = []

for address in labeled_wallets:

    rows.append(
        {
            "address": address,
            "label": int(
                features.loc[
                    features["address"] == address,
                    "label",
                ].iloc[0]
            ),
            "hop": 0,
            "node_type": "labeled",
        }
    )


for address in hop1:

    rows.append(
        {
            "address": address,
            "label": -1,
            "hop": 1,
            "node_type": "unlabeled",
        }
    )


for address in hop2:

    rows.append(
        {
            "address": address,
            "label": -1,
            "hop": 2,
            "node_type": "unlabeled",
        }
    )


nodes = pd.DataFrame(
    rows
)


# ============================================================
# SORT / NODE ID
# ============================================================

# Keep labeled nodes first, then 1-hop,
# then 2-hop. This makes the mapping easy
# to inspect later.

nodes = nodes.sort_values(
    [
        "hop",
        "address",
    ]
).reset_index(
    drop=True
)

nodes.insert(
    0,
    "node_id",
    range(
        len(nodes)
    ),
)


# ============================================================
# VALIDATION
# ============================================================

print()
print("=" * 70)
print("NODE UNIVERSE VALIDATION")
print("=" * 70)

if nodes["address"].duplicated().any():

    raise RuntimeError(
        "Duplicate addresses detected."
    )

if len(nodes) != len(all_nodes):

    raise RuntimeError(
        "Node count mismatch."
    )

if (
    nodes[
        nodes["hop"] == 0
    ]["label"] == -1
).any():

    raise RuntimeError(
        "Labeled nodes have invalid labels."
    )

if (
    nodes[
        nodes["hop"] > 0
    ]["label"] != -1
).any():

    raise RuntimeError(
        "Unlabeled nodes have labels."
    )


print(
    "Duplicate addresses: 0"
)

print(
    "Label validation   : PASSED"
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

nodes.to_csv(
    OUTPUT_FILE,
    index=False,
)


print()
print("=" * 70)
print("2-HOP NODE UNIVERSE SAVED")
print("=" * 70)

print(
    OUTPUT_FILE
)

print()
print(
    nodes[
        "node_type"
    ].value_counts()
)

print()
print(
    nodes[
        "hop"
    ].value_counts()
    .sort_index()
)