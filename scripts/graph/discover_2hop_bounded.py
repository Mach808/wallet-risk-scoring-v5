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
    "data/processed/graph/2hop_bounded_node_universe.csv"
)

TOP_K = 20


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
# LOAD LABELED WALLETS
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

if len(labeled_wallets) != 810:
    raise RuntimeError(
        f"Expected 810 labeled wallets, "
        f"found {len(labeled_wallets)}"
    )

print(
    f"Labeled wallets: {len(labeled_wallets)}"
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
# ADDRESS COLUMNS
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
# BUILD INTERACTION COUNTS
# ============================================================

print()
print("=" * 70)
print("BUILDING INTERACTION COUNTS")
print("=" * 70)


def build_interactions(
    df,
    from_col,
    to_col,
):

    interactions = {}

    for sender, receiver in zip(
        df[from_col],
        df[to_col],
    ):

        if not sender or not receiver:
            continue

        if sender == receiver:
            continue

        interactions[
            (sender, receiver)
        ] = (
            interactions.get(
                (sender, receiver),
                0,
            )
            + 1
        )

        # Treat the graph as undirected
        # for neighborhood discovery.
        interactions[
            (receiver, sender)
        ] = (
            interactions.get(
                (receiver, sender),
                0,
            )
            + 1
        )

    return interactions


eth_interactions = build_interactions(
    eth,
    eth_from,
    eth_to,
)

erc20_interactions = build_interactions(
    erc20,
    erc20_from,
    erc20_to,
)

print(
    f"Unique ETH wallet relationships   : "
    f"{len(eth_interactions):,}"
)

print(
    f"Unique ERC-20 wallet relationships: "
    f"{len(erc20_interactions):,}"
)


# ============================================================
# COMBINE INTERACTION COUNTS
# ============================================================

combined_interactions = {}

for pair, count in eth_interactions.items():

    combined_interactions[pair] = (
        combined_interactions.get(
            pair,
            0,
        )
        + count
    )

for pair, count in erc20_interactions.items():

    combined_interactions[pair] = (
        combined_interactions.get(
            pair,
            0,
        )
        + count
    )


# ============================================================
# BUILD ADJACENCY
# ============================================================

adjacency = {}

for (source, target), count in (
    combined_interactions.items()
):

    adjacency.setdefault(
        source,
        [],
    ).append(
        (
            target,
            count,
        )
    )


# Sort once so top-K selection is deterministic.

for wallet in adjacency:

    adjacency[wallet].sort(
        key=lambda x: (
            -x[1],
            x[0],
        )
    )


# ============================================================
# DISCOVER TOP-K NEIGHBORS
# ============================================================

def top_k_neighbors(
    seeds,
):

    neighbors = set()

    for wallet in seeds:

        candidates = adjacency.get(
            wallet,
            [],
        )

        # Remove already-known seeds from
        # the candidate ranking.

        candidates = [
            item
            for item in candidates
            if item[0] not in seeds
        ]

        for target, count in candidates[
            :TOP_K
        ]:

            neighbors.add(
                target
            )

    return neighbors


# ============================================================
# 1-HOP
# ============================================================

print()
print("=" * 70)
print(
    f"DISCOVERING 1-HOP "
    f"TOP-{TOP_K} NEIGHBORS"
)
print("=" * 70)

hop1 = top_k_neighbors(
    labeled_wallets
)

hop1 -= labeled_wallets

print(
    f"1-hop new wallets: "
    f"{len(hop1):,}"
)


# ============================================================
# 2-HOP
# ============================================================

print()
print("=" * 70)
print(
    f"DISCOVERING 2-HOP "
    f"TOP-{TOP_K} NEIGHBORS"
)
print("=" * 70)

hop2 = top_k_neighbors(
    hop1
)

hop2 -= labeled_wallets
hop2 -= hop1

print(
    f"2-hop new wallets: "
    f"{len(hop2):,}"
)


# ============================================================
# NODE UNIVERSE
# ============================================================

all_nodes = (
    labeled_wallets
    | hop1
    | hop2
)

print()
print("=" * 70)
print("FINAL BOUNDED 2-HOP NODE UNIVERSE")
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

label_lookup = (
    features
    .set_index("address")[
        "label"
    ]
    .to_dict()
)

rows = []

for address in labeled_wallets:

    rows.append(
        {
            "address": address,
            "label": int(
                label_lookup[address]
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
# SORT AND ASSIGN NODE IDS
# ============================================================

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

labeled_rows = nodes[
    nodes["hop"] == 0
]

unlabeled_rows = nodes[
    nodes["hop"] > 0
]

if (
    labeled_rows["label"] == -1
).any():

    raise RuntimeError(
        "Labeled nodes have label -1."
    )

if (
    unlabeled_rows["label"] != -1
).any():

    raise RuntimeError(
        "Unlabeled nodes have known labels."
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
print("BOUNDED 2-HOP NODE UNIVERSE SAVED")
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