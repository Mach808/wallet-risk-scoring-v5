from pathlib import Path
import importlib.util
import sys

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

NODE_FILE = (
    ROOT
    / "data"
    / "processed"
    / "graph"
    / "2hop_bounded_node_universe.csv"
)

ORIGINAL_FEATURE_FILE = (
    ROOT
    / "data"
    / "processed"
    / "combined_wallet_features_v03.csv"
)

ETH_TX_FILE = (
    ROOT
    / "data"
    / "raw"
    / "transactions.csv"
)

ERC20_FEATURE_FILE = (
    ROOT
    / "data"
    / "processed"
    / "erc20_wallet_features.csv"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "graph"
    / "2hop_bounded_features.csv"
)


# ============================================================
# FEATURE DEFINITIONS
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
# LOAD VALIDATED FEATURE FUNCTION
# ============================================================

FEATURE_ENGINEERING_FILE = (
    ROOT
    / "scripts"
    / "feature_engineering"
    / "feature_engineering.py"
)

spec = importlib.util.spec_from_file_location(
    "validated_feature_engineering",
    FEATURE_ENGINEERING_FILE,
)

feature_module = importlib.util.module_from_spec(
    spec
)

sys.modules[
    "validated_feature_engineering"
] = feature_module

spec.loader.exec_module(
    feature_module
)

compute_wallet_features = (
    feature_module.compute_wallet_features
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
# LOAD NODE UNIVERSE
# ============================================================

print("=" * 70)
print("LOADING 2-HOP NODE UNIVERSE")
print("=" * 70)

nodes = pd.read_csv(
    NODE_FILE
)

nodes["address"] = normalize(
    nodes["address"]
)

print(
    f"Total nodes: {len(nodes):,}"
)

print(
    nodes["hop"]
    .value_counts()
    .sort_index()
)


if len(nodes) != 8630:
    raise RuntimeError(
        f"Expected 8,630 nodes, "
        f"found {len(nodes)}"
    )

if nodes["address"].duplicated().any():
    raise RuntimeError(
        "Duplicate addresses in node universe."
    )


node_addresses = set(
    nodes["address"]
)


# ============================================================
# LOAD ORIGINAL 810 FEATURES
# ============================================================

print()
print("=" * 70)
print("LOADING ORIGINAL v0.3 FEATURES")
print("=" * 70)

original = pd.read_csv(
    ORIGINAL_FEATURE_FILE
)

original["address"] = normalize(
    original["address"]
)

print(
    f"Original wallets: "
    f"{len(original)}"
)

if len(original) != 810:
    raise RuntimeError(
        f"Expected 810 original wallets, "
        f"found {len(original)}"
    )

if original["address"].duplicated().any():
    raise RuntimeError(
        "Duplicate addresses in original feature dataset."
    )

missing_original_features = [
    f
    for f in ALL_FEATURES
    if f not in original.columns
]

if missing_original_features:
    raise RuntimeError(
        "Original dataset missing features: "
        + ", ".join(
            missing_original_features
        )
    )

original_lookup = (
    original
    .set_index("address")
)


# ============================================================
# VERIFY ORIGINAL NODES
# ============================================================

original_addresses = set(
    original["address"]
)

node_labeled_addresses = set(
    nodes.loc[
        nodes["hop"] == 0,
        "address"
    ]
)

if original_addresses != node_labeled_addresses:
    missing = (
        original_addresses
        - node_labeled_addresses
    )

    extra = (
        node_labeled_addresses
        - original_addresses
    )

    raise RuntimeError(
        "Original 810-wallet universe "
        "does not match node universe.\n"
        f"Missing: {len(missing)}\n"
        f"Extra: {len(extra)}"
    )

print(
    "Original 810-wallet alignment: PASSED"
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

print(
    f"Raw ETH rows: {len(eth):,}"
)


from_col = find_column(
    eth,
    [
        "from_address",
        "from",
        "sender",
    ],
    "ETH sender column",
)

to_col = find_column(
    eth,
    [
        "to_address",
        "to",
        "receiver",
    ],
    "ETH receiver column",
)

timestamp_col = find_column(
    eth,
    [
        "timestamp",
        "timeStamp",
        "time",
    ],
    "timestamp column",
)

value_col = find_column(
    eth,
    [
        "value",
        "eth_value",
        "amount",
    ],
    "value column",
)

category_col = find_column(
    eth,
    [
        "category",
        "type",
    ],
    "transaction category column",
)


# ============================================================
# NORMALIZE ETH DATA
# ============================================================

eth[from_col] = normalize(
    eth[from_col]
)

eth[to_col] = normalize(
    eth[to_col]
)

eth[value_col] = pd.to_numeric(
    eth[value_col],
    errors="coerce",
).fillna(0.0)

eth[timestamp_col] = pd.to_datetime(
    eth[timestamp_col],
    errors="coerce",
    utc=True,
)

eth[category_col] = (
    eth[category_col]
    .fillna("external")
    .astype(str)
    .str.lower()
    .str.strip()
)


# ============================================================
# CREATE WALLET-CENTRIC TRANSACTION ROWS
# ============================================================

print()
print("=" * 70)
print("BUILDING WALLET-CENTRIC ETH TRANSACTIONS")
print("=" * 70)

wallet_frames = []

# ------------------------------------------------------------
# Incoming
# ------------------------------------------------------------

incoming = eth[
    eth[to_col].isin(node_addresses)
].copy()

incoming = incoming[
    incoming[from_col] != incoming[to_col]
].copy()

incoming["wallet_address"] = incoming[
    to_col
]

incoming["direction"] = "incoming"

incoming["value"] = incoming[
    value_col
]

incoming["timestamp"] = incoming[
    timestamp_col
]

incoming["category"] = incoming[
    category_col
]

incoming = incoming[
    [
        "wallet_address",
        "direction",
        "value",
        from_col,
        to_col,
        "timestamp",
        "category",
    ]
].rename(
    columns={
        from_col: "from_address",
        to_col: "to_address",
    }
)

wallet_frames.append(
    incoming
)


# ------------------------------------------------------------
# Outgoing
# ------------------------------------------------------------

outgoing = eth[
    eth[from_col].isin(node_addresses)
].copy()

outgoing = outgoing[
    outgoing[from_col] != outgoing[to_col]
].copy()

outgoing["wallet_address"] = outgoing[
    from_col
]

outgoing["direction"] = "outgoing"

outgoing["value"] = outgoing[
    value_col
]

outgoing["timestamp"] = outgoing[
    timestamp_col
]

outgoing["category"] = outgoing[
    category_col
]

outgoing = outgoing[
    [
        "wallet_address",
        "direction",
        "value",
        from_col,
        to_col,
        "timestamp",
        "category",
    ]
].rename(
    columns={
        from_col: "from_address",
        to_col: "to_address",
    }
)

wallet_frames.append(
    outgoing
)


eth_wallet_tx = pd.concat(
    wallet_frames,
    ignore_index=True,
)

eth_wallet_tx["wallet_address"] = normalize(
    eth_wallet_tx["wallet_address"]
)

eth_wallet_tx["from_address"] = normalize(
    eth_wallet_tx["from_address"]
)

eth_wallet_tx["to_address"] = normalize(
    eth_wallet_tx["to_address"]
)

print(
    f"Wallet-centric ETH rows: "
    f"{len(eth_wallet_tx):,}"
)

print(
    f"Wallets with ETH activity: "
    f"{eth_wallet_tx['wallet_address'].nunique():,}"
)


# ============================================================
# COMPUTE ETH FEATURES
# ============================================================

print()
print("=" * 70)
print("COMPUTING ETH FEATURES FOR EXPANDED NODES")
print("=" * 70)

eth_feature_rows = []

for i, (wallet, group) in enumerate(
    eth_wallet_tx.groupby(
        "wallet_address",
        sort=False,
    ),
    start=1,
):

    result = compute_wallet_features(
        group
    )

    eth_feature_rows.append(
        result
    )

    if i % 1000 == 0:
        print(
            f"Processed ETH wallets: "
            f"{i:,}"
        )


eth_features = pd.DataFrame(
    eth_feature_rows
)

eth_features["address"] = normalize(
    eth_features["address"]
)

print(
    f"ETH feature wallets: "
    f"{len(eth_features):,}"
)


# ============================================================
# LOAD ERC-20 FEATURES
# ============================================================

print()
print("=" * 70)
print("LOADING ERC-20 FEATURES")
print("=" * 70)

erc20 = pd.read_csv(
    ERC20_FEATURE_FILE
)

erc20["address"] = normalize(
    erc20["address"]
)

print(
    f"ERC-20 feature wallets: "
    f"{len(erc20):,}"
)


missing_erc20_features = [
    f
    for f in ERC20_FEATURES
    if f not in erc20.columns
]

if missing_erc20_features:
    raise RuntimeError(
        "ERC-20 dataset missing features: "
        + ", ".join(
            missing_erc20_features
        )
    )


if erc20["address"].duplicated().any():
    raise RuntimeError(
        "Duplicate addresses in ERC-20 features."
    )


# ============================================================
# BUILD FEATURE DATASET
# ============================================================

print()
print("=" * 70)
print("BUILDING 2-HOP FEATURE DATASET")
print("=" * 70)


# Start with node universe.

result = nodes[
    [
        "node_id",
        "address",
        "label",
        "hop",
        "node_type",
    ]
].copy()


# ------------------------------------------------------------
# Add ETH features
# ------------------------------------------------------------

result = result.merge(
    eth_features[
        [
            "address"
        ]
        + ETH_FEATURES
    ],
    on="address",
    how="left",
)


# ------------------------------------------------------------
# Add ERC-20 features
# ------------------------------------------------------------

result = result.merge(
    erc20[
        [
            "address"
        ]
        + ERC20_FEATURES
    ],
    on="address",
    how="left",
)


# ============================================================
# FILL ABSENT ACTIVITY WITH ZERO
# ============================================================

for feature in ALL_FEATURES:

    result[feature] = pd.to_numeric(
        result[feature],
        errors="coerce",
    )

    result[feature] = (
        result[feature]
        .fillna(0.0)
    )


# ============================================================
# RESTORE EXACT ORIGINAL FEATURES
# ============================================================

print()
print(
    "Restoring exact v0.3 values "
    "for original 810 wallets..."
)

for feature in ALL_FEATURES:

    result.loc[
        result["hop"] == 0,
        feature,
    ] = result.loc[
        result["hop"] == 0,
        "address",
    ].map(
        original_lookup[feature]
    )


# ============================================================
# FEATURE ORDER
# ============================================================

result = result[
    [
        "node_id",
        "address",
        "label",
        "hop",
        "node_type",
    ]
    + ALL_FEATURES
]


# ============================================================
# VALIDATION
# ============================================================

print()
print("=" * 70)
print("FINAL DATASET VALIDATION")
print("=" * 70)

print(
    f"Rows     : {len(result):,}"
)

print(
    f"Features : {len(ALL_FEATURES)}"
)

print(
    f"Labeled  : "
    f"{(result['label'] != -1).sum():,}"
)

print(
    f"Unlabeled: "
    f"{(result['label'] == -1).sum():,}"
)

print(
    f"NaN values: "
    f"{result[ALL_FEATURES].isna().sum().sum()}"
)

print(
    f"Infinite values: "
    f"{np.isinf(
        result[ALL_FEATURES]
        .to_numpy(dtype=float)
    ).sum()}"
)

print(
    f"Duplicate addresses: "
    f"{result['address'].duplicated().sum()}"
)


# ------------------------------------------------------------
# Assertions
# ------------------------------------------------------------

if len(result) != 8630:
    raise RuntimeError(
        f"Expected 8,630 rows, "
        f"found {len(result)}"
    )

if len(ALL_FEATURES) != 38:
    raise RuntimeError(
        "Expected exactly 38 features."
    )

if (
    result["label"] != -1
).sum() != 810:
    raise RuntimeError(
        "Expected exactly 810 labeled nodes."
    )

if (
    result["label"] == -1
).sum() != 7820:
    raise RuntimeError(
        "Expected exactly 7,820 unlabeled nodes."
    )

if result["address"].duplicated().any():
    raise RuntimeError(
        "Duplicate addresses detected."
    )

if result[ALL_FEATURES].isna().any().any():
    raise RuntimeError(
        "NaN values detected."
    )

if np.isinf(
    result[
        ALL_FEATURES
    ].to_numpy(
        dtype=float
    )
).any():
    raise RuntimeError(
        "Infinite values detected."
    )


# ============================================================
# VERIFY ORIGINAL 810 FEATURE EQUIVALENCE
# ============================================================

print()
print("=" * 70)
print("VERIFYING ORIGINAL 810 FEATURE EQUIVALENCE")
print("=" * 70)

original_result = result[
    result["hop"] == 0
].set_index(
    "address"
)

max_difference = 0.0
failed_features = []

for feature in ALL_FEATURES:

    diff = (
        original_result[feature]
        - original_lookup.loc[
            original_result.index,
            feature,
        ]
    ).abs()

    feature_max = float(
        diff.max()
    )

    max_difference = max(
        max_difference,
        feature_max,
    )

    if not np.allclose(
        original_result[feature].to_numpy(),
        original_lookup.loc[
            original_result.index,
            feature,
        ].to_numpy(),
        rtol=1e-9,
        atol=1e-9,
    ):

        failed_features.append(
            feature
        )


if failed_features:

    raise RuntimeError(
        "Original feature equivalence failed: "
        + ", ".join(
            failed_features
        )
    )

print(
    "✓ Original 810 feature equivalence PASSED"
)

print(
    f"Maximum difference: "
    f"{max_difference:.3e}"
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

result.to_csv(
    OUTPUT_FILE,
    index=False,
)

print()
print("=" * 70)
print("2-HOP FEATURES SAVED")
print("=" * 70)

print(
    OUTPUT_FILE
)