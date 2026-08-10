from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import StratifiedKFold


# ============================================================
# CONFIG
# ============================================================

FEATURE_FILE = Path(
    "data/processed/wallet_features.csv"
)

TRAIN_SPLIT_FILE = Path(
    "data/splits/train_addresses.csv"
)

VAL_SPLIT_FILE = Path(
    "data/splits/val_addresses.csv"
)

RESULT_DIR = Path("results")

RANDOM_STATE = 42
N_SPLITS = 5
THRESHOLD = 0.50


# ============================================================
# FEATURE SETS
# ============================================================

V01_FEATURES = [
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
]


V02_FEATURES = [
    # v0.1
    *V01_FEATURES,

    # v0.2
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
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING DEVELOPMENT DATA")
print("=" * 70)

features = pd.read_csv(
    FEATURE_FILE
)

train_addresses = pd.read_csv(
    TRAIN_SPLIT_FILE
)

val_addresses = pd.read_csv(
    VAL_SPLIT_FILE
)


for frame in [
    features,
    train_addresses,
    val_addresses,
]:

    frame["address"] = (
        frame["address"]
        .astype(str)
        .str.strip()
        .str.lower()
    )


# ============================================================
# BUILD DEVELOPMENT SET
#
# IMPORTANT:
# Test addresses are NEVER loaded.
# ============================================================

development_addresses = pd.concat(
    [
        train_addresses[["address"]],
        val_addresses[["address"]],
    ],
    ignore_index=True,
)

development_addresses = (
    development_addresses
    .drop_duplicates(
        subset=["address"]
    )
)


development = (
    development_addresses
    .merge(
        features,
        on="address",
        how="left",
    )
)


# ============================================================
# VALIDATION
# ============================================================

ALL_FEATURES = list(
    dict.fromkeys(
        V01_FEATURES
        + V02_FEATURES
    )
)


if development[
    ALL_FEATURES
].isna().any().any():

    raise RuntimeError(
        "Some development wallets have missing features."
    )


print(
    f"Development wallets : {len(development)}"
)

print(
    f"v0.1 features       : {len(V01_FEATURES)}"
)

print(
    f"v0.2 features       : {len(V02_FEATURES)}"
)


print()
print("Label distribution:")

print(
    development["label"]
    .value_counts()
    .sort_index()
)


# ============================================================
# LABELS
# ============================================================

y = development[
    "label"
].to_numpy()


# ============================================================
# CROSS VALIDATION
# ============================================================

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)


# ============================================================
# MODEL FACTORY
#
# Exact same RF configuration for both versions.
# ============================================================

def create_model():

    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
):

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    return {

        "precision":
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            ),

        "recall":
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            ),

        "f1":
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            ),

        "roc_auc":
            roc_auc_score(
                y_true,
                probabilities,
            ),

        "pr_auc":
            average_precision_score(
                y_true,
                probabilities,
            ),
    }


# ============================================================
# RUN CV
# ============================================================

results = []


print()
print("=" * 70)
print("5-FOLD STRATIFIED CROSS-VALIDATION")
print("=" * 70)


for fold, (
    train_index,
    eval_index,
) in enumerate(
    skf.split(
        development,
        y,
    ),
    start=1,
):

    print()
    print(
        f"Fold {fold}/{N_SPLITS}"
    )

    y_train = y[
        train_index
    ]

    y_eval = y[
        eval_index
    ]


    # ========================================================
    # v0.1
    # ========================================================

    X_v01 = development[
        V01_FEATURES
    ]

    X01_train = X_v01.iloc[
        train_index
    ]

    X01_eval = X_v01.iloc[
        eval_index
    ]


    model_v01 = create_model()

    model_v01.fit(
        X01_train,
        y_train,
    )


    prob_v01 = (
        model_v01
        .predict_proba(
            X01_eval
        )[:, 1]
    )


    metrics_v01 = calculate_metrics(
        y_eval,
        prob_v01,
    )


    results.append(
        {
            "version":
                "v0.1",

            "fold":
                fold,

            **metrics_v01,
        }
    )


    # ========================================================
    # v0.2
    # ========================================================

    X_v02 = development[
        V02_FEATURES
    ]

    X02_train = X_v02.iloc[
        train_index
    ]

    X02_eval = X_v02.iloc[
        eval_index
    ]


    model_v02 = create_model()

    model_v02.fit(
        X02_train,
        y_train,
    )


    prob_v02 = (
        model_v02
        .predict_proba(
            X02_eval
        )[:, 1]
    )


    metrics_v02 = calculate_metrics(
        y_eval,
        prob_v02,
    )


    results.append(
        {
            "version":
                "v0.2",

            "fold":
                fold,

            **metrics_v02,
        }
    )


    # ========================================================
    # PRINT FOLD
    # ========================================================

    print(
        f"  v0.1 | "
        f"PR-AUC={metrics_v01['pr_auc']:.4f} | "
        f"ROC-AUC={metrics_v01['roc_auc']:.4f} | "
        f"F1={metrics_v01['f1']:.4f}"
    )

    print(
        f"  v0.2 | "
        f"PR-AUC={metrics_v02['pr_auc']:.4f} | "
        f"ROC-AUC={metrics_v02['roc_auc']:.4f} | "
        f"F1={metrics_v02['f1']:.4f}"
    )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# SUMMARY
# ============================================================

METRICS = [
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
]


summary_rows = []


for version in [
    "v0.1",
    "v0.2",
]:

    version_results = (
        results_df[
            results_df["version"]
            == version
        ]
    )

    row = {
        "version":
            version
    }

    for metric in METRICS:

        row[
            f"{metric}_mean"
        ] = (
            version_results[
                metric
            ].mean()
        )

        row[
            f"{metric}_std"
        ] = (
            version_results[
                metric
            ].std()
        )

    summary_rows.append(
        row
    )


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print()
print("=" * 70)
print("CROSS-VALIDATION SUMMARY")
print("=" * 70)


for version in [
    "v0.1",
    "v0.2",
]:

    row = summary_df[
        summary_df["version"]
        == version
    ].iloc[0]

    print()
    print(version)

    print(
        f"Precision : "
        f"{row['precision_mean']:.4f} "
        f"± {row['precision_std']:.4f}"
    )

    print(
        f"Recall    : "
        f"{row['recall_mean']:.4f} "
        f"± {row['recall_std']:.4f}"
    )

    print(
        f"F1        : "
        f"{row['f1_mean']:.4f} "
        f"± {row['f1_std']:.4f}"
    )

    print(
        f"ROC-AUC   : "
        f"{row['roc_auc_mean']:.4f} "
        f"± {row['roc_auc_std']:.4f}"
    )

    print(
        f"PR-AUC    : "
        f"{row['pr_auc_mean']:.4f} "
        f"± {row['pr_auc_std']:.4f}"
    )


# ============================================================
# DIRECT COMPARISON
# ============================================================

print()
print("=" * 70)
print("MEAN IMPROVEMENT — v0.2 OVER v0.1")
print("=" * 70)


v01 = summary_df[
    summary_df["version"] == "v0.1"
].iloc[0]

v02 = summary_df[
    summary_df["version"] == "v0.2"
].iloc[0]


for metric in METRICS:

    difference = (
        v02[f"{metric}_mean"]
        -
        v01[f"{metric}_mean"]
    )

    print(
        f"{metric:<12}: "
        f"{difference:+.4f}"
    )


# ============================================================
# PER-FOLD WINS
# ============================================================

print()
print("=" * 70)
print("PER-FOLD COMPARISON")
print("=" * 70)


for metric in [
    "pr_auc",
    "roc_auc",
    "f1",
]:

    pivot = results_df.pivot(
        index="fold",
        columns="version",
        values=metric,
    )

    v02_wins = int(
        (
            pivot["v0.2"]
            >
            pivot["v0.1"]
        ).sum()
    )

    ties = int(
        (
            pivot["v0.2"]
            ==
            pivot["v0.1"]
        ).sum()
    )

    print(
        f"{metric:<12}: "
        f"v0.2 wins {v02_wins}/{N_SPLITS} "
        f"(ties: {ties})"
    )


# ============================================================
# SAVE
# ============================================================

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


results_df.to_csv(
    RESULT_DIR /
    "cross_validation_folds.csv",
    index=False,
)


summary_df.to_csv(
    RESULT_DIR /
    "cross_validation_summary.csv",
    index=False,
)


# ============================================================
# DONE
# ============================================================

print()
print("=" * 70)
print("CROSS-VALIDATION COMPLETE")
print("=" * 70)

print()

print(
    "Saved:"
)

print(
    RESULT_DIR /
    "cross_validation_folds.csv"
)

print(
    RESULT_DIR /
    "cross_validation_summary.csv"
)

print()

print(
    "TEST SET HAS NOT BEEN LOADED OR EVALUATED."
)