from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# CONFIG
# ============================================================

FEATURE_FILE = Path(
    "data/processed/wallet_features.csv"
)

VAL_SPLIT_FILE = Path(
    "data/splits/val_addresses.csv"
)

MODEL_FILE = Path(
    "models/random_forest.pkl"
)

RESULT_DIR = Path("results")

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
    "active_days",
    "internal_tx_ratio",
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING VALIDATION DATA")
print("=" * 70)

features = pd.read_csv(
    FEATURE_FILE
)

val_addresses = pd.read_csv(
    VAL_SPLIT_FILE
)

features["address"] = (
    features["address"]
    .astype(str)
    .str.lower()
)

val_addresses["address"] = (
    val_addresses["address"]
    .astype(str)
    .str.lower()
)


# ============================================================
# RECONSTRUCT EXACT VALIDATION SET
# ============================================================

val = val_addresses[
    ["address"]
].merge(
    features,
    on="address",
    how="left",
)


if val[FEATURES].isna().any().any():

    raise RuntimeError(
        "Some validation wallets could not be "
        "matched to wallet_features.csv"
    )


print(
    f"Validation wallets: {len(val)}"
)

print()

print(
    val["label"]
    .value_counts()
    .sort_index()
)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("Loading Random Forest...")

with open(
    MODEL_FILE,
    "rb",
) as f:

    model = pickle.load(f)


# ============================================================
# PREDICT PROBABILITIES
# ============================================================

X_val = val[FEATURES]
y_val = val["label"]

probabilities = model.predict_proba(
    X_val
)[:, 1]

val["risk_probability"] = probabilities


# ============================================================
# THRESHOLD SEARCH
# ============================================================

print()
print("=" * 70)
print("THRESHOLD ANALYSIS")
print("=" * 70)

threshold_results = []

thresholds = np.arange(
    0.10,
    0.91,
    0.05,
)

for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_val,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_val,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_val,
        predictions,
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        predictions,
        labels=[0, 1],
    ).ravel()

    threshold_results.append({

        "threshold":
            round(float(threshold), 2),

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "true_negative":
            tn,

        "false_positive":
            fp,

        "false_negative":
            fn,

        "true_positive":
            tp,
    })


threshold_df = pd.DataFrame(
    threshold_results
)


# ============================================================
# PRINT THRESHOLDS
# ============================================================

print()

print(
    f"{'Threshold':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
    f"{'FP':<8}"
    f"{'FN':<8}"
)

print("-" * 64)

for row in threshold_df.itertuples():

    print(
        f"{row.threshold:<12.2f}"
        f"{row.precision:<12.4f}"
        f"{row.recall:<12.4f}"
        f"{row.f1:<12.4f}"
        f"{row.false_positive:<8}"
        f"{row.false_negative:<8}"
    )


# ============================================================
# BEST F1 THRESHOLD
# ============================================================

best_row = threshold_df.loc[
    threshold_df["f1"].idxmax()
]

best_threshold = float(
    best_row["threshold"]
)


print()
print("=" * 70)
print("BEST VALIDATION THRESHOLD BY F1")
print("=" * 70)

print(
    f"Threshold : "
    f"{best_threshold:.2f}"
)

print(
    f"Precision : "
    f"{best_row['precision']:.4f}"
)

print(
    f"Recall    : "
    f"{best_row['recall']:.4f}"
)

print(
    f"F1        : "
    f"{best_row['f1']:.4f}"
)

print(
    f"False Positives : "
    f"{int(best_row['false_positive'])}"
)

print(
    f"False Negatives : "
    f"{int(best_row['false_negative'])}"
)


# ============================================================
# PER-TYPE ANALYSIS
# ============================================================

print()
print("=" * 70)
print("MALICIOUS TYPE ANALYSIS")
print("=" * 70)

predictions = (
    probabilities >= best_threshold
).astype(int)

val["prediction"] = predictions


malicious = val[
    val["label"] == 1
].copy()


type_results = []

for wallet_type, group in malicious.groupby(
    "type"
):

    total = len(group)

    detected = int(
        group["prediction"].sum()
    )

    missed = (
        total - detected
    )

    recall = (
        detected / total
        if total > 0
        else 0
    )

    avg_risk = (
        group[
            "risk_probability"
        ].mean()
    )

    type_results.append({

        "type":
            wallet_type,

        "samples":
            total,

        "detected":
            detected,

        "missed":
            missed,

        "recall":
            recall,

        "avg_risk_probability":
            avg_risk,
    })


type_df = pd.DataFrame(
    type_results
).sort_values(
    "recall",
    ascending=False,
)


print()

print(
    f"{'Type':<25}"
    f"{'Samples':<10}"
    f"{'Detected':<12}"
    f"{'Missed':<10}"
    f"{'Recall':<12}"
    f"{'Avg Risk':<12}"
)

print("-" * 81)

for row in type_df.itertuples():

    print(
        f"{row.type:<25}"
        f"{row.samples:<10}"
        f"{row.detected:<12}"
        f"{row.missed:<10}"
        f"{row.recall:<12.4f}"
        f"{row.avg_risk_probability:<12.4f}"
    )


# ============================================================
# MISSED MALICIOUS WALLETS
# ============================================================

missed = malicious[
    malicious["prediction"] == 0
].copy()

missed = missed.sort_values(
    "risk_probability",
    ascending=False,
)


print()
print("=" * 70)
print("MISSED MALICIOUS WALLETS")
print("=" * 70)

if len(missed) == 0:

    print(
        "No malicious wallets missed."
    )

else:

    for row in missed.itertuples():

        print(
            f"{row.address} "
            f"| {row.type:<22} "
            f"| risk={row.risk_probability:.4f}"
        )


# ============================================================
# FALSE POSITIVES
# ============================================================

false_positives = val[
    (val["label"] == 0)
    &
    (val["prediction"] == 1)
].copy()

false_positives = (
    false_positives
    .sort_values(
        "risk_probability",
        ascending=False,
    )
)


print()
print("=" * 70)
print("FALSE POSITIVE BENIGN WALLETS")
print("=" * 70)

if len(false_positives) == 0:

    print(
        "No false positives."
    )

else:

    for row in false_positives.itertuples():

        print(
            f"{row.address} "
            f"| risk={row.risk_probability:.4f}"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


threshold_df.to_csv(
    RESULT_DIR /
    "threshold_analysis.csv",
    index=False,
)


type_df.to_csv(
    RESULT_DIR /
    "malicious_type_analysis.csv",
    index=False,
)


val[
    [
        "address",
        "type",
        "label",
        "risk_probability",
        "prediction",
    ]
].to_csv(
    RESULT_DIR /
    "validation_predictions.csv",
    index=False,
)


# ============================================================
# DONE
# ============================================================

print()
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print()

print(
    "Saved:"
)

print(
    RESULT_DIR /
    "threshold_analysis.csv"
)

print(
    RESULT_DIR /
    "malicious_type_analysis.csv"
)

print(
    RESULT_DIR /
    "validation_predictions.csv"
)

print()

print(
    "TEST SET remains untouched."
)