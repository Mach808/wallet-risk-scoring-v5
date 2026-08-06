from pathlib import Path
import pickle

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


# ============================================================
# CONFIG
# ============================================================

FEATURE_FILE = Path("data/processed/wallet_features.csv")

TRAIN_SPLIT_FILE = Path("data/splits/train_addresses.csv")
VAL_SPLIT_FILE = Path("data/splits/val_addresses.csv")

MODEL_DIR = Path("models")
RESULT_DIR = Path("results")

RANDOM_STATE = 42


FEATURES = [
    # v0.1
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
# LOAD FEATURE DATA
# ============================================================

print("=" * 70)
print("LOADING MVP v0.2 FEATURES")
print("=" * 70)

df = pd.read_csv(FEATURE_FILE)

df["address"] = (
    df["address"]
    .astype(str)
    .str.strip()
    .str.lower()
)

print(f"Wallets  : {len(df)}")
print(f"Features : {len(FEATURES)}")


# ============================================================
# CHECK FEATURES
# ============================================================

missing_features = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]

if missing_features:

    raise RuntimeError(
        f"Missing features: {missing_features}"
    )


# ============================================================
# LOAD FROZEN SPLITS
# ============================================================

print()
print("=" * 70)
print("LOADING FROZEN v0.1 SPLITS")
print("=" * 70)

train_addresses = pd.read_csv(
    TRAIN_SPLIT_FILE
)

val_addresses = pd.read_csv(
    VAL_SPLIT_FILE
)

for frame in [
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
# RECONSTRUCT TRAIN / VALIDATION DATA
# ============================================================

# Only merge by address.
# Labels/types come from wallet_features.csv.

train_df = (
    train_addresses[["address"]]
    .merge(
        df,
        on="address",
        how="left",
    )
)

val_df = (
    val_addresses[["address"]]
    .merge(
        df,
        on="address",
        how="left",
    )
)


# ============================================================
# VALIDATE MERGE
# ============================================================

if train_df[FEATURES].isna().any().any():

    raise RuntimeError(
        "Some TRAIN wallets could not be matched "
        "to wallet_features.csv"
    )


if val_df[FEATURES].isna().any().any():

    raise RuntimeError(
        "Some VALIDATION wallets could not be matched "
        "to wallet_features.csv"
    )


print(
    f"Train wallets      : {len(train_df)}"
)

print(
    f"Validation wallets : {len(val_df)}"
)


print()
print("Train distribution:")

print(
    train_df["label"]
    .value_counts()
    .sort_index()
)


print()
print("Validation distribution:")

print(
    val_df["label"]
    .value_counts()
    .sort_index()
)


# ============================================================
# X / Y
# ============================================================

X_train = train_df[FEATURES]
y_train = train_df["label"]

X_val = val_df[FEATURES]
y_val = val_df["label"]


# ============================================================
# TRAIN RANDOM FOREST v0.2
# ============================================================

print()
print("=" * 70)
print("TRAINING RANDOM FOREST — MVP v0.2")
print("=" * 70)


model = RandomForestClassifier(

    # EXACT SAME configuration as v0.1

    n_estimators=300,

    max_depth=None,

    min_samples_split=2,

    min_samples_leaf=1,

    class_weight="balanced",

    random_state=RANDOM_STATE,

    n_jobs=-1,
)


model.fit(
    X_train,
    y_train,
)

print("Training complete.")


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

probabilities = model.predict_proba(
    X_val
)[:, 1]


THRESHOLD = 0.50

predictions = (
    probabilities >= THRESHOLD
).astype(int)


# ============================================================
# METRICS
# ============================================================

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

roc_auc = roc_auc_score(
    y_val,
    probabilities,
)

pr_auc = average_precision_score(
    y_val,
    probabilities,
)

cm = confusion_matrix(
    y_val,
    predictions,
)


# ============================================================
# RESULTS
# ============================================================

print()
print("#" * 70)
print("MVP v0.2 VALIDATION RESULTS")
print("#" * 70)

print(
    f"Threshold : {THRESHOLD:.2f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1        : {f1:.4f}"
)

print(
    f"ROC-AUC   : {roc_auc:.4f}"
)

print(
    f"PR-AUC    : {pr_auc:.4f}"
)


print()
print("Confusion Matrix:")

print(cm)


print()
print("Classification Report:")

print(
    classification_report(
        y_val,
        predictions,
        digits=4,
        zero_division=0,
    )
)


# ============================================================
# COMPARE WITH v0.1
# ============================================================

V01 = {
    "precision": 0.7097,
    "recall": 0.7097,
    "f1": 0.7097,
    "roc_auc": 0.9118,
    "pr_auc": 0.8259,
}

V02 = {
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "roc_auc": roc_auc,
    "pr_auc": pr_auc,
}


print()
print("=" * 70)
print("v0.1 vs v0.2")
print("=" * 70)

print(
    f"{'Metric':<15}"
    f"{'v0.1':<12}"
    f"{'v0.2':<12}"
    f"{'Change':<12}"
)

print("-" * 51)


for metric in [
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
]:

    old = V01[metric]
    new = V02[metric]

    change = new - old

    print(
        f"{metric:<15}"
        f"{old:<12.4f}"
        f"{new:<12.4f}"
        f"{change:+.4f}"
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print()
print("=" * 70)
print("RANDOM FOREST v0.2 FEATURE IMPORTANCE")
print("=" * 70)


importance_df = pd.DataFrame(
    {
        "feature": FEATURES,
        "importance":
            model.feature_importances_,
    }
).sort_values(
    "importance",
    ascending=False,
)


for row in importance_df.itertuples():

    print(
        f"{row.feature:<30} "
        f"{row.importance:.4f}"
    )


# ============================================================
# MALICIOUS TYPE PERFORMANCE
# ============================================================

prediction_df = val_df[
    [
        "address",
        "type",
        "label",
    ]
].copy()

prediction_df[
    "risk_probability"
] = probabilities

prediction_df[
    "prediction"
] = predictions


malicious = prediction_df[
    prediction_df["label"] == 1
]


print()
print("=" * 70)
print("MALICIOUS TYPE RECALL @ 0.50")
print("=" * 70)


type_results = []


for wallet_type, group in malicious.groupby(
    "type"
):

    samples = len(group)

    detected = int(
        group["prediction"].sum()
    )

    missed = (
        samples - detected
    )

    type_recall = (
        detected / samples
        if samples > 0
        else 0
    )

    avg_risk = (
        group[
            "risk_probability"
        ].mean()
    )

    type_results.append(
        {
            "type":
                wallet_type,

            "samples":
                samples,

            "detected":
                detected,

            "missed":
                missed,

            "recall":
                type_recall,

            "avg_risk_probability":
                avg_risk,
        }
    )


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
# SAVE RESULTS
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# Model

with open(
    MODEL_DIR /
    "random_forest_v02.pkl",
    "wb",
) as f:

    pickle.dump(
        model,
        f,
    )


# Metrics

metrics_df = pd.DataFrame(
    [
        {
            "version": "v0.2",
            "threshold": THRESHOLD,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
        }
    ]
)

metrics_df.to_csv(
    RESULT_DIR /
    "v02_validation_metrics.csv",
    index=False,
)


# Feature importance

importance_df.to_csv(
    RESULT_DIR /
    "v02_rf_feature_importance.csv",
    index=False,
)


# Malicious type analysis

type_df.to_csv(
    RESULT_DIR /
    "v02_malicious_type_analysis.csv",
    index=False,
)


# Validation predictions

prediction_df.to_csv(
    RESULT_DIR /
    "v02_validation_predictions.csv",
    index=False,
)


# ============================================================
# DONE
# ============================================================

print()
print("=" * 70)
print("MVP v0.2 TRAINING COMPLETE")
print("=" * 70)

print()

print(
    "Saved model:"
)

print(
    MODEL_DIR /
    "random_forest_v02.pkl"
)

print()

print(
    "Saved results:"
)

print(
    RESULT_DIR /
    "v02_validation_metrics.csv"
)

print(
    RESULT_DIR /
    "v02_rf_feature_importance.csv"
)

print(
    RESULT_DIR /
    "v02_malicious_type_analysis.csv"
)

print(
    RESULT_DIR /
    "v02_validation_predictions.csv"
)

print()

print(
    "TEST SET HAS NOT BEEN TOUCHED."
)