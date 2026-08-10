from pathlib import Path
import pickle

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)


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

TEST_SPLIT_FILE = Path(
    "data/splits/test_addresses.csv"
)

MODEL_DIR = Path("models")
RESULT_DIR = Path("results")

RANDOM_STATE = 42
THRESHOLD = 0.50


# ============================================================
# FINAL v0.2 FEATURE SET
# ============================================================

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


# ============================================================
# LOAD FEATURES
# ============================================================

print("=" * 70)
print("FINAL MVP v0.2 TEST EVALUATION")
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


# ============================================================
# LOAD FROZEN SPLITS
# ============================================================

train_addresses = pd.read_csv(
    TRAIN_SPLIT_FILE
)

val_addresses = pd.read_csv(
    VAL_SPLIT_FILE
)

test_addresses = pd.read_csv(
    TEST_SPLIT_FILE
)


for frame in [
    train_addresses,
    val_addresses,
    test_addresses,
]:

    frame["address"] = (
        frame["address"]
        .astype(str)
        .str.strip()
        .str.lower()
    )


# ============================================================
# CHECK SPLIT OVERLAP
# ============================================================

train_set = set(
    train_addresses["address"]
)

val_set = set(
    val_addresses["address"]
)

test_set = set(
    test_addresses["address"]
)


assert train_set.isdisjoint(
    val_set
), "Train/validation overlap detected."

assert train_set.isdisjoint(
    test_set
), "Train/test overlap detected."

assert val_set.isdisjoint(
    test_set
), "Validation/test overlap detected."


print()
print("Split overlap check: PASSED")


# ============================================================
# BUILD FINAL DEVELOPMENT SET
#
# Train + validation
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
# BUILD TEST SET
# ============================================================

test = (
    test_addresses[["address"]]
    .merge(
        features,
        on="address",
        how="left",
    )
)


# ============================================================
# VALIDATE DATA
# ============================================================

if development[FEATURES].isna().any().any():

    raise RuntimeError(
        "Development set contains missing features."
    )


if test[FEATURES].isna().any().any():

    raise RuntimeError(
        "Test set contains missing features."
    )


print()
print("DATASET")

print("-" * 70)

print(
    f"Development wallets : {len(development)}"
)

print(
    f"Test wallets        : {len(test)}"
)

print(
    f"Features            : {len(FEATURES)}"
)


print()
print("Development labels:")

print(
    development["label"]
    .value_counts()
    .sort_index()
)


print()
print("Test labels:")

print(
    test["label"]
    .value_counts()
    .sort_index()
)


# ============================================================
# TRAIN FINAL MODEL
# ============================================================

X_development = development[
    FEATURES
]

y_development = development[
    "label"
]


print()
print("=" * 70)
print("TRAINING FINAL MVP MODEL")
print("=" * 70)

print(
    "Training on train + validation "
    f"({len(development)} wallets)..."
)


model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)


model.fit(
    X_development,
    y_development,
)


print("Training complete.")


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

X_test = test[
    FEATURES
]

y_test = test[
    "label"
]


probabilities = model.predict_proba(
    X_test
)[:, 1]


predictions = (
    probabilities >= THRESHOLD
).astype(int)


# ============================================================
# METRICS
# ============================================================

precision = precision_score(
    y_test,
    predictions,
    zero_division=0,
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0,
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0,
)

roc_auc = roc_auc_score(
    y_test,
    probabilities,
)

pr_auc = average_precision_score(
    y_test,
    probabilities,
)

cm = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1],
)


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print()
print("#" * 70)
print("FINAL HELD-OUT TEST RESULTS")
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
        y_test,
        predictions,
        digits=4,
        zero_division=0,
    )
)


# ============================================================
# TEST PREDICTIONS
# ============================================================

prediction_df = test[
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


# ============================================================
# MALICIOUS TYPE ANALYSIS
# ============================================================

malicious = prediction_df[
    prediction_df["label"] == 1
].copy()


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
        else 0.0
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
print("=" * 70)
print("FINAL MALICIOUS TYPE RECALL")
print("=" * 70)

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
# MISSED MALICIOUS
# ============================================================

missed = malicious[
    malicious["prediction"] == 0
].sort_values(
    "risk_probability",
    ascending=False,
)


print()
print("=" * 70)
print("MISSED MALICIOUS TEST WALLETS")
print("=" * 70)


if missed.empty:

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

false_positives = prediction_df[
    (prediction_df["label"] == 0)
    &
    (prediction_df["prediction"] == 1)
].sort_values(
    "risk_probability",
    ascending=False,
)


print()
print("=" * 70)
print("FALSE POSITIVE TEST WALLETS")
print("=" * 70)


if false_positives.empty:

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
# FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame(
    {
        "feature":
            FEATURES,

        "importance":
            model.feature_importances_,
    }
).sort_values(
    "importance",
    ascending=False,
)


print()
print("=" * 70)
print("FINAL FEATURE IMPORTANCE")
print("=" * 70)


for row in importance_df.itertuples():

    print(
        f"{row.feature:<30} "
        f"{row.importance:.4f}"
    )


# ============================================================
# SAVE FINAL MODEL + RESULTS
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


with open(
    MODEL_DIR /
    "wallet_risk_mvp_v02.pkl",
    "wb",
) as f:

    pickle.dump(
        model,
        f,
    )


metrics_df = pd.DataFrame(
    [
        {
            "model":
                "Random Forest",

            "version":
                "MVP v0.2",

            "features":
                len(FEATURES),

            "threshold":
                THRESHOLD,

            "development_wallets":
                len(development),

            "test_wallets":
                len(test),

            "precision":
                precision,

            "recall":
                recall,

            "f1":
                f1,

            "roc_auc":
                roc_auc,

            "pr_auc":
                pr_auc,
        }
    ]
)


metrics_df.to_csv(
    RESULT_DIR /
    "final_test_metrics.csv",
    index=False,
)


prediction_df.to_csv(
    RESULT_DIR /
    "final_test_predictions.csv",
    index=False,
)


type_df.to_csv(
    RESULT_DIR /
    "final_test_type_analysis.csv",
    index=False,
)


importance_df.to_csv(
    RESULT_DIR /
    "final_feature_importance.csv",
    index=False,
)


# ============================================================
# DONE
# ============================================================

print()
print("=" * 70)
print("MVP v0.2 COMPLETE")
print("=" * 70)

print()

print(
    "Final model:"
)

print(
    MODEL_DIR /
    "wallet_risk_mvp_v02.pkl"
)

print()

print(
    "Final results:"
)

print(
    RESULT_DIR /
    "final_test_metrics.csv"
)

print(
    RESULT_DIR /
    "final_test_predictions.csv"
)

print(
    RESULT_DIR /
    "final_test_type_analysis.csv"
)

print(
    RESULT_DIR /
    "final_feature_importance.csv"
)

print()
print(
    "The held-out test set has now been evaluated."
)

print(
    "Do NOT tune the model, features, or threshold "
    "against these test results."
)