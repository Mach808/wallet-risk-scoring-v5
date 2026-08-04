from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


# ============================================================
# CONFIG
# ============================================================

FEATURE_FILE = Path("data/processed/wallet_features.csv")

SPLIT_DIR = Path("data/splits")
MODEL_DIR = Path("models")
RESULT_DIR = Path("results")

RANDOM_STATE = 42

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

# Highly skewed non-negative financial features.
LOG_FEATURES = [
    "total_eth_received",
    "total_eth_sent",
    "avg_tx_value",
    "max_tx_value",
]

OTHER_FEATURES = [
    feature
    for feature in FEATURES
    if feature not in LOG_FEATURES
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

df = pd.read_csv(FEATURE_FILE)

print(f"Wallets  : {len(df)}")
print(f"Features : {len(FEATURES)}")

print("\nLabel distribution:")
print(df["label"].value_counts().sort_index())


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
#
# First:
#     70% train
#     30% temporary
#
# Then temporary:
#     50% validation
#     50% test
#
# Final:
#     70 / 15 / 15
# ============================================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=RANDOM_STATE,
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=RANDOM_STATE,
)


print("\n" + "=" * 70)
print("DATA SPLIT")
print("=" * 70)

print(f"Train      : {len(train_df)}")
print(f"Validation : {len(val_df)}")
print(f"Test       : {len(test_df)}")


def print_distribution(name, frame):

    print(f"\n{name}")

    counts = frame["label"].value_counts().sort_index()

    for label, count in counts.items():

        percentage = count / len(frame) * 100

        print(
            f"  Label {label}: "
            f"{count} ({percentage:.2f}%)"
        )


print_distribution("Train", train_df)
print_distribution("Validation", val_df)
print_distribution("Test", test_df)


# ============================================================
# SAVE SPLITS
# ============================================================

SPLIT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def save_split(frame, filename):

    frame[
        ["address", "type", "label"]
    ].to_csv(
        SPLIT_DIR / filename,
        index=False,
    )


save_split(
    train_df,
    "train_addresses.csv"
)

save_split(
    val_df,
    "val_addresses.csv"
)

save_split(
    test_df,
    "test_addresses.csv"
)

print("\nSaved dataset splits.")


# ============================================================
# X / Y
# ============================================================

X_train = train_df[FEATURES]
y_train = train_df["label"]

X_val = val_df[FEATURES]
y_val = val_df["label"]


# ============================================================
# LOGISTIC REGRESSION PREPROCESSING
# ============================================================

# Financial values span several orders of magnitude.
#
# log1p:
#
#     x -> log(1 + x)
#
# compresses extreme financial values before scaling.

log_transformer = Pipeline(
    steps=[
        (
            "log",
            FunctionTransformer(
                np.log1p,
                feature_names_out="one-to-one",
            ),
        ),
        (
            "scale",
            StandardScaler(),
        ),
    ]
)

normal_transformer = Pipeline(
    steps=[
        (
            "scale",
            StandardScaler(),
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "log_features",
            log_transformer,
            LOG_FEATURES,
        ),
        (
            "normal_features",
            normal_transformer,
            OTHER_FEATURES,
        ),
    ]
)


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 70)

logistic_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)

logistic_model.fit(
    X_train,
    y_train,
)

print("Training complete.")


# ============================================================
# RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)

random_forest = RandomForestClassifier(
    n_estimators=300,

    # Keep baseline reasonably conservative.
    max_depth=None,

    min_samples_split=2,
    min_samples_leaf=1,

    class_weight="balanced",

    random_state=RANDOM_STATE,

    n_jobs=-1,
)

random_forest.fit(
    X_train,
    y_train,
)

print("Training complete.")


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    name,
    model,
    X,
    y,
    threshold=0.50,
):

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y,
        probabilities,
    )

    pr_auc = average_precision_score(
        y,
        probabilities,
    )

    cm = confusion_matrix(
        y,
        predictions,
    )

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print(f"Threshold : {threshold:.2f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1        : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"PR-AUC    : {pr_auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")

    print(
        classification_report(
            y,
            predictions,
            digits=4,
            zero_division=0,
        )
    )

    return {
        "model": name,
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


# ============================================================
# VALIDATION EVALUATION
# ============================================================

print("\n")
print("#" * 70)
print("VALIDATION RESULTS")
print("#" * 70)

logistic_results = evaluate_model(
    "Logistic Regression",
    logistic_model,
    X_val,
    y_val,
)

rf_results = evaluate_model(
    "Random Forest",
    random_forest,
    X_val,
    y_val,
)


# ============================================================
# RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 70)

importance_df = pd.DataFrame(
    {
        "feature": FEATURES,
        "importance":
            random_forest.feature_importances_,
    }
)

importance_df = importance_df.sort_values(
    "importance",
    ascending=False,
)

for row in importance_df.itertuples():

    print(
        f"{row.feature:<30} "
        f"{row.importance:.4f}"
    )


# ============================================================
# SAVE FEATURE IMPORTANCE
# ============================================================

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

importance_df.to_csv(
    RESULT_DIR / "rf_feature_importance.csv",
    index=False,
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame(
    [
        logistic_results,
        rf_results,
    ]
)

metrics_df.to_csv(
    RESULT_DIR / "baseline_validation_metrics.csv",
    index=False,
)


# ============================================================
# SAVE MODELS
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    MODEL_DIR / "logistic_regression.pkl",
    "wb",
) as f:

    pickle.dump(
        logistic_model,
        f,
    )


with open(
    MODEL_DIR / "random_forest.pkl",
    "wb",
) as f:

    pickle.dump(
        random_forest,
        f,
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("BASELINE TRAINING COMPLETE")
print("=" * 70)

print(
    "\nModels:"
)

print(
    MODEL_DIR / "logistic_regression.pkl"
)

print(
    MODEL_DIR / "random_forest.pkl"
)

print(
    "\nResults:"
)

print(
    RESULT_DIR /
    "baseline_validation_metrics.csv"
)

print(
    RESULT_DIR /
    "rf_feature_importance.csv"
)

print(
    "\nIMPORTANT:"
)

print(
    "The TEST SET has NOT been evaluated."
)

print(
    "Keep it untouched until the MVP model and "
    "decision threshold are finalized."
)