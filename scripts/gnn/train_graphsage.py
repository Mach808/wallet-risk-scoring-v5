from pathlib import Path
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.preprocessing import RobustScaler
from torch_geometric.nn import SAGEConv


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

GRAPH_FILE = (
    ROOT / "data" / "processed" / "graph" / "graph_v03_2hop.pt"
)

MODEL_DIR = (
    ROOT / "models" / "graphsage" / "v03_2hop"
)

MODEL_FILE = MODEL_DIR / "model.pt"
SCALER_FILE = MODEL_DIR / "scaler.pkl"

SEED = 42

HIDDEN_DIM = 64
DROPOUT = 0.30

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

MAX_EPOCHS = 300
PATIENCE = 30

THRESHOLD = 0.50


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("GRAPHSAGE — 2-HOP SCALED EXPERIMENT")
print("=" * 70)

print(f"Device: {device}")


# ============================================================
# LOAD GRAPH
# ============================================================

print()
print("=" * 70)
print("LOADING GRAPH")
print("=" * 70)

data = torch.load(
    GRAPH_FILE,
    weights_only=False,
)

print(f"Nodes          : {data.num_nodes:,}")
print(f"Features/node  : {data.num_node_features}")
print(f"Edges          : {data.num_edges:,}")
print(f"Labeled        : {data.labeled_mask.sum().item():,}")
print(f"Train          : {data.train_mask.sum().item():,}")
print(f"Validation     : {data.val_mask.sum().item():,}")
print(f"Test           : {data.test_mask.sum().item():,}")


# ============================================================
# VALIDATION
# ============================================================

if torch.isnan(data.x).any():
    raise RuntimeError("NaN values found in node features.")

if torch.isinf(data.x).any():
    raise RuntimeError("Infinite values found in node features.")

if (data.train_mask & ~data.labeled_mask).any():
    raise RuntimeError(
        "Unlabeled nodes found in training mask."
    )

if (data.val_mask & ~data.labeled_mask).any():
    raise RuntimeError(
        "Unlabeled nodes found in validation mask."
    )

if (data.test_mask & ~data.labeled_mask).any():
    raise RuntimeError(
        "Unlabeled nodes found in test mask."
    )


# ============================================================
# FEATURE TRANSFORMATION
# ============================================================

print()
print("=" * 70)
print("FEATURE TRANSFORMATION")
print("=" * 70)

FEATURE_COLUMNS = [
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

COUNT_FEATURES = [
    "total_tx_count",
    "incoming_tx_count",
    "outgoing_tx_count",
    "unique_senders",
    "unique_receivers",
    "unique_counterparties",
    "distinct_active_days",

    "erc20_tx_count",
    "erc20_in_count",
    "erc20_out_count",
    "erc20_unique_tokens",
    "erc20_unique_senders",
    "erc20_unique_receivers",
    "erc20_unique_counterparties",
]

VALUE_FEATURES = [
    "total_eth_received",
    "total_eth_sent",
    "avg_tx_value",
    "max_tx_value",
    "median_tx_value",
    "std_tx_value",
]

TEMPORAL_FEATURES = [
    "activity_span_days",
    "tx_frequency",
    "avg_time_between_tx",
    "median_time_between_tx",
    "max_time_between_tx",

    "erc20_activity_span_days",
    "erc20_tx_frequency",
]

LOG_RATIO_FEATURES = [
    "in_out_tx_ratio",
    "counterparty_reuse_ratio",
    "erc20_in_out_ratio",
    "erc20_counterparty_reuse_ratio",
]

SIGNED_LOG_FEATURES = [
    "net_eth_flow",
]

# Everything else in this list is already bounded
# and will be left unchanged before standardization.
BOUNDED_FEATURES = [
    "internal_tx_ratio",
    "incoming_value_ratio",
    "zero_value_tx_ratio",
    "external_tx_ratio",
    "burstiness",
    "erc20_zero_value_ratio",
]


# ------------------------------------------------------------
# Verify feature definitions
# ------------------------------------------------------------

all_grouped_features = (
    COUNT_FEATURES
    + VALUE_FEATURES
    + TEMPORAL_FEATURES
    + LOG_RATIO_FEATURES
    + SIGNED_LOG_FEATURES
    + BOUNDED_FEATURES
)

if set(all_grouped_features) != set(FEATURE_COLUMNS):

    missing = (
        set(FEATURE_COLUMNS)
        - set(all_grouped_features)
    )

    extra = (
        set(all_grouped_features)
        - set(FEATURE_COLUMNS)
    )

    raise RuntimeError(
        f"Feature grouping mismatch.\n"
        f"Missing: {missing}\n"
        f"Extra: {extra}"
    )

if len(all_grouped_features) != len(
    set(all_grouped_features)
):

    raise RuntimeError(
        "A feature appears in multiple groups."
    )


# ------------------------------------------------------------
# Convert to DataFrame
# ------------------------------------------------------------

x_df = pd.DataFrame(
    data.x.cpu().numpy(),
    columns=FEATURE_COLUMNS,
)


# ------------------------------------------------------------
# Apply log1p to positive/skewed features
# ------------------------------------------------------------

for feature in (
    COUNT_FEATURES
    + VALUE_FEATURES
    + TEMPORAL_FEATURES
    + LOG_RATIO_FEATURES
):

    # All these features should be >= 0.
    if (
        x_df[feature] < 0
    ).any():

        raise RuntimeError(
            f"Negative value found in "
            f"non-negative feature: {feature}"
        )

    x_df[feature] = np.log1p(
        x_df[feature]
    )


# ------------------------------------------------------------
# Signed log transform for net ETH flow
# ------------------------------------------------------------

x_df["net_eth_flow"] = (
    np.sign(
        x_df["net_eth_flow"]
    )
    * np.log1p(
        np.abs(
            x_df["net_eth_flow"]
        )
    )
)


# ------------------------------------------------------------
# Fit scaler ONLY on training wallets
# ------------------------------------------------------------

train_indices = (
    data.train_mask.cpu().numpy()
)

train_x = x_df.loc[
    train_indices
].values

scaler = StandardScaler()

scaler.fit(
    train_x
)


# ------------------------------------------------------------
# Transform ALL graph nodes
# ------------------------------------------------------------

x_scaled = scaler.transform(
    x_df.values
).astype(
    np.float32
)


# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------

if not np.isfinite(
    x_scaled
).all():

    raise RuntimeError(
        "Feature transformation produced "
        "NaN or infinite values."
    )


print(
    f"Features transformed : "
    f"{len(FEATURE_COLUMNS)}"
)

print(
    f"Training rows        : "
    f"{train_indices.sum()}"
)

print(
    f"Raw global max       : "
    f"{np.max(np.abs(data.x.cpu().numpy())):.4e}"
)

print(
    f"Transformed max      : "
    f"{np.max(np.abs(x_scaled)):.4e}"
)

print(
    f"Transformed mean abs : "
    f"{np.mean(np.abs(x_scaled)):.4f}"
)

print(
    f"Transformed std      : "
    f"{np.std(x_scaled):.4f}"
)

data.x = torch.tensor(
    x_scaled,
    dtype=torch.float32,
)

# ============================================================
# CLASS DISTRIBUTION
# ============================================================

train_labels = data.y[
    data.train_mask
]

num_benign = (
    train_labels == 0
).sum().item()

num_malicious = (
    train_labels == 1
).sum().item()

print()
print("=" * 70)
print("TRAINING LABEL DISTRIBUTION")
print("=" * 70)

print(f"Benign    : {num_benign}")
print(f"Malicious : {num_malicious}")

pos_weight = (
    num_benign / num_malicious
)

print(
    f"Positive class weight: "
    f"{pos_weight:.4f}"
)


# ============================================================
# MODEL
# ============================================================

class GraphSAGE(nn.Module):

    def __init__(
        self,
        in_channels,
        hidden_channels,
        dropout,
    ):
        super().__init__()

        self.conv1 = SAGEConv(
            in_channels,
            hidden_channels,
        )

        self.conv2 = SAGEConv(
            hidden_channels,
            hidden_channels,
        )

        self.classifier = nn.Linear(
            hidden_channels,
            1,
        )

        self.dropout = dropout

    def forward(
        self,
        x,
        edge_index,
    ):

        x = self.conv1(
            x,
            edge_index,
        )

        x = F.relu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        x = self.conv2(
            x,
            edge_index,
        )

        x = F.relu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        return self.classifier(
            x
        ).squeeze(-1)


# ============================================================
# INITIALIZE
# ============================================================

model = GraphSAGE(
    data.num_node_features,
    HIDDEN_DIM,
    DROPOUT,
).to(device)

data = data.to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(
        pos_weight,
        dtype=torch.float32,
        device=device,
    )
)


# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 70)
print("TRAINING")
print("=" * 70)

best_val_pr_auc = -1.0
best_epoch = 0
epochs_without_improvement = 0
best_state = None


for epoch in range(
    1,
    MAX_EPOCHS + 1,
):

    model.train()

    optimizer.zero_grad()

    logits = model(
        data.x,
        data.edge_index,
    )

    loss = criterion(
        logits[data.train_mask],
        data.y[data.train_mask].float(),
    )

    loss.backward()

    optimizer.step()


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        logits = model(
            data.x,
            data.edge_index,
        )

        val_probs = torch.sigmoid(
            logits[data.val_mask]
        ).cpu().numpy()

        val_labels = data.y[
            data.val_mask
        ].cpu().numpy()


    val_pr_auc = average_precision_score(
        val_labels,
        val_probs,
    )

    val_roc_auc = roc_auc_score(
        val_labels,
        val_probs,
    )

    val_predictions = (
        val_probs >= THRESHOLD
    ).astype(int)

    _, _, val_f1, _ = (
        precision_recall_fscore_support(
            val_labels,
            val_predictions,
            average="binary",
            zero_division=0,
        )
    )


    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    if val_pr_auc > best_val_pr_auc:

        best_val_pr_auc = val_pr_auc
        best_epoch = epoch
        epochs_without_improvement = 0

        best_state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }

    else:

        epochs_without_improvement += 1


    if (
        epoch == 1
        or epoch % 10 == 0
        or epoch == best_epoch
    ):

        print(
            f"Epoch {epoch:03d} | "
            f"Loss {loss.item():.4f} | "
            f"Val PR-AUC {val_pr_auc:.4f} | "
            f"Val ROC-AUC {val_roc_auc:.4f} | "
            f"Val F1 {val_f1:.4f}"
        )


    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print()
        print(
            f"Early stopping at epoch {epoch}"
        )

        break


# ============================================================
# RESTORE BEST
# ============================================================

if best_state is None:
    raise RuntimeError(
        "No best model was produced."
    )

model.load_state_dict(
    best_state
)

model.eval()


# ============================================================
# TEST
# ============================================================

print()
print("=" * 70)
print("FINAL TEST")
print("=" * 70)

with torch.no_grad():

    logits = model(
        data.x,
        data.edge_index,
    )

    test_probs = torch.sigmoid(
        logits[data.test_mask]
    ).cpu().numpy()

    test_labels = data.y[
        data.test_mask
    ].cpu().numpy()


test_pr_auc = average_precision_score(
    test_labels,
    test_probs,
)

test_roc_auc = roc_auc_score(
    test_labels,
    test_probs,
)

test_predictions = (
    test_probs >= THRESHOLD
).astype(int)

precision, recall, f1, _ = (
    precision_recall_fscore_support(
        test_labels,
        test_predictions,
        average="binary",
        zero_division=0,
    )
)

cm = confusion_matrix(
    test_labels,
    test_predictions,
)


print(f"Best epoch      : {best_epoch}")
print(f"Best Val PR-AUC : {best_val_pr_auc:.4f}")
print(f"Test PR-AUC     : {test_pr_auc:.4f}")
print(f"Test ROC-AUC    : {test_roc_auc:.4f}")
print(f"Test Precision   : {precision:.4f}")
print(f"Test Recall      : {recall:.4f}")
print(f"Test F1          : {f1:.4f}")


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()
print("CONFUSION MATRIX")
print()

print(
    "                 Predicted"
)

print(
    "                 Benign  Malicious"
)

print(
    f"Actual Benign    "
    f"{cm[0, 0]:7d}  "
    f"{cm[0, 1]:9d}"
)

print(
    f"Actual Malicious "
    f"{cm[1, 0]:7d}  "
    f"{cm[1, 1]:9d}"
)


# ============================================================
# SAVE MODEL + SCALER
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "in_channels":
            data.num_node_features,

        "hidden_channels":
            HIDDEN_DIM,

        "dropout":
            DROPOUT,

        "threshold":
            THRESHOLD,

        "best_epoch":
            best_epoch,

        "best_val_pr_auc":
            best_val_pr_auc,

        "test_pr_auc":
            test_pr_auc,

        "test_roc_auc":
            test_roc_auc,

        "test_f1":
            f1,
    },
    MODEL_FILE,
)



# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("MODEL SAVED")
print("=" * 70)

print(MODEL_FILE)
print(SCALER_FILE)

print()
print("GraphSAGE 2-hop scaled experiment complete.")