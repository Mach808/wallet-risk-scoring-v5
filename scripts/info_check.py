import torch
import pandas as pd

data = torch.load(
    "data/processed/graph/graph_v03_2hop.pt",
    weights_only=False
)

features = pd.read_csv(
    "data/processed/graph/2hop_bounded_features.csv"
)

labeled = features.loc[
    features["label"] != -1,
    ["node_id", "address", "label"]
]

masked = (
    data.train_mask
    | data.val_mask
    | data.test_mask
)

missing = labeled[
    ~masked[labeled["node_id"].values].numpy()
]

print(missing)