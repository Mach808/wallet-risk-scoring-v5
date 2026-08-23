from live_fetch import fetch_wallet_data
from live_features import build_live_features

import pandas as pd


ADDRESS = "0xf9e4faab0b92657933632413cb2a6d13be9b8778"


# ------------------------------------------------------------
# FETCH
# ------------------------------------------------------------

data = fetch_wallet_data(
    ADDRESS
)

print(
    f"ETH transactions   : {len(data['eth'])}"
)

print(
    f"ERC20 transactions : {len(data['erc20'])}"
)


# ------------------------------------------------------------
# LIVE FEATURES
# ------------------------------------------------------------

live = build_live_features(
    ADDRESS,
    data["eth"],
    data["erc20"],
)

live_df = pd.DataFrame(
    [live]
)

live_df["address"] = ADDRESS


# ------------------------------------------------------------
# STORED FEATURES
# ------------------------------------------------------------

stored = pd.read_csv(
    "data/processed/combined_wallet_features_v03.csv"
)

stored["address"] = (
    stored["address"]
    .astype(str)
    .str.strip()
    .str.lower()
)

stored_row = stored[
    stored["address"] == ADDRESS.lower()
]


if stored_row.empty:

    print(
        "\nWallet not found in stored dataset."
    )

else:

    stored_row = stored_row.iloc[0]

    print(
        "\n" + "=" * 70
    )

    print(
        "LIVE VS STORED FEATURES"
    )

    print(
        "=" * 70
    )

    for feature in live:

        if feature == "address":
            continue

        stored_value = stored_row[feature]
        live_value = live[feature]

        difference = (
            float(live_value)
            - float(stored_value)
        )

        print(
            f"{feature:35s} "
            f"stored={float(stored_value):12.6f} "
            f"live={float(live_value):12.6f} "
            f"diff={difference:12.6f}"
        )