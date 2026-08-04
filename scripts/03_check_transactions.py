from pathlib import Path

import pandas as pd


# ============================================================
# FILES
# ============================================================

TRANSACTIONS_FILE = Path("data/raw/transactions.csv")

MALICIOUS_FILE = Path("data/labels/malicious.csv")
BENIGN_FILE = Path("data/labels/benign.csv")

PROCESSED_FILE = Path("data/raw/processed_wallets.txt")


# ============================================================
# LOAD LABELS
# ============================================================

malicious = pd.read_csv(MALICIOUS_FILE)
benign = pd.read_csv(BENIGN_FILE)

wallets = pd.concat(
    [malicious, benign],
    ignore_index=True
)

wallets["address"] = (
    wallets["address"]
    .astype(str)
    .str.strip()
    .str.lower()
)

wallets = wallets.drop_duplicates(
    subset=["address"]
)

all_wallets = set(wallets["address"])


# ============================================================
# LOAD TRANSACTIONS
# ============================================================

transactions = pd.read_csv(
    TRANSACTIONS_FILE
)

transactions["wallet_address"] = (
    transactions["wallet_address"]
    .astype(str)
    .str.strip()
    .str.lower()
)


# ============================================================
# BASIC STATS
# ============================================================

total_transactions = len(transactions)

wallets_with_transactions = set(
    transactions["wallet_address"].unique()
)

wallets_without_transactions = (
    all_wallets - wallets_with_transactions
)


# ============================================================
# CATEGORY COUNTS
# ============================================================

category_counts = (
    transactions["category"]
    .value_counts()
)


# ============================================================
# DIRECTION COUNTS
# ============================================================

direction_counts = (
    transactions["direction"]
    .value_counts()
)


# ============================================================
# TRANSACTIONS PER WALLET
# ============================================================

tx_per_wallet = (
    transactions
    .groupby("wallet_address")
    .size()
)


# ============================================================
# PROCESSED COUNT
# ============================================================

processed = set()

if PROCESSED_FILE.exists():

    with open(PROCESSED_FILE) as f:

        processed = {
            line.strip().lower()
            for line in f
            if line.strip()
        }


# ============================================================
# PRINT REPORT
# ============================================================

print()
print("=" * 60)
print("TRANSACTION DATASET REPORT")
print("=" * 60)

print()

print("LABEL DATASET")
print("-" * 60)

print(f"Malicious wallets       : {len(malicious)}")
print(f"Benign wallets          : {len(benign)}")
print(f"Total labeled wallets   : {len(all_wallets)}")


print()
print("FETCH STATUS")
print("-" * 60)

print(f"Processed wallets       : {len(processed)}")
print(f"Unprocessed wallets     : {len(all_wallets - processed)}")


print()
print("TRANSACTIONS")
print("-" * 60)

print(f"Total transactions      : {total_transactions}")
print(
    f"Wallets with tx         : "
    f"{len(wallets_with_transactions)}"
)
print(
    f"Wallets with ZERO tx    : "
    f"{len(wallets_without_transactions)}"
)


print()
print("TRANSACTION CATEGORY")
print("-" * 60)

for category, count in category_counts.items():

    print(
        f"{category:<20}: {count}"
    )


print()
print("TRANSACTION DIRECTION")
print("-" * 60)

for direction, count in direction_counts.items():

    print(
        f"{direction:<20}: {count}"
    )


print()
print("TRANSACTIONS PER WALLET")
print("-" * 60)

print(
    f"Minimum                 : "
    f"{tx_per_wallet.min()}"
)

print(
    f"Median                  : "
    f"{tx_per_wallet.median():.0f}"
)

print(
    f"Mean                    : "
    f"{tx_per_wallet.mean():.2f}"
)

print(
    f"Maximum                 : "
    f"{tx_per_wallet.max()}"
)


# ============================================================
# ZERO TRANSACTION BREAKDOWN
# ============================================================

zero_df = wallets[
    wallets["address"].isin(
        wallets_without_transactions
    )
]

print()
print("ZERO-TRANSACTION WALLETS BY LABEL")
print("-" * 60)

if len(zero_df) > 0:

    print(
        zero_df["label"]
        .value_counts()
        .sort_index()
    )

else:

    print("None")


# ============================================================
# SAVE ZERO TRANSACTION WALLETS
# ============================================================

if len(zero_df) > 0:

    output = Path(
        "data/raw/zero_transaction_wallets.csv"
    )

    zero_df.to_csv(
        output,
        index=False
    )

    print()
    print(
        f"Zero-transaction wallets saved to:"
    )

    print(output)


print()
print("=" * 60)