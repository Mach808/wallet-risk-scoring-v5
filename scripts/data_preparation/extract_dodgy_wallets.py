import pandas as pd
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path(
    "data/labels/eth_addresses.csv"
)

OUTPUT_FILE = Path(
    "data/labels/external_dodgy_wallets.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print(f"Total rows: {len(df):,}")


# ============================================================
# KEEP ONLY DODGY EOAs / WALLETS
# ============================================================

df = df[
    (df["Label"].astype(str).str.strip().str.lower() == "dodgy")
    &
    (df["Account Type"].astype(str).str.strip().str.lower() == "wallet")
].copy()

print(f"Dodgy wallets: {len(df):,}")


# ============================================================
# CATEGORY MAPPING
# ============================================================

TAG_MAPPING = {
    "Phishing": "phishing",

    "Phish / Hack": "phish_hack",

    "Upbit Hack": "hack",
    "Heist": "hack",
    "Cryptopia Hack": "hack",
    "bZx Exploit": "hack",
    "Lendf.Me Hack": "hack",
    "EtherDelta Hack": "hack",

    "Compromised": "compromised",

    "Fake ICO": "scam",
    "Plus Token Scam": "scam",
    "Scam": "scam",
}


# ============================================================
# FIND ALL TAG COLUMNS
# ============================================================

tag_columns = [
    column
    for column in df.columns
    if "tag" in str(column).lower()
]

print("\nTag columns:")
print(tag_columns)


# ============================================================
# EXTRACT ONE ROW PER MATCHING TAG
# ============================================================

rows = []

for _, row in df.iterrows():

    address = (
        str(row["Address"])
        .strip()
        .lower()
    )

    if not address.startswith("0x"):
        continue

    for column in tag_columns:

        value = row[column]

        if pd.isna(value):
            continue

        tag = str(value).strip()

        if tag not in TAG_MAPPING:
            continue

        rows.append(
            {
                "address": address,
                "type": TAG_MAPPING[tag],
                "label": 1,
                "source_tag": tag,
                "name": row.get("Name", ""),
            }
        )


# ============================================================
# BUILD OUTPUT
# ============================================================

result = pd.DataFrame(rows)

if result.empty:
    raise RuntimeError(
        "No matching malicious wallets found."
    )


# Normalize addresses
result["address"] = (
    result["address"]
    .astype(str)
    .str.strip()
    .str.lower()
)


# Remove duplicate address/tag combinations
result = result.drop_duplicates(
    subset=[
        "address",
        "source_tag",
    ]
)


# One category per address
#
# If an address has multiple matching tags,
# keep the first one for now.
result = result.drop_duplicates(
    subset=["address"],
    keep="first",
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


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 60)
print("EXTRACTION COMPLETE")
print("=" * 60)

print(f"Unique wallets extracted: {len(result):,}")

print("\nTYPE DISTRIBUTION:")
print(
    result["type"]
    .value_counts()
    .to_string()
)

print("\nSOURCE TAG DISTRIBUTION:")
print(
    result["source_tag"]
    .value_counts()
    .to_string()
)

print(f"\nSaved to: {OUTPUT_FILE}")