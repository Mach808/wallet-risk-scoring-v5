import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

ALCHEMY_API_KEY = os.getenv("ALCHEMY_API_KEY")

if not ALCHEMY_API_KEY:
    raise RuntimeError("ALCHEMY_API_KEY not found in .env")

ALCHEMY_URL = f"https://eth-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}"

MALICIOUS_FILE = Path("data/labels/malicious.csv")
BENIGN_FILE = Path("data/labels/benign.csv")

OUTPUT_FILE = Path("data/raw/transactions.csv")
PROCESSED_FILE = Path("data/raw/processed_wallets.txt")

MAX_TX_PER_WALLET = 500
PAGE_SIZE = 100
MAX_RETRIES = 5

CATEGORIES = [
    "external",
    "internal"
]


# ============================================================
# LOAD WALLETS
# ============================================================

def load_wallets():

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

    print("=" * 60)
    print("WALLETS")
    print("=" * 60)

    print(f"Malicious : {len(malicious)}")
    print(f"Benign    : {len(benign)}")
    print(f"Total     : {len(wallets)}")

    return wallets


# ============================================================
# PROCESSED WALLETS
# ============================================================

def load_processed():

    if not PROCESSED_FILE.exists():
        return set()

    with open(PROCESSED_FILE, "r") as f:

        return {
            line.strip().lower()
            for line in f
            if line.strip()
        }


def mark_processed(address):

    PROCESSED_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(PROCESSED_FILE, "a") as f:
        f.write(address + "\n")


# ============================================================
# ALCHEMY REQUEST
# ============================================================

def alchemy_request(params):

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "alchemy_getAssetTransfers",
        "params": [params]
    }

    for attempt in range(MAX_RETRIES):

        try:

            response = requests.post(
                ALCHEMY_URL,
                json=payload,
                timeout=30
            )

            if response.status_code == 429:

                wait = 2 ** attempt

                print(
                    f"Rate limited. Waiting {wait}s..."
                )

                time.sleep(wait)

                continue

            response.raise_for_status()

            data = response.json()

            if "error" in data:

                print(
                    "Alchemy error:",
                    data["error"]
                )

                time.sleep(2 ** attempt)

                continue

            return data.get(
                "result",
                {}
            )

        except requests.RequestException as e:

            wait = 2 ** attempt

            print(
                f"Request error: {e}"
            )

            print(
                f"Retrying in {wait}s..."
            )

            time.sleep(wait)

    return None


# ============================================================
# FETCH TRANSFERS
# ============================================================

def fetch_direction(
    address,
    direction
):

    transfers = []

    page_key = None

    while True:

        params = {

            "fromBlock": "0x0",
            "toBlock": "latest",

            "category": CATEGORIES,

            "withMetadata": True,

            # Keep zero-value external transactions too.
            #
            # These may represent contract interactions,
            # which are useful behavioral information.
            "excludeZeroValue": False,

            "maxCount": hex(
                PAGE_SIZE
            )
        }

        if direction == "incoming":

            params["toAddress"] = address

        elif direction == "outgoing":

            params["fromAddress"] = address

        else:

            raise ValueError(
                "Invalid direction"
            )

        if page_key:

            params["pageKey"] = page_key

        result = alchemy_request(
            params
        )

        if result is None:
            break

        batch = result.get(
            "transfers",
            []
        )

        transfers.extend(
            batch
        )

        # We don't need unlimited history.
        #
        # Fetch a little more than 500 because
        # incoming/outgoing may contain duplicate hashes.
        if len(transfers) >= MAX_TX_PER_WALLET:
            break

        page_key = result.get(
            "pageKey"
        )

        if not page_key:
            break

    return transfers


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_transfer(
    wallet,
    transfer,
    direction
):

    metadata = transfer.get(
        "metadata",
        {}
    )

    return {

        "wallet_address":
            wallet,

        "hash":
            transfer.get(
                "hash"
            ),

        "block_num":
            transfer.get(
                "blockNum"
            ),

        "timestamp":
            metadata.get(
                "blockTimestamp"
            ),

        "from_address":
            (
                transfer.get("from")
                or ""
            ).lower(),

        "to_address":
            (
                transfer.get("to")
                or ""
            ).lower(),

        "value":
            transfer.get(
                "value"
            ),

        "asset":
            transfer.get(
                "asset"
            ),

        "category":
            transfer.get(
                "category"
            ),

        "direction":
            direction
    }


# ============================================================
# FETCH ONE WALLET
# ============================================================

def fetch_wallet(address):

    incoming = fetch_direction(
        address,
        "incoming"
    )

    outgoing = fetch_direction(
        address,
        "outgoing"
    )

    rows = []

    for tx in incoming:

        rows.append(
            normalize_transfer(
                address,
                tx,
                "incoming"
            )
        )

    for tx in outgoing:

        rows.append(
            normalize_transfer(
                address,
                tx,
                "outgoing"
            )
        )

    # ========================================================
    # DEDUPLICATION
    #
    # Internal transactions can share the same transaction
    # hash, so HASH ALONE is NOT sufficient.
    #
    # We use several fields to identify the transfer.
    # ========================================================

    unique = {}

    for row in rows:

        key = (
            row["hash"],
            row["from_address"],
            row["to_address"],
            row["value"],
            row["category"]
        )

        unique[key] = row

    rows = list(
        unique.values()
    )

    # ========================================================
    # SORT NEWEST FIRST
    # ========================================================

    rows.sort(
        key=lambda x: (
            x["timestamp"] or ""
        ),
        reverse=True
    )

    # ========================================================
    # TOTAL CAP
    # ========================================================

    return rows[
        :MAX_TX_PER_WALLET
    ]


# ============================================================
# SAVE
# ============================================================

def save_transactions(rows):

    if not rows:
        return

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.DataFrame(
        rows
    )

    exists = OUTPUT_FILE.exists()

    df.to_csv(
        OUTPUT_FILE,
        mode="a",
        header=not exists,
        index=False
    )


# ============================================================
# MAIN
# ============================================================

def main():

    wallets = load_wallets()

    processed = load_processed()

    print()
    print(
        f"Already processed: "
        f"{len(processed)}"
    )

    remaining = wallets[
        ~wallets["address"].isin(
            processed
        )
    ]

    print(
        f"Remaining: "
        f"{len(remaining)}"
    )

    print()

    print("=" * 60)
    print("FETCHING EXTERNAL + INTERNAL TRANSACTIONS")
    print("=" * 60)

    start_time = time.time()

    try:

        for i, row in enumerate(
            remaining.itertuples(),
            start=1
        ):

            address = row.address

            print()
            print(
                f"[{i}/{len(remaining)}] "
                f"{address}"
            )

            try:

                transactions = fetch_wallet(
                    address
                )

                # Count categories for debugging
                external_count = sum(
                    tx["category"] == "external"
                    for tx in transactions
                )

                internal_count = sum(
                    tx["category"] == "internal"
                    for tx in transactions
                )

                save_transactions(
                    transactions
                )

                mark_processed(
                    address
                )

                print(
                    f"Transactions: "
                    f"{len(transactions)}"
                )

                print(
                    f"  External: "
                    f"{external_count}"
                )

                print(
                    f"  Internal: "
                    f"{internal_count}"
                )

            except Exception as e:

                print(
                    f"FAILED: {e}"
                )

                continue

            elapsed = (
                time.time()
                - start_time
            )

            rate = (
                i / elapsed
                if elapsed > 0
                else 0
            )

            remaining_count = (
                len(remaining) - i
            )

            eta = (
                remaining_count / rate
                if rate > 0
                else 0
            )

            print(
                f"Progress: "
                f"{i}/{len(remaining)} "
                f"| Rate: {rate:.2f} wallets/s "
                f"| ETA: {eta / 60:.1f} min"
            )

    except KeyboardInterrupt:

        print()
        print("Interrupted by user.")

    print()

    print("=" * 60)
    print("DONE")
    print("=" * 60)

    processed = load_processed()

    print(
        f"Processed wallets: "
        f"{len(processed)}"
    )

    print(
        f"Transactions saved to: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()