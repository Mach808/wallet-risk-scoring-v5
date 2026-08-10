import os
import time

import requests
from dotenv import load_dotenv


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

ALCHEMY_API_KEY = os.getenv(
    "ALCHEMY_API_KEY"
)

if not ALCHEMY_API_KEY:
    raise RuntimeError(
        "ALCHEMY_API_KEY not found in .env"
    )


ALCHEMY_URL = (
    "https://eth-mainnet.g.alchemy.com/v2/"
    f"{ALCHEMY_API_KEY}"
)


MAX_TX_PER_WALLET = 500
PAGE_SIZE = 100
MAX_RETRIES = 5

CATEGORIES = [
    "external",
    "internal",
]


# ============================================================
# ALCHEMY REQUEST
# ============================================================

def alchemy_request(params):

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "alchemy_getAssetTransfers",
        "params": [params],
    }

    for attempt in range(
        MAX_RETRIES
    ):

        try:

            response = requests.post(
                ALCHEMY_URL,
                json=payload,
                timeout=30,
            )

            if response.status_code == 429:

                wait = 2 ** attempt

                print(
                    f"Rate limited. "
                    f"Waiting {wait}s..."
                )

                time.sleep(wait)

                continue

            response.raise_for_status()

            data = response.json()

            if "error" in data:

                print(
                    "Alchemy error:",
                    data["error"],
                )

                time.sleep(
                    2 ** attempt
                )

                continue

            return data.get(
                "result",
                {},
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
# FETCH ONE DIRECTION
# ============================================================

def fetch_direction(
    address,
    direction,
):

    transfers = []

    page_key = None

    while True:

        params = {

            "fromBlock":
                "0x0",

            "toBlock":
                "latest",

            "category":
                CATEGORIES,

            "withMetadata":
                True,

            "excludeZeroValue":
                False,

            "maxCount":
                hex(PAGE_SIZE),
        }

        if direction == "incoming":

            params[
                "toAddress"
            ] = address

        elif direction == "outgoing":

            params[
                "fromAddress"
            ] = address

        else:

            raise ValueError(
                "Invalid direction"
            )

        if page_key:

            params[
                "pageKey"
            ] = page_key

        result = alchemy_request(
            params
        )

        if result is None:

            break

        batch = result.get(
            "transfers",
            [],
        )

        transfers.extend(
            batch
        )

        if (
            len(transfers)
            >= MAX_TX_PER_WALLET
        ):

            break

        page_key = result.get(
            "pageKey"
        )

        if not page_key:

            break

    return transfers


# ============================================================
# NORMALIZE TRANSFER
# ============================================================

def normalize_transfer(
    wallet,
    transfer,
    direction,
):

    metadata = transfer.get(
        "metadata",
        {},
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
            direction,
    }


# ============================================================
# FETCH WALLET
# ============================================================

def fetch_wallet(
    address,
):

    address = (
        str(address)
        .strip()
        .lower()
    )

    incoming = fetch_direction(
        address,
        "incoming",
    )

    outgoing = fetch_direction(
        address,
        "outgoing",
    )

    rows = []

    for transfer in incoming:

        rows.append(
            normalize_transfer(
                address,
                transfer,
                "incoming",
            )
        )

    for transfer in outgoing:

        rows.append(
            normalize_transfer(
                address,
                transfer,
                "outgoing",
            )
        )


    # ========================================================
    # DEDUPLICATION
    # ========================================================

    unique = {}

    for row in rows:

        key = (
            row["hash"],
            row["from_address"],
            row["to_address"],
            row["value"],
            row["category"],
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
            x["timestamp"]
            or ""
        ),
        reverse=True,
    )


    # ========================================================
    # TOTAL CAP
    # ========================================================

    return rows[
        :MAX_TX_PER_WALLET
    ]