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

ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")

if not ETHERSCAN_API_KEY:
    raise RuntimeError(
        "ETHERSCAN_API_KEY not found. Add it to your .env file."
    )

BASE_URL = "https://api.etherscan.io/v2/api"
CHAIN_ID = 1  # Ethereum Mainnet

TARGET_WALLETS = 600

# We don't need to inspect every transaction from every block.
# Recent blocks normally contain plenty of addresses.
MAX_BLOCKS_TO_SCAN = 500

# Save periodically so progress isn't lost.
SAVE_EVERY = 25

MALICIOUS_FILE = Path("data/labels/malicious.csv")
OUTPUT_FILE = Path("data/labels/benign.csv")


# ============================================================
# API HELPER
# ============================================================

def etherscan_request(params, retries=5):
    """
    Send request to Etherscan with basic retry/rate-limit handling.
    """

    params["chainid"] = CHAIN_ID
    params["apikey"] = ETHERSCAN_API_KEY

    for attempt in range(retries):

        try:
            response = requests.get(
                BASE_URL,
                params=params,
                timeout=30
            )

            response.raise_for_status()

            data = response.json()

            # Some Etherscan errors are returned as JSON.
            if "error" in data:
                print(f"[API ERROR] {data['error']}")

                time.sleep(1 + attempt)
                continue

            return data

        except requests.RequestException as e:
            print(f"[REQUEST ERROR] {e}")
            time.sleep(2 + attempt)

    return None


# ============================================================
# ETHEREUM FUNCTIONS
# ============================================================

def get_latest_block():
    """
    Get latest Ethereum block number.
    """

    data = etherscan_request({
        "module": "proxy",
        "action": "eth_blockNumber"
    })

    if not data or "result" not in data:
        raise RuntimeError("Could not fetch latest block.")

    return int(data["result"], 16)


def get_block(block_number):
    """
    Fetch a block including full transaction objects.
    """

    data = etherscan_request({
        "module": "proxy",
        "action": "eth_getBlockByNumber",
        "tag": hex(block_number),
        "boolean": "true"
    })

    if not data:
        return None

    return data.get("result")


def get_code(address):
    """
    Get bytecode stored at an address.

    EOA:
        0x

    Contract:
        non-empty bytecode
    """

    data = etherscan_request({
        "module": "proxy",
        "action": "eth_getCode",
        "address": address,
        "tag": "latest"
    })

    if not data:
        return None

    return data.get("result")


def is_eoa(address):
    """
    Return True if address currently has no contract bytecode.
    """

    code = get_code(address)

    if code is None:
        return False

    return code in ("0x", "0x0", "")


# ============================================================
# DATA FUNCTIONS
# ============================================================

def load_malicious():
    """
    Load known malicious addresses.
    """

    if not MALICIOUS_FILE.exists():
        raise FileNotFoundError(
            f"{MALICIOUS_FILE} does not exist."
        )

    df = pd.read_csv(MALICIOUS_FILE)

    if "address" not in df.columns:
        raise ValueError(
            "malicious.csv must contain an 'address' column."
        )

    addresses = (
        df["address"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    malicious = set(addresses)

    print(f"Loaded {len(malicious)} malicious addresses.")

    return malicious


def load_existing_benign():
    """
    Load previous progress if benign.csv already exists.
    """

    if not OUTPUT_FILE.exists():
        return set()

    df = pd.read_csv(OUTPUT_FILE)

    if "address" not in df.columns:
        return set()

    addresses = (
        df["address"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    benign = set(addresses)

    print(f"Resuming with {len(benign)} benign wallets.")

    return benign


def save_benign(addresses):
    """
    Save benign wallets using the same schema as malicious.csv.
    """

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.DataFrame({
        "address": sorted(addresses),
        "type": "benign",
        "label": 0
    })

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )


# ============================================================
# COLLECTION
# ============================================================

def main():

    malicious = load_malicious()
    benign = load_existing_benign()

    if len(benign) >= TARGET_WALLETS:
        print(
            f"Already have {len(benign)} benign wallets."
        )
        return

    latest_block = get_latest_block()

    print(f"Latest Ethereum block: {latest_block}")
    print(f"Target benign wallets: {TARGET_WALLETS}")
    print()

    checked = set()

    blocks_scanned = 0

    block_number = latest_block

    try:

        while (
            len(benign) < TARGET_WALLETS
            and blocks_scanned < MAX_BLOCKS_TO_SCAN
        ):

            print(
                f"\nScanning block {block_number} "
                f"| benign={len(benign)}/{TARGET_WALLETS}"
            )

            block = get_block(block_number)

            if not block:

                print("Could not retrieve block.")

                block_number -= 1
                blocks_scanned += 1

                continue

            transactions = block.get(
                "transactions",
                []
            )

            print(
                f"Transactions: {len(transactions)}"
            )

            # ------------------------------------------------
            # IMPORTANT:
            #
            # We primarily sample transaction SENDERS.
            #
            # A normal Ethereum transaction sender must be an
            # account capable of signing the transaction.
            #
            # We still run eth_getCode to ensure the address
            # currently has no deployed contract code.
            # ------------------------------------------------

            for tx in transactions:

                if len(benign) >= TARGET_WALLETS:
                    break

                address = tx.get("from")

                if not address:
                    continue

                address = address.lower()

                # Already known malicious
                if address in malicious:
                    continue

                # Already accepted
                if address in benign:
                    continue

                # Already checked during this run
                if address in checked:
                    continue

                checked.add(address)

                print(
                    f"Checking {address[:10]}...",
                    end=" "
                )

                if is_eoa(address):

                    benign.add(address)

                    print(
                        f"EOA ✓ "
                        f"[{len(benign)}/{TARGET_WALLETS}]"
                    )

                    if len(benign) % SAVE_EVERY == 0:

                        save_benign(benign)

                        print(
                            f"Saved progress: "
                            f"{len(benign)} wallets"
                        )

                else:

                    print("contract / rejected")

                # Be polite to API limits.
                time.sleep(0.22)

            block_number -= 1
            blocks_scanned += 1

    except KeyboardInterrupt:

        print("\nInterrupted by user.")

    finally:

        save_benign(benign)

        print()
        print("=" * 50)
        print("COLLECTION COMPLETE")
        print("=" * 50)
        print(f"Benign wallets : {len(benign)}")
        print(f"Malicious      : {len(malicious)}")
        print(f"Blocks scanned : {blocks_scanned}")
        print(f"Saved to       : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()