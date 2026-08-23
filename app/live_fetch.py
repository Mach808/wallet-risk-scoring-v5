import sys
from pathlib import Path

# Add project root to Python path
ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data_collection.transaction_fetcher import (
    fetch_wallet as fetch_eth_wallet,
)

from scripts.data_collection.fetch_token_transactions import (
    fetch_wallet as fetch_erc20_wallet,
)




def fetch_wallet_data(address):

    address = (
        str(address)
        .strip()
        .lower()
    )

    if not address.startswith("0x") or len(address) != 42:
        raise ValueError(
            "Invalid Ethereum address."
        )

    eth_rows = fetch_eth_wallet(
        address
    )

    erc20_rows = fetch_erc20_wallet(
        address
    )

    return {
        "eth": eth_rows,
        "erc20": erc20_rows,
    }