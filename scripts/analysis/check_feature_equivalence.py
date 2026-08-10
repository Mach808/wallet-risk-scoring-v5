from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ============================================================
# IMPORT PATH
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent

sys.path.insert(
    0,
    str(SCRIPTS_DIR)
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

from feature_engineering.feature_engineering import (
    compute_features,
    FEATURE_COLUMNS,
)


# ============================================================
# CONFIG
# ============================================================

TRANSACTION_FILE = Path(
    "data/raw/transactions.csv"
)

OLD_FEATURE_FILE = Path(
    "data/processed/wallet_features.csv"
)

TOLERANCE = 1e-9


# ============================================================
# LOAD ORIGINAL FEATURES
# ============================================================

print("=" * 70)
print("FEATURE EQUIVALENCE CHECK")
print("=" * 70)

print()
print("Loading original feature dataset...")

old = pd.read_csv(
    OLD_FEATURE_FILE
)

old["address"] = (
    old["address"]
    .astype(str)
    .str.strip()
    .str.lower()
)


# ============================================================
# LOAD TRANSACTIONS
# ============================================================

print(
    "Loading transactions..."
)

tx = pd.read_csv(
    TRANSACTION_FILE
)

print(
    f"Transactions: {len(tx):,}"
)


# ============================================================
# COMPUTE NEW FEATURES
# ============================================================

print()
print(
    "Computing features using "
    "feature_engineering.py..."
)

new = compute_features(
    tx
)

new["address"] = (
    new["address"]
    .astype(str)
    .str.strip()
    .str.lower()
)


# ============================================================
# WALLET COUNTS
# ============================================================

print()
print("=" * 70)
print("WALLET COUNT")
print("=" * 70)

print(
    f"Original feature wallets : {len(old)}"
)

print(
    f"New feature wallets      : {len(new)}"
)


# ============================================================
# ADDRESS SET CHECK
# ============================================================

old_addresses = set(
    old["address"]
)

new_addresses = set(
    new["address"]
)


missing_from_new = (
    old_addresses - new_addresses
)

extra_in_new = (
    new_addresses - old_addresses
)


print()
print(
    f"Original wallets missing "
    f"from new features : "
    f"{len(missing_from_new)}"
)

print(
    f"Additional wallets in "
    f"new features       : "
    f"{len(extra_in_new)}"
)


# ============================================================
# MISSING WALLET CHECK
# ============================================================

if missing_from_new:

    print()
    print(
        "ERROR: Original feature dataset "
        "contains wallets missing from "
        "the new feature dataset."
    )

    print()
    print(
        "First missing addresses:"
    )

    for address in list(
        missing_from_new
    )[:10]:

        print(
            f"  {address}"
        )


# ============================================================
# COMPARE ONLY ORIGINAL WALLETS
# ============================================================

common_addresses = old_addresses & new_addresses


old_common = (
    old[
        old["address"].isin(
            common_addresses
        )
    ]
    .copy()
)

new_common = (
    new[
        new["address"].isin(
            common_addresses
        )
    ]
    .copy()
)


# ============================================================
# ALIGN ROWS
# ============================================================

old_common = (
    old_common
    .set_index("address")
    .sort_index()
)

new_common = (
    new_common
    .set_index("address")
    .sort_index()
)


# ============================================================
# CHECK ROW COUNT
# ============================================================

print()
print("=" * 70)
print("COMMON WALLET COMPARISON")
print("=" * 70)

print(
    f"Wallets being compared: "
    f"{len(common_addresses)}"
)


if len(old_common) != len(
    new_common
):

    raise RuntimeError(
        "Common wallet counts do not match."
    )


# ============================================================
# FEATURE COMPARISON
# ============================================================

print()
print("=" * 70)
print("FEATURE COMPARISON")
print("=" * 70)

differences = []


for feature in FEATURE_COLUMNS:

    old_values = pd.to_numeric(
        old_common[feature],
        errors="coerce",
    )

    new_values = pd.to_numeric(
        new_common[feature],
        errors="coerce",
    )


    # --------------------------------------------------------
    # NaN pattern
    # --------------------------------------------------------

    old_nan = old_values.isna()
    new_nan = new_values.isna()

    nan_mismatch = int(
        (old_nan != new_nan).sum()
    )


    # --------------------------------------------------------
    # Numerical comparison
    # --------------------------------------------------------

    valid = (
        ~old_nan
        &
        ~new_nan
    )


    if valid.any():

        old_array = (
            old_values[valid]
            .to_numpy(
                dtype=float
            )
        )

        new_array = (
            new_values[valid]
            .to_numpy(
                dtype=float
            )
        )

        abs_difference = np.abs(
            old_array
            - new_array
        )

        max_difference = float(
            np.max(
                abs_difference
            )
        )

        different_values = int(
            np.sum(
                abs_difference
                > TOLERANCE
            )
        )

    else:

        max_difference = 0.0
        different_values = 0


    differences.append(
        {
            "feature":
                feature,

            "max_absolute_difference":
                max_difference,

            "different_values":
                different_values,

            "nan_mismatch":
                nan_mismatch,
        }
    )


difference_df = pd.DataFrame(
    differences
)


# ============================================================
# PRINT FEATURE RESULTS
# ============================================================

for row in difference_df.itertuples():

    status = (
        "PASS"
        if (
            row.different_values == 0
            and row.nan_mismatch == 0
        )
        else "FAIL"
    )

    print(
        f"{status:<6} "
        f"{row.feature:<30} "
        f"max_diff="
        f"{row.max_absolute_difference:.12g} "
        f"diff_values="
        f"{row.different_values}"
    )


# ============================================================
# FINAL RESULT
# ============================================================

failed_features = difference_df[
    (
        difference_df[
            "different_values"
        ]
        > 0
    )
    |
    (
        difference_df[
            "nan_mismatch"
        ]
        > 0
    )
]


print()
print("=" * 70)
print("FINAL RESULT")
print("=" * 70)


if (
    len(missing_from_new) == 0
    and failed_features.empty
):

    print()
    print(
        "✓ FEATURE EQUIVALENCE PASSED"
    )

    print()
    print(
        f"All {len(FEATURE_COLUMNS)} features "
        f"match for all {len(common_addresses)} "
        f"original wallets."
    )

    print()
    print(
        f"The new feature module also contains "
        f"{len(extra_in_new)} additional wallets, "
        f"which is expected."
    )

else:

    print()
    print(
        "✗ FEATURE EQUIVALENCE FAILED"
    )

    if missing_from_new:

        print()
        print(
            f"Missing original wallets: "
            f"{len(missing_from_new)}"
        )

    if not failed_features.empty:

        print()
        print(
            f"Features with differences: "
            f"{len(failed_features)}"
        )

    raise SystemExit(1)