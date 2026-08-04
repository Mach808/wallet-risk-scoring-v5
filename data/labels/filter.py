"""
Builds a clean, deduplicated address,type,label CSV from one or more
per-category source files.

Each source file can be:
  - a plain list of addresses (one per line, header optional)
  - a CSV with an 'address' column (extra columns are ignored)

USAGE:
    Edit the SOURCES list below, then run:
        python3 build_dataset.py

CONFIG:
    Each entry maps a source file -> the category "type" to assign.
    All rows get label=1 (malicious) by default -- change per-source
    if you ever add a benign/negative class.

If the same address appears in more than one source file, it is kept
ONCE per (address, type) pair -- so a wallet can legitimately carry
multiple types (e.g. rugpull AND laundering_via_mixer), but won't be
duplicated within the same type.
"""

import csv
import sys
from pathlib import Path

# ---- CONFIGURE YOUR SOURCE FILES HERE ----------------------------------
SOURCES = [
    # (file_path, type_label)
    ("rugpull.csv", "rugpull"),
    ("phishing.csv", "phishing"),
    ("sanctioned.csv", "sanctioned"),
    ("laundering_via_mixer_labels.csv", "laundering_via_mixer"),
]
OUTPUT_PATH = "final_dataset.csv"
DEFAULT_LABEL = 1  # 1 = malicious/risky wallet
# --------------------------------------------------------------------------


def extract_addresses(path: str) -> list[str]:
    """Read addresses from a file, whether it's a plain list or has an
    'address' column among others. Skips a header row if present."""
    p = Path(path)
    if not p.exists():
        print(f"  [skip] {path} not found")
        return []

    addresses = []
    with open(p, newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        has_header = "address" in sample.splitlines()[0].lower() if sample else False

        if has_header:
            reader = csv.DictReader(f)
            # find the column that looks like an address column
            addr_col = None
            for col in reader.fieldnames or []:
                if col.strip().lower() == "address":
                    addr_col = col
                    break
            if addr_col is None:
                print(f"  [warn] {path} has a header but no 'address' column, skipping")
                return []
            for row in reader:
                val = (row.get(addr_col) or "").strip()
                if val:
                    addresses.append(val)
        else:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                val = row[0].strip()
                if val:
                    addresses.append(val)

    return addresses


def main():
    seen = set()  # (address_lower, type) pairs already written
    rows = []
    total_raw = 0

    for path, type_label in SOURCES:
        print(f"Reading {path} -> type={type_label}")
        addrs = extract_addresses(path)
        total_raw += len(addrs)
        added = 0
        for addr in addrs:
            addr_norm = addr.lower()
            if not addr_norm.startswith("0x") or len(addr_norm) != 42:
                print(f"  [warn] skipping malformed address: {addr}")
                continue
            key = (addr_norm, type_label)
            if key in seen:
                continue
            seen.add(key)
            rows.append((addr_norm, type_label, DEFAULT_LABEL))
            added += 1
        print(f"  -> {added} unique new rows added")

    rows.sort(key=lambda r: (r[1], r[0]))

    with open(OUTPUT_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["address", "type", "label"])
        w.writerows(rows)

    unique_addresses = len(set(r[0] for r in rows))
    print(f"\nDone. {len(rows)} (address,type) rows | {unique_addresses} unique addresses")
    print(f"Written to {OUTPUT_PATH}")

    # flag addresses that appear under more than one type (multi-label wallets)
    from collections import defaultdict
    by_addr = defaultdict(list)
    for addr, t, _ in rows:
        by_addr[addr].append(t)
    multi = {a: ts for a, ts in by_addr.items() if len(ts) > 1}
    if multi:
        print(f"\n{len(multi)} address(es) appear under multiple types:")
        for a, ts in multi.items():
            print(f"  {a}: {', '.join(ts)}")


if __name__ == "__main__":
    main()