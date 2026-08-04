# Wallet Risk Scoring — Feature Definitions

This document defines the features used in the Wallet Risk Scoring MVP v0.1.

The current model uses **12 behavioral features** derived from external and internal Ethereum transactions.

---

## 1. `total_tx_count`

**Description:**
Total number of observed incoming and outgoing transactions associated with the wallet.

**Formula:**

[
\text{total_tx_count}
=====================

N_{incoming} + N_{outgoing}
]

Where:

* (N_{incoming}) = number of incoming transactions
* (N_{outgoing}) = number of outgoing transactions

---

## 2. `incoming_tx_count`

**Description:**
Number of transactions where the wallet is the receiver.

**Formula:**

[
\text{incoming_tx_count}
========================

N(T_{in})
]

where:

[
T_{in} = {t \mid t.direction = incoming}
]

---

## 3. `outgoing_tx_count`

**Description:**
Number of transactions where the wallet is the sender.

**Formula:**

[
\text{outgoing_tx_count}
========================

N(T_{out})
]

where:

[
T_{out} = {t \mid t.direction = outgoing}
]

---

## 4. `total_eth_received`

**Description:**
Total ETH value received by the wallet across the observed incoming external and internal transfers.

**Formula:**

[
\text{total_eth_received}
=========================

\sum_{t \in T_{in}} value(t)
]

Where:

* (T_{in}) = incoming transactions
* (value(t)) = ETH value of transaction/transfer (t)

---

## 5. `total_eth_sent`

**Description:**
Total ETH value sent by the wallet across the observed outgoing external and internal transfers.

**Formula:**

[
\text{total_eth_sent}
=====================

\sum_{t \in T_{out}} value(t)
]

---

## 6. `avg_tx_value`

**Description:**
Average ETH value of all observed transactions associated with the wallet.

**Formula:**

[
\text{avg_tx_value}
===================

\frac{\sum_{t \in T} value(t)}
{|T|}
]

Where:

[
T = T_{in} \cup T_{out}
]

and (|T|) is the total number of observed transactions.

---

## 7. `max_tx_value`

**Description:**
Largest ETH value observed in a single transaction associated with the wallet.

**Formula:**

[
\text{max_tx_value}
===================

\max_{t \in T}(value(t))
]

---

## 8. `unique_senders`

**Description:**
Number of distinct addresses that sent ETH to the wallet.

**Formula:**

[
\text{unique_senders}
=====================

\left|
{from(t) \mid t \in T_{in}}
\right|
]

The wallet's own address is excluded if present.

---

## 9. `unique_receivers`

**Description:**
Number of distinct addresses that received ETH from the wallet.

**Formula:**

[
\text{unique_receivers}
=======================

\left|
{to(t) \mid t \in T_{out}}
\right|
]

The wallet's own address is excluded if present.

---

## 10. `unique_counterparties`

**Description:**
Total number of distinct addresses that interacted with the wallet as either sender or receiver.

This avoids double-counting an address that interacted with the wallet in both directions.

**Formula:**

[
S =
{from(t) \mid t \in T_{in}}
]

[
R =
{to(t) \mid t \in T_{out}}
]

Then:

[
\text{unique_counterparties}
============================

|S \cup R|
]

---

## 11. `active_days`

**Description:**
Observed activity span of the wallet in days.

It measures the time between the earliest and latest transaction available in the MVP dataset.

**Formula:**

[
\text{active_days}
==================

(t_{last} - t_{first})_{days} + 1
]

Where:

* (t_{first}) = timestamp of the earliest observed transaction
* (t_{last}) = timestamp of the latest observed transaction

If only one valid timestamp exists:

[
\text{active_days} = 1
]

If no valid timestamp exists:

[
\text{active_days} = 0
]

### Important

Despite the name, this feature currently represents the **activity span**, not the number of distinct calendar days on which transactions occurred.

---

## 12. `internal_tx_ratio`

**Description:**
Fraction of the wallet's observed transactions that are internal Ethereum transfers.

Internal transfers are value movements generated during smart-contract execution.

**Formula:**

[
\text{internal_tx_ratio}
========================

\frac{N_{internal}}
{N_{total}}
]

Where:

* (N_{internal}) = number of internal transactions
* (N_{total}) = total observed transactions

The value ranges from:

[
0 \leq \text{internal_tx_ratio} \leq 1
]

For example:

```text
Total transactions    = 100
Internal transactions = 25

internal_tx_ratio = 25 / 100 = 0.25
```

---

# Feature Summary

| Feature                 | Formula / Operation                          |          |    |
| ----------------------- | -------------------------------------------- | -------- | -- |
| `total_tx_count`        | (N_{incoming} + N_{outgoing})                |          |    |
| `incoming_tx_count`     | (                                            | T_{in}   | )  |
| `outgoing_tx_count`     | (                                            | T_{out}  | )  |
| `total_eth_received`    | (\sum_{t \in T_{in}} value(t))               |          |    |
| `total_eth_sent`        | (\sum_{t \in T_{out}} value(t))              |          |    |
| `avg_tx_value`          | (\frac{\sum value(t)}{                       | T        | }) |
| `max_tx_value`          | (\max(value(t)))                             |          |    |
| `unique_senders`        | Number of unique incoming sender addresses   |          |    |
| `unique_receivers`      | Number of unique outgoing receiver addresses |          |    |
| `unique_counterparties` | (                                            | S \cup R | )  |
| `active_days`           | ((t_{last}-t_{first})_{days}+1)              |          |    |
| `internal_tx_ratio`     | (\frac{N_{internal}}{N_{total}})             |          |    |

---

# Data Scope

The MVP v0.1 features are calculated using:

* Ethereum Mainnet
* External ETH transfers
* Internal ETH transfers
* Maximum of 500 observed transfers per wallet
* Known benign and malicious labeled wallets

The malicious class currently contains:

* Phishing wallets
* Sanctioned wallets
* Wallets associated with laundering through mixers

Rug-pull addresses are excluded from MVP v0.1 because the original rug-pull dataset contained an uncertain mixture of EOAs and smart-contract addresses.

---

# Planned Future Features

These are intentionally excluded from MVP v0.1 and may be introduced in later versions:

* Transaction frequency
* Incoming/outgoing ratio
* Net ETH flow
* Median transaction value
* Transaction-value standard deviation
* Wallet age
* Dormancy periods
* Transaction burstiness
* Risky counterparty count
* Risky counterparty ratio
* Contract interaction count
* Unique contracts interacted with
* ERC-20 activity
* Token diversity
* Approval behavior
* Suspicious/unlimited approvals
* Spam-token exposure
* 1-hop risky-neighbor features
* 2-hop risky-neighbor features
* Degree-based graph features
* PageRank
* Community features
* GNN embeddings

These features should only be introduced after the MVP baseline has been trained and evaluated.
