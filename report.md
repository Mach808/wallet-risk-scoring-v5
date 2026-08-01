# Wallet Risk Scoring MVP — Project Report

## 1. Project Goal

Build one small, reliable, end-to-end MVP for Ethereum wallet risk scoring and improve the same project incrementally instead of repeatedly starting new projects.

### MVP Input

An Ethereum wallet address.

### MVP Output

- Risk score from `0–100`
- Risk level: `LOW`, `MEDIUM`, or `HIGH`
- Human-readable reasons contributing to the score

Example:

```text
Wallet: 0xabc...

Risk Score: 78/100
Risk Level: HIGH

Reasons:
- Interacted with 2 known malicious wallets
- 31% of counterparties are risky
- Very high transaction burstiness
- Suspicious recent activity pattern
```

---

## 2. MVP Philosophy

The first version should be deliberately small.

For MVP v0.1, avoid:

- GNNs
- Huge transaction graphs
- Hundreds of thousands of wallets
- Large feature sets
- Complex token/contract analysis
- Multiple repositories
- Constantly rebuilding the dataset

The goal is to first prove that the complete pipeline works:

```text
Labels
  ↓
Transactions
  ↓
Features
  ↓
Model
  ↓
Risk Score
  ↓
API
```

Once this pipeline is reliable, improve each component inside the same repository.

---

## 3. Scope of MVP v0.1

### Included

- Ethereum wallets
- Real labeled wallet addresses
- Limited transaction history
- Small set of explainable features
- Logistic Regression baseline
- Random Forest model
- Probability-based risk score
- Risk categories
- Basic explanation/reason generation
- Single-wallet inference
- FastAPI scoring endpoint

### Not Included Initially

- Graph Neural Networks
- GraphSAGE / GCN
- Full Ethereum transaction graph
- ERC-20 approval-risk analysis
- Scam-token detection
- Multi-hop graph analysis
- PageRank/community detection
- Smart-contract bytecode analysis
- Complex temporal models
- Ensemble models

These can be added after the MVP works.

---

## 4. Dataset Strategy

The MVP will use a small labeled dataset of real Ethereum addresses.

### Initial Target

| Class | Target |
|---|---:|
| Malicious | 100–500 |
| Benign | 500–1000 |

The MVP can begin with fewer addresses if the labels are trustworthy.

For example:

```text
100 malicious
300 benign
```

is sufficient for an initial baseline.

### Label Files

```text
data/
└── labels/
    ├── malicious.csv
    └── benign.csv
```

Minimum format:

```csv
address,label
0x...,1
```

and:

```csv
address,label
0x...,0
```

Prefer storing additional provenance information:

```csv
address,label,source,category,notes
```

Example categories for malicious addresses may include:

- phishing
- scam
- exploit
- rug pull
- laundering
- drainer
- other confirmed malicious behavior

### Important Dataset Rule

Every label should have a defensible source.

Do not automatically treat an unknown wallet as benign.

For MVP v0.1, once the initial labeled dataset is accepted, freeze it long enough to build and evaluate the complete pipeline instead of continuously rebuilding the dataset.

---

## 5. Transaction Collection

Fetch only a bounded amount of transaction history for each labeled wallet.

Initial target:

```text
200–500 transactions per wallet
```

This keeps:

- API usage manageable
- processing time manageable
- storage manageable
- debugging simple

Raw transactions should be stored rather than repeatedly fetched.

Suggested output:

```text
data/raw/transactions.csv
```

Transaction collection should initially focus on information required by the MVP features.

---

## 6. MVP Features

Start with approximately 10 explainable features.

| Feature | Description |
|---|---|
| `tx_count` | Total observed transaction count |
| `total_received` | Total ETH/value received |
| `total_sent` | Total ETH/value sent |
| `unique_senders` | Number of unique incoming counterparties |
| `unique_receivers` | Number of unique outgoing counterparties |
| `active_days` | Number of days between first and last observed activity |
| `avg_tx_value` | Average transaction value |
| `max_tx_value` | Maximum observed transaction value |
| `risky_counterparty_count` | Number of known risky counterparties interacted with |
| `risky_counterparty_ratio` | Fraction of counterparties that are known risky |

Processed features:

```text
data/processed/wallet_features.csv
```

Expected structure:

```csv
address,tx_count,total_received,total_sent,unique_senders,unique_receivers,active_days,avg_tx_value,max_tx_value,risky_counterparty_count,risky_counterparty_ratio,label
```

---

## 7. Data Quality Requirements

Before training, verify:

- No duplicate wallet addresses
- Valid Ethereum address format
- No accidental label conflicts
- No `NaN` or infinite model inputs
- No impossible numeric values
- ETH/Wei conversion is handled consistently
- Features have expected ranges
- No leakage from the target label
- Train/test splitting is reproducible

Special attention should be given to value conversion to avoid extremely large values caused by incorrectly interpreting raw blockchain units.

---

## 8. Models

### Baseline 1 — Logistic Regression

Use Logistic Regression as a simple interpretable baseline.

Its purpose is to answer:

> Can these features separate risky and benign wallets at all?

### Baseline 2 — Random Forest

Random Forest will be the primary MVP model.

Benefits:

- Works well on tabular data
- Handles nonlinear relationships
- Requires relatively little data
- Provides feature importance
- Produces class probabilities
- Easy to debug and deploy

---

## 9. Risk Score

The model's malicious-class probability becomes the risk score.

Example:

```python
malicious_probability = model.predict_proba(features)[0][1]
risk_score = malicious_probability * 100
```

If:

```text
P(malicious) = 0.73
```

then:

```text
Risk Score = 73/100
```

Initial risk categories can be:

```text
0–39   → LOW
40–69  → MEDIUM
70–100 → HIGH
```

These thresholds are provisional and should later be calibrated using validation results.

A model probability is not automatically a real-world probability of criminality. The score should be presented as model-estimated risk based on the available data and features.

---

## 10. Explainability

The API should not return only a number.

Example:

```json
{
  "address": "0x...",
  "risk_score": 78,
  "risk_level": "HIGH",
  "reasons": [
    "Interacted with known malicious wallets",
    "High risky-counterparty ratio",
    "Unusual transaction-value pattern"
  ]
}
```

For the first version, explanations can be rule-based using feature values.

Later versions can use:

- feature importance
- permutation importance
- SHAP
- graph-based explanations

---

## 11. Model Evaluation

Do not judge the model only by accuracy.

Track:

- Precision
- Recall
- F1 score
- PR-AUC
- ROC-AUC
- Confusion matrix

PR-AUC is particularly important when malicious wallets are much less common than benign wallets.

Always compare future models against the MVP baseline.

---

## 12. Repository Structure

```text
wallet-risk-scoring/
│
├── data/
│   ├── labels/
│   │   ├── malicious.csv
│   │   └── benign.csv
│   │
│   ├── raw/
│   │   └── transactions.csv
│   │
│   └── processed/
│       └── wallet_features.csv
│
├── scripts/
│   ├── 01_build_labels.py
│   ├── 02_fetch_transactions.py
│   ├── 03_build_features.py
│   └── 04_train_model.py
│
├── models/
│   └── random_forest.pkl
│
├── app/
│   ├── main.py
│   ├── scoring.py
│   └── features.py
│
├── tests/
│
├── report.md
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

---

## 13. Development Milestones

### Milestone 1 — Labels

Goal:

```text
address → trustworthy label
```

Requirements:

- Collect malicious wallets
- Collect benign wallets
- Normalize addresses
- Remove duplicates
- Detect conflicting labels
- Record label provenance

Success condition:

A clean labeled CSV exists and can be loaded without errors.

---

### Milestone 2 — Transaction Fetching

Goal:

```text
wallet → transactions
```

Start by testing approximately 10 wallets.

Success condition:

Transaction fetching works reliably and produces consistent raw data.

---

### Milestone 3 — Feature Engineering

Goal:

```text
transactions → feature vector
```

Success condition:

Each labeled wallet produces one valid row of model-ready features.

No:

- NaNs
- infinity
- overflow
- obviously corrupted values

---

### Milestone 4 — Baseline Training

Goal:

```text
features → trained model
```

Train:

1. Logistic Regression
2. Random Forest

Generate:

- Precision
- Recall
- F1
- PR-AUC
- ROC-AUC
- Confusion matrix

Save the selected model to:

```text
models/random_forest.pkl
```

---

### Milestone 5 — Single-Wallet Inference

Target interface:

```bash
python score.py 0x...
```

Example output:

```text
Wallet: 0x...

Risk Score: 74/100
Risk Level: HIGH

Reasons:
- 4 risky counterparties
- Risky counterparty ratio: 18%
- Unusual transaction behavior
```

Success condition:

A previously unseen wallet can be processed through the complete pipeline.

---

### Milestone 6 — API

Use FastAPI.

Target endpoint:

```text
GET /score/{address}
```

Example response:

```json
{
  "address": "0x...",
  "risk_score": 74,
  "risk_level": "HIGH",
  "reasons": [
    "Interacted with risky counterparties",
    "High risky-counterparty ratio"
  ]
}
```

At this point, MVP v0.1 is complete.

---

## 14. Version Roadmap

### v0.1 — Basic MVP

- Small real dataset
- Basic transaction features
- Logistic Regression
- Random Forest
- Risk score
- Explanations
- API

### v0.2 — Better Dataset

Improve:

- number of labeled wallets
- label quality
- source diversity
- class balance
- deduplication
- provenance tracking

### v0.3 — Temporal Features

Potential additions:

- transaction burstiness
- average time between transactions
- inactivity periods
- wallet age
- recent activity ratio
- transaction-frequency changes

### v0.4 — Contract and Token Features

Potential additions:

- contract interaction count
- unique contracts interacted with
- ERC-20 transfer behavior
- verified/unverified token interactions
- suspicious-token exposure

### v0.5 — Graph Features

Potential additions:

- risky 1-hop neighbor count
- risky 1-hop ratio
- degree
- PageRank
- neighborhood statistics

### v0.6 — Approval and Scam Features

Potential additions:

- unlimited approvals granted
- approvals to suspicious contracts
- active dangerous approvals
- spam-token exposure
- suspicious-token sender count

### v0.7 — GNN

Only after a strong tabular baseline exists.

Potential models:

- GCN
- GraphSAGE

The GNN must be evaluated against the Random Forest baseline.

### v1.0 — Ensemble

Potential final architecture:

```text
Tabular Features ──→ Random Forest ─┐
                                   ├─→ Final Risk Score
Graph Features ────→ GNN ──────────┘
```

---

## 15. Project Rules

### Rule 1 — One Repository

Do not restart the project in a new repository when an experiment fails.

Fix or improve the existing project.

### Rule 2 — Small Before Large

Never scale a broken pipeline.

Test:

```text
10 wallets
↓
50 wallets
↓
100 wallets
↓
full MVP dataset
```

### Rule 3 — Cache Blockchain Data

Do not repeatedly spend API credits fetching the same transaction history.

### Rule 4 — Keep Raw Data

Raw blockchain/API responses should remain separate from processed features.

### Rule 5 — Every Feature Must Have a Reason

Do not add features simply because they are available.

For each feature, document:

- definition
- calculation
- expected range
- why it may indicate risk

### Rule 6 — Prevent Data Leakage

Information derived directly from the known label must not accidentally become a model feature.

Counterparty-risk features need special care because the known malicious-address set is related to the labels used for training.

### Rule 7 — Baseline Before GNN

A GNN is useful only if it provides measurable value over the simpler baseline.

### Rule 8 — Reproducibility

Keep:

- fixed random seeds
- versioned feature definitions
- model metrics
- dataset statistics
- experiment results

---

## 16. Current MVP Definition

The project is considered a successful MVP when the following works:

```text
Ethereum Address
       ↓
Fetch Transactions
       ↓
Build ~10 Features
       ↓
Load Trained Model
       ↓
Predict Risk
       ↓
Risk Score 0–100
       ↓
LOW / MEDIUM / HIGH
       ↓
Human-Readable Reasons
```

The MVP does **not** need to be production-grade or state-of-the-art.

Its purpose is to establish a stable foundation that can be measured and improved.

---

## 17. Immediate Next Step

The first task is **Milestone 1: build the labeled dataset**.

Before model development:

1. Decide trustworthy sources for malicious labels.
2. Decide trustworthy sources/methods for benign labels.
3. Create a normalized label schema.
4. Collect a small initial dataset.
5. Deduplicate addresses.
6. detect label conflicts.
7. Record label provenance.
8. Freeze the first MVP dataset.

Only after the labeled dataset is usable should transaction collection begin.

---

## 18. Progress Tracker

| Stage | Status |
|---|---|
| Project scope defined | ✅ |
| Repository structure defined | ✅ |
| Dataset collected | ⬜ |
| Labels validated | ⬜ |
| Transactions fetched | ⬜ |
| Features generated | ⬜ |
| Logistic Regression trained | ⬜ |
| Random Forest trained | ⬜ |
| Evaluation completed | ⬜ |
| Single-wallet inference | ⬜ |
| Risk explanations | ⬜ |
| FastAPI endpoint | ⬜ |
| MVP v0.1 complete | ⬜ |

---

## 19. Core Principle

> Build the smallest wallet-risk system that works end-to-end, establish a measurable baseline, and improve that same system one component at a time.

The priority is not complexity.

The priority is a **working, reproducible and improvable wallet risk scoring pipeline**.
