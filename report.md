# Ethereum Wallet Risk Scoring — MVP Report

## 1. Project Overview

This project develops a small, reproducible MVP for **Ethereum wallet risk scoring** using supervised machine learning.

The objective is to determine whether transaction-level behavioral characteristics of an Ethereum address can be used to distinguish between:

* Benign wallets
* Sanctioned wallets
* Phishing wallets
* Wallets associated with laundering through mixers

Rather than continuously increasing dataset size, changing models, or introducing increasingly complex architectures, the MVP focuses on establishing a reliable baseline using a controlled dataset, reproducible feature engineering, fixed dataset splits, and a held-out test set.

The final MVP uses a **Random Forest classifier with 26 behavioral and temporal features**.

---

# 2. Dataset

## 2.1 Labeled Wallet Collection

The original labeled dataset consisted of:

| Category                       | Wallets |
| ------------------------------ | ------: |
| Benign                         |     600 |
| Rugpull                        |      55 |
| Phishing                       |      64 |
| Sanctioned                     |      96 |
| Laundering                     |      78 |
| **Total originally collected** | **893** |

All collected addresses were unique.

After preprocessing and transaction availability checks, the usable dataset changed slightly.

---

## 2.2 Rugpull Removal

The rugpull dataset contained a mixture of externally owned accounts (EOAs) and smart-contract addresses.

Because the MVP is intended to model **wallet behavior**, mixing contract addresses with EOAs could introduce a strong confounding signal. A model might learn the behavioral differences between contracts and EOAs rather than characteristics associated with malicious wallet activity.

Since the EOA status of all rugpull addresses was uncertain, the rugpull category was removed entirely from the MVP.

The remaining malicious categories were:

* Phishing
* Sanctioned
* Laundering via mixers

---

# 3. Transaction Collection

Ethereum transaction history was collected for the labeled addresses.

Both:

* External transactions
* Internal transactions

were included.

The transaction collection produced:

| Statistic             |   Value |
| --------------------- | ------: |
| Total transactions    | 202,482 |
| External transactions | 171,049 |
| Internal transactions |  31,433 |
| Incoming transactions |  98,621 |
| Outgoing transactions | 103,861 |

The original transaction-fetching stage processed 887 labeled wallets.

Of these:

* 861 had at least one transaction
* 26 had zero retrieved transactions

All 26 zero-transaction wallets belonged to the malicious class.

A maximum of **500 transactions per wallet** was collected during the MVP data collection process.

---

# 4. Final Feature Dataset

After removing the rugpull category and excluding wallets without usable transaction history, the final feature dataset contained:

| Class     | Wallets |
| --------- | ------: |
| Benign    |     600 |
| Malicious |     209 |
| **Total** | **809** |

The malicious wallets consisted of:

| Type                 | Wallets |
| -------------------- | ------: |
| Sanctioned           |      91 |
| Laundering via mixer |      74 |
| Phishing             |      44 |
| **Total malicious**  | **209** |

This results in a moderately imbalanced binary classification dataset, with approximately 25.8% malicious wallets.

---

# 5. Feature Engineering

Two feature versions were evaluated.

## 5.1 MVP v0.1 — Basic Transaction Features

The initial model used 12 features:

1. `total_tx_count`
2. `incoming_tx_count`
3. `outgoing_tx_count`
4. `total_eth_received`
5. `total_eth_sent`
6. `avg_tx_value`
7. `max_tx_value`
8. `unique_senders`
9. `unique_receivers`
10. `unique_counterparties`
11. `activity_span_days`
12. `internal_tx_ratio`

These features primarily capture:

* Transaction volume
* ETH flow
* Counterparty diversity
* Wallet activity duration
* Internal transaction usage

---

## 5.2 MVP v0.2 — Behavioral and Temporal Features

Analysis of v0.1 indicated that basic transaction statistics alone were insufficient to represent several types of malicious behavior, particularly phishing.

Fourteen additional features were therefore introduced:

13. `in_out_tx_ratio`
14. `net_eth_flow`
15. `median_tx_value`
16. `std_tx_value`
17. `distinct_active_days`
18. `tx_frequency`
19. `incoming_value_ratio`
20. `zero_value_tx_ratio`
21. `external_tx_ratio`
22. `counterparty_reuse_ratio`
23. `avg_time_between_tx`
24. `median_time_between_tx`
25. `max_time_between_tx`
26. `burstiness`

The new features capture additional aspects of wallet behavior including:

* Incoming versus outgoing transaction patterns
* Net ETH movement
* Transaction-value distributions
* Actual active-day counts
* Transaction frequency
* Directional value flow
* Repeated interaction with counterparties
* Transaction timing
* Bursty versus regular transaction behavior

The final MVP therefore uses **26 features**.

Detailed definitions and formulas are maintained separately in `features.md`.

---

# 6. Dataset Splitting

The 809-wallet dataset was divided using a stratified split so that the malicious/benign class distribution remained approximately consistent.

| Split      | Wallets |
| ---------- | ------: |
| Training   |     566 |
| Validation |     121 |
| Test       |     122 |
| **Total**  | **809** |

Class distributions were:

### Training

* Benign: 420
* Malicious: 146

### Validation

* Benign: 90
* Malicious: 31

### Test

* Benign: 90
* Malicious: 32

The split addresses were saved and frozen.

All subsequent model versions used the **same train, validation, and test wallets**.

The test set was not used during feature engineering, model selection, threshold analysis, or cross-validation.

---

# 7. MVP v0.1 Baseline

Two initial baseline models were trained:

* Logistic Regression
* Random Forest

## Logistic Regression Validation Results

| Metric    | Result |
| --------- | -----: |
| Precision | 0.6000 |
| Recall    | 0.6774 |
| F1        | 0.6364 |
| ROC-AUC   | 0.8455 |
| PR-AUC    | 0.7389 |

## Random Forest Validation Results

| Metric    | Result |
| --------- | -----: |
| Precision | 0.7097 |
| Recall    | 0.7097 |
| F1        | 0.7097 |
| ROC-AUC   | 0.9118 |
| PR-AUC    | 0.8259 |

Random Forest substantially outperformed Logistic Regression and was therefore selected as the primary MVP model.

---

# 8. v0.1 Risk-Type Analysis

Further validation analysis showed that model performance varied substantially between malicious wallet categories.

At the validation threshold selected during the v0.1 analysis:

| Type                 | Samples | Detected | Recall |
| -------------------- | ------: | -------: | -----: |
| Laundering via mixer |       9 |        9 |   100% |
| Sanctioned           |      14 |        8 |  57.1% |
| Phishing             |       8 |        3 |  37.5% |

The model was highly effective at identifying laundering wallets but substantially weaker at detecting phishing and sanctioned wallets.

This motivated the development of the additional behavioral and temporal features in v0.2.

---

# 9. MVP v0.2 Validation Results

The same Random Forest configuration and frozen dataset split were used with the expanded 26-feature representation.

At a threshold of 0.50:

| Metric    |   v0.1 |   v0.2 |  Change |
| --------- | -----: | -----: | ------: |
| Precision | 0.7097 | 0.6970 | -0.0127 |
| Recall    | 0.7097 | 0.7419 | +0.0322 |
| F1        | 0.7097 | 0.7188 | +0.0091 |
| ROC-AUC   | 0.9118 | 0.9382 | +0.0264 |
| PR-AUC    | 0.8259 | 0.8702 | +0.0443 |

The expanded feature set improved recall, F1, ROC-AUC, and PR-AUC.

---

# 10. v0.2 Malicious-Type Performance

At threshold 0.50, validation performance by malicious category was:

| Type                 | Samples | Detected | Recall |
| -------------------- | ------: | -------: | -----: |
| Laundering via mixer |       9 |        9 |   100% |
| Sanctioned           |      14 |        9 |  64.3% |
| Phishing             |       8 |        5 |  62.5% |

Compared with the original feature set, phishing detection improved substantially.

---

# 11. Threshold Analysis

The v0.2 model was evaluated across thresholds from 0.10 to 0.90.

Selected operating points included:

| Threshold | Precision | Recall |     F1 | FP | FN |
| --------: | --------: | -----: | -----: | -: | -: |
|      0.30 |    0.5800 | 0.9355 | 0.7160 | 21 |  2 |
|      0.40 |    0.6341 | 0.8387 | 0.7222 | 15 |  5 |
|      0.45 |    0.6857 | 0.7742 | 0.7273 | 11 |  7 |
|      0.50 |    0.6970 | 0.7419 | 0.7188 | 10 |  8 |
|      0.55 |    0.8400 | 0.6774 | 0.7500 |  4 | 10 |
|      0.60 |    0.9500 | 0.6129 | 0.7451 |  1 | 12 |

The highest validation F1 occurred at a threshold of **0.55**.

However, the final MVP retained a threshold of **0.50** because it provides a more recall-oriented operating point and was the fixed threshold used during the cross-validation comparison.

The predicted probability can also be interpreted as a continuous risk score rather than forcing every application to use the same binary threshold.

---

# 12. Five-Fold Cross-Validation

Before evaluating the held-out test set, v0.1 and v0.2 were compared using stratified five-fold cross-validation.

Only the original training and validation sets were combined for this experiment:

**566 + 121 = 687 development wallets**

The 122 test wallets remained completely excluded.

## v0.1 Cross-Validation

| Metric    |      Mean ± Std |
| --------- | --------------: |
| Precision | 0.7040 ± 0.0692 |
| Recall    | 0.6789 ± 0.1363 |
| F1        | 0.6879 ± 0.0943 |
| ROC-AUC   | 0.9094 ± 0.0374 |
| PR-AUC    | 0.8124 ± 0.0860 |

## v0.2 Cross-Validation

| Metric    |          Mean ± Std |
| --------- | ------------------: |
| Precision | **0.7771 ± 0.0480** |
| Recall    | **0.7578 ± 0.1231** |
| F1        | **0.7650 ± 0.0809** |
| ROC-AUC   | **0.9381 ± 0.0318** |
| PR-AUC    | **0.8707 ± 0.0625** |

Mean improvement from v0.1 to v0.2:

| Metric    | Improvement |
| --------- | ----------: |
| Precision |     +0.0731 |
| Recall    |     +0.0789 |
| F1        |     +0.0771 |
| ROC-AUC   |     +0.0286 |
| PR-AUC    |     +0.0583 |

Most importantly, v0.2 outperformed v0.1 on:

* PR-AUC: **5/5 folds**
* ROC-AUC: **5/5 folds**
* F1: **5/5 folds**

This provided stronger evidence that the additional behavioral and temporal features improved model performance rather than the improvement being specific to one validation split.

---

# 13. Final Model

After model and feature selection were completed, the MVP configuration was frozen.

Final configuration:

| Component          | Configuration |
| ------------------ | ------------- |
| Model              | Random Forest |
| Number of features | 26            |
| Number of trees    | 300           |
| Class weighting    | Balanced      |
| Random state       | 42            |
| Decision threshold | 0.50          |

The training and validation datasets were then combined:

**687 development wallets**

A new final Random Forest was trained on all development wallets.

Only after training was complete was the held-out test set evaluated.

---

# 14. Final Held-Out Test Results

The final test set contained:

* 90 benign wallets
* 32 malicious wallets
* 122 wallets total

The final model achieved:

| Metric    |     Result |
| --------- | ---------: |
| Precision | **0.9259** |
| Recall    | **0.7812** |
| F1        | **0.8475** |
| ROC-AUC   | **0.9865** |
| PR-AUC    | **0.9669** |
| Accuracy  | **0.9262** |

The confusion matrix was:

|                   | Predicted Benign | Predicted Risky |
| ----------------- | ---------------: | --------------: |
| **Actual Benign** |               88 |               2 |
| **Actual Risky**  |                7 |              25 |

Therefore:

* True negatives: 88
* False positives: 2
* False negatives: 7
* True positives: 25

The model correctly classified **113 of 122 test wallets**.

---

# 15. Final Performance by Malicious Type

The 32 malicious test wallets consisted of:

| Type                 | Samples | Detected | Missed |    Recall |
| -------------------- | ------: | -------: | -----: | --------: |
| Laundering via mixer |      15 |       13 |      2 | **86.7%** |
| Sanctioned           |      14 |       10 |      4 | **71.4%** |
| Phishing             |       3 |        2 |      1 | **66.7%** |

Laundering wallets remained the easiest category for the model to identify.

Results for phishing should be interpreted cautiously because only three phishing wallets were present in the held-out test set.

---

# 16. Final Random Forest Feature Importance

The highest impurity-based Random Forest feature importances in the final model were:

| Rank | Feature                    | Importance |
| ---: | -------------------------- | ---------: |
|    1 | `total_eth_sent`           |     10.52% |
|    2 | `incoming_value_ratio`     |     10.03% |
|    3 | `avg_tx_value`             |      7.96% |
|    4 | `max_tx_value`             |      5.98% |
|    5 | `std_tx_value`             |      5.64% |
|    6 | `total_eth_received`       |      4.42% |
|    7 | `median_time_between_tx`   |      4.26% |
|    8 | `net_eth_flow`             |      4.20% |
|    9 | `avg_time_between_tx`      |      4.14% |
|   10 | `counterparty_reuse_ratio` |      3.87% |

The results suggest that the model uses a combination of:

* ETH flow
* Directional flow
* Transaction-value distribution
* Transaction timing
* Counterparty interaction behavior

rather than relying exclusively on transaction counts.

These values represent Random Forest impurity-based feature importance and should **not** be interpreted as evidence that any individual feature causes malicious behavior.

---

# 17. Key Findings

The MVP produced several important findings.

First, relatively simple transaction-level features are sufficient to create a useful baseline for Ethereum wallet risk classification.

Second, behavioral and temporal features substantially improved the model.

The v0.2 feature set improved mean five-fold cross-validation performance from:

* PR-AUC: 0.8124 → 0.8707
* ROC-AUC: 0.9094 → 0.9381
* F1: 0.6879 → 0.7650

Furthermore, v0.2 outperformed v0.1 on all five folds for PR-AUC, ROC-AUC, and F1.

Third, malicious wallet categories exhibit different behavioral patterns. Laundering wallets were consistently easier to identify than phishing and sanctioned wallets.

Finally, the held-out test results demonstrate that the final model generalized well to the reserved subset, achieving:

* **98.65% ROC-AUC**
* **96.69% PR-AUC**
* **84.75% F1**
* **92.59% malicious-class precision**
* **78.12% malicious-class recall**

---

# 18. Limitations

Despite the encouraging results, this MVP has several important limitations.

## 18.1 Small Labeled Dataset

The final dataset contains only **809 wallets**, including 209 malicious wallets.

This is sufficient for an MVP but small relative to the scale and diversity of the Ethereum network.

The model therefore should not be considered production-ready.

---

## 18.2 Limited Phishing Test Data

Only **three phishing wallets** occurred in the held-out test set.

The observed 66.7% phishing recall corresponds to detecting two of three wallets and therefore cannot be treated as a reliable estimate of general phishing detection performance.

A substantially larger phishing dataset is required.

---

## 18.3 Malicious-Class Imbalance

The malicious categories are not evenly represented.

The final malicious dataset consists of:

* 91 sanctioned wallets
* 74 laundering wallets
* 44 phishing wallets

Consequently, overall binary classification metrics may hide substantial differences between individual malicious behaviors.

---

## 18.4 Transaction Collection Cap

A maximum of **500 transactions per wallet** was collected.

Highly active wallets may therefore have only a partial transaction history represented in the dataset.

Features such as transaction counts, total transferred value, active duration, counterparty diversity, and temporal behavior may consequently represent only the collected portion of a wallet's complete history.

---

## 18.5 External and Internal Transactions Only

The MVP focuses on the collected external and internal Ethereum transactions.

More complete blockchain behavior could include:

* ERC-20 transfers
* ERC-721/NFT activity
* ERC-1155 activity
* Contract interactions
* Token approvals
* DeFi protocol interactions
* Bridge activity

These behaviors may contain additional risk signals not represented by the current feature set.

---

## 18.6 No Graph-Based Features

The MVP intentionally avoids graph-neighborhood features.

Features such as:

* Number of risky neighbors
* Risky-counterparty ratio
* One-hop risky ratio
* Two-hop risky ratio
* PageRank
* Graph centrality
* Community structure

could improve performance.

However, label-derived graph features also introduce significant risk of data leakage if not constructed carefully.

They should therefore be investigated separately with strict fold-aware feature generation.

---

## 18.7 Random Stratified Evaluation

The dataset was evaluated using stratified random splitting and stratified cross-validation.

This tests generalization to unseen wallets drawn from approximately the same collected dataset but does not necessarily measure generalization across:

* Different time periods
* Newly emerging attack patterns
* Different label sources
* Different blockchain environments

Future experiments should include temporal and source-separated evaluation.

---

## 18.8 Potential Dataset and Source Bias

Malicious wallets originate from known labeled sources.

Known malicious addresses may represent particularly visible or well-documented attacks and may not reflect the complete distribution of malicious Ethereum activity.

Likewise, the benign dataset may not represent every type of legitimate Ethereum user.

The model may therefore partially learn characteristics associated with how the datasets were collected.

---

## 18.9 Risk Probability Is Not a Calibrated Probability

The Random Forest output is used as a continuous risk score, but a value such as `0.80` should not automatically be interpreted as an 80% real-world probability that a wallet is malicious.

Probability calibration was not performed in this MVP.

Calibration methods such as isotonic regression or Platt scaling could be investigated later.

---

## 18.10 Feature Importance Is Not Causal

Random Forest impurity-based feature importance indicates which features were useful for splitting the training data.

It does not demonstrate that those features cause malicious behavior.

Correlated features may also share or distort importance.

Future analysis could use permutation importance and SHAP to better understand model behavior.

---

## 18.11 MVP Is Binary

The model currently predicts:

`Benign vs Malicious`

It does not attempt to classify the malicious activity as phishing, laundering, sanctioned activity, or another risk category.

A future system could investigate multiclass classification or separate specialized risk detectors.

---

# 19. Future Work

Future versions can extend the MVP while retaining the current model as a reproducible baseline.

Potential improvements include:

1. Expand the labeled wallet dataset.
2. Increase phishing representation.
3. Collect more complete transaction histories.
4. Add ERC-20 transfer behavior.
5. Add token approval and allowance features.
6. Add contract-interaction features.
7. Investigate graph-based wallet features.
8. Evaluate graph models such as GraphSAGE or GCN only after establishing a sufficiently large graph dataset.
9. Add temporal train/test evaluation.
10. Evaluate across independent label sources.
11. Use permutation importance and SHAP for model interpretation.
12. Investigate probability calibration.
13. Compare additional classical ML models such as XGBoost or LightGBM.
14. Investigate multiclass risk categorization.
15. Build an inference pipeline that accepts an Ethereum address and produces a continuous wallet risk score.

---

# 20. MVP Conclusion

The objective of this MVP was not to build a production-ready blockchain intelligence platform, but to establish a small, controlled, reproducible foundation for Ethereum wallet risk scoring.

Starting with 12 basic transaction features, the project identified weaknesses in detecting certain malicious behaviors and introduced additional behavioral and temporal features.

The resulting 26-feature Random Forest consistently outperformed the original feature representation across five cross-validation folds.

The final model achieved **0.9865 ROC-AUC, 0.9669 PR-AUC, and 0.8475 F1** on a previously untouched 122-wallet test set.

These results demonstrate that transaction-level behavioral and temporal characteristics contain useful signals for distinguishing known malicious Ethereum wallets from benign wallets.

At the same time, limitations related to dataset size, class representation, transaction-history completeness, dataset bias, and evaluation methodology mean that these results should be interpreted as a **proof-of-concept MVP**, not as evidence of production-level wallet-risk detection.

The current MVP therefore provides a stable baseline from which larger datasets, richer blockchain features, graph-based methods, and more rigorous generalization experiments can be developed.
