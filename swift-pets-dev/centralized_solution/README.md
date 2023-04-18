# Centralized Solution

 ## Overview

The centralized solution is based on [XGBoost Model]([XGBoost Documentation — xgboost 1.7.5 documentation](https://xgboost.readthedocs.io/en/stable/)). We join the account level data with transaction level data, and then compute features that will be used to build the gradient boosting tree model. To preserve the privacy, we rely on Differential Privacy to add noise to sensitive features so that the privacy can be preserved.

## Run Program

```shell
Python solution_centralized.py
```

## Features Computed

- **Sender Bank Hourly Frequency**: how many transactions sender bank makes in each hour, indexed as 0 - 23
- **Receiver Bank Hourly Frequency**: how many transactions receiver bank makes in each hour, indexed as 0 - 23
- **Sender Currency Frequency:** how often a currency is used in transactions by a sender bank to send funds
- **Receiver Currency Frequency:** how often a currency is used in transactions for a receiver bank to receive funds
- **Sender Currency Average Amount:** Average amount (per currency) per transaction by a sender bank
- **Receiver Currency Average Amount:** Average amount (per currency) per transaction received by a receiver bank
- **Sender Receiver Frequency:** how many transaction between each pair of sender bank and receiver bank
- **Receiver In Degree:** for each bank, how many unique banks have sent funds to it
- **Date Diff:** date difference between `Timestamp`, `SettlementDate` 



## Our Solution Overview

Code files related:

  - `model2.py`

Following shows in training phase, what PNS and banks do

1. 1st STEP:
   - PNS joins Banks account level data and computes flag-based feature (which is `1` if the beneficiary/ordering is invalid or flagged)
     `model2.py add_BF_feature()`<br>
   - Compute features: `generate_feature(), extract_feature()`
   - PNS adds DP noise to sensitive features.  `model2.py laplace_mech(), ls_at_distance(), smooth_sens()
2. 2nd STEP
   - PNS trains XGBoost model  `model2.py PNSModel()`

## Entry Point

`solution_centralized.py` - main program to run
