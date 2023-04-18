# Centralized Solution

 ## Overview

The centralized solution is based on [XGBoost Model]([XGBoost Documentation — xgboost 1.7.5 documentation](https://xgboost.readthedocs.io/en/stable/)). We join the accounts information into transaction data and then compute the serveral important features used to build the gradient boosting tree model. To preserve the privacy, we rely on Differential Privacy to add noise to sensitive features so that the privacy can be preserved.

## Run Program

```shell
Python solution_centralized.py
```

## Features Computed

- **Sender Bank Hourly Frequency**: how many transactions sender bank make for each hour index from 0 - 23
- **Receiver Bank Hourly Frequency**: how many transactions recevier bank make for each hour index from 0 - 23
- **Sender Currency Frequency:** how many currencies of each sender bank among all its transactions across the whole time range of training data phase
- **Receiver Currency Frequency:** how many currencies of each receiver bank among all its transactions across the whole time range of training data phase
- **Sender Currency Average Amount:** Average amount of each currency of each sender among all its transactions across the whole time range of training data phase
- **Receiver Currency Average Amount:** Average amount of each currency of each receiver among all its transactions across the whole time range of training data phase
- **Sender Receiver Frequency:** how many transaction between each pair of sender bank and receiver bank
- **Receiver In Degree:** how many unique banks receiver receive money from
- **Date Diff:** date difference between `Timestamp`, `SettlementDate` 



## Solution Inllustration

Code files related:

  - `model2.py`

Following shows in training phase, what PNS and banks do

1. 1st STEP:
   - PNS joins Banks account data and identify flag-based feature. 
     `model2.py add_BF_feature()`<br>
   - Compute features: `generate_feature(), extract_feature()`
   - PNS adds DP noise to sensitive features.  `model2.py laplace_mech(), ls_at_distance(), smooth_sens()
2. 2nd STEP
   - PNS trains XGBoost model  `model2.py PNSModel()`

## Entry Point

`solution_centralized.py` - main program to run