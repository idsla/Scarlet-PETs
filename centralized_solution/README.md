# Centralized Solution

 ## Overview

The centralized solution is based on [XGBoost Model]([XGBoost Documentation — xgboost 1.7.5 documentation](https://xgboost.readthedocs.io/en/stable/)). We join the account-level data with transaction-level data, and then compute features that will be used to build the gradient boosting tree model. To preserve privacy, we rely on Differential Privacy to add noise to sensitive features so that privacy can be preserved. The detailed solution description can be found [here](https://rutgers.box.com/s/q84zjo3edv5d1e1eu67ypihiw8cb2djq).

## Hoe to run the program:

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
- **Sender Receiver Frequency:** how many transactions between each pair of sender bank and receiver bank
- **Receiver In Degree:** for each bank, how many unique banks have sent funds to it
- **Date Diff:** date difference between `Timestamp`, `SettlementDate` 



## Our Solution Overview

Code files related:

  - `model2.py`

Following shows in the training phase, what PNS and banks do

1. 1st STEP:
   - PNS joins Banks account level data and computes flag-based feature (which is `1` if the beneficiary/ordering is invalid or flagged)
     `model2.py add_BF_feature()`<br>
   - Compute features: `generate_feature(), extract_feature()`
   - PNS adds DP noise to sensitive features.  `model2.py laplace_mech(), ls_at_distance(), smooth_sens()
2. 2nd STEP
   - PNS trains XGBoost model  `model2.py PNSModel()`

## Privacy and Security Parameters

JSON file `parameters.json` contains several **security and privacy parameters** that are essential for ensuring the strength and robustness of the privacy and security of our solution. These parameters have been carefully selected and tuned to provide strong encryption, secure communication, and reliable privacy protection for sensitive data. Below is a list of each parameter, along with an explanation of its meaning:

- **`bf_error_rate`**: Represents the acceptable false positive rate for the Bloom filter. The false positive rate affects the accuracy of the filter and must be carefully balanced against its computational cost.
- **`DP_epsilon`**: Specifies the level of privacy protection provided by the differential privacy mechanism used to protect sensitive features.



## Entry Point

`solution_centralized.py` - main program to run
