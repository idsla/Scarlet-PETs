# Centralized Solution
 Code files related:
  - `model2.py`

Following shows in training phase, what swift and banks do (__Algorithm 1__ in the report)

1. 1st STEP:
   - __Alg1.line2__ Swift joins Banks account data and identify flag-based feature. 
    `model2.py add_BF_feature()`<br>
   - __Alg1.line3__ Swift adds DP noise to sensitive features.  `model2.py laplace_mech(), ls_at_distance(), smooth_sens(),
   generate_feature(), extract_feature()`
2. 2nd STEP
   - __Alg1.line4__ Swift trains XGBoost model  `model2.py SwiftModel()`

## Entry Point
`solution_centralized.py` - contains api which needs to be submitted

# Federated Solution
## Train Strategy and Bank/TrainSwift Client

Following shows the mapping between __Algorithm 2__ and the sourcecode

---

 Code files related:
  - `BankClient.py`
  - `TrainStrategy.py`
  - `TrainSwiftClient.py`
  


1. 1st STEP 
   -  __Alg2.line2__ All banks securely generate the common random key, _k_, for PRF.
      - _Round 4_ to _Round 4+num_banks +1_
       `BankClient.py " task == 'secure_sum', 'compute_final_sum', ' "`
          and `TrainStrategy.py " 'server_round < 4 + num_banks', 'server_round == 4 + num_banks' "`
   - __Alg2.line4__ All banks securely compute the total number of accounts to be added in  the bloom filters
     - _Round 4_ to _Round 4 + num_banks + 1_
          `BankClient.py " task == 'secure_sum', 'compute_final_sum', ' "`
          and `TrainStrategy.py " 'server_round < 4 + num_banks', 'server_round == 4 + num_banks' "`
   - __Alg2.line5-7__ Each bank builds local secure bloom filter
     - _Round 1_ Swift and bank clients send their public key to Strategy 
     `TrainSwiftClient.py BankClient.py " task == 'send_public_key'"` and `TrainStrategy.py " server_round == 1 "`
     - _Round 2_ Bank clients tell Strategy which banks are present using Swift public key
     `BankClient.py " task == 'share_bank_in_partition'"` and `TrainStrategy.py " server_round == 2 "`
     - _Round 3_ Strategy sends encrypted bank names of each bank client to Swift and Swift decrypts them 
       swift compute hash of accounts `TrainSwiftClient.py " task == 'send_swift_banks_in_partitions'"` and `TrainStrategy.py " server_round == 3 "`
     - _Round 4 + num_banks + 2_ Bank clients encrypts local bloom filter swift public key and encrypts the hash of accounts with _k_. 
     `TrainSwiftClient.py " task == 'send_swift_banks_in_partitions'"` and `TrainStrategy.py " server_round == Round 4 + num_banks + 1 "`
2. 2nd STEP 
   - __Alg2.line10__ Swift securely aggregates all the local Bloom Filters
        - _Final Round_ Swift decrypts and aggregates all local bloom filter to get global bloomfilter
     `TrainSwiftClient.py " task == 'swift_run_train'"` and `TrainStrategy.py " server_round == 4 + num_banks + 2 "`
   - __Alg2.line11-13__ _Final Round_ Swifts use global bloomfilter and encrypted hash of accounts info to build bloomfilter feature.
   Swift add DP noise to sensitive features and train XGBoost with the augmented data
   `TrainSwiftClient.py " task == 'swift_run_train'"` and `TrainStrategy.py " server_round == 4 + num_banks + 2 "`

---
Following shows in training phase in every round, what server and clients do (__Algorithm 2__)

-   Code files related:

    -   BankClient.py
    -   TrainStrategy.py
    -   TrainSwiftClient.py

-   Round 1
    -   client selected - all banks and swift
    -   task: collect public key
        -   swift and banks send their public key to server and server store it.
-   Round 2
    -   client selected - all banks client
    -   task: collect encrypted bank names from bank client
        -   every bank client sent banks it has in its partition encrypted using swift public key to server
-   Round 3
    -   client selected - swift
    -   task: send hashed accounts info belong to each bank partition to server
        -   server send encrypted bank names of each bank client to swift and swift decrypt them
        -   swift compute hash of accounts and send them based on banks of each bank client
-   Round 4 - 4 + num_banks - **secure sum and computation**
    -   client selected - bank round - 4
    -   task: securely compute common key and total accounts of all bank clients
        -   every bank first decrypt the result from server to get current sum of accounts and xor of keys
        -   every client encrypts its number of accounts + current sum and its key using next bank public key and send it to server, server send it to next bank
-   Round 4 + num_banks + 1
    -   client selected - bank 0
    -   task: compute final total accounts and common key
        -   bank0 decrypted result from server
        -   bank0 compute final total accounts and common key from results
        -   bank0 encrypted final total accounts and common key using public key of each other banks and send to server
-   Round 4 + num_banks + 2
    -   client selected - all banks
    -   task: build local bloom filter and encrypted hash of accounts
        -   server sends encrypted final sum and common key to every bank
        -   server sends hash of accounts info belongs to each bank client to each bank client
        -   every bank client decrypt to get final total accounts and use it to build bloom filter
        -   every bank client decrypt to get final common computed key and use it to encrypt the hash value of accounts info received from server
        -   every bank encrypt local bloomfilter with swift public key and send them with encrpted hash of accounts info to server
-   Final Round
    -   client selected - swift
    -   task: **aggregate secure bloom filter** and extract features and train model
        -   swift decrypt and aggregate all local bloom filter of every bank sent by server to get global bloomfilter
        -   swift use global bloomfilter and encrypted hash of accounts info to build bloomfilter feature - whether accounts info is valid or not based on Flags of bank data
        -   swift compute some extra features
        -   swift use bloomfilter feature and extra features to train dp version of xgboot model
        -   swift save model
---
## Test Strategy and Bank/TestSwift Client

The only different of test phase compare to training phase is in final round, swift load saved model and use it to do prediction
- Code file related:
  - `BankClient.py`
  - `TestStrategy.py`
  - `TestSwiftClient.py`

## Usage of Other Files
- `bloom_filter.py` -  contains our implemented of bloom filter, it will be used by bank clients to build local bloom filter based on Flags information
- `model3.py` -  contains swift training and test model and function to extract features and compute bloomfilter features from bloomfilter
- `utils.py` and `utils_new.py` - contains utility functions such as encryption, decryption, aggregation of bloom filter, compute accounts hash value etc.

## Entry Point
 `solution_federated.py` - contains api which needs to be submitted

## Packages Used
- mmh3 -  to compute hash value
- pycryptodome - to encrypt and decrypt data
- xgboost - train model
