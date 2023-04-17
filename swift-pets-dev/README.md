# Private Preserving Federated Learning for Financial Transaction Anomaly Detection

## Link to the Competition
- https://www.drivendata.org/competitions/group/nist-federated-learning/
- https://petsprizechallenges.com/
- https://www.drivendata.org/competitions/105/nist-federated-learning-2-financial-crime-federated/
- https://www.drivendata.org/competitions/144/nist-federated-learning-2-financial-crime-centralized/

## Installation Instruction

- Install python 3.8
- Installing packages by building conda environment from `environment.yml` file
  ```shell
  conda env create -f environment.yml
  conda activate ftad_pets
  ```
- Execute program
  ```shell
    # centralized_solution solution
    cd centralized_solution
    python solution_centralized.py
    # federated solution 
    cd federated_solution
    python solution_federated.py
  ```

### Packages Used

- mmh3 - to compute hash value
- pycryptodome - to encrypt and decrypt data
- xgboost - train model
- flower - federated learning framework

## US-PETS Financial Crime Prize: Problem Description

The United Nations estimates that up to $2 trillion of cross-border money laundering takes place each year, financing
organized crime and undermining economic prosperity. Financial institutions such as banks and credit agencies, along
with organizations that process transactions between institutions must protect personal and financial data, while also trying to report and deter illicit financial activities.
Under this context, we will design and later develop innovative privacy-preserving federated
learning solutions that facilitate cross-institution and cross-border anomaly detection to combat financial crime. This
use case features both vertical and horizontal data partitioning.
![alt text](./image/problem.png)

Details see the following link:
- https://www.drivendata.org/competitions/98/nist-federated-learning-1/page/524/
- https://www.drivendata.org/competitions/105/nist-federated-learning-2-financial-crime-federated/
- https://www.drivendata.org/competitions/144/nist-federated-learning-2-financial-crime-centralized/

## Dataset Description

Details see the following link:
- https://www.drivendata.org/competitions/98/nist-federated-learning-1/page/524/

### General Format of Account Dataset owned by Bank Clients

The general format of dataset for bank clients is as follows:

- **Bank_ID**: ID of bank
- **Account_ID**: ID of account
- **Account_info**: information of account
- **Flag**: categorical or ordinal value indicating whether risk of account, for example, 0 means no risk, 1 means low
  risk, 2 means high risk

Example of a synthetic fake dataset used in demo is as follows:

 | Bank_ID | Account_ID | Account_info    | Flag |
|---------|------------|-----------------|------|
| AAA     | AAA293939  | Bob, 11 st CA   | 0    |
| AAA     | AAA293948  | Alice, 12 st CA | 1    |

### Format of Transaction Dataset owned by Payment System Orgnization Client

The general format of dataset for payment system organization client is as follows:

- MessageID: ID of message
- UETR: ID of transaction
- Sender: sender bank ID
- Receiver: receiver bank ID
- Ordering_Account_ID: ID of ordering account
- Beneficiary_Account_ID: ID of beneficiary account
- Amount: amount of transaction
- Ordering_Currency: currency of transaction
- Beneficiary_Currency: currency of transaction
- Transaction_Date: date of transaction
- Finalization_Date: date of transaction finalization

Example of a synthetic fake dataset used in demo is as follows:
![alt text](./image/transaction_data.png)

## Solution Design - Two Stage Private Federated Anomaly Detection

### Overview

### Centralized Solution Design

### Federated Learning Solution Design

#### Training Phase

Following shows in training phase in every round, what server and clients do

- Code files related:
    - BankClient.py
    - TrainStrategy.py
    - TrainSwiftClient.py

- Round 1
    - client selected - all banks and swift
    - task: collect public key
        - swift and banks send their public key to server and server store it.
- Round 2
    - client selected - all banks client
    - task: collect encrypted bank names from bank client
        - every bank client sent banks it has in its partition encrypted using swift public key to server
- Round 3
    - client selected - swift
    - task: send hashed accounts info belong to each bank partition to server
        - server send encrypted bank names of each bank client to swift and swift decrypt them
        - swift compute hash of accounts and send them based on banks of each bank client
- Round 4 - 4 + num_banks
    - client selected - bank round - 4
    - task: securely compute common key and total accounts of all bank clients
        - every bank first decrypt the result from server to get current sum of accounts and xor of keys
        - every client encrypts its number of accounts + current sum and its key using next bank public key and send it
          to server, server send it to next bank
- Round 4 + num_banks + 1
    - client selected - bank 0
    - task: compute final total accounts and common key
        - bank0 decrypted result from server
        - bank0 compute final total accounts and common key from results
        - bank0 encrypted final total accounts and common key using public key of each other banks and send to server
- Round 4 + num_banks + 2
    - client selected - all banks
    - task: build local bloom filter and encrypted hash of accounts
        - server sends encrypted final sum and common key to every bank
        - server sends hash of accounts info belongs to each bank client to each bank client
        - every bank client decrypt to get final total accounts and use it to build bloom filter
        - every bank client decrypt to get final common computed key and use it to encrypt the hash value of accounts
          info received from server
        - every bank encrypt local bloomfilter with swift public key and send them with encrpted hash of accounts info
          to server
- Final Round
    - client selected - swift
    - task: build bloom filter feature and train model
        - swift decrypt and aggregate all local bloom filter of every bank sent by server to get global bloomfilter
        - swift use global bloomfilter and encrypted hash of accounts info to build bloomfilter feature - whether
          accounts info is valid or not based on Flags of bank data
        - swift compute some extra features
        - swift use bloomfilter feature and extra features to train dp version of xgboot model
        - swift save model

#### Test Phase

The only difference of test phase compare to training phase is in final round, swift load saved model and use it to do
prediction

- Code file related:
    - BankClient.py
    - TestStrategy.py
    - TestSwiftClient.py

## Code Structure

### Usage of Other Files

- bloom_filter.py - contains our implemented of bloom filter, it will be used by bank clients to build local bloom
  filter based on Flags information
- model3.py - contains swift training and test model and function to extract features and compute bloomfilter features
  from bloomfilter
- utils.py and utils_new.py - contains utility functions such as encryption, decryption, aggregation of bloom filter,
  compute accounts hash value etc.
