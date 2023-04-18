# Private Preserving Federated Learning for Financial Transaction Anomaly Detection

## Link to the Competition
- https://www.drivendata.org/competitions/group/nist-federated-learning/
- https://petsprizechallenges.com/
- https://www.drivendata.org/competitions/105/nist-federated-learning-2-financial-crime-federated/
- https://www.drivendata.org/competitions/144/nist-federated-learning-2-financial-crime-centralized/

## Installation Instruction

- Install python > 3.8

- Set up and activate virtualenv

  ```shell
  python -m venv ./venv
  
  # window gitbash
  source ./venv/Scripts/activate
  
  # linux/unix
  source ./venv/bin/activate
  ```

- Install Flower framework

  ```shell
  pip install -U flwr["simulation"]
  ```

- Install all other packages

  ```shell
  pip install -r requirements.txt
  ```

- Packages Used

  - mmh3 - to compute hash value
  - pycryptodome - to encrypt and decrypt data
  - xgboost - train model
  - flower - federated learning framework
  - others: joblib, loguru, hyperopt, scikit-learn

## Run program

- **Centralized Solution**

  ```shell
  # centralized_solution solution
  cd centralized_solution
  python solution_centralized.py
  ```

- **Federated Solution**

  ```shell
  # federated solution 
  cd federated_solution
  python solution_federated.py
  ```

## Problem Description: US-PETS Financial Crime Prize

The United Nations estimates that up to $2 trillion of cross-border money laundering takes place each year, financing
organized crime and undermining economic prosperity. Financial institutions such as banks and credit agencies, along
with organizations that process transactions between institutions must protect personal and financial data, while also trying to report and deter illicit financial activities. Under this context, we will design and later develop innovative privacy-preserving federated
learning solutions that facilitate cross-institution and cross-border anomaly detection to combat financial crime. This
use case features both vertical and horizontal data partitioning.
![alt text](./image/problem.png)

Details see the following link:
- https://www.drivendata.org/competitions/98/nist-federated-learning-1/page/524/
- https://www.drivendata.org/competitions/105/nist-federated-learning-2-financial-crime-federated/
- https://www.drivendata.org/competitions/144/nist-federated-learning-2-financial-crime-centralized/

## Dataset Requirement and Description

Details see the following link:
- https://www.drivendata.org/competitions/98/nist-federated-learning-1/page/524/

### Demo Datasets

Because the competition organization will not release the data used in competition to public. We provide a simple self generated fake synthetic demo dataset under **`new_data/`** folder of `/federated_solutution` and `/centralized_solution` so that one can run our program. In federated solution, **`new_data/`** folder contains subfolders whose names are denoted by name of clients, each of them contains the the data owned by client, for example, a bank client will has its own accounts dataset and payment system organization client will have it own transaction data.

#### Account Dataset owned by Bank Clients

The general format of dataset for bank clients is as follows:

- **Account information:** `Bank`, `Account`, `Name`, `Street`, `CountryCityZip`
- **Flags**: categorical or ordinal value indicating whether risk of account, for example, 0 means no risk, 1 means low risk

Example of a synthetic fake dataset used in demo is as follows:

| Bank     | Account        | Name        | Street      | CountryCityZip       | Flags |
| -------- | -------------- | ----------- | ----------- | -------------------- | ----- |
| BANK1682 | ACCOUNT74089   | acctname25  | address 200 | country_city_zip 175 | 0     |
| BANK1682 | ACCOUNT91126   | acctname186 | address 107 | country_city_zip 22  | 0     |
| BANK1682 | ACCOUNT2352638 | acctname131 | address 157 | country_city_zip 155 | 0     |

#### Transaction Dataset owned by Payment System Organization

The general format of dataset for payment system organization client is as follows:

- **Transaction IDs:** `TransactionId`

- **Transaction Time:** `Timestamp`, `SettlementDate`
- **Transaction Accounts:** `Sender`, `Receiver`, `OrderingAccount`, `BeneficaryAccount`, ...
- **Transaction Currency and Amount:** `SettlementAmount`, `InstructedAmount`, `SettlementCurrency`, `InstructedCurrency`
- **Label:** whether transaction is anomalous or not

Example of a synthetic fake dataset used in demo is as follows:

| TransactionId | Timestamp      | Sender   | Receiver | OrderingAccount | OrderingName | OrderingStreet | OrderingCountryCityZip | BeneficiaryAccount | BeneficiaryName | BeneficiaryStreet | BeneficiaryCountryCityZip | SettlementDate | SettlementCurrency | SettlementAmount | InstructedCurrency | InstructedAmount | Label |
| ------------- | -------------- | -------- | -------- | --------------- | ------------ | -------------- | ---------------------- | ------------------ | --------------- | ----------------- | ------------------------- | -------------- | ------------------ | ---------------- | ------------------ | ---------------- | ----- |
| M1            | 2/6/2055 17:44 | BANK6242 | BANK1682 | ACCOUNT421876   | acctname60   | address 111    | country_city_zip 90    | ACCOUNT551369      | acctname154     | address 197       | country_city_zip 132      | 550206         | currency 41        | 1                | currency 4         | 1                | 0     |
| M2            | 2/9/2055 23:21 | BANK2501 | BANK1682 | ACCOUNT3176524  | acctname192  | address 38     | country_city_zip 160   | ACCOUNT1           | acctname86      | address 76        | country_city_zip 99       | 550209         | currency 4         | 1                | currency 4         | 1                | 1     |
| M3            | 2/3/2055 4:10  | BANK65   | BANK2602 | ACCOUNT6331626  | acctname152  | address 127    | country_city_zip 49    | ACCOUNT857376      | acctname37      | address 34        | country_city_zip 101      | 290218         | currency 141       | 1                | currency 6         | 1                | 1     |

#### Training and Testing Data of Payment System Organization

Payment system owns the transaction data, and the anormal transaction detection will be operated at its side, so it also contains the **label** which denotes whether a transaction is abnormal or not. The after containing the labels, the transaction data will be split into training and test data for training and evaluate a machine learning model for detecting abnormal transaction. `Transaction_train.csv` and `Transaction_test.csv` are training and test data.

## Solution

### Centralized Solution

[Centralized Solution Documentation](./centralized_solution/README.md)

### Federated Learning Solution

[Federated Solution Documentation](./federated_solution/README.md)
