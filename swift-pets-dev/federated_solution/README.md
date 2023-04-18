# Anomaly Detection via Privacy-Enhanced Two-Step Federated Learning for Financial Crime Prevention
## Overview

We propose a novel privacy-preserving federated learning approach to identify anomalous financial transactions in Payment Network Systems (PNS), e.g., SWIFT. Our approach utilizes a two-step anomaly detection methodology to solve the problem. In the first step, we mine features based on account-level data and labels, and then use a privacy-preserving encoding scheme to augment these features to the data held by PNS. In the second step, PNS now learns a highly accurate classifier from the augmented data.

## Solution Illustration



![pets](../image/pets.png)

## Run Program

```shell
python federated_solution.py
```

## Solution Framework

Our solution for federated learning (FL) is built upon the [Flower framework]([Flower 1.4.0](https://flower.dev/docs/)), which is well-known for its efficient and scalable approach for FL. We have utilized the `Client` and `Strategy` classes, inherited from Flower, and implemented the interface, ensuring seamless integration with the framework. For more information on the Flower framework and its capabilities, please refer to the provided links. The following link shows how clients and the server communicate in Flower framework - [Flower Architecture]([Implementing Strategies - Flower 1.4.0](https://flower.dev/docs/implementing-strategies.html)) 

**Folders**

- `./state/<client_id>/` used to persist all internal state (e.g. public key and private key) of all clients and server
- `./data/scenario_name/<client_id>/` contains all data owned by the client

**Functionality of Client and Strategy Classes**

- `BankClient.py` implements functionality of bank clients following interfaces provided by Flower framework
  - Share its public key to the server for secure communication
  - Compute secure sum with all other clients to count total number of accounts, needed to build the bloom filter
  - Collaboratively build privacy-preserving bloom filter to encode account-level info. (e.g., `flag` and name) and aggregate them at PNS client
- `TrainPNSClient.py` and `TestPNSClient.py` implement functionality of PNS clients following interfaces provided by Flower framework
  - Share its public key to the server for secure communication
  - Construct global bloom filter from local privacy-preserving bloom filters (of bank clients)
  -  Train a XGboost model to detect anomalous financial transactions
- `TrainStrategy.py` implements functionality of server for secure and privacy-preserving communication and computation with all the clients (training phase)
  - Collects public key from all clients
  - Transfer encrypted information to assist communication between all clients

- `TestStrategy.py` implements functionality of server for privacy-preserving communication and computation for all clients (testing pahse)
  - Collects public key from all clients
  - Transfer encrypted information to assist communication between all clients

**Functionality of other files**

- `model3.py` contains a [XGBoost]([XGBoost Documentation — xgboost 1.7.5 documentation](https://xgboost.readthedocs.io/en/stable/)) model related functionalities, which are used to identify anomalous financial transactions
- `bloomfilter.py` contains our implementation of bloom filter
- `utils_basic.py` and `utils_advanced.py` contain functions to compute hashes, encryption and decryption, and some data (pre-)processing functions

## Our Solution: Step-by-Step Overview

###  Setup Phase (Training and Testing)

The program will first setup all private and public keys, random number (seeds) owned by each client for generating keys for encryption and decrytion. Then the program will start the collaborative part of the computation by using the built-in function from Flower framework `fl.simulate()`. Which will use different processes for each client and the server to simulate the federated learning setting. 

###  Training Phase
 
In Flower, computation protocols for federated learning are implemented in rounds. The following gives an overview of the tasks parties are performing in each round.

- **Code files related:**
  - BankClient.py
  - TrainStrategy.py
  - TrainPNSClient.py

- **Round 1**
  - client selected - all banks and PNS client
  - task: collect public key
    - PNS client and banks send their public keys to the server who stores them and keep track of them.
- **Round 2**
  - client selected - all banks client
  - task: collect bank IDs from bank clients that they are responsible for
    - a bank (Flower) client may be responsible for more than one banks; thus, it sends to the server the banks IDs for all such banks (the communication is encrypted using PNS client public key)
- **Round 3**
  - client selected - PNS client
  - task: send hashed accounts info belong to each bank partition to server
    - server sends to PNS client the set of bank IDs for each bank client (received in the previous round)
    - PNS client computes hashes of the account info. (of beneficiary/ordering) from transactions, attach them to approperiate bank clients (using banks IDs ownership), and sends them to the server
- **Round 4 to 4 + num_banks**
  - client selected - bank round - 4 (`round - 4` gives the index of bank client)
  - task: securely compute a common random string, i.e., key, and the total accounts across all banks
    - secure sum is computed using additive secrete sharing
    - the key is generated by XORing random string independently picked by all the bank clients
- **Round 4 + num_banks + 1**
  - client selected - bank 0
  - task: compute final sum and common key
    - bank0 compute final sum (total accounts) and the common key from the secret shares
    - bank0 encrypted final sum and common key using public key of each bank client and sends them to the server
- **Round 4 + num_banks + 2**
  - client selected - all banks
  - task: build secure local bloom filter and encrypt hashed account info. 
    - server shares the final sum and common key with the bank clients
    - server sends to each bank client the hashed account info. (received in Round 3) of the banks it's responsible for
    - every bank client performs secure information encoding by building its local bloom filter; this is done by adding the encrypted hashes of account info. of the banks it's responsible for (encryption key is the common key)
    - every bank client encrypts the hashed account info. (sent by PNS)
    - every bank client sends its bloom filter and encrypted account info to the server
- **Final Round**
  - client selected - PNS client
  - task: build global bloom filter, feature augmentation, and model training
    - PNS client aggregates all local bloom filters into a global bloom filter
    - PNS client uses the global bloom filter and encrypted hashed accounts info. (associated with each transaction) to build the privacy-preserving account-level feature and augment it to its data
    - PNS client computes features from the augmented data
    - PNS client uses the data to train an xgboot model
    - PNS client saves the model

###  Testing Phase

The testing phase differs from the training phase in only the final round where: PNS client loads the trained model and uses it for prediction

- **Code file related:**
  - BankClient.py
  - TestStrategy.py
  - TestPNSClient.py
