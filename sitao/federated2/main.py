from pathlib import Path
import flwr as fl
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    Status,
    Code
)
from flwr.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays
import pandas as pd
import numpy as np
from BankClient import BankClient
from TrainSwiftClient import TrainSwiftClient
from TestSwiftClient import TestSwiftClient
from utils import convert_bank_valid_accounts_list
from utils_new import get_unique_banks


def client_fn_train(cid: str) -> fl.client.Client:
    """Create a Flower client representing a single organization."""
    session_key_length = 16
    # Load model
    if cid == 'swift':
        data_path = './data/scenario01/train/swift/dev_swift_transaction_train_dataset.csv'
        client_dir = Path('./state/swift/')
        data = pd.read_csv(data_path, index_col="MessageId")
        return TrainSwiftClient(
            cid, client_dir, data, session_key_length, error_rate=0.001
        )
    else:
        # generate public and private key
        data_path = './data/scenario01/train/{}/bank_dataset.csv'.format(cid)
        client_dir = Path('./state/{}/'.format(cid))
        data = pd.read_csv(data_path)
        accounts_list = convert_bank_valid_accounts_list(data)
        banks = get_unique_banks(data)
        return BankClient(
            cid, accounts_list, data.shape[0], banks, client_dir, session_key_length, error_rate=0.001
        )


def client_fn_test(cid: str) -> fl.client.Client:
    """Create a Flower client representing a single organization."""
    session_key_length = 16
    # Load model
    if cid == 'swift':
        data_path = './data/scenario01/test/swift/dev_swift_transaction_test_dataset.csv'
        client_dir = Path('./state/swift/')
        preds_dest_path = Path('./data/scenario01/test/swift/')
        preds_format_path = Path('./data/scenario01/test/swift/predictions_format.csv')
        data = pd.read_csv(data_path, index_col="MessageId")
        return TestSwiftClient(
            cid, data, client_dir, session_key_length = session_key_length,error_rate=0.001,
            preds_dest_path=preds_dest_path, preds_format_path=preds_format_path, evaluation=True
        )
    else:
        # generate public and private key
        data_path = './data/scenario01/test/{}/bank_dataset.csv'.format(cid)
        client_dir = Path('./state/{}/'.format(cid))
        data = pd.read_csv(data_path)
        accounts_list = convert_bank_valid_accounts_list(data)
        banks = get_unique_banks(data)
        return BankClient(
            cid, accounts_list, data.shape[0], banks, client_dir, session_key_length, error_rate=0.001
        )


if __name__ == '__main__':
    from TrainStrategy import TrainStrategy
    from TestStrategy import TestStrategy
    import time

    start = time.time()
    train_strategy = TrainStrategy(server_dir=Path('.'))
    # Start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn_train,
        clients_ids=['swift', 'bank01', 'bank02', 'bank03', 'bank04'],
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=train_strategy,
    )

    test_strategy = TestStrategy(server_dir=Path('.'))
    fl.simulation.start_simulation(
        client_fn=client_fn_test,
        clients_ids=['swift', 'bank01', 'bank02', 'bank03', 'bank04'],
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=test_strategy,
    )
    end = time.time()
    print("Total time running: {}".format(end - start))
