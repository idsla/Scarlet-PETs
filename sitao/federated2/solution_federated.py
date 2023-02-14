import pathlib
import typing
import flwr
from loguru import logger
import pandas as pd
from pathlib import Path
from typing import Dict
import json
from Crypto.PublicKey import RSA
import pickle
import random
from Crypto.Random import get_random_bytes

from .TrainSwiftClient import TrainSwiftClient
from .TestSwiftClient import TestSwiftClient
from .BankClient import BankClient
from .TrainStrategy import TrainStrategy
from .TestStrategy import TestStrategy
from .utils import convert_bank_valid_accounts_list
from .utils_new import get_unique_banks


def train_client_factory(
        cid: str,
        data_path: pathlib.Path,
        client_dir: pathlib.Path,
) -> typing.Union[flwr.client.Client, flwr.client.NumPyClient]:
    """
    Factory function that instantiates and returns a Flower Client for training.
    The federated learning simulation engine will use this function to
    instantiate clients with all necessary dependencies.

    Args:
        cid (str): Identifier for a client node/federation unit. Will be
            constant over the simulation and between train and test stages. The
            SWIFT node will always be named 'swift'.
        data_path (Path): Path to CSV data file specific to this client.
        client_dir (Path): Path to a directory specific to this client that is
            available over the simulation. Clients can use this directory for
            saving and reloading client state.

    Returns:
        (Union[Client, NumPyClient]): Instance of a Flower Client.
    """
    error_rate = 0.001
    session_key_length = 32
    prime = 20358416231591
    public_key_size = 2048

    if cid == "swift":
        #logger.info("Initializing SWIFT client for {}", cid)
        swift_df = pd.read_csv(data_path, index_col="MessageId")
        return TrainSwiftClient(
            cid, client_dir=client_dir, data=swift_df, session_key_length=session_key_length,
            public_key_size=public_key_size
        )
    else:
        #logger.info("Initializing bank client for {}", cid)
        data = pd.read_csv(data_path)
        #logger.info(data['Flags'].dtype)
        accounts_list = convert_bank_valid_accounts_list(data)
        total_accounts = data.shape[0]
        banks = get_unique_banks(data)
        del data
        return BankClient(
            cid, client_dir=client_dir, valid_accounts=accounts_list, total_accounts=total_accounts, unique_banks=banks,
            session_key_length=session_key_length, error_rate=error_rate, prime=prime,public_key_size=public_key_size
        )


def train_strategy_factory(
        server_dir: pathlib.Path,
) -> typing.Tuple[flwr.server.strategy.Strategy, int]:
    """
    Factory function that instantiates and returns a Flower Strategy, plus the
    number of federated learning rounds to run.

    Args:
        server_dir (Path): Path to a directory specific to the server/aggregator
            that is available over the simulation. The server can use this
            directory for saving and reloading server state. Using this
            directory is required for the trained model to be persisted between
            training and test stages.

    Returns:
        (Strategy): Instance of a Flower Strategy.
        (int): Number of federated learning rounds to execute.
    """
    training_strategy = TrainStrategy(server_dir=server_dir)
    with open(server_dir / 'train_server_config.json', 'r') as f:
        server_config = json.load(f)
    num_rounds = server_config['num_rounds']
    #logger.info('num_rounds: {}'.format(num_rounds))

    return training_strategy, num_rounds


def test_client_factory(
        cid: str,
        data_path: pathlib.Path,
        client_dir: pathlib.Path,
        preds_format_path: typing.Optional[pathlib.Path],
        preds_dest_path: typing.Optional[pathlib.Path],
) -> typing.Union[flwr.client.Client, flwr.client.NumPyClient]:
    """
    Factory function that instantiates and returns a Flower Client for test-time
    inference. The federated learning simulation engine will use this function
    to instantiate clients with all necessary dependencies.

    Args:
        cid (str): Identifier for a client node/federation unit. Will be
            constant over the simulation and between train and test stages. The
            SWIFT node will always be named 'swift'.
        data_path (Path): Path to CSV test data file specific to this client.
        client_dir (Path): Path to a directory specific to this client that is
            available over the simulation. Clients can use this directory for
            saving and reloading client state.
        preds_format_path (Optional[Path]): Path to CSV file matching the format
            you must write your predictions with, filled with dummy values. This
            will only be non-None for the 'swift' client—bank clients should not
            write any predictions and receive None for this argument.
        preds_dest_path (Optional[Path]): Destination path that you must write
            your test predictions to as a CSV file. This will only be non-None
            for the 'swift' client—bank clients should not write any predictions
            and will receive None for this argument.

    Returns:
        (Union[Client, NumPyClient]): Instance of a Flower Client.
    """
    error_rate = 0.001
    session_key_length = 32
    prime = 20358416231591
    public_key_size = 2048
    if cid == "swift":
        #logger.info("Initializing SWIFT client for {}", cid)
        swift_df = pd.read_csv(data_path, index_col="MessageId")
        return TestSwiftClient(
            cid, client_dir=client_dir, data=swift_df, preds_dest_path=preds_dest_path,
            preds_format_path=preds_format_path, evaluation=False, error_rate=error_rate,
            session_key_length=session_key_length, public_key_size=public_key_size
        )
    else:
        #logger.info("Initializing bank client for {}", cid)
        data = pd.read_csv(data_path)
        accounts_list = convert_bank_valid_accounts_list(data)
        total_accounts = data.shape[0]
        banks = get_unique_banks(data)
        del data
        return BankClient(
            cid, client_dir=client_dir, valid_accounts=accounts_list, total_accounts=total_accounts,
            unique_banks=banks, session_key_length=session_key_length,
            error_rate=error_rate, prime=prime, public_key_size=public_key_size
        )


def test_strategy_factory(
        server_dir: pathlib.Path,
) -> typing.Tuple[flwr.server.strategy.Strategy, int]:
    """
    Factory function that instantiates and returns a Flower Strategy, plus the
    number of federated learning rounds to run.

    Args:
        server_dir (Path): Path to a directory specific to the server/aggregator
            that is available over the simulation. The server can use this
            directory for saving and reloading server state. Using this
            directory is required for the trained model to be persisted between
            training and test stages.

    Returns:
        (Strategy): Instance of a Flower Strategy.
        (int): Number of federated learning rounds to execute.
    """
    test_strategy = TestStrategy(server_dir=server_dir)
    with open(server_dir / 'test_server_config.json', 'r') as f:
        server_config = json.load(f)
    num_rounds = server_config['num_rounds']
    #logger.info('num_rounds: {}'.format(num_rounds))
    return test_strategy, num_rounds


def train_setup(server_dir: Path, client_dirs_dict: Dict[str, Path]):
    """
    Perform initial setup between parties before federated training.

    Args:
        server_dir (Path): Path to a directory specific to the server/aggregator
            that is available over the simulation. The server can use this
            directory for saving and reloading server state. Using this
            directory is required for the trained model to be persisted between
            training and test stages.
        client_dirs_dict (Dict[str, Path]): Dictionary of paths to the directories
            specific to each client that is available over the simulation. Clients
            can use these directory for saving and reloading client state. This
            dictionary is keyed by the client ID.
    """
    num_rounds = len(client_dirs_dict.keys()) + 7
    public_key_size = 2048
    with open(server_dir / 'train_server_config.json', 'w') as f:
        json.dump({'num_rounds': num_rounds}, f)

    for client, client_dir in client_dirs_dict.items():

        private_key_file = Path.joinpath(client_dir, "private.pem")  # rename
        public_key_file = Path.joinpath(client_dir, "public.pem")

        if private_key_file.exists():
            private_key_file.unlink()

        if public_key_file.exists():
            public_key_file.unlink()

        # generate public and private key
        key = RSA.generate(public_key_size)
        private_key = key.export_key()
        file_out = Path.joinpath(client_dir, "private.pem").open(mode='wb')
        file_out.write(private_key)
        file_out.close()

        public_key = key.publickey().export_key()
        file_out = Path.joinpath(client_dir, "public.pem").open(mode="wb")
        file_out.write(public_key)
        file_out.close()

        file_path = Path.joinpath(client_dir, 'client_config.pkl')
        if file_path.exists():
            file_path.unlink()
        prime = 20358416231591
        random_key = random.randint(0, prime - 1)
        key2 = get_random_bytes(32)  # TODO:parameters
        client_config = {'random_key': random_key, 'key2': key2}
        with file_path.open(mode='wb') as f:
            pickle.dump(client_config, f)


def test_setup(server_dir: Path, client_dirs_dict: Dict[str, Path]):
    """
    Perform initial setup between parties before federated test inference.
    """
    num_rounds = len(client_dirs_dict.keys()) + 7
    with open(server_dir / 'test_server_config.json', 'w') as f:
        json.dump({'num_rounds': num_rounds}, f)

    for client, client_dir in client_dirs_dict.items():
        # generate public and private key
        private_key_file = Path.joinpath(client_dir, "private.pem")  # rename
        public_key_file = Path.joinpath(client_dir, "public.pem")

        if private_key_file.exists():
            private_key_file.unlink()

        if public_key_file.exists():
            public_key_file.unlink()

        key = RSA.generate(2048)
        private_key = key.export_key()
        file_out = Path.joinpath(client_dir, "private.pem").open(mode='wb')
        file_out.write(private_key)
        file_out.close()

        public_key = key.publickey().export_key()
        file_out = Path.joinpath(client_dir, "public.pem").open(mode="wb")
        file_out.write(public_key)
        file_out.close()

        file_path = Path.joinpath(client_dir, 'client_config.pkl')
        if file_path.exists():
            file_path.unlink()
        prime = 20358416231591
        random_key = random.randint(0, prime - 1)
        key2 = get_random_bytes(32)  # TODO:parameters
        client_config = {'random_key': random_key, 'key2': key2}
        with file_path.open(mode='wb') as f:
            pickle.dump(client_config, f)
