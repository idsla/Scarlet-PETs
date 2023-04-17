import json
import pickle
from pathlib import Path
import random
from typing import Dict

import flwr as fl
import pandas as pd
from Crypto.PublicKey import RSA
from Crypto.Random import get_random_bytes

from federated_solution.BankClient import BankClient
from federated_solution.TrainSwiftClient import TrainSwiftClient
from federated_solution.TestSwiftClient import TestSwiftClient
from federated_solution.utils_basic import convert_bank_valid_accounts_list
from federated_solution.utils_advanced import get_unique_banks


def client_fn_train(cid: str) -> fl.client.Client:
	"""Create a Flower client representing a single organization."""
	session_key_length = 16
	# Load model
	if cid == 'swift':
		data_path = './federated_solution/data/scenario01/train/swift/dev_swift_transaction_train_dataset.csv'
		client_dir = Path('./federated_solution/state/swift/')
		data = pd.read_csv(data_path, index_col="MessageId")
		return TrainSwiftClient(
			cid, client_dir, data, session_key_length, error_rate=0.1
		)
	else:
		# generate public and private key
		data_path = './federated_solution/data/scenario01/train/{}/bank_dataset.csv'.format(cid)
		client_dir = Path('./federated_solution/state/{}/'.format(cid))
		data = pd.read_csv(data_path)
		accounts_list = convert_bank_valid_accounts_list(data)
		banks = get_unique_banks(data)
		return BankClient(
			cid, accounts_list, data.shape[0], banks, client_dir, session_key_length, error_rate=0.1
		)


def client_fn_test(cid: str) -> fl.client.Client:
	"""Create a Flower client representing a single organization."""
	session_key_length = 16
	# Load model
	if cid == 'swift':
		data_path = './federated_solution/data/scenario01/test/swift/dev_swift_transaction_test_dataset.csv'
		client_dir = Path('./federated_solution/state/swift/')
		preds_dest_path = Path('./federated_solution/data/scenario01/test/swift/')
		preds_format_path = Path('./federated_solution/data/scenario01/test/swift/predictions_format.csv')
		data = pd.read_csv(data_path, index_col="MessageId")
		return TestSwiftClient(
			cid, data, client_dir, session_key_length=session_key_length, error_rate=0.1,
			preds_dest_path=preds_dest_path, preds_format_path=preds_format_path, evaluation=True
		)
	else:
		# generate public and private key
		data_path = './federated_solution/data/scenario01/test/{}/bank_dataset.csv'.format(cid)
		client_dir = Path('./federated_solution/state/{}/'.format(cid))
		data = pd.read_csv(data_path)
		accounts_list = convert_bank_valid_accounts_list(data)
		banks = get_unique_banks(data)
		return BankClient(
			cid, accounts_list, data.shape[0], banks, client_dir, session_key_length, error_rate=0.1
		)


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


if __name__ == '__main__':

	from federated_solution.TrainStrategy import TrainStrategy
	from federated_solution.TestStrategy import TestStrategy
	import time

	server_dir = Path('./federated_solution/state/server/')
	clients = ['swift', 'bank01', 'bank02', 'bank03', 'bank04']
	num_rounds = len(clients) + 7
	client_dirs_dict = {
		client: Path('./federated_solution/state/{}/'.format(client))
		for client in clients
	}

	train_setup(server_dir, client_dirs_dict)
	test_setup(server_dir, client_dirs_dict)

	start = time.time()
	train_strategy = TrainStrategy(server_dir=server_dir)
	# Start simulation
	fl.simulation.start_simulation(
		client_fn=client_fn_train,
		clients_ids=clients,
		config=fl.server.ServerConfig(num_rounds=num_rounds),
		strategy=train_strategy,
	)
	end = time.time()
	print("Total time running: {}".format(end - start))

	test_strategy = TestStrategy(server_dir=server_dir)
	fl.simulation.start_simulation(
		client_fn=client_fn_test,
		clients_ids=clients,
		config=fl.server.ServerConfig(num_rounds=num_rounds),
		strategy=test_strategy,
	)
	end = time.time()
	print("Total time running: {}".format(end - start))
