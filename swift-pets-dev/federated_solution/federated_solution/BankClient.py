"""
This file contains the BankClient class which is used to simulate a bank client.
"""
import json
import random
from pathlib import Path
from typing import List

import flwr as fl
from loguru import logger
from flwr.common import (
	EvaluateIns,
	EvaluateRes,
	FitIns,
	FitRes,
	Status,
	Code,
)
from Crypto.PublicKey import RSA
from Crypto.Random import get_random_bytes
from flwr.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays
import numpy as np
import pickle
import time
import tracemalloc

from .utils_basic import (
	str_to_bytes, decrypt_data_and_session_key, encrypt_data_and_session_key, search_data_enc,
	build_bloomfilter, encryption_bf,
)
from .utils_advanced import (
	ndarray_to_byte_string,
	encrypt_bytes_with_public_key,
	decrypt_bytes_with_private_key,
	bytes_xor,
	enc_hashed_banks,
	compute_data_capacity,
)


class BankClient(fl.client.Client):
	"""
	BankClient class

	Parameters
	----------
	cid: str
		Client ID
	valid_accounts: List[str]
		List of valid accounts
	total_accounts: int
		Total number of accounts
	unique_banks: List[str]
		List of unique banks
	client_dir: Path
		Client directory
	session_key_length: int
		Session key length
	error_rate: float
		Error rate
	prime: int
		Prime number
	public_key_size: int
		Public key size

	Attributes
	----------
	cid: str
		Client ID
	client_dir: Path
		Client directory
	cid_bytes: bytes
		Client ID in bytes
	valid_accounts: List[str]
		List of valid accounts
	total_accounts: int
		Total number of accounts
	unique_banks: List[str]
		List of unique banks
	error_rate: float
		Error rate
	session_key_length: int
		Session key length
	prime: int
		Prime number
	session_key_dict: dict
		Dictionary of session keys
	public_key_dict: dict
		Dictionary of public keys
	public_key: RsaKey
		Public key
	private_key: RsaKey
		Private key
	random_key: bytes
		Random key
	public_key_size: int
		Public key size
	key2: bytes
		Key 2
	key3: bytes
		Key 3

	Methods
	-------
	get_parameters()
		Get parameters
	evaluate(parameters, config)
		Execute certain task in evaluation phase
	fit(parameters, config)
		Execute certain task in training phase
	load_states()
		Load states
	save_states()
		Save states
	save_stats()
		Save statistics

	"""
	def __init__(
			self,
			cid: str,
			valid_accounts: List[str],
			total_accounts: int,
			unique_banks: List[str],
			client_dir: Path,
			session_key_length: int = 16,
			error_rate: float = 0.001,
			prime: int = 20358416231591,
			public_key_size: int = 2048
	):
		super().__init__()
		self.cid = cid
		self.client_dir = client_dir
		self.cid_bytes = str_to_bytes(self.cid)
		self.valid_accounts = valid_accounts
		self.total_accounts = np.int64(total_accounts)
		self.unique_banks = unique_banks
		self.error_rate = error_rate
		self.session_key_length = session_key_length

		# load states
		self.prime = prime
		self.session_key_dict = None
		self.public_key_dict = None
		self.public_key = None
		self.private_key = None
		self.random_key = None
		self.public_key_size = public_key_size
		self.key2 = None
		self.stats = None
		self.load_state()


	def load_state(self):
		"""Load states"""
		client_dir = self.client_dir

		###########################################################################################
		# load own private and public key
		private_key_file = Path.joinpath(client_dir, "private.pem")  # rename
		public_key_file = Path.joinpath(client_dir, "public.pem")
		# if it has private and public key read and load them
		if private_key_file.exists() and public_key_file.exists():
			private_key = RSA.import_key(private_key_file.open().read())
			public_key = RSA.import_key(public_key_file.open().read())
		# if it does not have public and private key -> generate them
		else:
			key = RSA.generate(self.public_key_size)
			private_key = key.export_key()
			file_out = Path.joinpath(client_dir, "private.pem").open(mode='wb')
			file_out.write(private_key)
			file_out.close()

			public_key = key.publickey().export_key()
			file_out = Path.joinpath(client_dir, "public.pem").open(mode="wb")
			file_out.write(public_key)
			file_out.close()

			public_key = RSA.import_key(public_key)
			private_key = RSA.import_key(private_key)

		self.private_key = private_key
		self.public_key = public_key

		##################################################################################
		# load public key list
		file_path = Path.joinpath(client_dir, 'client_config.pkl')
		if file_path.exists():
			with file_path.open(mode='rb') as f:
				config = pickle.load(f)
				self.random_key = config['random_key']
				self.key2 = config['key2']
		else:
			self.random_key = random.randint(0, self.prime - 1)
			self.key2 = get_random_bytes(32)  # TODO:parameters
			client_config = {'random_key': self.random_key, 'key2': self.key2}
			with file_path.open(mode='wb') as f:
				pickle.dump(client_config, f)

		file_path = Path.joinpath(client_dir, '/stats.json')
		if file_path.exists():
			with file_path.open(mode='r') as f:
				self.stats = json.load(f)
		else:
			self.stats = {}

	def save_state(self, data, file_name):
		"""Save states"""
		file_path = Path.joinpath(self.client_dir, file_name)
		if not file_path.exists():
			with file_path.open('wb') as f:
				pickle.dump(data, f)

	def save_stats(self):
		"""Save stats"""
		file_path = Path.joinpath(self.client_dir, 'state/server/stats.json')
		with file_path.open('w') as f:
			json.dump(self.stats, f)

	# Flower internal function will call this to get local parameters e.g., print them
	def get_parameters(self, ins):
		return ndarrays_to_parameters([])

	def fit(self, ins: FitIns) -> FitRes:
		"""
		Execute certain task in training phase
		Parameters
		----------
		ins: FitIns -> Fit instruction

		Returns
		-------
		FitRes -> Fit result
		"""

		# get global parameters (key) and config
		global_parameters = ins.parameters
		config = ins.config
		task = config['task']

		tracemalloc.start()

		# Instruction1: send public key
		if task == 'send_public_key':
			data_sent = [np.array([self.public_key.export_key()])]
			tracemalloc.stop()
			return FitRes(
				status=Status(code=Code.OK, message='Success'),
				parameters=ndarrays_to_parameters(data_sent),
				num_examples=1,
				metrics={}
			)

		# Instruction2: share bank ids to server
		elif task == 'share_bank_in_partition':
			pub_key = config['key']
			recipient_key = RSA.import_key(pub_key)
			# logger.info("{} {}".format(self.cid, self.unique_banks))
			# encrypt bank ids
			data_enc = encrypt_bytes_with_public_key(
				ndarray_to_byte_string(self.unique_banks), recipient_key,
				self.session_key_length
			)
			# send encrypted bank ids to server
			data_sent = [np.array(item) for item in data_enc]
			tracemalloc.stop()

			return FitRes(
				status=Status(code=Code.OK, message='Success'),
				parameters=ndarrays_to_parameters(data_sent),
				num_examples=1,
				metrics={}
			)

		# Instruction3: collaborative compute secure sum
		elif task == 'secure_sum':
			data_received = parameters_to_ndarrays(global_parameters)
			# data_received = [value[0] for value in data_received]
			pub_key = config['key']
			recipient_key = RSA.import_key(pub_key)
			# empty -> initialization
			if len(data_received) == 0:
				# set data
				data = np.mod(self.total_accounts + self.random_key, self.prime)
				# encryption
				data_enc = encrypt_data_and_session_key(data, recipient_key, self.session_key_length)

				data2 = self.key2
				data_enc2 = encrypt_bytes_with_public_key(data2, recipient_key, self.session_key_length)

				end = time.time()
				data_sent = [np.array(value) for value in data_enc + data_enc2]
				tracemalloc.stop()

				return FitRes(
					status=Status(code=Code.OK, message='Success'),
					parameters=ndarrays_to_parameters(data_sent),
					num_examples=1,
					metrics={}
				)
			# decrypt data and add own total_accounts and then encrypt with public key
			else:
				data_received1 = data_received[0:3]
				data1 = decrypt_data_and_session_key(data_received1, self.private_key)

				data_received2 = data_received[3:]
				data2 = decrypt_bytes_with_private_key(data_received2, self.private_key)

				if data1 is not None and data2 is not None:
					data1 = np.mod(data1 + self.total_accounts, self.prime)
					data2 = bytes_xor(data2, self.key2)
					# encrypt update sum with session key created by other's publik key
					data_sent = encrypt_data_and_session_key(data1, recipient_key, self.session_key_length)
					data_sent2 = encrypt_bytes_with_public_key(data2, recipient_key, self.session_key_length)

					data_sent = [np.array(value) for value in data_sent + data_sent2]
					tracemalloc.stop()

					return FitRes(
						status=Status(code=Code.OK, message='Success'),
						parameters=ndarrays_to_parameters(data_sent),
						num_examples=1,
						metrics={}
					)
				else:
					raise ValueError('Decrypt error')
		# Instruction4: compute final sum of total accounts
		elif task == 'compute_final_sum':
			data_received = parameters_to_ndarrays(global_parameters)
			sum_data = data_received[0:3]
			key_data = data_received[3:]

			data = decrypt_data_and_session_key(sum_data, self.private_key)
			final_sum = np.mod(data - self.random_key, self.prime)

			data2 = decrypt_bytes_with_private_key(key_data, self.private_key)
			final_key = data2

			# broadcast to all
			final_sum_result = []
			for key, value in config.items():
				if key != 'task':
					cid = key
					pub_key = RSA.import_key(value)
					data_enc = encrypt_data_and_session_key(final_sum, pub_key, self.session_key_length)
					if cid != 'swift':
						data_enc2 = encrypt_bytes_with_public_key(final_key, pub_key, self.session_key_length)
					else:
						data_enc2 = ['', '', '']
					final_sum_result.append(cid)
					final_sum_result.extend(data_enc + data_enc2)

			end = time.time()
			data_sent = [np.array(value) for value in final_sum_result]
			tracemalloc.stop()

			return FitRes(
				status=Status(code=Code.OK, message='Success'),
				parameters=ndarrays_to_parameters(data_sent),
				num_examples=1,
				metrics={}
			)
		# Instruction5: build local bloomfilter
		elif task == 'build-local-bloomfilter':
			# get total sum and xor key
			data_received = parameters_to_ndarrays(global_parameters)
			data_enc = search_data_enc(data_received, self.cid)
			final_sum_enc = data_enc[0:3]
			total_sum = decrypt_data_and_session_key(final_sum_enc, self.private_key)

			xor_key_enc = data_enc[3:6]
			xor_key = decrypt_bytes_with_private_key(xor_key_enc, self.private_key)

			# build bloomfilter
			bf_array = build_bloomfilter(total_sum, self.error_rate, self.valid_accounts, key=xor_key, verbose=False)

			# encrypt hashed accounts
			hash_accounts_array1 = data_enc[6]
			hash_accounts_array2 = data_enc[7]
			enc_hash_accounts1 = enc_hashed_banks(hash_accounts_array1, key=xor_key)
			enc_hash_accounts2 = enc_hashed_banks(hash_accounts_array2, key=xor_key)

			# encrypt bloomfilter
			swift_pub_key = RSA.import_key(config['key'])
			data_enc_bf = encryption_bf(bf_array, swift_pub_key, self.session_key_length)
			data_enc_bf = [np.array(value) for value in data_enc_bf]

			# data sent
			data_sent = data_enc_bf + [
				hash_accounts_array1, enc_hash_accounts1, hash_accounts_array2, enc_hash_accounts2
			]

			tracemalloc.stop()

			return FitRes(
				status=Status(code=Code.OK, message="Success"),
				parameters=ndarrays_to_parameters(data_sent),
				num_examples=1,
				metrics={},
			)
		# Instruction6: do nothing
		else:
			# do nothing
			return FitRes(
				status=Status(code=Code.OK, message="Success"),
				parameters=ndarrays_to_parameters([]),
				num_examples=1,
				metrics={},
			)

	def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
		"""
		Execute certain task in evaluation phase
		Parameters
		----------
		ins: EvaluateIns -> evaluation instruction

		Returns
		-------
		EvaluateRes -> evaluation result
		"""
		return EvaluateRes(
			status=Status(code=Code.OK, message="Success"),
			loss=1.0,
			num_examples=1,
			metrics={},
		)
