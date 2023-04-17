"""
This file contains class of client for pns in training phase.
"""

from pathlib import Path
from typing import Union

import flwr as fl
from array import array

import pandas as pd
from loguru import logger
import pickle
from flwr.common import (
	EvaluateIns,
	EvaluateRes,
	FitIns,
	FitRes,
	Status,
	Code,
)
from flwr.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays
import numpy as np
from Crypto.PublicKey import RSA
import json
import tracemalloc

from .utils_basic import (
	str_to_bytes, decrypt_data_and_session_key, search_data_enc, OR_arrays, decryption_bf,
)
from .bloom_filter import BloomFilter
from .model3 import (
	extract_feature, PNSModel, add_BF_feature2,
)
from .utils_advanced import (
	decrypt_bytes_with_private_key,
	byte_string_to_ndarray,
	get_hashed_accounts_of_banks,
	update_hashed_accounts_dict,
	compute_data_capacity,
)
import time


class TrainPNSClient(fl.client.Client):
	"""
	Client for pns in training phase

	Parameters
	----------
	cid: client id
	client_dir: client directory
	data: data of client
	session_key_length: session key length
	error_rate: error rate
	public_key_size: public key size

	Attributes
	----------
	cid: client id
	client_dir: client directory
	cid_bytes: client id in bytes
	data: data of client
	error_rate: error rate
	bloom: bloom filter
	total_accounts: total number of accounts
	session_key_length: session key length
	session_key_dict: session key dictionary
	public_key: public key
	private_key: private key
	internal_state: internal state
	public_key_size: public key size
	stats: statistics

	Methods
    -------
    load_state()
        load states of the client
    get_parameters()
        get parameters of the model
    set_parameters(parameters)
        set parameters of the model
    fit(
        ins: FitIns,
        parameters: List[np.ndarray],
        config: Dict[str, Any]
    ) -> FitRes
        execute certain task based on fit instruction
    evaluate(
        ins: EvaluateIns,
        parameters: List[np.ndarray],
        config: Dict[str, Any]
    ) -> EvaluateRes
        execute certain task based on evaluate instruction


	"""

	def __init__(
			self,
			cid: str,
			client_dir: Path,
			data: pd.DataFrame,
			session_key_length: int = 16,
			error_rate: float = 0.001,
			public_key_size: int = 2048
	):
		super().__init__()
		self.cid: str = cid
		self.client_dir: Path = client_dir
		self.cid_bytes: bytes = str_to_bytes(self.cid)
		self.data: pd.DataFrame = data
		self.error_rate: float = error_rate
		self.bloom: Union[BloomFilter, None] = None
		self.total_accounts: int = 0
		self.session_key_length: int = session_key_length

		# load states
		self.session_key_dict = None
		self.public_key = None
		self.private_key = None
		self.internal_state = None
		self.public_key_size = public_key_size
		self.stats = None
		self.load_state()

	def load_state(self):
		"""Load states of clients"""
		client_dir = self.client_dir

		###########################################################################################
		# load own private and public key
		private_key_file = Path.joinpath(client_dir, "private.pem")
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
		# load internal state
		file_path = Path.joinpath(client_dir, 'internal_state.pkl')
		if file_path.exists():
			with file_path.open(mode='rb') as f:
				self.internal_state = pickle.load(f)
		else:
			self.internal_state = {}

		file_path = Path.joinpath(client_dir, 'stats.json')
		if file_path.exists():
			with file_path.open(mode='r') as f:
				self.stats = json.load(f)
		else:
			self.stats = {}

	def save_state(self, data, file_name):
		"""Save states of clients"""
		file_path = Path.joinpath(self.client_dir, file_name)
		if not file_path.exists():
			with file_path.open('wb') as f:
				pickle.dump(data, f)

	def save_stats(self):
		"""Save stats of clients"""
		file_path = Path.joinpath(self.client_dir, 'state/server/stats.json')
		with file_path.open('w') as f:
			json.dump(self.stats, f)

	# Flower internal function will call this to get local parameters e.g., print them
	def get_parameters(self, ins):
		"""Get parameters"""
		return ndarrays_to_parameters([])

	def fit(self, ins: FitIns) -> FitRes:
		"""
		Fit function for pns, execute certain actions based on fit instruction and return fit result to server
		Parameters
		----------
		ins: FitIns -  Fit instruction from server

		Returns
		-------
		FitRes - Fit result to server
		"""
		# get global parameters (key) and config
		global_parameters = ins.parameters
		config = ins.config
		task = config['task']

		# start memory usage

		# Instruction1: send public key -> return public key of client to server
		if task == 'send_public_key':
			data_sent = [np.array([self.public_key.export_key()])]

			return FitRes(
				status=Status(code=Code.OK, message='Success'),
				parameters=ndarrays_to_parameters(data_sent),
				num_examples=1,
				metrics={}
			)
		# Instruction2: send encrypted and hashed banks list in pns side -> return banks list of client to server
		elif task == 'send_pns_banks_in_partitions':
			client_banks_partition = {}
			data_received = parameters_to_ndarrays(global_parameters)
			for i in range(len(data_received) // 4):
				cid = data_received[4 * i].item()
				data_enc = [data_received[4 * i + 1], data_received[4 * i + 2], data_received[4 * i + 3]]
				data_dec = decrypt_bytes_with_private_key(data_enc, self.private_key)
				banks = byte_string_to_ndarray(data_dec)
				client_banks_partition[cid] = banks  # make it to be a list

			# send hashed accounts of each bank to server
			data_sent = []
			for key, public_key in config.items():
				if key != 'task':
					cid = key
					banks = client_banks_partition[cid]
					hash1_array, hash2_array = get_hashed_accounts_of_banks(self.data, banks)
					data_sent.append(np.array(cid))
					data_sent.append(hash1_array)
					data_sent.append(hash2_array)

			return FitRes(
				status=Status(code=Code.OK, message='Success'),
				parameters=ndarrays_to_parameters(data_sent),
				num_examples=1,
				metrics={}
			)

		# Instruction3: build local bloom filter -> return empty list to server
		elif task == 'build-local-bloomfilter':
			# get total sum
			data_received = parameters_to_ndarrays(global_parameters)
			data_enc = search_data_enc(data_received, self.cid)[0:3]
			total_sum = decrypt_data_and_session_key(data_enc, self.private_key)
			if total_sum is None:
				raise ValueError('Total sum cannot be decryption in {}'.format(self.cid))

			pns_internal_state = {'total_sum': total_sum}
			self.save_state(pns_internal_state, 'internal_state.pkl')


			return FitRes(
				status=Status(code=Code.OK, message='Success'),
				parameters=ndarrays_to_parameters([]),
				num_examples=1,
				metrics={}
			)

		# Instruction4: train model to detection anomaly after receiving all necessary information from server
		elif task == 'pns_run_train':
			if 'total_sum' in self.internal_state:
				total_sum = self.internal_state['total_sum']
			else:
				raise ValueError('internal state do not have total sum something wrong.')

			bloom = BloomFilter(total_sum, error_rate=self.error_rate)

			############################################################################################################
			# decrypt and OR all bloom arrays
			############################################################################################################
			data_received = parameters_to_ndarrays(global_parameters)
			bloom_arrays = []
			hashed_accounts_dict1 = {}
			hashed_accounts_dict2 = {}
			for i in range(len(data_received) // 7):
				bf_enc = data_received[7 * i:7 * i + 3]
				bf_array = decryption_bf(bf_enc, self.private_key)
				bloom_arrays.append(bf_array)
				hashed_accounts_enc = data_received[7 * i + 3:7 * i + 7]
				dict1, dict2 = update_hashed_accounts_dict(hashed_accounts_enc)
				hashed_accounts_dict1.update(dict1)
				hashed_accounts_dict2.update(dict2)

			global_bloom_array = OR_arrays(bloom_arrays)
			bloom.backend.array_ = array("L", global_bloom_array)

			############################################################################################################
			# Training model
			############################################################################################################
			logger.info("Start Training ...")
			pns_df = self.data
			logger.info("Adding BF and extracting features")
			pns_df = add_BF_feature2(pns_df, bloom, hashed_accounts_dict1, hashed_accounts_dict2)
			logger.info("Adding BF and extracting features")
			pns_df = extract_feature(pns_df, self.client_dir, phase='train', epsilon=0.25, dp_flag=True)
			pns_model = PNSModel()
			logger.info("Fitting pns model...")
			pns_model.fit(X=pns_df.drop(['Label'], axis=1), y=pns_df["Label"])
			pns_model.save(Path.joinpath(self.client_dir, "pns_model.joblib"))



			return FitRes(
				status=Status(code=Code.OK, message="Success"),
				parameters=ndarrays_to_parameters([]),
				num_examples=1,
				metrics={},
			)
		else:
			return FitRes(
				status=Status(code=Code.OK, message='Success'),
				parameters=ndarrays_to_parameters([]),
				num_examples=1,
				metrics={}
			)

	def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
		"""
		evaluation function for pns
		Parameters
		----------
		ins: EvaluateIns - evaluation instruction

		Returns
		-------
		EvaluateRes - evaluation result
		"""
		return EvaluateRes(
			status=Status(code=Code.OK, message="Success"),
			loss=1.0,
			num_examples=1,
			metrics={},
		)
