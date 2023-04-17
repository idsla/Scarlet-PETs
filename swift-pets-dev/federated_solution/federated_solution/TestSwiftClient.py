"""
This file is the swift client side of the federated learning system in testing phase.
"""
from pathlib import Path
from typing import Union

import flwr as fl
from array import array
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
import pandas as pd
import numpy as np
from Crypto.PublicKey import RSA
from sklearn import metrics

from .utils_basic import (
	str_to_bytes, decrypt_data_and_session_key, search_data_enc, OR_arrays, decryption_bf,
)
from .bloom_filter import BloomFilter
from .model3 import (
	extract_feature, SwiftModel, add_BF_feature2,
)
from .utils_advanced import (
	decrypt_bytes_with_private_key,
	byte_string_to_ndarray,
	get_hashed_accounts_of_banks,
	update_hashed_accounts_dict,
)


class TestSwiftClient(fl.client.Client):
	"""
	This class is the swift client side of the federated learning system in testing phase.

	Parameters
	----------
	cid: str
		client id
	data: pd.DataFrame
		data of the client
	client_dir: Path
		client directory
	preds_format_path: Path
		path to the prediction format file
	preds_dest_path: Path
		path to the destination of the prediction file
	session_key_length: int
		session key length
	error_rate: float
		error rate of the bloom filter
	public_key_size: int
		public key size
	evaluation: bool

	Attributes
	----------
	cid: str
		client id
	client_dir: Path
		client directory
	preds_format_path: Path
		path to the prediction format file
	preds_dest_path: Path
		path to the destination of the prediction file
	cid_bytes: bytes
		client id in bytes
	data: pd.DataFrame
		data of the client
	error_rate: float
		error rate of the bloom filter
	bloom: BloomFilter
		bloom filter of the client
	total_accounts: int
		total number of accounts of the client
	session_key_length: int
		session key length
	evaluation: bool
		whether to evaluate the model
	session_key_dict: dict
		session key dictionary
	public_key: RSA
		public key
	private_key: RSA
		private key
	internal_state: dict
		internal state of the client
	public_key_size: int
		public key size

	Methods
	-------
	load_state()
		load states of the client
	get_parameters()
		get parameters of the model
	fit(
		ins: FitIns,
		parameters: List[np.ndarray],
		config: Dict[str, Any]
	) -> FitRes
		fit the model
	evaluate(
		ins: EvaluateIns,
		parameters: List[np.ndarray],
		config: Dict[str, Any]
	) -> EvaluateRes
		evaluate the model

	"""

	def __init__(
			self,
			cid: str,
			data: pd.DataFrame,
			client_dir: Path,
			preds_format_path: Path,
			preds_dest_path: Path,
			session_key_length: int = 16,
			error_rate: float = 0.001,
			public_key_size: int = 2048,
			evaluation: bool = True
	):
		super().__init__()
		self.cid: str = cid
		self.client_dir: Path = client_dir
		self.preds_format_path: Path = preds_format_path
		self.preds_dest_path: Path = preds_dest_path
		self.cid_bytes: bytes = str_to_bytes(self.cid)
		self.data: pd.DataFrame = data
		self.error_rate: float = error_rate
		self.bloom: Union[BloomFilter, None] = None
		self.total_accounts: int = 0
		self.session_key_length: int = session_key_length
		self.evaluation: bool = evaluation

		# load states
		self.session_key_dict = None
		self.public_key = None
		self.private_key = None
		self.internal_state = None
		self.public_key_size = public_key_size
		self.load_state()

	def load_state(self):
		"""Load states of the client."""
		client_dir = self.client_dir

		###########################################################################################
		# load own private and public key
		private_key_file = Path.joinpath(client_dir, "private.pem")
		public_key_file = Path.joinpath(client_dir, "public.pem")
		# print(private_key_file)
		# print(public_key_file)
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

	def save_state(self, data, file_name):
		"""Save states of the client."""
		file_path = Path.joinpath(self.client_dir, file_name)
		if not file_path.exists():
			with file_path.open('wb') as f:
				pickle.dump(data, f)

	# Flower internal function will call this to get local parameters e.g., print them
	def get_parameters(self, ins):
		"""Get parameters."""
		return ndarrays_to_parameters([])

	def fit(self, ins: FitIns) -> FitRes:
		"""Execute certain task based on fit instruction and return fit result."""
		# get global parameters (key) and config
		global_parameters = ins.parameters
		config = ins.config
		task = config['task']

		# Instruction1: send public key
		if task == 'send_public_key':
			return FitRes(
				status=Status(code=Code.OK, message='Success'),
				parameters=ndarrays_to_parameters([np.array([self.public_key.export_key()])]),
				num_examples=1,
				metrics={}
			)
		# Instruction2: send encrypted and hashed banks of client to server
		elif task == 'send_swift_banks_in_partitions':
			# receive banks for each partition and store it in dict {cid: list of banks at cid}
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
		# Instruction3: build local bloom filter
		elif task == 'build-local-bloomfilter':
			# get total sum
			data_received = parameters_to_ndarrays(global_parameters)
			data_enc = search_data_enc(data_received, self.cid)[0:3]
			total_sum = decrypt_data_and_session_key(data_enc, self.private_key)
			if total_sum is None:
				raise ValueError('Total sum cannot be decryption in {}'.format(self.cid))

			swift_internal_state = {'total_sum': total_sum}
			# print(f"{self.cid} total sum - {total_sum}")
			self.save_state(swift_internal_state, 'internal_state.pkl')
			return FitRes(
				status=Status(code=Code.OK, message='Success'),
				parameters=ndarrays_to_parameters([]),
				num_examples=1,
				metrics={}
			)
		# Instruction4: do nothing
		else:
			return FitRes(
				status=Status(code=Code.OK, message='Success'),
				parameters=ndarrays_to_parameters([]),
				num_examples=1,
				metrics={}
			)

	def evaluate(self, ins: EvaluateIns) -> EvaluateRes:

		global_parameters = ins.parameters
		config = ins.config
		task = config['task']

		# Instruction1: test model based on trained model
		if task == 'swift_run_test':
			# get total sum
			if 'total_sum' in self.internal_state:
				total_sum = self.internal_state['total_sum']
			else:
				raise ValueError('internal state do not have total sum something wrong.')

			bloom = BloomFilter(total_sum, error_rate=self.error_rate)

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

			# Testing
			logger.info("Start Testing ...")
			swift_df = self.data
			logger.info("Adding BF and extracting features")
			swift_df = add_BF_feature2(swift_df, bloom, hashed_accounts_dict1, hashed_accounts_dict2)
			swift_df = extract_feature(swift_df, self.client_dir, phase='test', epsilon=0.25, dp_flag=False)

			logger.info("Loading SWIFT model...")
			if self.evaluation:
				swift_model = SwiftModel.load(Path.joinpath(self.client_dir, "swift_model.joblib"))
				swift_preds = swift_model.predict(swift_df.drop(['Label'], axis=1))

				preds_format_df = pd.read_csv(self.preds_format_path, index_col="MessageId")
				preds_format_df["Score"] = preds_format_df.index.map(swift_preds)

				print(
					"AUPRC:",
					metrics.average_precision_score(y_true=swift_df['Label'], y_score=preds_format_df["Score"])
					)

				logger.info("Writing out test predictions...")
				preds_format_df.to_csv(self.preds_dest_path)
				logger.info("Done.")
			else:
				swift_model = SwiftModel.load(Path.joinpath(self.client_dir, "swift_model.joblib"))
				swift_preds = swift_model.predict(swift_df)

				preds_format_df = pd.read_csv(self.preds_format_path, index_col="MessageId")
				preds_format_df["Score"] = preds_format_df.index.map(swift_preds)

				logger.info("Writing out test predictions...")
				preds_format_df.to_csv(self.preds_dest_path)
				logger.info("Done.")

			return EvaluateRes(
				status=Status(code=Code.OK, message="Success"),
				loss=1.0,
				num_examples=1,
				metrics={},
			)
		else:
			return EvaluateRes(
				status=Status(code=Code.OK, message="Success"),
				loss=1.0,
				num_examples=1,
				metrics={},
			)
