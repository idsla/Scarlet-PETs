"""
This file contains basic helper functions for federated learning including:
- Data encryption and decryption
- Data type conversion
- Bloom filter construction
"""
from typing import List, Any, Union

import pandas as pd
from Crypto.PublicKey.RSA import RsaKey
from flwr.common.typing import Parameters
from array import array
from Crypto.Random import get_random_bytes
from Crypto.Cipher import AES, PKCS1_OAEP
from Crypto.Util.Padding import pad, unpad
import numpy as np
from base64 import b64encode, b64decode
import hashlib
import binascii

from .bloom_filter import BloomFilter


########################################################################################################################
# Helper functions for type conversion
########################################################################################################################

def strings_to_parameters(strings: List[str]) -> Parameters:
	"""Convert list of string to parameters object."""
	tensors = [bytes(string, 'utf-8') for string in strings]
	return Parameters(tensors=tensors, tensor_type="string")


def parameters_to_strings(parameters: Parameters) -> List[str]:
	"""Convert parameters object to list of string."""
	return [byte_string.decode('utf-8') for byte_string in parameters.tensors]


def convert_bank_valid_accounts_list(df: pd.DataFrame) -> List[str]:
	"""
	Convert bank valid accounts list to string list
	Parameters
	----------
	df: pd.DataFrame

	Returns
	-------
	List[str]

	Examples
	--------
	>>> df = pd.DataFrame({'Bank': [1, 2, 3], 'Account': [1, 2, 3], 'Name': ['a', 'b', 'c'], 'Street': ['d', 'e', 'f']\
	, 'CountryCityZip': ['g', 'h', 'i'], 'Flags': [0, 0, 1]})
	>>> convert_bank_valid_accounts_list(df)
	['11adg', '22beh']
	"""
	df['Account'] = df['Account'].astype(str)
	df = df[df['Flags'] == 0]
	ret = df['Account'] + df['Name'] + df['Street'] + df['CountryCityZip']
	return ret.values.tolist()


def convert_bank_invalid_accounts_list(df: pd.DataFrame) -> List[str]:
	"""
	Convert bank invalid accounts list to string list
	Parameters
	----------
	df: pd.DataFrame

	Returns
	-------
	List[str]

	Examples
	--------
	>>> df = pd.DataFrame({'Bank': [1, 2, 3], 'Account': [1, 2, 3], 'Name': ['a', 'b', 'c'], 'Street': ['d', 'e', 'f']\
	, 'CountryCityZip': ['g', 'h', 'i'], 'Flags': [1, 1, 0]})
	>>> convert_bank_invalid_accounts_list(df)
	['11adg', '22beh']
	"""
	df['Bank'] = df['Bank'].astype(str)
	df['Account'] = df['Account'].astype(str)
	df = df[df['Flags'] != 0]
	ret = df['Bank'] + df['Account'] + df['Name'] + df['Street'] + df['CountryCityZip']
	return ret.values.tolist()


def bf_array_to_bytestring(bf_array: array) -> bytes:
	"""Convert bloom filter array to bytestring"""
	ret = bf_array.tobytes()
	return ret


def bytestring_to_bf_array(byte_string: bytes) -> array:
	"""Convert bytestring to bloom filter array"""
	bf_array = array('L', [])
	bf_array.frombytes(byte_string)
	return bf_array


def str_to_bytes(string: str) -> bytes:
	"""Convert string to bytes"""
	return bytes(string, 'utf-8')


def bytes_to_str(bytes_: bytes) -> str:
	"""Convert bytes to string"""
	return bytes_.decode()


########################################################################################################################
# Helper functions for encryption and decryption
########################################################################################################################
def XOR(string1: str, string2: str) -> str:
	"""XOR two strings of binary digits"""
	res = []
	for _a, _b in zip(string1, string2):
		res.append(str(int(_a) | int(_b)))
	return ''.join(res)


def XOR_array(array1: array, array2: array) -> List[str]:
	"""XOR two arrays of binary digits"""
	res = []
	for integer1, integer2 in zip(array1, array2):
		res.append(integer1 | integer2)
	return res


def encrypt_data_and_session_key(data: Any, public_key: RsaKey, session_key_length: int = 16) -> List[Any]:
	"""
	Encrypt data from session key using public key
	Parameters
	----------
	data: data to be encrypted
	public_key: public key
	session_key_length: session key length

	Returns
	-------
	data_enc: encrypted data and session key
	"""

	# encrypt session key using other's pubkey
	session_key = get_random_bytes(session_key_length)
	cipher_rsa = PKCS1_OAEP.new(public_key)
	enc_session_key = cipher_rsa.encrypt(session_key)

	# data
	data = np.int64(data)
	data = bytes(str(data), 'utf-8')

	# encrypt data
	cipher_aes = AES.new(session_key, AES.MODE_CBC)  # change CBC mode
	ct_bytes = cipher_aes.encrypt(pad(data, AES.block_size))
	iv = b64encode(cipher_aes.iv).decode('utf-8')
	ct = b64encode(ct_bytes).decode('utf-8')

	data_enc = [enc_session_key, iv, ct]

	return data_enc


def decrypt_data_and_session_key(data_enc: List[Any], private_key: RsaKey) -> Any:
	"""
	Decrypt data from session key using private key
	Parameters
	----------
	data_enc: encrypted data and session key
	private_key: private key

	Returns
	-------
	data: decrypted data
	"""
	enc_session_key, iv, ct = data_enc[0].item(), data_enc[1].item(), data_enc[2].item()
	try:
		cipher_rsa = PKCS1_OAEP.new(private_key)
		session_key = cipher_rsa.decrypt(enc_session_key)
	except ValueError:
		return None
	else:
		try:
			iv = b64decode(iv)
			ct = b64decode(ct)
			cipher = AES.new(session_key, AES.MODE_CBC, iv)
			pt = unpad(cipher.decrypt(ct), AES.block_size)
			pt_int = np.array(pt.decode('utf-8')).astype('int64')
			return pt_int
		except (ValueError, KeyError):
			print("error decrypt data")


def search_data_enc(ret_array: Any, cid: str) -> Any:
	"""
	Search data from encrypted data
	Parameters
	----------
	ret_array: encrypted data
	cid: client id

	Returns
	-------
	data_enc: encrypted data searched for client cid
	"""
	data_enc = []
	for i in range(len(ret_array)):
		item = ret_array[i]
		if item.size == 1:
			if item.item() == cid:
				data_enc.extend(ret_array[i + 1: i + 9])
	if len(data_enc) == 0:
		raise ValueError('Cannot find cid from result')

	return data_enc


########################################################################################################################
#   Bloom filter
########################################################################################################################

def build_bloomfilter(
		bf_size: int, error_rate: float, valid_accounts: List[str], key: Union[bytes, None] = None,
		verbose: bool = False
) -> array:
	"""
	Build bloom filter
	Parameters
	----------
	bf_size: bloom filter size
	error_rate: bloom filter error rate
	valid_accounts: valid accounts
	key: key for encryption hash of account
	verbose: whether to print out progress

	Returns
	-------
	res : bloom filter array
	"""
	# build bloomfilter and return backend array
	bloom = BloomFilter(max_elements=bf_size, error_rate=error_rate)

	# if key is not None, encrypt hash of account
	if key:
		for account in valid_accounts:
			cipher = AES.new(key, AES.MODE_ECB)
			hashed_account = hashlib.sha3_256(account.encode()).hexdigest()
			hashed_account_unhex = binascii.unhexlify(hashed_account)
			ct_bytes = cipher.encrypt(pad(hashed_account_unhex, AES.block_size))
			ct = binascii.hexlify(ct_bytes)
			ct = ct.decode()
			bloom.add(ct)

		if verbose:
			total_in, total_notin = 0, 0
			for item in valid_accounts:
				item_hash = hashlib.sha3_256(item.encode()).digest()
				ct_bytes = cipher.encrypt(pad(item_hash, AES.block_size))
				ct = binascii.hexlify(ct_bytes)
				ct = ct.decode()
				if ct in bloom:
					total_in += 1
				else:
					total_notin += 1
			print("Total accounts: {}".format(len(valid_accounts)))
			print("Total accounts can be found in bloom filter: {}".format(total_in))
			print("Total accounts cannot be found in bloom filter: {}".format(total_notin))
	# if key is None, hash account directly
	else:
		for account in valid_accounts:
			hashed_account = hashlib.sha3_256(account).hexdigest()
			bloom.add(hashed_account)

	res = bloom.backend.array_

	return res


def OR_arrays(array_list: List[array]) -> List[int]:
	"""
	OR operation on arrays
	Parameters
	----------
	array_list: list of arrays

	Returns
	-------
	res : OR result
	"""
	res = []
	for integers in zip(*array_list):
		init = 0
		for integer in integers:
			init = init | integer
		res.append(init)
	return res


def encryption_bf(bf_array:array, public_key:RsaKey, session_key_length:int):
	"""
	Encrypt bloom filter array using session key
	Parameters
	----------
	bf_array: bloom filter array
	public_key: public key
	session_key_length: session key length

	Returns
	-------
	data_enc: encrypted bloom filter array and session key
	"""
	# encrypt session key using other's pubkey
	session_key = get_random_bytes(session_key_length)
	cipher_rsa = PKCS1_OAEP.new(public_key)
	enc_session_key = cipher_rsa.encrypt(session_key)

	# data
	bf_array = bf_array_to_bytestring(bf_array)

	# encrypt data
	cipher_aes = AES.new(session_key, AES.MODE_CBC)  # change CBC mode
	ct_bytes = cipher_aes.encrypt(pad(bf_array, AES.block_size))
	iv = b64encode(cipher_aes.iv).decode('utf-8')
	ct = b64encode(ct_bytes).decode('utf-8')

	data_enc = [enc_session_key, iv, ct]

	return data_enc


def decryption_bf(bf_array_enc: List[Any], private_key: RsaKey):
	"""
	Decrypt bloom filter array using session key
	Parameters
	----------
	bf_array_enc: encrypted bloom filter array and session key
	private_key: private key

	Returns
	-------
	bf_array: decrypted bloom filter array
	"""
	enc_session_key, iv, ct = bf_array_enc[0].item(), bf_array_enc[1].item(), bf_array_enc[2].item()
	try:
		cipher_rsa = PKCS1_OAEP.new(private_key)
		session_key = cipher_rsa.decrypt(enc_session_key)
	except ValueError:
		print('error decrypt session key')
		return None
	else:
		try:
			iv = b64decode(iv)
			ct = b64decode(ct)
			cipher = AES.new(session_key, AES.MODE_CBC, iv)
			pt = unpad(cipher.decrypt(ct), AES.block_size)
			bf_array = bytestring_to_bf_array(pt)
			return bf_array
		except (ValueError, KeyError):
			print("error decrypt data")
