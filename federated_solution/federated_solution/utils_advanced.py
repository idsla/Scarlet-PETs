"""
This file contains advanced helper functions for federated data transfer including:
- Data encryption and decryption
- Data type conversion
- Hashing functions and encryption for bank accounts
- Utility functions for compute memory usage
"""
import hashlib
import numpy as np
import pickle

import pandas as pd
from Crypto.PublicKey.RSA import RsaKey
from Crypto.Random import get_random_bytes
from Crypto.Cipher import AES, PKCS1_OAEP
from Crypto.Util.Padding import pad, unpad
from base64 import b64encode, b64decode
import binascii
import sys

from typing import List, Any, Union


########################################################################################################################
# Helper functions for type conversion and encryption
########################################################################################################################\
def bytes_xor(ba1: bytes, ba2: bytes) -> bytes:
    """XOR two byte array"""
    return bytes([_a ^ _b for _a, _b in zip(ba1, ba2)])


def ndarray_to_byte_string(data: np.ndarray) -> bytes:
    """Convert ndarray to byte string"""
    return pickle.dumps(data)


def byte_string_to_ndarray(byte_string: bytes) -> np.ndarray:
    """Convert byte string to ndarray"""
    return pickle.loads(byte_string)


def encrypt_bytes_with_public_key(data_bytes: bytes, public_key: RsaKey, session_key_length: int) -> List[Any]:
    """
    Encrypt data with session key using public key
    Parameters
    ----------
    data_bytes: data to be encrypted
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

    # encrypt data
    cipher_aes = AES.new(session_key, AES.MODE_CBC)  # change CBC mode
    ct_bytes = cipher_aes.encrypt(pad(data_bytes, AES.block_size))
    iv = b64encode(cipher_aes.iv).decode('utf-8')
    ct = b64encode(ct_bytes).decode('utf-8')

    data_enc = [enc_session_key, iv, ct]

    return data_enc


def decrypt_bytes_with_private_key(data_enc: List, private_key: RsaKey) -> Union[bytes, None]:
    """
    Decrypt data with session key using private key
    Parameters
    ----------
    data_enc: data to be decrypted
    private_key: private key

    Returns
    -------
    data_dec: decrypted data
    """
    enc_session_key, iv, ct = data_enc[0].item(), data_enc[1].item(), data_enc[2].item()
    # try to decrypt session key using own private key
    try:
        cipher_rsa = PKCS1_OAEP.new(private_key)
        session_key = cipher_rsa.decrypt(enc_session_key)
    # if session key cannot be decrypted, pop up error message and return None
    except ValueError:
        print('error decrypt session key')
        return None
    # if session key is valid, try to decrypt data using session key
    else:
        try:
            iv = b64decode(iv)
            ct = b64decode(ct)
            cipher = AES.new(session_key, AES.MODE_CBC, iv)
            data_dec = unpad(cipher.decrypt(ct), AES.block_size)
            return data_dec
        # if data cannot be decrypted, pop up error message and return None
        except (ValueError, KeyError):
            print("error decrypt data")
            return None


########################################################################################################################
# Helper functions for sender and receiver bank accounts hashing
########################################################################################################################
def get_hashed_accounts_of_banks(data: pd.DataFrame, banks: List[str]):
    """
    Get hashed accounts of banks
    Parameters
    ----------
    data: data frame of transactions
    banks: list of banks

    Returns
    -------
    hash1_array: hashed accounts of banks
    """

    sender = data[data['Sender'].isin(list(banks))]

    def get_hash(value):
        """compute hash value of a string"""
        hash_ = hashlib.sha3_256(value.encode()).hexdigest()
        return hash_

    func = np.vectorize(get_hash)

    # if sender is not empty then compute hash value of accounts
    if sender.shape[0] > 0:
        hash1 = sender['OrderingAccount'].astype(str) + sender['OrderingName'].astype(str) + sender[
            'OrderingStreet'].astype(str) + sender['OrderingCountryCityZip'].astype(str)
        hash1_array = hash1.unique()
        hash1_array_ = func(hash1_array)
        hash1_array = hash1_array_

        hash2 = sender['BeneficiaryAccount'].astype(str) + sender['BeneficiaryName'].astype(str) + \
                sender['BeneficiaryStreet'].astype(str) + sender['BeneficiaryCountryCityZip'].astype(str)
        hash2_array = hash2.unique()
        hash2_array_ = func(hash2_array)
        hash2_array = hash2_array_
    else:
        hash1_array = np.array([])
        hash2_array = np.array([])

    return hash1_array, hash2_array


def assembling_final_sum_with_hashed_accounts(final_sum_list: List, hashed_accounts_list: List):
    """
    Assembling results including encrypted final sum and value of hashed accounts and align in one linear array
    Parameters
    ----------
    final_sum_list: List of computed results
    hashed_accounts_list: List of hashed accounts

    Returns
    -------
    ret: assembled results in format [cid, final sum encrypted,  hashed accounts array] for each client
    """
    ret = []
    for i in range(len(final_sum_list) // 7):
        cid = final_sum_list[7 * i].item()
        if cid == 'pns':
            # hash array for pns is empty
            hash_array1 = np.array([])
            hash_array2 = np.array([])
            # assemble final results including cid, encrypted final sum, and hash array
            ret.append(np.array(cid))
            ret.extend(final_sum_list[(7 * i + 1): (7 * i + 4)])
            ret.extend([np.array(b'') for _ in range(3)])
            ret.append(hash_array1)
            ret.append(hash_array2)
        else:
            hash_array1 = np.array([])
            hash_array2 = np.array([])
            # Find the index of cid in hashed_accounts_list and get hash array for cid
            for j in range(len(hashed_accounts_list)):
                item = hashed_accounts_list[j]
                if item.size == 1:
                    if item.item() == cid:
                        hash_array1 = hashed_accounts_list[j + 1]
                        hash_array2 = hashed_accounts_list[j + 2]
            # assemble final results including cid, encrypted final sum, and hash array
            ret.append(np.array(cid))
            ret.extend(final_sum_list[7 * i + 1:7 * i + 7])
            ret.append(hash_array1)
            ret.append(hash_array2)

    return ret


def enc_hashed_banks(accounts_hash_ndarray: np.ndarray, key: bytes):
    """
    Encrypt hashed accounts of banks
    Parameters
    ----------
    accounts_hash_ndarray: hashed accounts of banks
    key: key for encryption

    Returns
    -------
    ret: encrypted hashed accounts of banks
    """
    def encrypt(value):  # value -> bytes
        """encrypt value"""
        value_unhex = binascii.unhexlify(value)
        cipher = AES.new(key, AES.MODE_ECB)  # change to ECB
        ct_bytes = cipher.encrypt(pad(value_unhex, AES.block_size))
        ct = binascii.hexlify(ct_bytes)
        ct = ct.decode()
        return ct

    func = np.vectorize(encrypt)
    if accounts_hash_ndarray.size > 0:
        ret = func(accounts_hash_ndarray)
    else:
        ret = np.array([])
    return ret


def update_hashed_accounts_dict(hash_accounts_list):
    """
    Construct dictionary of two array of hashed accounts of banks: sender, receiver
    Parameters
    ----------
    hash_accounts_list: list of hashed accounts

    Returns
    -------
    dict1: dictionary of hashed accounts of banks
    dict2: dictionary of hashed accounts of banks
    """

    hash1_array = hash_accounts_list[0]
    hash1_array_enc = hash_accounts_list[1]
    hash2_array = hash_accounts_list[2]
    hash2_array_enc = hash_accounts_list[3]

    dict1 = {}
    dict2 = {}

    if hash1_array.size > 0:
        for i in range(hash1_array.shape[0]):
            # hash_value = b64encode(hash1_array[i]).decode()
            dict1[hash1_array[i]] = hash1_array_enc[i]

    if hash2_array.size > 0:
        for i in range(hash2_array.shape[0]):
            # hash_value = b64encode(hash2_array[i]).decode()
            dict2[hash2_array[i]] = hash2_array_enc[i]

    return dict1, dict2


########################################################################################################################
# Other functions
########################################################################################################################
def get_unique_banks(data: pd.DataFrame) -> List[str]:
    """
    Get unique banks
    Parameters
    ----------
    data: data frame of transactions

    Returns
    -------
    unique banks
    """
    return data['Bank'].unique()


def compute_data_capacity(data: List):
    """Compute the capacity of data in MB"""
    size = 0
    for item in data:
        if not isinstance(item, np.ndarray):
            size += sys.getsizeof(item)
        else:
            size += item.size * item.itemsize
    print(size)
    return size / (1024 * 1024)