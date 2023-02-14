import hashlib
import numpy as np
import pickle
from Crypto.Random import get_random_bytes
from Crypto.Cipher import AES, PKCS1_OAEP
from Crypto.Util.Padding import pad, unpad
from base64 import b64encode, b64decode
from typing import List
import binascii


def bytes_xor(ba1, ba2):
    return bytes([_a ^ _b for _a, _b in zip(ba1, ba2)])


def ndarray_to_byte_string(data: np.ndarray):
    return pickle.dumps(data)


def byte_string_to_ndarray(byte_string):
    return pickle.loads(byte_string)


def encrypt_bytes_with_public_key(data_bytes, public_key, session_key_length):
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


def decrypt_bytes_with_private_key(data_enc: List, private_key):
    enc_session_key, iv, ct = data_enc[0].item(), data_enc[1].item(), data_enc[2].item()
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
            data_dec = unpad(cipher.decrypt(ct), AES.block_size)
            return data_dec
        except (ValueError, KeyError):
            print("error decrypt data")


def get_hashed_accounts_of_banks(data, banks):

    sender = data[data['Sender'].isin(list(banks))]
    receiver = data[data['Receiver'].isin(list(banks))]

    def get_hash(value):
        hash = hashlib.sha3_256(value.encode()).hexdigest()
        return hash

    func = np.vectorize(get_hash)

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
    # if receiver.shape[0] > 0:
    #     hash2 = receiver['BeneficiaryAccount'].astype(str) + receiver['BeneficiaryName'].astype(str) + receiver[
    #         'BeneficiaryStreet'].astype(str) + receiver['BeneficiaryCountryCityZip'].astype(str)
    #
    #     hash2_array = hash2.unique()
    #     hash2_array_ = func(hash2_array)
    #     hash2_array = hash2_array_
    # else:
    #     hash2_array = np.array([])

    return hash1_array, hash2_array


def assembling_final_sum_with_hashed_accounts(final_sum_list, hashed_accounts_list):
    ret = []
    for i in range(len(final_sum_list) // 7):
        cid = final_sum_list[7 * i].item()
        if cid == 'swift':
            hash_array1 = np.array([])
            hash_array2 = np.array([])
            ret.append(np.array(cid))
            ret.extend(final_sum_list[(7 * i + 1): (7 * i + 4)])
            ret.extend([np.array(b'') for _ in range(3)])
            ret.append(hash_array1)
            ret.append(hash_array2)
        else:
            #print(cid)
            # find two hash array for cid
            hash_array1 = np.array([])
            hash_array2 = np.array([])
            for j in range(len(hashed_accounts_list)):
                item = hashed_accounts_list[j]
                #print(j, item)
                if item.size == 1:
                    if item.item() == cid:
                        hash_array1 = hashed_accounts_list[j + 1]
                        hash_array2 = hashed_accounts_list[j + 2]
            ret.append(np.array(cid))
            ret.extend(final_sum_list[7 * i + 1:7 * i + 7])
            ret.append(hash_array1)
            ret.append(hash_array2)

    return ret


def enc_hashed_banks(accounts_hash_ndarray: np.ndarray, key):

    #cipher = AES.new(key, AES.MODE_ECB)  # change to ECB

    def encrypt(value):  # value -> bytes
        value_unhex = binascii.unhexlify(value)
        cipher = AES.new(key, AES.MODE_ECB)  # change to ECB
        ct_bytes = cipher.encrypt(pad(value_unhex, AES.block_size))
        ct = binascii.hexlify(ct_bytes)
        ct = ct.decode()
        # ct = b64encode(ct_bytes).decode('utf-8')
        # hash_ = hashlib.sha3_256(ct).hexdigest()
        return ct

    func = np.vectorize(encrypt)
    if accounts_hash_ndarray.size > 0:
        ret = func(accounts_hash_ndarray)
    else:
        ret = np.array([])
    return ret


def update_hashed_accounts_dict(hash_accounts_list):

    hash1_array = hash_accounts_list[0]
    hash1_array_enc = hash_accounts_list[1]
    hash2_array = hash_accounts_list[2]
    hash2_array_enc = hash_accounts_list[3]

    dict1 = {}
    dict2 = {}

    if hash1_array.size > 0:
        for i in range(hash1_array.shape[0]):
            #hash_value = b64encode(hash1_array[i]).decode()
            dict1[hash1_array[i]] = hash1_array_enc[i]

    if hash2_array.size > 0:
        for i in range(hash2_array.shape[0]):
            #hash_value = b64encode(hash2_array[i]).decode()
            dict2[hash2_array[i]] = hash2_array_enc[i]

    return dict1, dict2


def get_unique_banks(data):
    return data['Bank'].unique()
