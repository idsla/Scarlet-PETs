from typing import List
from flwr.common.typing import Parameters
from array import array
from Crypto.Random import get_random_bytes
from Crypto.Cipher import AES, PKCS1_OAEP
from Crypto.Util.Padding import pad, unpad
import numpy as np
from base64 import b64encode, b64decode
import hashlib   # TODO:import hash function as crypt_hash
import binascii

from .bloom_filter import BloomFilter


def strings_to_parameters(strings: List[str]) -> Parameters:
    """Convert list of string to parameters object."""
    tensors = [bytes(string, 'utf-8') for string in strings]
    return Parameters(tensors=tensors, tensor_type="string")


def parameters_to_strings(parameters: Parameters) -> List[str]:
    """Convert parameters object to list of string."""
    return [byte_string.decode('utf-8') for byte_string in parameters.tensors]


def XOR(string1: str, string2: str):
    res = []
    for _a, _b in zip(string1, string2):
        res.append(str(int(_a) | int(_b)))
    return ''.join(res)


def XOR_array(array1, array2):
    res = []
    for integer1, integer2 in zip(array1, array2):
        res.append(integer1 | integer2)
        # print('int1: {:032b}'.format(integer1))
        # print('int2: {:032b}'.format(integer2))
        # print('res : {:032b}'.format(integer1 | integer2))
    return res


def convert_bank_valid_accounts_list(df):
    df['Account'] = df['Account'].astype(str)
    df = df[df['Flags'] == 0]
    ret = df['Account'] + df['Name'] + df['Street'] + df['CountryCityZip']
    return ret.values.tolist()


def convert_bank_invalid_accounts_list(df):
    df['Bank'] = df['Bank'].astype(str)
    df['Account'] = df['Account'].astype(str)
    df = df[df['Flags'] != 0]
    ret = df['Bank'] + df['Account'] + df['Name'] + df['Street'] + df['CountryCityZip']
    return ret.values.tolist()


def bf_array_to_bytestring(bf_array):
    ret = bf_array.tobytes()
    return ret


def bytestring_to_bf_array(byte_string):
    bf_array = array('L', [])
    bf_array.frombytes(byte_string)
    return bf_array


def str_to_bytes(string):
    return bytes(string, 'utf-8')


def bytes_to_str(bytes_):
    return bytes_.decode()


def encrypt_data_and_session_key(data, public_key, session_key_length):
    # TODO: hafiz need to check iv, ct -> encrypt iv too
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


def decrypt_data_and_session_key(data_enc, private_key):
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


def search_data_enc(ret_array, cid):
    data_enc = []
    for i in range(len(ret_array)):
        item = ret_array[i]
        if item.size == 1:
            if item.item() == cid:
                data_enc.extend(ret_array[i + 1: i + 9])
    if len(data_enc) == 0:
        raise ValueError('Cannot find cid from result')

    return data_enc


def build_bloomfilter(bf_size, error_rate, valid_accounts, key=None, verbose=False):
    #print(key)
    # build bloomfilter and return backend array
    bloom = BloomFilter(max_elements=bf_size, error_rate=error_rate)

    if key:
        #cipher = AES.new(key, AES.MODE_ECB)  # change to ECB
        for account in valid_accounts:
            cipher = AES.new(key, AES.MODE_ECB)
            hashed_account = hashlib.sha3_256(account.encode()).hexdigest()
            hashed_account_unhex = binascii.unhexlify(hashed_account)
            ct_bytes = cipher.encrypt(pad(hashed_account_unhex, AES.block_size))
            ct = binascii.hexlify(ct_bytes)
            ct = ct.decode()
            #ct = b64encode(ct_bytes).decode('utf-8')
            # hashed_account, tag = cipher.encrypt_and_digest(hashed_account)
            # hashed_account = hashlib.sha3_256(hashed_account).hexdigest()
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
    else:
        for account in valid_accounts:
            hashed_account = hashlib.sha3_256(account).hexdigest()
            bloom.add(hashed_account)

    res = bloom.backend.array_

    return res


def OR_arrays(array_list):
    res = []
    for integers in zip(*array_list):
        init = 0
        for integer in integers:
            init = init | integer
        res.append(init)
    return res


def encryption_bf(bf_array, public_key, session_key_length):
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


def decryption_bf(bf_array_enc, private_key):
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
