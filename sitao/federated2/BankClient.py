import random
from pathlib import Path
import flwr as fl
from loguru import logger
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Status,
    Code
)
from Crypto.PublicKey import RSA
from Crypto.Random import get_random_bytes
from flwr.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays
import numpy as np
import pickle

from  .utils import (
    bytes_to_str, str_to_bytes, decrypt_data_and_session_key, encrypt_data_and_session_key, search_data_enc,
    build_bloomfilter, bf_array_to_bytestring, bytestring_to_bf_array, encryption_bf, decryption_bf
)
from .utils_new import (
    ndarray_to_byte_string,
    byte_string_to_ndarray,
    encrypt_bytes_with_public_key,
    decrypt_bytes_with_private_key,
    bytes_xor,
    enc_hashed_banks
)


class BankClient(fl.client.Client):
    def __init__(
            self,
            cid,
            valid_accounts,
            total_accounts,
            unique_banks,
            client_dir,
            session_key_length=16,
            error_rate=0.001,
            prime=20358416231591,
            public_key_size=2048
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
        self.load_state()

        #logger.info(
        #    "============================================================================================================")
        #logger.info("{} - {}".format(self.cid, self.total_accounts))
        #logger.info("{} - {}".format(self.cid, self.random_key))
        #logger.info("{} - {}".format(self.cid, self.key2))
        #logger.info(
        #    "============================================================================================================")

    def load_state(self):
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

    def save_state(self, data, file_name):
        file_path = Path.joinpath(self.client_dir, file_name)
        if not file_path.exists():
            with file_path.open('wb') as f:
                pickle.dump(data, f)

    # Flower internal function will call this to get local parameters e.g., print them
    def get_parameters(self, ins):
        return ndarrays_to_parameters([])

    def fit(self, ins: FitIns) -> FitRes:

        # get global parameters (key) and config
        global_parameters = ins.parameters
        config = ins.config

        task = config['task']
        if task == 'send_public_key':
            return FitRes(
                status=Status(code=Code.OK, message='Success'),
                parameters=ndarrays_to_parameters([np.array([self.public_key.export_key()])]),
                num_examples=1,
                metrics={}
            )

        elif task == 'share_bank_in_partition':
            pub_key = config['key']
            recipient_key = RSA.import_key(pub_key)
            #logger.info("{} {}".format(self.cid, self.unique_banks))
            # encrypt bank ids
            data_enc = encrypt_bytes_with_public_key(ndarray_to_byte_string(self.unique_banks), recipient_key,
                                                     self.session_key_length)
            # send encrypted bank ids to server
            return FitRes(
                status=Status(code=Code.OK, message='Success'),
                parameters=ndarrays_to_parameters([np.array(item) for item in data_enc]),
                num_examples=1,
                metrics={}
            )

        elif task == 'secure_sum':
            data_received = parameters_to_ndarrays(global_parameters)
            # data_received = [value[0] for value in data_received]
            # print("data received: {}".format(data_received))
            pub_key = config['key']
            recipient_key = RSA.import_key(pub_key)
            # print(data_received)
            # empty -> initialization
            if len(data_received) == 0:
                # set data
                data = np.mod(self.total_accounts + self.random_key, self.prime)
                # print("{} - {} {}".format(self.cid, data, self.total_accounts))
                # encryption
                data_enc = encrypt_data_and_session_key(data, recipient_key, self.session_key_length)
                # print(data_enc)

                data2 = self.key2
                data_enc2 = encrypt_bytes_with_public_key(data2, recipient_key, self.session_key_length)

                return FitRes(
                    status=Status(code=Code.OK, message='Success'),
                    parameters=ndarrays_to_parameters([np.array(value) for value in data_enc + data_enc2]),
                    num_examples=1,
                    metrics={}
                )
            # decrypt data and add own total_accounts and then encrypt with public key
            else:
                #print(data_received)
                data_received1 = data_received[0:3]
                data1 = decrypt_data_and_session_key(data_received1, self.private_key)
                #logger.info(f"{self.cid} - {data1}")

                data_received2 = data_received[3:]
                data2 = decrypt_bytes_with_private_key(data_received2, self.private_key)

                if data1 is not None and data2 is not None:
                    data1 = np.mod(data1 + self.total_accounts, self.prime)
                    data2 = bytes_xor(data2, self.key2)
                    # encrypt update sum with session key created by other's publik key
                    data_sent = encrypt_data_and_session_key(data1, recipient_key, self.session_key_length)
                    data_sent2 = encrypt_bytes_with_public_key(data2, recipient_key, self.session_key_length)

                    return FitRes(
                        status=Status(code=Code.OK, message='Success'),
                        parameters=ndarrays_to_parameters([np.array(value) for value in data_sent + data_sent2]),
                        num_examples=1,
                        metrics={}
                    )
                else:
                    raise ValueError('Decrypt error')
        elif task == 'compute_final_sum':
            data_received = parameters_to_ndarrays(global_parameters)
            sum_data = data_received[0:3]
            key_data = data_received[3:]

            data = decrypt_data_and_session_key(sum_data, self.private_key)
            final_sum = np.mod(data - self.random_key, self.prime)

            data2 = decrypt_bytes_with_private_key(key_data, self.private_key)
            final_key = data2

            # logger.info("===============================================================")
            # logger.info("{} - total accounts {}".format(self.cid, self.total_accounts))
            # logger.info("{} - final sum {}".format(self.cid, final_sum))
            # logger.info("{} - final key {}".format(self.cid, final_key))
            # logger.info("===============================================================")

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

            return FitRes(
                status=Status(code=Code.OK, message='Success'),
                parameters=ndarrays_to_parameters([np.array(value) for value in final_sum_result]),
                num_examples=1,
                metrics={}
            )
        elif task == 'build-local-bloomfilter':
            # get total sum and xor key
            data_received = parameters_to_ndarrays(global_parameters)
            data_enc = search_data_enc(data_received, self.cid)
            final_sum_enc = data_enc[0:3]
            total_sum = decrypt_data_and_session_key(final_sum_enc, self.private_key)

            xor_key_enc = data_enc[3:6]
            xor_key = decrypt_bytes_with_private_key(xor_key_enc, self.private_key)

            #print("{} - final sum {}".format(self.cid, total_sum))
            #print("{} - xor key {}".format(self.cid, xor_key))

            # build bloomfilter
            bf_array = build_bloomfilter(total_sum, self.error_rate, self.valid_accounts, key=xor_key, verbose=False)
            #logger.info("{} - {}".format(self.cid, len(bf_array)))

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

            # if self.cid == 'bank01':
            #     hash_accounts_array1.tofile("./hash1_accounts_bank1.csv")
            #     hash_accounts_array2.tofile("./hash2_accounts_bank1.csv")
            #     enc_hash_accounts1.tofile("./enc_hash1_accounts_bank1.csv")
            #     enc_hash_accounts2.tofile("./enc_hash2_accounts_bank1.csv")

            return FitRes(
                status=Status(code=Code.OK, message="Success"),
                parameters=ndarrays_to_parameters(data_sent),
                num_examples=1,
                metrics={},
            )
        else:
            # do nothing
            return FitRes(
                status=Status(code=Code.OK, message="Success"),
                parameters=ndarrays_to_parameters([]),
                num_examples=1,
                metrics={},
            )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        return EvaluateRes(
            status=Status(code=Code.OK, message="Success"),
            loss=1.0,
            num_examples=1,
            metrics={},
        )
