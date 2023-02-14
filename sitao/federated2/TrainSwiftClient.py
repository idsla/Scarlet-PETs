from pathlib import Path
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
    Code
)
from flwr.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays
import numpy as np
from Crypto.PublicKey import RSA

from .utils import (
    bytes_to_str, str_to_bytes, decrypt_data_and_session_key, encrypt_data_and_session_key, search_data_enc,
    build_bloomfilter, bf_array_to_bytestring, bytestring_to_bf_array, OR_arrays, encryption_bf, decryption_bf
)
from .bloom_filter import BloomFilter
from .model3 import (
    extract_feature, SwiftModel, add_BF_feature2
)
from .utils_new import (
    decrypt_bytes_with_private_key,
    encrypt_bytes_with_public_key,
    ndarray_to_byte_string,
    byte_string_to_ndarray,
    get_hashed_accounts_of_banks,
    update_hashed_accounts_dict
)


class TrainSwiftClient(fl.client.Client):
    def __init__(
            self,
            cid,
            client_dir,
            data,
            session_key_length=16,
            error_rate=0.001,
            public_key_size=2048
    ):
        super().__init__()
        self.cid = cid
        self.client_dir = client_dir
        self.cid_bytes = str_to_bytes(self.cid)
        self.data = data
        self.error_rate = error_rate
        self.bloom = None
        self.total_accounts = 0
        self.session_key_length = session_key_length

        # load states
        self.session_key_dict = None
        self.public_key = None
        self.private_key = None
        self.internal_state = None
        self.public_key_size = public_key_size
        self.load_state()

    def load_state(self):
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

        # send public key
        if task == 'send_public_key':
            return FitRes(
                status=Status(code=Code.OK, message='Success'),
                parameters=ndarrays_to_parameters([np.array([self.public_key.export_key()])]),
                num_examples=1,
                metrics={}
            )
        elif task == 'send_swift_banks_in_partitions':
            client_banks_partition = {}
            data_received = parameters_to_ndarrays(global_parameters)
            for i in range(len(data_received) // 4):
                cid = data_received[4 * i].item()
                data_enc = [data_received[4 * i + 1], data_received[4 * i + 2], data_received[4 * i + 3]]
                data_dec = decrypt_bytes_with_private_key(data_enc, self.private_key)
                banks = byte_string_to_ndarray(data_dec)
                #print(cid, banks)
                client_banks_partition[cid] = banks  # make it to be a list

            # send hashed accounts of each bank to server
            data_sent = []
            for key, public_key in config.items():
                if key != 'task':
                    cid = key
                    banks = client_banks_partition[cid]
                    hash1_array, hash2_array = get_hashed_accounts_of_banks(self.data, banks)
                    #print(cid, hash1_array.shape, hash2_array.shape)
                    data_sent.append(np.array(cid))
                    data_sent.append(hash1_array)
                    data_sent.append(hash2_array)

            return FitRes(
                status=Status(code=Code.OK, message='Success'),
                parameters=ndarrays_to_parameters(data_sent),
                num_examples=1,
                metrics={}
            )

        elif task == 'build-local-bloomfilter':
            # get total sum
            data_received = parameters_to_ndarrays(global_parameters)
            data_enc = search_data_enc(data_received, self.cid)[0:3]
            total_sum = decrypt_data_and_session_key(data_enc, self.private_key)
            if total_sum is None:
                raise ValueError('Total sum cannot be decryption in {}'.format(self.cid))

            swift_internal_state = {'total_sum': total_sum}
            self.save_state(swift_internal_state, 'internal_state.pkl')
            return FitRes(
                status=Status(code=Code.OK, message='Success'),
                parameters=ndarrays_to_parameters([]),
                num_examples=1,
                metrics={}
            )

        elif task == 'swift_run_train':
            if 'total_sum' in self.internal_state:
                total_sum = self.internal_state['total_sum']
            else:
                raise ValueError('internal state do not have total sum something wrong.')

            bloom = BloomFilter(total_sum, error_rate=self.error_rate)

            # decrypt and OR all bloom arrays
            data_received = parameters_to_ndarrays(global_parameters)
            bloom_arrays = []
            hashed_accounts_dict1 = {}
            hashed_accounts_dict2 = {}
            #print(len(data_received))
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

            # Training code here
            #logger.info("Start Training ...")
            swift_df = self.data
            #logger.info("Adding BF and extracting features")
            swift_df = add_BF_feature2(swift_df, bloom, hashed_accounts_dict1, hashed_accounts_dict2)
            swift_df = extract_feature(swift_df, self.client_dir, phase='train', epsilon=0.25, dp_flag=True)
            swift_model = SwiftModel()
            #logger.info("Fitting SWIFT model...")
            swift_model.fit(X=swift_df.drop(['Label'], axis=1), y=swift_df["Label"])
            swift_model.save(Path.joinpath(self.client_dir, "swift_model.joblib"))

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
        return EvaluateRes(
            status=Status(code=Code.OK, message="Success"),
            loss=1.0,
            num_examples=1,
            metrics={},
        )
