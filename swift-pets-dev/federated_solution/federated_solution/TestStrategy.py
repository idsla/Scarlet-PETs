import flwr as fl
from flwr.common import (
    EvaluateIns,
    FitIns,
    Parameters
)
from flwr.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays
import numpy as np
from .utils_advanced import (
    assembling_final_sum_with_hashed_accounts
)

def empty_parameters() -> Parameters:
    """Utility function that generates empty Flower Parameters dataclass instance."""
    return fl.common.ndarrays_to_parameters([])


class TestStrategy(fl.server.strategy.Strategy):
    """
    TestStrategy is a custom testing strategy from Flower framework. It is used by the Flower server.

    Args:
        server_dir (Path): Path to the server directory.

    Attributes:
        total_accounts (int): Total number of accounts.
        bloom_filters (dict): Encrypted bloom filters for each bank.
        hashed_accounts_enc (dict): Encrypted hashed accounts for each bank.
        bank_ids (list): Bank ids.
        public_keys_dict (dict): Public key dictionary of all clients.
        secure_sum (ndarray): Save encrypted secure sum computation result.
        final_sum_all (ndarray): Collection of encrtped sum using different session keys.
        banks_partition (dict): Store banks of each client partition.
        hashed_accounts (ndarray): Hashed accounts from swift.
        server_dir (Path): Path to the server directory.

    Methods
    -------
    save_stats()
        Save statistics to disk.
    initialize_parameters(client_manager)
        Initialize parameters.
    configure_fit(server_round, parameters, client_manager)
        Configure fit.
    fit(server_round, parameters, config, client_manager)
        Fit.
    evaluate(parameters, config, client_manager)
        Evaluate.
    """

    def __init__(self, server_dir):
        self.total_accounts = 0
        self.bloom_filters = {}  # encrypted bloom filters for each bank
        self.hashed_accounts_enc = {}
        self.bank_ids = None  # bank ids
        self.public_keys_dict = {}  # public key dictionary of all clients
        self.secure_sum = None  # save encrypted secure sum computation result
        self.final_sum_all = None  # collection of encrtped sum using different session keys
        self.banks_partition = None  # store banks of each client partition
        self.hashed_accounts = None  # hashed accounts from swift
        self.server_dir = server_dir
        super().__init__()

    def initialize_parameters(self, client_manager):
        """
        Initialize parameters.
        Parameters
        ----------
        client_manager: fl.server.client_manager.ClientManager

        Returns
        -------
        fl.common.Parameters
        """
        # Your implementation here
        # empty parameters
        server_parameters = [np.array([])]
        clients = client_manager.all()
        self.bank_ids = [cid for cid in clients.keys() if cid != 'swift']
        return ndarrays_to_parameters(server_parameters)

    def configure_fit(self, server_round, parameters, client_manager):
        """
        Configure fit. This method is called by the Flower server before each round of training.
        Parameters
        ----------
        server_round : int
        parameters : fl.common.Parameters
        client_manager : fl.server.client_manager.ClientManager

        Returns
        -------
        List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitIns]]
        """
        # get all clients
        clients = client_manager.all()
        bank_cids = [cid for cid in clients.keys() if cid != 'swift']
        num_banks = len(bank_cids)

        # first round -> tell clients to send public key
        if server_round == 1:
            ret = []
            for cid in clients.keys():
                config = {'task': 'send_public_key'}
                ret.append((clients[cid], FitIns(parameters, config)))
            return ret
        # second round -> tell clients to share bank in partition
        elif server_round == 2:
            config = {'task': 'share_bank_in_partition', 'key': self.public_keys_dict['swift']}
            ret = []
            for cid in clients.keys():
                if cid != 'swift':
                    ret.append((clients[cid], FitIns(parameters, config)))
            return ret
        # third round -> tell swift clients bank partitions
        elif server_round == 3:
            config = {'task': 'send_swift_banks_in_partitions'}
            for cid in bank_cids:
                config[cid] = self.public_keys_dict[cid]
            ret = [
                (clients['swift'], FitIns(ndarrays_to_parameters(self.banks_partition), config))
            ]
            return ret
        # next num_banks rounds -> compute secure sum -> bank1 L = s0 + R -> bank2 L + S1 + -> ... -> bankN L + SN
        elif server_round < 4 + num_banks:
            curr_bank_idx = server_round - 4
            curr_bank_cid = bank_cids[curr_bank_idx]
            next_bank_cid = bank_cids[(curr_bank_idx + 1) % num_banks]
            config = {'task': 'secure_sum', 'key': self.public_keys_dict[next_bank_cid]}
            if self.secure_sum is None:
                parameters = ndarrays_to_parameters([])
            else:
                parameters = ndarrays_to_parameters(self.secure_sum)
            ret = [(clients[curr_bank_cid], FitIns(parameters, config))]
            return ret

        # next round send bank1 (SUM + R) to subtract R, then encrypt it with all other's public key
        elif server_round == 4 + num_banks:
            curr_bank_cid = bank_cids[0]
            config = {'task': 'compute_final_sum'}
            for k, v in self.public_keys_dict.items():  # send all other parties public keys
                config[k] = v
            parameters = ndarrays_to_parameters(self.secure_sum)
            ret = [(clients[curr_bank_cid], FitIns(parameters, config))]
            return ret

        # build local bloomfilter
        elif server_round == 4 + num_banks + 1:
            ret = []
            for cid in clients.keys():
                config = {'task': 'build-local-bloomfilter', 'key': self.public_keys_dict['swift']}
                data_sent = assembling_final_sum_with_hashed_accounts(self.final_sum_all, self.hashed_accounts)
                parameters = ndarrays_to_parameters(data_sent)
                ret.append(
                    (clients[cid], FitIns(parameters, config))
                )
            return ret
        else:
            pass

    def aggregate_fit(self, server_round, results, failures):
        """
        Aggregate fit.
        Parameters
        ----------
        server_round : int
        results : List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]]
        failures : List[Tuple[fl.server.client_proxy.ClientProxy, Exception]]

        Returns
        -------
        fl.common.Parameters, Dict[str, Any]
        """

        num_banks = len(self.bank_ids)

        # collect all public keys from clients
        if server_round == 1:
            # [party_id1, party_public_key1,...]
            for client, FitRes in results:
                public_key = parameters_to_ndarrays(FitRes.parameters)[0][0]
                cid = client.cid
                self.public_keys_dict[cid] = public_key
            return empty_parameters(), {}
        # aggregate banks of all clients partition
        elif server_round == 2:
            ret = []
            for client, FitRes in results:
                data_enc = parameters_to_ndarrays(FitRes.parameters)
                ret.append(np.array(client.cid))
                ret.extend(data_enc)
            self.banks_partition = ret
            return empty_parameters(), {}
        elif server_round == 3:
            # print("Agg round {} - Results: {}".format(server_round, results))
            for client, FitRes in results:
                if client.cid == 'swift':
                    data_received = parameters_to_ndarrays(FitRes.parameters)
                    self.hashed_accounts = data_received
            return empty_parameters(), {}
        # update secure sum
        elif server_round < 4 + num_banks:
            received_value = False
            for client, FitRes in results:
                received_data_enc = parameters_to_ndarrays(FitRes.parameters)
                # print(received_data)
                if len(received_data_enc) > 0:
                    self.secure_sum = received_data_enc
                    # print("secure_sum {}".format(self.secure_sum))
                    received_value = True
            if not received_value:
                raise ValueError("Error in aggregation at server {}".format(server_round))
            return empty_parameters(), {}

        # save final sum all from bank1
        elif server_round == 4 + num_banks:
            received_value = False
            for client, FitRes in results:
                received_data_enc = parameters_to_ndarrays(FitRes.parameters)
                if len(received_data_enc) > 0:
                    self.final_sum_all = received_data_enc
                    received_value = True
            if not received_value:
                raise ValueError("Error in aggregation at server {}".format(server_round))
            return empty_parameters(), {}

        # aggregate local bloom filters
        elif server_round == 4 + num_banks + 1:
            for client, FitRes in results:
                if client.cid != 'swift':
                    data_enc = parameters_to_ndarrays(FitRes.parameters)
                    self.bloom_filters[client.cid] = data_enc  # bloom filter + encrypted hash accounts
            return empty_parameters(), {}
        else:
            return empty_parameters(), {}

    def configure_evaluate(self, server_round, parameters, client_manager):
        """
        Configure evaluate. This method is called before the evaluate method.
        Parameters
        ----------
        server_round : int
        parameters : fl.common.Parameters
        client_manager : fl.server.client_manager.ClientManager

        Returns
        -------
        List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitIns]]
        """

        # get all clients
        clients = client_manager.all()
        bank_cids = [cid for cid in clients.keys() if cid != 'swift']
        num_banks = len(bank_cids)

        # send encrypted bloom filter to swift
        if server_round == 4 + num_banks + 1:
            ret = []
            config = {'task': 'swift_run_test'}
            data_sent = []
            #print(len(self.bloom_filters))
            for cid, bf_enc in self.bloom_filters.items():
                data_sent.extend(bf_enc)
            parameters = ndarrays_to_parameters(data_sent)
            ret.append((clients['swift'], FitIns(parameters, config)))
            return ret

        else:
            return []

    def aggregate_evaluate(self, server_round, results, failures):
        # Your implementation here
        return empty_parameters(), {}

    def evaluate(self, server_round, parameters):
        # Your implementation here
        return None
