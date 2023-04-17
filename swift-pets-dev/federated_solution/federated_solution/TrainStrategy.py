import flwr as fl
from flwr.common import (
    EvaluateIns,
    FitIns,
    Parameters,
)
from flwr.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays
import numpy as np
from pathlib import Path
import json
from .utils_advanced import (
    assembling_final_sum_with_hashed_accounts,
    compute_data_capacity
)
import time
import tracemalloc

def empty_parameters() -> Parameters:
    """Utility function that generates empty Flower Parameters dataclass instance."""
    return fl.common.ndarrays_to_parameters([])


class TrainStrategy(fl.server.strategy.Strategy):
    """
    TrainStrategy is a custom training strategy from Flower framework. It is used by the Flower server.

    Args:
        server_dir (Path): Path to the server directory.

    Attributes:
        total_accounts (int): Total number of accounts.
        bloom_filters (dict): Encrypted bloom filters for each bank.
        hashed_accounts_enc (dict): Encrypted hashed accounts for each bank.
        bank_ids (list): Bank ids.
        public_keys_dict (dict): Public key dictionary of all clients.
        secure_sum (ndarray): Save encrypted secure sum computation result.
        final_sum_all (list): Collection of encrtped sum using different session keys.
        banks_partition (dict): Store banks of each client partition.
        hashed_accounts (list): Hashed accounts from swift.
        server_dir (Path): Path to the server directory.
        stats (dict): Dictionary to store statistics.

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

        file_path = Path.joinpath(server_dir, "stats.json")
        if file_path.exists():
            with file_path.open(mode='r') as f:
                self.stats = json.load(f)
        else:
            self.stats = {}
        super().__init__()

    def save_stats(self):
        """Save statistics to disk."""
        file_path = Path.joinpath(self.server_dir, 'stats.json')
        with file_path.open('w') as f:
            json.dump(self.stats, f)

    def initialize_parameters(self, client_manager):
        """Initialize parameters."""
        # Your implementation here
        # empty parameters
        server_parameters = [np.array([])]
        clients = client_manager.all()
        self.bank_ids = [cid for cid in clients.keys() if cid != 'swift']
        return ndarrays_to_parameters(server_parameters)

    def configure_fit(self, server_round, parameters, client_manager):
        """
        Configure fit.
        This function is called by the Flower server before the fit function is called to give client fit instructions.

        Parameters
        ----------
        server_round: int server round
        parameters: Parameters global parameters
        client_manager: ClientManager client manager

        Returns
        -------
        list of tuples (client, FitIns)
        """
        tracemalloc.start()
        start = time.time()
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

            end = time.time()
            self.stats['configure_fit']['time'][str(server_round)] = end - start
            self.stats['configure_fit']['memory'][str(server_round)] = tracemalloc.get_traced_memory()[1]/(1024*1024)
            self.stats['configure_fit']['network'][str(server_round)] = compute_data_capacity([])
            self.save_stats()
            tracemalloc.stop()

            return ret
        # second round -> tells banks to share their banks partition
        elif server_round == 2:
            print(self.public_keys_dict)
            config = {'task': 'share_bank_in_partition', 'key': self.public_keys_dict['swift']}
            ret = []
            for cid in clients.keys():
                if cid != 'swift':
                    ret.append((clients[cid], FitIns(parameters, config)))

            end = time.time()
            self.stats['configure_fit']['time'][str(server_round)] = end - start
            self.stats['configure_fit']['memory'][str(server_round)] = tracemalloc.get_traced_memory()[1] / (
                        1024 * 1024)
            self.stats['configure_fit']['network'][str(server_round)] = compute_data_capacity([])
            self.save_stats()
            tracemalloc.stop()

            return ret
        # third round -> send swift banks partition
        elif server_round == 3:
            config = {'task': 'send_swift_banks_in_partitions'}
            for cid in bank_cids:
                config[cid] = self.public_keys_dict[cid]
            ret = [
                (clients['swift'], FitIns(ndarrays_to_parameters(self.banks_partition), config))
            ]

            end = time.time()
            self.stats['configure_fit']['time'][str(server_round)] = end - start
            self.stats['configure_fit']['memory'][str(server_round)] = tracemalloc.get_traced_memory()[1] / (
                    1024 * 1024)
            self.stats['configure_fit']['network'][str(server_round)] = compute_data_capacity([self.banks_partition])
            tracemalloc.stop()
            self.save_stats()

            return ret
        # next num_banks rounds -> compute secure sum -> bank1 L = s0 + R -> bank2 L + S1 + -> ... -> bankN L + SN
        elif server_round < 4 + num_banks:
            curr_bank_idx = server_round - 4
            curr_bank_cid = bank_cids[curr_bank_idx]
            next_bank_cid = bank_cids[(curr_bank_idx + 1) % num_banks]
            config = {'task': 'secure_sum', 'key': self.public_keys_dict[next_bank_cid]}
            if self.secure_sum is None:
                parameters = ndarrays_to_parameters([])
                self.stats['configure_fit']['network'][str(server_round)] = compute_data_capacity([])
            else:
                parameters = ndarrays_to_parameters(self.secure_sum)
                self.stats['configure_fit']['network'][str(server_round)] = compute_data_capacity([self.secure_sum])
            ret = [(clients[curr_bank_cid], FitIns(parameters, config))]

            end = time.time()
            self.stats['configure_fit']['time'][str(server_round)] = end - start
            self.stats['configure_fit']['memory'][str(server_round)] = tracemalloc.get_traced_memory()[1] / (
                    1024 * 1024)
            tracemalloc.stop()
            self.save_stats()

            return ret

        # next round send bank1 (SUM + R) to subtract R, then encrypt it with all other's public key
        elif server_round == 4 + num_banks:
            curr_bank_cid = bank_cids[0]
            config = {'task': 'compute_final_sum'}
            for k, v in self.public_keys_dict.items():  # send all other parties public keys
                config[k] = v
            parameters = ndarrays_to_parameters(self.secure_sum)
            ret = [(clients[curr_bank_cid], FitIns(parameters, config))]

            end = time.time()
            self.stats['configure_fit']['time'][str(server_round)] = end - start
            self.stats['configure_fit']['memory'][str(server_round)] = tracemalloc.get_traced_memory()[1] / (
                        1024 * 1024)
            self.stats['configure_fit']['network'][str(server_round)] = compute_data_capacity([self.secure_sum])
            self.save_stats()
            tracemalloc.stop()

            return ret

        # build local bloomfilter
        elif server_round == 4 + num_banks + 1:
            ret = []
            size = 0
            for cid in clients.keys():
                config = {'task': 'build-local-bloomfilter', 'key': self.public_keys_dict['swift']}
                data_sent = assembling_final_sum_with_hashed_accounts(self.final_sum_all, self.hashed_accounts)
                parameters = ndarrays_to_parameters(data_sent)
                size += compute_data_capacity(data_sent)
                ret.append(
                    (clients[cid], FitIns(parameters, config))
                )

            end = time.time()
            self.stats['configure_fit']['time'][str(server_round)] = end - start
            self.stats['configure_fit']['memory'][str(server_round)] = tracemalloc.get_traced_memory()[1] / (
                    1024 * 1024)
            self.stats['configure_fit']['network'][str(server_round)] = size
            self.save_stats()
            tracemalloc.stop()

            return ret

        # send encrypted bloom filter to swift
        elif server_round == 4 + num_banks + 2:
            ret = []
            config = {'task': 'swift_run_train'}
            data_sent = []
            for cid, bf_enc in self.bloom_filters.items():
                data_sent.extend(bf_enc)
            parameters = ndarrays_to_parameters(data_sent)
            ret.append((clients['swift'], FitIns(parameters, config)))

            end = time.time()
            self.stats['configure_fit']['time'][str(server_round)] = end - start
            self.stats['configure_fit']['memory'][str(server_round)] = tracemalloc.get_traced_memory()[1] / (
                    1024 * 1024)
            self.stats['configure_fit']['network'][str(server_round)] = compute_data_capacity(data_sent)
            self.save_stats()
            tracemalloc.stop()

            return ret
        else:
            pass

    def aggregate_fit(self, server_round, results, failures):
        """
        Aggregate the results from the clients
        Parameters
        ----------
        server_round: int - current round
        results: list of (client, FitRes) - results from clients
        failures: list of (client, exception) - failed clients

        Returns
        -------
        global parameters, global config
        """
        start = time.time()
        tracemalloc.start()
        num_banks = len(self.bank_ids)

        # collect all public keys from clients
        if server_round == 1:
            size = 0
            # [party_id1, party_public_key1,...]
            for client, FitRes in results:
                public_key = parameters_to_ndarrays(FitRes.parameters)[0][0]
                cid = client.cid
                self.public_keys_dict[cid] = public_key
                size += compute_data_capacity(public_key)

            end = time.time()
            self.stats['aggregate_fit']['time'][str(server_round)] = end - start
            self.stats['aggregate_fit']['memory'][str(server_round)] = tracemalloc.get_traced_memory()[1] / (
                    1024 * 1024)
            self.stats['aggregate_fit']['network'][str(server_round)] = size
            self.save_stats()
            tracemalloc.stop()

            return empty_parameters(), {}
        # aggregate banks of all clients partition
        elif server_round == 2:
            ret = []
            for client, FitRes in results:
                data_enc = parameters_to_ndarrays(FitRes.parameters)
                ret.append(np.array(client.cid))
                ret.extend(data_enc)

            end = time.time()
            self.stats['aggregate_fit']['time'][str(server_round)] = end - start
            self.stats['aggregate_fit']['memory'][str(server_round)] = tracemalloc.get_traced_memory()[1] / (
                    1024 * 1024)
            self.stats['aggregate_fit']['network'][str(server_round)] = compute_data_capacity(ret)
            self.save_stats()
            tracemalloc.stop()

            self.banks_partition = ret
            return empty_parameters(), {}
        elif server_round == 3:
            for client, FitRes in results:
                if client.cid == 'swift':
                    data_received = parameters_to_ndarrays(FitRes.parameters)
                    self.hashed_accounts = data_received

            end = time.time()
            self.stats['aggregate_fit']['time'][str(server_round)] = end - start
            self.stats['aggregate_fit']['memory'][str(server_round)] = tracemalloc.get_traced_memory()[1] / (
                    1024 * 1024)
            self.stats['aggregate_fit']['network'][str(server_round)] = compute_data_capacity(data_received)
            self.save_stats()
            tracemalloc.stop()

            return empty_parameters(), {}
        # update secure sum
        elif server_round < 4 + num_banks:
            received_value = False
            for client, FitRes in results:
                received_data_enc = parameters_to_ndarrays(FitRes.parameters)
                if len(received_data_enc) > 0:
                    self.secure_sum = received_data_enc
                    received_value = True

            end = time.time()
            self.stats['aggregate_fit']['time'][str(server_round)] = end - start
            self.stats['aggregate_fit']['memory'][str(server_round)] = tracemalloc.get_traced_memory()[1] / (
                    1024 * 1024)
            self.stats['aggregate_fit']['network'][str(server_round)] = compute_data_capacity(received_data_enc)
            self.save_stats()
            tracemalloc.stop()

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

            end = time.time()
            self.stats['aggregate_fit']['time'][str(server_round)] = end - start
            self.stats['aggregate_fit']['memory'][str(server_round)] = tracemalloc.get_traced_memory()[1] / (
                    1024 * 1024)
            self.stats['aggregate_fit']['network'][str(server_round)] = compute_data_capacity(received_data_enc)
            self.save_stats()
            tracemalloc.stop()

            if not received_value:
                raise ValueError("Error in aggregation at server {}".format(server_round))
            return empty_parameters(), {}

        # aggregate local bloom filters
        elif server_round == 4 + num_banks + 1:
            size = 0
            for client, FitRes in results:
                if client.cid != 'swift':
                    data_enc = parameters_to_ndarrays(FitRes.parameters)
                    self.bloom_filters[client.cid] = data_enc  # bloom filter + encrypted hash accounts
                    size += compute_data_capacity(data_enc)

            end = time.time()
            self.stats['aggregate_fit']['time'][str(server_round)] = end - start
            self.stats['aggregate_fit']['memory'][str(server_round)] = tracemalloc.get_traced_memory()[1] / (
                    1024 * 1024)
            self.stats['aggregate_fit']['network'][str(server_round)] = size
            self.save_stats()
            tracemalloc.stop()

            return empty_parameters(), {}
        else:
            return empty_parameters(), {}

    def configure_evaluate(self, server_round, parameters, client_manager):
        # Your implementation here
        return []

    def aggregate_evaluate(self, server_round, results, failures):
        # Your implementation here
        return empty_parameters(), {}

    def evaluate(self, server_round, parameters):
        # Your implementation here
        return None


if __name__ == '__main__':
    pass
