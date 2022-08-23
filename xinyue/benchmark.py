import os
import logging; logging.basicConfig(level=logging.WARNING)
import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
import time
import gc
from keras import backend as K

from myutils import Utils

class RunPipeline():
    def __init__(self, suffix:str=None, mode:str='rla', parallel:str=None,X_train, Y_train, X_test, Y_test,
                 generate_duplicates=True, n_samples_threshold=1000,
                 realistic_synthetic_mode:str=None,
                 noise_type=None):
        '''
        :param suffix: saved file suffix (including the model performance result and model weights)
        :param mode: rla or nla —— ratio of labeled anomalies or number of labeled anomalies
        :param parallel: unsupervise, semi-supervise or supervise, choosing to parallelly run the code
        :param generate_duplicates: whether to generate duplicated samples when sample size is too small
        :param n_samples_threshold: threshold for generating the above duplicates, if generate_duplicates is False, then datasets with sample size smaller than n_samples_threshold will be dropped
        :param realistic_synthetic_mode: local, global, dependency or cluster —— whether to generate the realistic synthetic anomalies to test different algorithms
        :param noise_type: duplicated_anomalies, irrelevant_features or label_contamination —— whether to test the model robustness
        '''

        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

        # utils function
        self.utils = Utils()

        self.mode = mode
        self.parallel = parallel

        # the suffix of all saved files
        if not os.path.exists('result'):
            os.makedirs('result')

        # ratio of labeled anomalies

        # model_dict (model_name: clf)
        self.model_dict = {}

        # unsupervised algorithms
        if self.parallel == 'unsupervise':
            from baseline.PyOD import PYOD
            from baseline.DAGMM.run import DAGMM

            # from pyod
            for _ in ['IForest', 'OCSVM', 'CBLOF', 'COF', 'COPOD', 'ECOD', 'FeatureBagging', 'HBOS', 'KNN', 'LODA',
                      'LOF', 'LSCP', 'MCD', 'PCA', 'SOD', 'SOGAAL', 'MOGAAL']:
                self.model_dict[_] = PYOD

            # DAGMM
            self.model_dict['DAGMM'] = DAGMM

        # semi-supervised algorithms
        elif self.parallel == 'semi-supervise':
            from baseline.PyOD import PYOD
            from baseline.GANomaly.run import GANomaly
            from baseline.DeepSAD.src.run import DeepSAD
            from baseline.REPEN.run import REPEN
            from baseline.DevNet.run import DevNet
            from baseline.PReNet.run import PReNet
            from baseline.FEAWAD.run import FEAWAD

            self.model_dict = {'REPEN': REPEN,
                               'XGBOD': PYOD}

        # fully-supervised algorithms
        elif self.parallel == 'supervise':
            from baseline.Supervised import supervised
            from baseline.FTTransformer.run import FTTransformer

            # from sklearn
            for _ in ['LR', 'NB', 'SVM', 'MLP', 'RF', 'LGB', 'XGB', 'CatB']:
                self.model_dict[_] = supervised
        else:
            raise NotImplementedError

        # We remove the following model for considering the computational cost
        for _ in ['SOGAAL', 'MOGAAL', 'LSCP', 'MCD', 'FeatureBagging']:
            if _ in self.model_dict.keys():
                self.model_dict.pop(_)


    # whether the dataset in the NLP / CV dataset
    # currently we have 5 NLP datasets and 5 CV datasets
    def isin_NLPCV(self, dataset):
        NLPCV_list = ['agnews', 'amazon', 'imdb', 'yelp', '20news',
                      'MNIST-C', 'FashionMNIST', 'CIFAR10', 'SVHN', 'MVTec-AD']

        return any([_ in dataset for _ in NLPCV_list])

    # model fitting function
    def model_fit(self):
        try:
            # model initialization, if model weights are saved, the save_suffix should be specified
            if self.model_name in ['DevNet', 'FEAWAD', 'REPEN']:
                self.clf = self.clf(seed=self.seed, model_name=self.model_name, save_suffix=self.suffix)
            else:
                self.clf = self.clf(seed=self.seed, model_name=self.model_name)

        except Exception as error:
            print(f'Error in model initialization. Model:{self.model_name}, Error: {error}')
            pass

        try:
            # fitting
            start_time = time.time()
            self.clf = self.clf.fit(X_train=self.X_train, y_train=self.Y_train,
                                    ratio=sum(self.Y_test) / len(self.Y_test))
            end_time = time.time(); time_fit = end_time - start_time


            # predicting score (inference)
            start_time = time.time()
            if self.model_name == 'DAGMM':
                score_test = self.clf.predict_score(self.X_train, self.X_test)
            else:
                score_test = self.clf.predict_score(self.X_test)
            end_time = time.time(); time_inference = end_time - start_time

            # performance
            result = self.utils.metric(y_true=self.Y_test, y_score=score_test, pos_label=1)

            K.clear_session()
            print(f"Model: {self.model_name}, AUC-ROC: {result['aucroc']}, AUC-PR: {result['aucpr']}")

            del self.clf
            gc.collect()

        except Exception as error:
            print(f'Error in model fitting. Model:{self.model_name}, Error: {error}')
            time_fit, time_inference = None, None
            result = {'TP':np.nan, 'aucroc': np.nan, 'aucpr': np.nan}
            pass

        return time_fit, time_inference, result

    # run the experiment
    def run(self):
        #  filteting dataset that does not meet the experimental requirements
        dataset_list = ['With_flag']

        # experimental parameters
        if self.mode == 'nla':
            if self.noise_type is not None:
                experiment_params = list(product(dataset_list, self.nla_list, self.noise_params_list, self.seed_list))
            else:
                experiment_params = list(product(dataset_list, self.nla_list, self.seed_list))
        else:
            if self.noise_type is not None:
                experiment_params = list(product(dataset_list, self.rla_list, self.noise_params_list, self.seed_list))
            else:
                experiment_params = dataset_list

        print(f'{len(dataset_list)} datasets, {len(self.model_dict.keys())} models')

        # save the results
        df_TP = pd.DataFrame(data=None, index=experiment_params, columns=list(self.model_dict.keys()))
        df_AUCROC = pd.DataFrame(data=None, index=experiment_params, columns=list(self.model_dict.keys()))
        df_AUCPR = pd.DataFrame(data=None, index=experiment_params, columns=list(self.model_dict.keys()))
        df_time_fit = pd.DataFrame(data=None, index=experiment_params, columns=list(self.model_dict.keys()))
        df_time_inference = pd.DataFrame(data=None, index=experiment_params, columns=list(self.model_dict.keys()))

        for i, params in tqdm(enumerate(experiment_params)):
            if self.noise_type is not None:
                dataset, la, noise_param, self.seed = params
            else:
                dataset, la, self.seed = params

            if self.parallel == 'unsupervise' and la != 0.0 and self.noise_type is None:
                continue

            # We only run one time on CV / NLP datasets for considering computational cost
            # The final results are the average performance on different classes
            if self.isin_NLPCV(dataset) and self.seed > 1:
                continue

            print(f'Current experiment parameters: {params}')

            # generate data


            for model_name in tqdm(self.model_dict.keys()):
                self.model_name = model_name
                self.clf = self.model_dict[self.model_name]

                # fit model
                time_fit, time_inference, result = self.model_fit()

                # store and save the result (AUC-ROC, AUC-PR and runtime / inference time)
                df_TP[model_name].iloc[i] = result['TP']
                df_AUCROC[model_name].iloc[i] = result['aucroc']
                df_AUCPR[model_name].iloc[i] = result['aucpr']
                df_time_fit[model_name].iloc[i] = time_fit
                df_time_inference[model_name].iloc[i] = time_inference

                df_AUCROC.to_csv(os.path.join(os.getcwd(), 'result', 'AUCROC_' + self.suffix + '.csv'), index=True)
                df_AUCPR.to_csv(os.path.join(os.getcwd(), 'result', 'AUCPR_' + self.suffix + '.csv'), index=True)
                df_time_fit.to_csv(os.path.join(os.getcwd(), 'result', 'Time(fit)_' + self.suffix + '.csv'), index=True)
                df_time_inference.to_csv(os.path.join(os.getcwd(), 'result', 'Time(inference)_' + self.suffix + '.csv'), index=True)

# run the above pipeline for reproducing the results in the paper
pipeline = RunPipeline(suffix='ADBench', parallel='unsupervise', realistic_synthetic_mode=None, noise_type=None, X_train, Y_train, X_test, Y_test)
pipeline.run()