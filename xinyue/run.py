import os
import logging;

logging.basicConfig(level=logging.WARNING)
import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
import time
import gc
from keras import backend as K
from sklearn.preprocessing import StandardScaler
from myutils import Utils


class RunPipeline():
    def __init__(self, X_train, Y_train, X_test, Y_test, suffix = None, mode = 'rla', parallel= None,
                 generate_duplicates=True, n_samples_threshold=1000, seed = 0,
                 realistic_synthetic_mode = None,
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

        self.seed = seed
        # the suffix of all saved files
        if not os.path.exists('../../Desktop/Swift-PETs/result'):
            os.makedirs('../../Desktop/Swift-PETs/result')

        # ratio of labeled anomalies

        # model_dict (model_name: clf)
        self.model_dict = {}

        # unsupervised algorithms
        if self.parallel == 'unsupervise':
            from baseline.PyOD import PYOD
            from baseline.DAGMM.run import DAGMM

            # from pyod
            for _ in ['IForest', 'CBLOF', 'COF', 'COPOD', 'ECOD', 'FeatureBagging', 'HBOS', 'KNN', 'LODA',
                      'LOF', 'LSCP', 'MCD', 'PCA', 'SOD', 'SOGAAL', 'MOGAAL']:
                self.model_dict[_] = PYOD

            # DAGMM
            self.model_dict['DAGMM'] = DAGMM

        # semi-supervised algorithms
        elif self.parallel == 'semi-supervise':
            from baseline.PyOD import PYOD
            from baseline.GANomaly.run import GANomaly
            from baseline.REPEN.run import REPEN
            from baseline.DevNet.run import DevNet
            from baseline.PReNet.run import PReNet
            from baseline.FEAWAD.run import FEAWAD

            self.model_dict = {'GANomaly': GANomaly,
                                   'DevNet': DevNet,
                                   'PReNet': PReNet,
                                   'FEAWAD': FEAWAD}

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
                self.clf = self.clf(seed=self.seed, model_name=self.model_name)
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
            end_time = time.time();
            time_fit = end_time - start_time

            # predicting score (inference)
            start_time = time.time()
            if self.model_name == 'DAGMM':
                score_test = self.clf.predict_score(self.X_train, self.X_test)
            else:
                score_test = self.clf.predict_score(self.X_test)
            end_time = time.time();
            time_inference = end_time - start_time

            # performance
            result = self.utils.metric( y_true=self.Y_test, y_score=score_test, pos_label=1)

            K.clear_session()
            print(f"Model: {self.model_name}, AUC-ROC: {result['aucroc']}, AUC-PR: {result['aucpr']}")

            del self.clf
            gc.collect()

        except Exception as error:
            print(f'Error in model fitting. Model:{self.model_name}, Error: {error}')
            time_fit, time_inference = None, None
            result = {'TP': np.nan, 'aucroc': np.nan, 'aucpr': np.nan}
            pass

        return time_fit, time_inference, result

    # run the experiment
    def run(self):
        # save the results
        df_res = pd.DataFrame(data=None, index =list(self.model_dict.keys()),
                              columns = ['AUCROC', 'AUCPR','Time','Time_inference'])


        for i, model_name in tqdm(enumerate(self.model_dict.keys())):
            self.model_name = model_name
            print(model_name)
            self.clf = self.model_dict[self.model_name]

            # fit model
            time_fit, time_inference, result = self.model_fit()

            # save model
             # not implemented yet

            # store and save the result (AUC-ROC, AUC-PR and runtime / inference time)
            #df_res['TP'].iloc[i] = result['TP']
            df_res['AUCROC'].iloc[i] = result['aucroc']
            df_res['AUCPR'].iloc[i] = result['aucpr']
            df_res['Time'].iloc[i] = time_fit
            df_res['Time_inference'].iloc[i] = time_inference

            df_res.to_csv(
                os.path.join(os.getcwd(), '../../Desktop/Swift-PETs/result', self.parallel + '.csv'),
                index=True)

# run the above pipeline for reproducing the results in the paper

# train and test are the SWIFT_train and SWIFT_test with two more features: order_flag and Bene_flag

train = pd.read_csv('../../Desktop/Swift-PETs/data/train.csv')
test = pd.read_csv('../../Desktop/Swift-PETs/data/test.csv')

# NaN in the order_flag and Bene_flag is filled with (-1) this time
train = train.fillna(-1)
test = test.fillna(-1)

Y_train = train["Label"].values
X_train = train.drop(["Label"], axis=1).values
Y_test = test["Label"].values
X_test = test.drop(["Label"], axis=1).values

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

pipeline = RunPipeline(X_train, Y_train, X_test, Y_test, suffix='ADBench', parallel='semi-supervise', realistic_synthetic_mode=None, noise_type=None)
pipeline.run()