import os

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from loguru import logger
from hyperopt import fmin, tpe, hp, anneal, Trials
import numpy as np
from pathlib import Path


def join_flags_to_swift_data(swift_df, bank_df):
    acc_flag = pd.Series(bank_df.Flags.values, index=bank_df.Account).to_dict()
    swift_df['order_flag'] = swift_df['OrderingAccount'].map(acc_flag)
    swift_df['bene_flag'] = swift_df['BeneficiaryAccount'].map(acc_flag)

    return swift_df


def rule_mining(data, threshold):
    df = data.copy()
    df['BOFlags'] = df['bene_flag'].astype(str) + df['order_flag'].astype(str)
    df.groupby('BOFlags')['Label'].apply(lambda x: x.value_counts(normalize=True))

    all_rules = df.groupby('BOFlags')['Label'].apply(lambda x: x.value_counts(normalize=True)).to_dict()

    for key in all_rules.copy().keys():
        if key[1] == 0:  # remove the rules applied on normal transactions
            all_rules.pop(key, None)
        elif key[0].startswith('0.0') or key[0].endswith('0.0'):  # remove the rules by the one-side flag
            all_rules.pop(key, None)
        else:
            if all_rules[key] < threshold:  # remove the rules w/o enough support
                all_rules.pop(key, None)
            else:
                all_rules[key[0]] = all_rules.pop(key)

    # add new feature based on the rule dic
    if all_rules.keys():
        for i in range(len(list(all_rules.keys()))):
            df['Rule_' + str(i)] = [0 for _ in range(len(df))]
            ano_index = df[df['BOFlags'] == list(all_rules.keys())[i]].index
            df['Rule_' + str(i)].mask(df.index.isin(ano_index), 1, inplace=True)

    return df, all_rules


def generate_feature(df, pivot_name, new_feature_name, func, agg_col=None):
    if func == 'value_count':
        d = df[pivot_name].value_counts()
        d.name = new_feature_name
        df = df.merge(d, left_on=pivot_name, right_index=True)
    elif func == 'frequency':
        d = df[pivot_name].value_counts(normalize=True)
        d.name = new_feature_name
        df = df.merge(d, left_on=pivot_name, right_index=True)
    elif func == 'mean':
        d = df.groupby(pivot_name).agg(**{
            new_feature_name: pd.NamedAgg(column=agg_col, aggfunc='mean')})
        df = df.merge(d, left_on=pivot_name, right_index=True)
    elif func == 'n_unique':
        d = df.groupby(pivot_name).agg(**{
            new_feature_name: pd.NamedAgg(column=agg_col, aggfunc=lambda x: len(x.unique()))})
        df = df.merge(d, left_on=pivot_name, right_index=True)
        # test = df.merge(d, left_on=pivot_name, right_index=True)
    else:
        raise ValueError("func is not a valid option.")

    return df, d


def extract_feature(df, model_dir, phase):
    """
    from swift train and test data, extract more features
    """
    if phase == 'train':

        df["Timestamp"] = df["Timestamp"].astype("datetime64[ns]")

        logger.info("Extracting features")

        df["hour"] = df["Timestamp"].dt.hour.astype(str)

        ###########################################################################################
        # Hour frequency for each sender
        df['sender_hour'] = df['Sender'] + df['hour']
        df_merged, d_dict_df = generate_feature(
            df, pivot_name='sender_hour', new_feature_name='sender_hour_freq', func='value_count')

        # persistent pivot table to hdf5
        d_dict_df.to_pickle(Path.joinpath(model_dir, 'f1.pkl'))

        ###########################################################################################
        # currency freq
        df_merged["sender_currency"] = df_merged["Sender"] + df_merged["InstructedCurrency"]
        name = 'sender_currency'

        df_merged, d_dict_df = generate_feature(
            df_merged, pivot_name=name, new_feature_name=name + "_freq", func='value_count')

        # persistent pivot table to hdf5
        d_dict_df.to_pickle(Path.joinpath(model_dir, 'f2.pkl'))

        ###########################################################################################
        # currency avg amount
        df_merged, d_dict_df = generate_feature(
            df_merged, pivot_name=name, new_feature_name=name + "_avg_amount", func='mean', agg_col='InstructedAmount')

        # persistent pivot table to hdf5
        d_dict_df.to_pickle(Path.joinpath(model_dir, 'f3.pkl'))

        ###########################################################################################
        # Sender-Receiver Frequency
        df_merged["sender_receiver"] = df_merged["Sender"] + df_merged["Receiver"]

        df_merged, d_dict_df = generate_feature(
            df_merged, pivot_name='sender_receiver',
            new_feature_name="sender_receiver_freq", func='value_count')

        # persistent pivot table to hdf5
        d_dict_df.to_pickle(Path.joinpath(model_dir, 'f4.pkl'))

    elif phase == 'test':
        df["Timestamp"] = df["Timestamp"].astype("datetime64[ns]")
        df["hour"] = df["Timestamp"].dt.hour.astype(str)

        # read feature lookup table from file

        # sender_hour_freq
        d = pd.read_pickle(Path.joinpath(model_dir, 'f1.pkl'))
        pivot_name = 'sender_hour'
        df[pivot_name] = df['Sender'] + df['hour']
        df_merged = df.merge(d, how='left', left_on=pivot_name, right_index=True)

        # sender currency freq
        d = pd.read_pickle(Path.joinpath(model_dir, 'f2.pkl'))
        pivot_name = 'sender_currency'
        df_merged[pivot_name] = df_merged['Sender'] + df_merged["InstructedCurrency"]
        df_merged = df_merged.merge(d, how='left', left_on=pivot_name, right_index=True)

        # sender currency avg amount
        d = pd.read_pickle(Path.joinpath(model_dir, 'f3.pkl'))
        df_merged = df_merged.merge(d, how='left', left_on=pivot_name, right_index=True)

        # sender receiver freq
        d = pd.read_pickle(Path.joinpath(model_dir, 'f4.pkl'))
        pivot_name = 'sender_receiver'
        df_merged[pivot_name] = df_merged['Sender'] + df_merged['Receiver']
        df_merged = df_merged.merge(d, how='left', left_on=pivot_name, right_index=True)

        df_merged = df_merged.fillna(value=-1)

    else:
        logger.error("Phase is not correct, it should be 'train' or 'test'")
        raise ValueError

    columns_to_drop = [
        "UETR",
        "Sender",
        "Receiver",
        "TransactionReference",
        "OrderingAccount",
        "OrderingName",
        "OrderingStreet",
        "OrderingCountryCityZip",
        "BeneficiaryAccount",
        "BeneficiaryName",
        "BeneficiaryStreet",
        "BeneficiaryCountryCityZip",
        "SettlementDate",
        "SettlementCurrency",
        "InstructedCurrency",
        "Timestamp",
        "sender_hour",
        "sender_currency",
        'sender_receiver',
        #'Hash1',
        #'Hash2'
        # 'order_flag',
        # 'bene_flag',
        # 'BOFlags'
    ]

    df_merged['hour'] = df_merged['hour'].astype('int')

    return df_merged.drop(columns_to_drop, axis=1)


def compute_flag(row, bloom):
    hash1 = row['Hash1']
    hash2 = row['Hash2']
    if hash1 in bloom and hash2 in bloom:
        return False
    else:
        return True


def add_BF_feature2(swift_df, bank):
    logger.info("Extracting bloom filter features")
    from bloom_filter import BloomFilter
    bloom = BloomFilter(max_elements=bank.shape[0], error_rate=0.001)
    bank['Account'] = bank['Account'].astype(str)
    bank_nonflags = bank[bank['Flags'] == 0]
    ret = bank_nonflags['Account'].astype(str) + bank_nonflags['Name'].astype(str) + bank_nonflags['Street'].astype(
        str) + bank_nonflags['CountryCityZip'].astype(str)
    valid_accounts = ret.values.tolist()
    for account in valid_accounts:
        bloom.add(account)

    swift_df['BeneficiaryAccount'] = swift_df['BeneficiaryAccount'].astype(str)
    swift_df['OrderingAccount'] = swift_df['OrderingAccount'].astype(str)
    swift_df['Hash1'] = swift_df['BeneficiaryAccount'] + swift_df['BeneficiaryName'].astype(str) + swift_df[
        'BeneficiaryStreet'].astype(str) + swift_df['BeneficiaryCountryCityZip'].astype(str)
    swift_df['Hash2'] = swift_df['OrderingAccount'] + swift_df['OrderingName'].astype(str) + swift_df[
        'OrderingStreet'].astype(str) + swift_df['OrderingCountryCityZip'].astype(str)

    swift_df['BF'] = swift_df.apply(lambda row: compute_flag(row, bloom), axis=1).astype('bool')
    print(swift_df['BF'].value_counts())
    swift_df = swift_df.drop(['Hash1', 'Hash2'], axis=1)

    return swift_df


def add_BF_feature(swift_df, bank):
    logger.info("Extracting bloom filter features")

    swift_df['Hash1'] = swift_df['BeneficiaryAccount'] + swift_df['BeneficiaryName'] + swift_df[
        'BeneficiaryStreet'] + swift_df['BeneficiaryCountryCityZip']
    swift_df['Hash2'] = swift_df['OrderingAccount'] + swift_df['OrderingName'] + swift_df[
        'OrderingStreet'] + swift_df['OrderingCountryCityZip']

    bank_non_flagged = bank[bank['Flags'] == 0]
    bank_non_flagged['Hash'] = bank_non_flagged['Account'] + bank_non_flagged['Name'] + bank_non_flagged['Street'] + \
                               bank_non_flagged['CountryCityZip']

    swift_join1 = pd.merge(swift_df, bank_non_flagged, how='left', left_on=['BeneficiaryAccount'],
                           right_on=['Account'])
    swift_join2 = pd.merge(swift_df, bank_non_flagged, how='left', left_on=['OrderingAccount'],
                           right_on=['Account'])

    simple_anos_1 = swift_join1[(swift_join1['Hash1'] != swift_join1['Hash'])].index
    simple_anos_2 = swift_join2[(swift_join2['Hash2'] != swift_join2['Hash'])].index

    simple_anos = list(set(simple_anos_1).union(simple_anos_2))

    swift_df['BF'] = [0 for _ in range(len(swift_df))]
    swift_df['BF'].mask(swift_df.reset_index().index.isin(simple_anos), 1, inplace=True)

    swift_df = swift_df.drop(['Hash1', 'Hash2'], axis=1)

    return swift_df


def gb_xgb_cv(params, X, Y, random_state):
    params = {'n_estimators': int(params['n_estimators']),
              'max_depth': int(params['max_depth']),
              'learning_rate': params['learning_rate']
              }

    kfold = StratifiedKFold(n_splits=2, random_state=random_state, shuffle=True)
    model = XGBClassifier(random_state=random_state, **params)

    score = -cross_val_score(model, X, Y, cv=kfold, scoring="f1", n_jobs=-1).mean()

    return score


class SwiftModel:
    def __init__(self):
        pass

    def cv(self, X, Y):
        random_state = 40
        trials = Trials()
        space = {'n_estimators': hp.quniform('n_estimators', 100, 200, 40),
                 'max_depth': hp.quniform('max_depth', 2, 10, 2),
                 'learning_rate': hp.loguniform('learning_rate', -2, 0)
                 }
        best = fmin(fn=gb_xgb_cv,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=50,
                    trials=trials,
                    rstate=np.random.default_rng(random_state)
                    )
        return best

    def fit(self, X, y, cv_flag=None):
        if cv_flag:
            params = self.cv(X, y)
        else:
            params = {'learning_rate': 0.12419136825319058, 'max_depth': 6.0, 'n_estimators': 180.0}

        self.model = XGBClassifier(n_estimators=int(params['n_estimators']),
                                   max_depth=int(params['max_depth']), learning_rate=params['learning_rate'])

        self.model.fit(X, y)

        return self

    def predict(self, X):
        return pd.Series(self.pipeline.predict_proba(X)[:, 1], index=X.index)

    def save(self, path):
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path):
        inst = cls()
        inst.pipeline = joblib.load(path)
        return inst
