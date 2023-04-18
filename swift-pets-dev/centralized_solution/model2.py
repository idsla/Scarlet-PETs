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
import hashlib  # TODO:import hash function as crypt_hash


def join_flags_to_pns_data(pns_df, bank_df):
	"""
	joint the flags to the pns data
	Parameters
	----------
	pns_df : pd.DataFrame
	bank_df : pd.DataFrame

	Returns
	-------
	pns_df : pd.DataFrame
	"""
	acc_flag = pd.Series(bank_df.Flags.values, index=bank_df.Account).to_dict()
	pns_df['order_flag'] = pns_df['OrderingAccount'].map(acc_flag)
	pns_df['bene_flag'] = pns_df['BeneficiaryAccount'].map(acc_flag)

	return pns_df


def rule_mining(data, threshold):
	"""
	Rule mining based on the flags
	Parameters
	----------
	data: pd.DataFrame
	threshold: float

	Returns
	-------
	all_rules: dict
	"""
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


def laplace_mech(v, sensitivity, epsilon):
	return np.round(v + np.random.laplace(loc=0, scale=sensitivity / epsilon))


def ls_at_distance(df, u, k):
	return np.abs(u / (len(df) - k + 1))


def smooth_sens(df, k, epsilon):
	delta = 1 / len(df) ** 2
	u = df.max()

	beta = epsilon / (2 * np.log(2 / delta))
	r = [np.exp(- beta * k) * ls_at_distance(df, u, k) for k in range(0, k)]
	return 2 * np.max(r)


def generate_feature(
		df, pivot_name, new_feature_name, func, count_columns, mean_columns, agg_col=None, epsilon=0.25, dp_flag=False
):
	if func == 'value_count':
		d = df[pivot_name].value_counts()
		d.name = new_feature_name
		df = df.merge(d, left_on=pivot_name, right_index=True)
		if dp_flag:  # if use DP
			if new_feature_name in count_columns:  # add laplace noise with sensitivity = 1
				d_dp = d.copy()
				for i in range(len(d_dp)):  # for each transaction, add laplace noise based on 1/eps
					d_dp.iloc[i] = laplace_mech(d_dp.iloc[i], 1, epsilon)
				df = df.drop(new_feature_name, axis=1).merge(d_dp, left_on=pivot_name, right_index=True)  # replace
			# with the dp version
			else:
				pass

	elif func == 'frequency':
		d = df[pivot_name].value_counts(normalize=True)
		d.name = new_feature_name
		df = df.merge(d, left_on=pivot_name, right_index=True)

	elif func == 'mean':
		d = df.groupby(pivot_name).agg(
			**{
				new_feature_name: pd.NamedAgg(column=agg_col, aggfunc='mean')
			}
		)
		df = df.merge(d, left_on=pivot_name, right_index=True)

		if dp_flag:  # if use DP
			if new_feature_name in mean_columns:  # add laplace noise with smooth sensitivity
				d_dp = d.copy()
				ss = smooth_sens(df[new_feature_name], 500, epsilon)
				for i in range(len(d_dp)):  # for each transaction, add laplace noise based on 1/eps
					d_dp.iloc[i] = laplace_mech(d_dp.iloc[i], ss, epsilon)
				df = df.drop(new_feature_name, axis=1).merge(
					d_dp, left_on=pivot_name, right_index=True
				)  # replace with the dp version
			else:
				pass

	elif func == 'n_unique':
		d = df.groupby(pivot_name).agg(
			**{
				new_feature_name: pd.NamedAgg(column=agg_col, aggfunc=lambda x: len(x.unique()))
			}
		)
		df = df.merge(d, left_on=pivot_name, right_index=True)
	# test = df.merge(d, left_on=pivot_name, right_index=True)
	else:
		raise ValueError("func is not a valid option.")

	return df, d


def extract_feature(df, model_dir, phase, epsilon=0.25, dp_flag=False):
	"""
	from pns train and test data, extract more features
	"""
	df["Timestamp"] = df["Timestamp"].astype("datetime64[ns]")
	df["hour"] = df["Timestamp"].dt.hour.astype(str)
	df['day'] = df["Timestamp"].dt.dayofweek.astype(str)

	invalid_pair = {
		'BAMEUR', 'BAMGBP', 'BDTUSD', 'BOBUSD', 'BRLUSD', 'BWPCZK', 'BWPEUR', 'BWPJPY', 'EGPEUR', 'EGPGBP',
		'EGPUSD', 'FJDUSD', 'HRKAUD', 'HRKCAD', 'HRKUSD', 'JODAUD', 'JODCHF', 'JODDKK', 'JODEUR', 'JODGBP',
		'JODUSD', 'KESCZK', 'KESEUR', 'KESUSD', 'KRWUSD', 'LKRCAD', 'LKREUR', 'LKRSGD', 'LKRUSD', 'NADEUR',
		'NADNZD', 'NPRUSD', 'RSDEUR', 'RSDUSD', 'TZSEUR', 'TZSGBP', 'XOFCAD', 'XOFCHF', 'XOFEUR', 'XOFGBP', 'XOFUSD'
	}

	df['F2'] = df.apply(
		lambda row: 1 if (row['InstructedCurrency'] + row['SettlementCurrency']) in invalid_pair else 0, axis=1
	)

	features_needed = [
		{
			"feature_name": "sender_hour_freq",
			"pivot_features": "Sender,hour", "function": "value_count", "agg_col": "None"
		},
		{
			"feature_name": "receiver_hour_freq",
			"pivot_features": "Receiver,hour", "function": "value_count", "agg_col": "None"
		},
		{
			"feature_name": "sender_currency_freq",
			"pivot_features": "Sender,InstructedCurrency", "function": "value_count", "agg_col": "None"
		},
		{
			"feature_name": "receiver_currency_freq",
			"pivot_features": "Receiver,InstructedCurrency", "function": "value_count", "agg_col": "None"
		},
		{
			"feature_name": "sender_currency_avg_amount",
			"pivot_features": "Sender,InstructedCurrency", "function": "mean", "agg_col": "InstructedAmount"
		},
		{
			"feature_name": "receiver_currency_avg_amount",
			"pivot_features": "Receiver,InstructedCurrency", "function": "mean", "agg_col": "InstructedAmount"
		},
		{
			"feature_name": "sender_receiver_freq",
			"pivot_features": "Sender,Receiver", "function": "value_count", "agg_col": "None"
		},
		{
			"feature_name": "recevier_in_degree",
			"pivot_features": "Receiver", "function": "n_unique", "agg_col": "Sender"
		}
	]

	count_columns = ['sender_hour_freq', 'receiver_hour_freq']
	mean_columns = ['sender_currency_avg_amount', 'receiver_currency_avg_amount']
	for row in features_needed:

		# pivot column
		features = [feature.strip() for feature in row['pivot_features'].split(',')]
		pivot_name = '_'.join([feature.lower() for feature in features])

		if len(features) == 2:
			s = df[features[0]] + df[features[1]]
		elif len(features) == 3:
			s = df[features[0]] + df[features[1]] + df[features[2]]
		else:
			s = df[features[0]]

		if pivot_name in df.columns:
			pivot_name = pivot_name + "_p"

		df[pivot_name] = s

		# aggregation col
		agg_cols = [agg_col.strip() for agg_col in row['agg_col'].split(',')]

		if agg_cols[0] != 'None':
			if len(agg_cols) == 2:
				a = df[agg_cols[0]] + df[agg_cols[1]]
			elif len(agg_cols) == 3:
				a = df[agg_cols[0]] + df[agg_cols[1]] + df[agg_cols[2]]
			else:
				a = df[agg_cols[0]]

			agg_col_name = '_'.join([agg_col.lower() for agg_col in agg_cols])

			if agg_col_name in df.columns:
				agg_col_name = agg_col_name + "_p"

			df[agg_col_name] = a
		else:
			agg_col_name = 'None'

		# feature name and filename
		new_feature_name = row['feature_name']
		filename = hashlib.sha256(new_feature_name.encode()).hexdigest()
		function_name = row['function']

		if phase == 'train':

			if agg_col_name == 'None':
				df, d_dict_df = generate_feature(
					df, pivot_name=pivot_name, new_feature_name=new_feature_name,
					func=function_name, count_columns=count_columns, mean_columns=mean_columns,
					epsilon=epsilon, dp_flag=dp_flag
				)
			else:
				df, d_dict_df = generate_feature(
					df, pivot_name=pivot_name, new_feature_name=new_feature_name,
					func=function_name, agg_col=agg_col_name, count_columns=count_columns, mean_columns=mean_columns,
					epsilon=epsilon, dp_flag=dp_flag
				)

			if len(df[new_feature_name].unique()) == 1:
				df = df.drop([new_feature_name], axis=1)
			else:
				d_dict_df.to_pickle(Path.joinpath(model_dir, '{}.pkl'.format(filename)))


		elif phase == 'test':

			file_path = Path.joinpath(model_dir, '{}.pkl'.format(filename))
			if file_path.exists():
				d = pd.read_pickle(Path.joinpath(model_dir, '{}.pkl'.format(filename)))
				df = df.merge(d, how='left', left_on=pivot_name, right_index=True)
				df = df.fillna(value=-1)

		if agg_col_name != 'None':
			df = df.drop([pivot_name], axis=1)
			df = df.drop([agg_col_name], axis=1)
		else:
			df = df.drop([pivot_name], axis=1)

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
		"Timestamp"
	]

	df['hour'] = df['hour'].astype('int')
	df['day'] = df['day'].astype('int')

	logger.info('Total features: {}'.format(df.shape[1] - 1))

	return df.drop(columns_to_drop, axis=1)


def compute_flag(row, bloom):
	hash1 = row['Hash1']
	hash2 = row['Hash2']
	if hash1 in bloom and hash2 in bloom:
		return False
	else:
		return True


def add_BF_feature2(pns_df, bank):
	logger.info("Extracting bloom filter features")
	from .bloom_filter import BloomFilter
	bloom = BloomFilter(max_elements=bank.shape[0], error_rate=0.001)
	bank['Account'] = bank['Account'].astype(str)
	bank_nonflags = bank[bank['Flags'] == 0]
	ret = bank_nonflags['Account'].astype(str) + bank_nonflags['Name'].astype(str) + bank_nonflags['Street'].astype(
		str
	) + bank_nonflags['CountryCityZip'].astype(str)
	valid_accounts = ret.values.tolist()
	for account in valid_accounts:
		bloom.add(account)

	pns_df['BeneficiaryAccount'] = pns_df['BeneficiaryAccount'].astype(str)
	pns_df['OrderingAccount'] = pns_df['OrderingAccount'].astype(str)
	pns_df['Hash1'] = pns_df['BeneficiaryAccount'] + pns_df['BeneficiaryName'].astype(str) + pns_df[
		'BeneficiaryStreet'].astype(str) + pns_df['BeneficiaryCountryCityZip'].astype(str)
	pns_df['Hash2'] = pns_df['OrderingAccount'] + pns_df['OrderingName'].astype(str) + pns_df[
		'OrderingStreet'].astype(str) + pns_df['OrderingCountryCityZip'].astype(str)

	pns_df['BF'] = pns_df.apply(lambda row: compute_flag(row, bloom), axis=1).astype('bool')
	print(pns_df['BF'].value_counts())
	pns_df = pns_df.drop(['Hash1', 'Hash2'], axis=1)

	return pns_df


def add_BF_feature(pns_df, bank):
	logger.info("Extracting bloom filter features")

	pns_df['Hash1'] = pns_df['BeneficiaryAccount'] + pns_df['BeneficiaryName'] + pns_df[
		'BeneficiaryStreet'] + pns_df['BeneficiaryCountryCityZip']
	pns_df['Hash2'] = pns_df['OrderingAccount'] + pns_df['OrderingName'] + pns_df[
		'OrderingStreet'] + pns_df['OrderingCountryCityZip']

	bank_non_flagged = bank[bank['Flags'] == 0]
	bank_non_flagged['Hash'] = bank_non_flagged['Account'] + bank_non_flagged['Name'] + bank_non_flagged['Street'] + \
	                           bank_non_flagged['CountryCityZip']

	pns_join1 = pd.merge(
		pns_df, bank_non_flagged, how='left', left_on=['BeneficiaryAccount'],
		right_on=['Account']
	)
	pns_join2 = pd.merge(
		pns_df, bank_non_flagged, how='left', left_on=['OrderingAccount'],
		right_on=['Account']
	)

	simple_anos_1 = pns_join1[(pns_join1['Hash1'] != pns_join1['Hash'])].index
	simple_anos_2 = pns_join2[(pns_join2['Hash2'] != pns_join2['Hash'])].index

	simple_anos = list(set(simple_anos_1).union(simple_anos_2))

	pns_df['BF'] = [0 for _ in range(len(pns_df))]
	pns_df['BF'].mask(pns_df.reset_index().index.isin(simple_anos), 1, inplace=True)

	pns_df = pns_df.drop(['Hash1', 'Hash2'], axis=1)

	return pns_df


def gb_xgb_cv(params, X, Y, random_state):
	params = {
		'n_estimators': int(params['n_estimators']),
		'max_depth': int(params['max_depth']),
		'learning_rate': params['learning_rate']
	}

	kfold = StratifiedKFold(n_splits=2, random_state=random_state, shuffle=True)
	model = XGBClassifier(random_state=random_state, **params)

	score = -cross_val_score(model, X, Y, cv=kfold, scoring="f1", n_jobs=-1).mean()

	return score


class PNSModel:
	def __init__(self):
		pass

	def cv(self, X, Y):
		random_state = 40
		trials = Trials()
		space = {
			'n_estimators': hp.quniform('n_estimators', 100, 200, 40),
			'max_depth': hp.quniform('max_depth', 2, 10, 2),
			'learning_rate': hp.loguniform('learning_rate', -2, 0)
		}
		best = fmin(
			fn=gb_xgb_cv,
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

		self.model = XGBClassifier(
			n_estimators=int(params['n_estimators']),
			max_depth=int(params['max_depth']), learning_rate=params['learning_rate']
		)

		self.model.fit(X, y)

		return

	def predict(self, X):
		return pd.Series(self.pipeline.predict_proba(X)[:, 1], index=X.index)

	def save(self, path):
		joblib.dump(self.model, path)

	@classmethod
	def load(cls, path):
		inst = cls()
		inst.pipeline = joblib.load(path)
		return inst
