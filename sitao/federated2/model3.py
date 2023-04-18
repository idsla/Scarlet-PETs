import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from loguru import logger
from hyperopt import fmin, tpe, hp, anneal, Trials
import numpy as np
from pathlib import Path
import hashlib


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
				df = df.drop(new_feature_name, axis=1).merge(
					d_dp, left_on=pivot_name, right_index=True
				)  # replace with the dp version
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
	from swift train and test data, extract more features
	"""
	# df["Timestamp"] = df["Timestamp"].astype("datetime64[ns]")
	# df["hour"] = df["Timestamp"].dt.hour.astype(str)
	# df['day'] = df["Timestamp"].dt.dayofweek.astype(str)
	# df['wom'] = df['Timestamp'].apply(lambda d: (d.day - 1) // 7 + 1).astype(str)
	df['Timestamp'] = pd.to_datetime(df['Timestamp'])
	x= pd.to_datetime(df['SettlementDate'], format = "%y%m%d").dt.strftime("%m%d")
	x1 = pd.to_datetime(x, format = "%m%d")
	x = df['Timestamp'].dt.strftime("%m%d")
	x2 = pd.to_datetime(x, format = "%m%d")
	df['datediff'] = ((x2 - x1).dt.days > 1).astype(int)
	df["hour"] = df["Timestamp"].dt.hour.astype(str)
	df['day'] = df["Timestamp"].dt.dayofweek.astype(str)

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
		},
		# {"feature_name": "sender_wom_out_degree",
		# "pivot_features": "Sender,wom", "function": "n_unique", "agg_col": "Receiver,wom"},
		# {"feature_name": "receiver_wom_in_degree",
		# "pivot_features": "Receiver,wom", "function": "n_unique", "agg_col": "Sender,wom"},
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
		# new_feature_name = '{}_{}'.format(pivot_name, function_name[0:4])
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
	# df['wom'] = df['wom'].astype('int')

	logger.info('Total features: {}'.format(df.shape[1] - 1))

	return df.drop(columns_to_drop, axis=1)


def compute_flag(row, bloom, hashed_account_dict1, hashed_account_dict2):

	hash1 = hashlib.sha3_256(row['Hash1'].encode()).hexdigest()
	# hash1 = binascii.hexlify(hash1).decode()
	if hash1 not in hashed_account_dict1.keys():
		# print("hash1 {} {}".format(row['Sender'], row['Hash1']))
		# print("hash1 {} {}".format(row['Sender'], hash1))
		hash1_enc = 'a'
	else:
		hash1_enc = hashed_account_dict1[hash1]

	hash2 = hashlib.sha3_256(row['Hash2'].encode()).hexdigest()
	# hash2 = binascii.hexlify(hash2).decode()
	if hash2 not in hashed_account_dict2.keys():
		# print("hash2 {} {} ".format(row['Receiver'], row['Hash2']))
		# print("hash2 {} {} ".format(row['Receiver'], hash2))
		hash2_enc = 'a'
	else:
		hash2_enc = hashed_account_dict2[hash2]

	if hash1_enc in bloom and hash2_enc in bloom:
		return False
	else:
		return True


def add_BF_feature2(swift_df, bloom, hashed_account_dict1, hashed_account_dict2):
	# logger.info("Extracting bloom filter features")

	# count_hash1, count_hash2 = 0, 0
	# print(len(hashed_account_dict1), len(hashed_account_dict2))
	# for k, v in hashed_account_dict1.items():
	#     print(k, v)

	swift_df['Hash2'] = swift_df['BeneficiaryAccount'].astype(str) + swift_df['BeneficiaryName'].astype(str) + \
	                    swift_df[
		'BeneficiaryStreet'].astype(str) + swift_df['BeneficiaryCountryCityZip'].astype(str)

	swift_df['Hash1'] = swift_df['OrderingAccount'].astype(str) + swift_df['OrderingName'].astype(str) + swift_df[
		'OrderingStreet'].astype(str) + swift_df['OrderingCountryCityZip'].astype(str)

	swift_df['BF'] = swift_df.apply(
		lambda row: compute_flag(row, bloom, hashed_account_dict1, hashed_account_dict2), axis=1
		).astype('bool')

	# logger.info(swift_df['BF'].value_counts().to_string())
	# logger.info("swift ordering accounts: {}".format(len(swift_df['OrderingAccount'].unique())))
	# logger.info("swift beneficiary accounts: {}".format(len(swift_df['BeneficiaryAccount'].unique())))

	swift_df = swift_df.drop(['Hash1', 'Hash2'], axis=1)

	return swift_df


def add_BF_feature(swift_df, bank):
	# logger.info("Extracting bloom filter features")

	swift_df['Hash1'] = swift_df['BeneficiaryAccount'] + swift_df['BeneficiaryName'] + swift_df[
		'BeneficiaryStreet'] + swift_df['BeneficiaryCountryCityZip']
	swift_df['Hash2'] = swift_df['OrderingAccount'] + swift_df['OrderingName'] + swift_df[
		'OrderingStreet'] + swift_df['OrderingCountryCityZip']

	bank_non_flagged = bank[bank['Flags'] == 0]
	bank_non_flagged['Hash'] = bank_non_flagged['Account'] + bank_non_flagged['Name'] + bank_non_flagged['Street'] + \
	                           bank_non_flagged['CountryCityZip']

	swift_join1 = pd.merge(
		swift_df, bank_non_flagged, how='left', left_on=['BeneficiaryAccount'],
		right_on=['Account']
		)
	swift_join2 = pd.merge(
		swift_df, bank_non_flagged, how='left', left_on=['OrderingAccount'],
		right_on=['Account']
		)

	simple_anos_1 = swift_join1[(swift_join1['Hash1'] != swift_join1['Hash'])].index
	simple_anos_2 = swift_join2[(swift_join2['Hash2'] != swift_join2['Hash'])].index

	simple_anos = list(set(simple_anos_1).union(simple_anos_2))

	swift_df['BF'] = [0 for _ in range(len(swift_df))]
	swift_df['BF'].mask(swift_df.reset_index().index.isin(simple_anos), 1, inplace=True)

	swift_df = swift_df.drop(['Hash1', 'Hash2'], axis=1)

	return swift_df


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


class SwiftModel:
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