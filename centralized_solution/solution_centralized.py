import numpy as np
import pandas as pd
from loguru import logger
import time
from pathlib import Path
from sklearn import metrics

from model2 import (
    add_BF_feature, add_BF_feature2,
    extract_feature,
    PNSModel,
    join_flags_to_pns_data
)


# TODO: track running time and peak mry usage

# Workflow: load bank data -> BF; load pns data ->add BF feature -> extract other features -> drop uneccessary features
#           Fit pns model -> predict


def fit(pns_data_path, bank_data_path, model_dir):
    logger.info("Preparing data")
    pns_df = pd.read_csv(pns_data_path, index_col="TransactionId")
    bank = pd.read_csv(bank_data_path)
    t0 = time.time()
    pns_df = add_BF_feature(pns_df, bank)
    t1 = time.time()

    print('add BF costs %s ' % (t1 - t0))

    #pns_df = join_flags_to_pns_data(pns_df, bank)
    #pns_df, _ = rule_mining(pns_df, 0.05)
    t2 = time.time()
    print('rule mining %s ' % (t2 - t1))

    pns_df = extract_feature(pns_df, model_dir, phase='train')
    #pns_df.to_csv(Path.joinpath(model_dir, 'pns_df_v2.csv'))

    t3 = time.time()
    print('extract features costs %s ' % (t3 - t2))
    logger.info("Fitting pns model...")

    pns_model = PNSModel()
    pns_model.fit(X=pns_df.drop(['Label'], axis=1), y=pns_df["Label"])

    logger.info("...done fitting")
    pns_model.save(Path.joinpath(model_dir, "pns_model.joblib"))


def predict(pns_data_path, bank_data_path, model_dir):
    logger.info("Preparing data")
    pns_df = pd.read_csv(pns_data_path, index_col="TransactionId")
    bank = pd.read_csv(bank_data_path)

    pns_df = add_BF_feature(pns_df, bank)
    #pns_df = join_flags_to_pns_data(pns_df, bank)
    #pns_df, _ = rule_mining(pns_df, 0.05)
    pns_df = extract_feature(pns_df, model_dir, phase='test')
    logger.info("EF {}".format(pns_df.shape))

    logger.info("Loading pns model...")
    pns_model = PNSModel.load(Path.joinpath(model_dir , "pns_model.joblib"))
    pns_preds = pns_model.predict(pns_df.drop(['Label'], axis=1))

    # preds_format_df = pd.read_csv(preds_format_path, index_col="TransactionId")
    # preds_format_df["Score"] = preds_format_df.index.map(pns_preds)

    print("AUPRC:", metrics.average_precision_score(y_true=pns_df['Label'], y_score=pns_preds))

    # logger.info("Writing out test predictions...")
    # preds_format_df.to_csv(preds_dest_path)
    # logger.info("Done.")


def main():
    pns_data_path = './new_data/pns_transaction_train.csv'
    bank_data_path = './new_data/dev_bank_dataset.csv'
    model_dir = Path('./new_data/')

    fit(pns_data_path, bank_data_path, model_dir)

    predict('./new_data/pns_transaction_test.csv', bank_data_path, model_dir)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
