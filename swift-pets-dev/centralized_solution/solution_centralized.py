import numpy as np
import pandas as pd
from loguru import logger
import time
from pathlib import Path
from sklearn import metrics

from .model2 import (
    add_BF_feature,
    extract_feature,
    PNSModel,
    join_flags_to_pns_data,
    rule_mining
)



def fit(pns_data_path, bank_data_path, model_dir):
    logger.info("Preparing data")
    pns_df = pd.read_csv(pns_data_path, index_col="MessageId")
    bank = pd.read_csv(bank_data_path)
    pns_df = add_BF_feature(pns_df, bank)

    pns_df = extract_feature(pns_df, model_dir, phase='train', epsilon = 0.25, dp_flag = True)
    logger.info("Fitting pns model...")

    pns_model = PNSModel()
    pns_model.fit(X=pns_df.drop(['Label'], axis=1), y=pns_df["Label"])

    logger.info("...done fitting")
    pns_model.save(Path.joinpath(model_dir, "pns_model.joblib"))


def predict(pns_data_path, bank_data_path, model_dir, preds_format_path, preds_dest_path):
    logger.info("Preparing data")
    pns_df = pd.read_csv(pns_data_path, index_col="MessageId")
    bank = pd.read_csv(bank_data_path)

    pns_df = add_BF_feature(pns_df, bank)
    pns_df = extract_feature(pns_df, model_dir, phase='test', epsilon = 0.25, dp_flag = False)
    logger.info("EF {}".format(pns_df.shape))

    logger.info("Loading pns model...")
    pns_model = PNSModel.load(Path.joinpath(model_dir , "pns_model.joblib"))
    pns_preds = pns_model.predict(pns_df)

    preds_format_df = pd.read_csv(preds_format_path, index_col="MessageId")
    preds_format_df["Score"] = preds_format_df.index.map(pns_preds)


    logger.info("Writing out test predictions...")
    preds_format_df.to_csv(preds_dest_path)
    logger.info("Done.")


def main():
    pns_data_path = './new_data/dev_pns_transaction_train_dataset.csv'
    bank_data_path = './new_data/dev_bank_dataset.csv'
    model_dir = Path('./new_data/')
    preds_format_path = './new_data/fincrime/scenario01/test/pns/predictions_format.csv'
    preds_dest_path = './new_data/predictions_test.csv'

    fit(pns_data_path, bank_data_path, model_dir)

    predict('./new_data/dev_pns_transaction_test_dataset.csv', bank_data_path, model_dir, preds_format_path,
            preds_dest_path)


if __name__ == "__main__":
    main()
