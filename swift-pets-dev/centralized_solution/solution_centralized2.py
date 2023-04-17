import numpy as np
import pandas as pd
from loguru import logger
import time
from pathlib import Path
from sklearn import metrics

from model2 import (
    add_BF_feature, add_BF_feature2,
    extract_feature,
    SwiftModel,
    join_flags_to_swift_data,
    rule_mining
)


# TODO: track running time and peak mry usage

# Workflow: load bank data -> BF; load Swift data ->add BF feature -> extract other features -> drop uneccessary features
#           Fit Swift model -> predict


def fit(swift_data_path, bank_data_path, model_dir):
    logger.info("Preparing data")
    swift_df = pd.read_csv(swift_data_path, index_col="MessageId")
    bank = pd.read_csv(bank_data_path)
    t0 = time.time()
    swift_df = add_BF_feature(swift_df, bank)
    t1 = time.time()

    print('add BF costs %s ' % (t1 - t0))

    #swift_df = join_flags_to_swift_data(swift_df, bank)
    #swift_df, _ = rule_mining(swift_df, 0.05)
    t2 = time.time()
    print('rule mining %s ' % (t2 - t1))

    swift_df = extract_feature(swift_df, model_dir, phase='train')
    #swift_df.to_csv(Path.joinpath(model_dir, 'swift_df_v2.csv'))

    t3 = time.time()
    print('extract features costs %s ' % (t3 - t2))
    print(swift_df.columns)
    print(swift_df.shape)
    logger.info("Fitting SWIFT model...")

    swift_model = SwiftModel()
    swift_model.fit(X=swift_df.drop(['Label'], axis=1), y=swift_df["Label"])

    logger.info("...done fitting")
    swift_model.save(Path.joinpath(model_dir, "swift_model.joblib"))


def predict(swift_data_path, bank_data_path, model_dir, preds_format_path, preds_dest_path):
    logger.info("Preparing data")
    swift_df = pd.read_csv(swift_data_path, index_col="MessageId")
    bank = pd.read_csv(bank_data_path)

    swift_df = add_BF_feature(swift_df, bank)
    #swift_df = join_flags_to_swift_data(swift_df, bank)
    #swift_df, _ = rule_mining(swift_df, 0.05)
    swift_df = extract_feature(swift_df, model_dir, phase='test')
    logger.info("EF {}".format(swift_df.shape))

    logger.info("Loading SWIFT model...")
    swift_model = SwiftModel.load(Path.joinpath(model_dir , "swift_model.joblib"))
    swift_preds = swift_model.predict(swift_df.drop(['Label'], axis=1))

    preds_format_df = pd.read_csv(preds_format_path, index_col="MessageId")
    preds_format_df["Score"] = preds_format_df.index.map(swift_preds)

    print("AUPRC:", metrics.average_precision_score(y_true=swift_df['Label'], y_score=preds_format_df["Score"]))

    logger.info("Writing out test predictions...")
    preds_format_df.to_csv(preds_dest_path)
    logger.info("Done.")


def main():
    swift_data_path = '../new_data/dev_swift_transaction_train_dataset.csv'
    bank_data_path = '../new_data/dev_bank_dataset.csv'
    model_dir = Path('../new_data/')
    preds_format_path = '../new_data/fincrime/scenario01/test/swift/predictions_format.csv'
    preds_dest_path = '../new_data/predictions_test.csv'

    fit(swift_data_path, bank_data_path, model_dir)

    predict('../new_data/dev_swift_transaction_test_dataset.csv', bank_data_path, model_dir, preds_format_path,
            preds_dest_path)


if __name__ == "__main__":
    main()
