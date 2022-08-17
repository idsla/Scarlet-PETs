import pandas as pd
from pathlib import Path

def load_swift_data(DATA_DIR):
    train = pd.read_csv(
        DATA_DIR / "swift_transaction_train_dataset.csv", index_col="MessageId"
    )
    train["Timestamp"] = train["Timestamp"].astype("datetime64[ns]")
    test = pd.read_csv(DATA_DIR / "swift_transaction_test_dataset.csv", index_col="MessageId")
    test["Timestamp"] = test["Timestamp"].astype("datetime64[ns]")

    return train, test


def load_bank_data(DATA_DIR):
    bank_data = pd.read_csv(DATA_DIR / "bank_dataset.csv")
    return bank_data
