import pandas as pd


def load_swift_data(train_data_path, test_data_path):
    """
    Load swift train and test data to dataframe
    :param DATA_DIR: folder path containing data
    :return: swift train dataframe, swift test dataframe
    """
    train = pd.read_csv(train_data_path, index_col="MessageId")
    train["Timestamp"] = train["Timestamp"].astype("datetime64[ns]")
    test = pd.read_csv(test_data_path, index_col="MessageId")
    test["Timestamp"] = test["Timestamp"].astype("datetime64[ns]")

    return train, test


def load_bank_data(DATA_DIR):
    """
    Load bank data to dataframe
    :param DATA_DIR: folder path containing data
    :return: bank dataframe
    """
    bank_data = pd.read_csv(DATA_DIR / "bank_dataset.csv")
    return bank_data


def merge_swift_bank_data(df_swift, df_bank):
    """
    merge bank account information to swift data
    :param df_swift: swift dataframe
    :param df_bank: bank dataframe
    :return: merged dataframe
    """
    df_bank_filtered = df_bank.drop(['Bank', 'Name', 'Street', 'CountryCityZip'], axis=1)
    # merge order account information
    df_swift_order = df_swift.merge(df_bank_filtered, how='left', left_on='OrderingAccount', right_on='Account')
    df_swift_order['Flags'] = df_swift_order['Flags'].astype('str')
    df_swift_order = df_swift_order.rename({'Flags': 'Flag_ordering'}, axis=1)
    df_swift_order = df_swift_order.drop(['Account'], axis=1)
    # merge beneficial account information
    df_swift_bene = df_swift_order.merge(df_bank_filtered, how='left', left_on='BeneficiaryAccount', right_on='Account')
    df_swift_bene['Flags'] = df_swift_bene['Flags'].astype('str')
    df_swift_bene = df_swift_bene.rename({'Flags': 'Flag_beneficiary'}, axis=1)
    df_swift_bene = df_swift_bene.drop(['Account'], axis=1)

    return df_swift_bene
