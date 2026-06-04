# This file handles feature scaling, selection and class imbalance before training

import pandas as pd  # data manipulation
import numpy as np  # numerical operations
from pathlib import Path
from sqlalchemy import create_engine  # database connection
from sklearn.preprocessing import StandardScaler  # scale numerical features
from sklearn.feature_selection import SelectKBest, f_classif  # select top k important features
from imblearn.over_sampling import SMOTE  # handle class imbalance by oversampling minority class
import pickle  # save scaler and selector to disk
from dotenv import load_dotenv  # load env variables
import os  # access env variables

load_dotenv()  # load .env file

from etl.transform import transform_data  # import transform functions


def resolve_path(path: str):
    if not path:
        return None
    file_path = Path(path.strip())
    if not file_path.is_absolute():
        file_path = Path(__file__).resolve().parent.parent / file_path
    return file_path


def get_pg_engine():
    # build postgresql connection string for data warehouse
    url = (
        f"postgresql+psycopg2://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD')}"
        f"@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_NAME')}"
    )
    return create_engine(url)  # return postgresql engine


def load_transformed_data():
    transformed_csv = os.getenv('TRANSFORMED_CSV_PATH')
    if transformed_csv:
        transformed_file = resolve_path(transformed_csv)
        if transformed_file and transformed_file.exists():
            df = pd.read_csv(transformed_file)
            print(f"Loaded {len(df)} rows from transformed CSV {transformed_file}")
            return df
        raise FileNotFoundError(f"Transformed CSV path '{transformed_file}' not found.")

    raw_csv = os.getenv('DATA_CSV_PATH')
    if raw_csv:
        raw_file = resolve_path(raw_csv)
        if raw_file and raw_file.exists():
            df = pd.read_csv(raw_file)
            print(f"Loaded {len(df)} rows from raw CSV {raw_file}")
            if 'Month' in df.columns and 'VisitorType' in df.columns:
                df = transform_data(df)
            else:
                print("Input CSV appears already transformed; skipping raw transform.")
            return df
        raise FileNotFoundError(f"CSV path '{raw_file}' not found. Set DATA_CSV_PATH in .env or pass a valid file path.")

    engine = get_pg_engine()  # get postgresql connection
    df = pd.read_sql("SELECT * FROM transformed_shoppers", engine)  # fetch transformed data from warehouse
    print(f"Loaded {len(df)} rows from PostgreSQL warehouse")  # log count
    return df  # return dataframe


def preprocess(df):
    X = df.drop('Revenue', axis=1)  # features
    y = df['Revenue']  # target

    # check class distribution before SMOTE
    print(f"Before SMOTE — Revenue 0: {(y==0).sum()}, Revenue 1: {(y==1).sum()}")

    # scale numerical features to same range (mean=0, std=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # fit and transform features

    # select top 10 most important features using ANOVA f-test
    selector = SelectKBest(score_func=f_classif, k=10)
    X_selected = selector.fit_transform(X_scaled, y)  # fit and select features

    # get selected feature names for reference
    selected_features = X.columns[selector.get_support()].tolist()
    print(f"Selected Features: {selected_features}")  # log selected features

    # if the data contains only one class, abort training because classifier cannot learn two classes
    if y.nunique() < 2:
        raise ValueError(
            "Training data contains only one class for Revenue. "
            "Please provide a dataset with both Revenue=0 and Revenue=1 examples."
        )

    # apply SMOTE to balance classes
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_selected, y)  # oversample minority class

    # check class distribution after SMOTE
    print(f"After SMOTE — Revenue 0: {(y_resampled==0).sum()}, Revenue 1: {(y_resampled==1).sum()}")

    # save scaler to disk for use in prediction
    with open('./ml/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)  # serialize scaler

    # save selector to disk for use in prediction
    with open('./ml/selector.pkl', 'wb') as f:
        pickle.dump(selector, f)  # serialize selector

    # save selected feature names for reference in prediction
    with open('./ml/selected_features.pkl', 'wb') as f:
        pickle.dump(selected_features, f)  # serialize feature names

    print("Scaler, Selector saved to ./ml/")  # log save
    return X_resampled, y_resampled  # return preprocessed data


if __name__ == "__main__":
    df = load_transformed_data()  # load data from postgresql or CSV
    X, y = preprocess(df)  # run preprocessing
    print(f"Final shape after preprocessing: {X.shape}")  # log final shape