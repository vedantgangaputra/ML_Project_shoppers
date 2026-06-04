# This file trains ML models and logs results to MLflow

import pandas as pd  # data manipulation
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from sklearn.model_selection import train_test_split  # split data into train and test sets
from sklearn.ensemble import RandomForestClassifier  # random forest model
from xgboost import XGBClassifier  # xgboost model
from sklearn.metrics import (
    accuracy_score,    # overall accuracy
    precision_score,   # precision score
    recall_score,      # recall score
    f1_score,          # f1 score
    roc_auc_score      # roc auc score
)
import pickle  # save model to disk
from dotenv import load_dotenv  # load env variables
import os  # access env variables
from preprocess import load_transformed_data, preprocess  # import preprocess functions
from mlflow_tracker import log_run  # import mlflow logging function

load_dotenv()  # load .env file

def train():
    df = load_transformed_data()  # load transformed data from mysql
    X, y = preprocess(df)  # apply scaling, feature selection and SMOTE

    if y.nunique() < 2:
        raise ValueError(
            "Training data contains only one class for Revenue. "
            "Please provide both 0 and 1 examples before training."
        )

    # split into 80% train and 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # train random forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)  # fit on training data
    rf_acc = accuracy_score(y_test, rf_model.predict(X_test))  # evaluate accuracy
    print(f"Random Forest Accuracy: {rf_acc:.4f}")  # log accuracy

    # train xgboost model
    xgb_model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)  # fit on training data
    xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test))  # evaluate accuracy
    print(f"XGBoost Accuracy: {xgb_acc:.4f}")  # log accuracy

    # select best model based on accuracy
    if xgb_acc >= rf_acc:
        best_model = xgb_model  # xgboost better or equal
        best_name = "XGBoost"
    else:
        best_model = rf_model  # random forest better
        best_name = "RandomForest"

    print(f"Best Model: {best_name}")  # log best model name

    # evaluate best model metrics for mlflow
    y_pred = best_model.predict(X_test)  # predictions
    y_prob = best_model.predict_proba(X_test)[:, 1]  # probabilities

    acc = accuracy_score(y_test, y_pred)   # accuracy
    pre = precision_score(y_test, y_pred, zero_division=0)  # precision
    rec = recall_score(y_test, y_pred, zero_division=0)     # recall
    f1  = f1_score(y_test, y_pred, zero_division=0)         # f1 score
    if y_test.nunique() < 2:
        auc = None
        print("Warning: ROC AUC skipped because y_test contains only one class.")
    else:
        auc = roc_auc_score(y_test, y_prob)    # roc auc

    # log all metrics and model to mlflow
    log_run(best_model, best_name, acc, pre, rec, f1, auc)

    # save best model to disk
    os.makedirs('./ml', exist_ok=True)  # create ml dir if not exists
    with open(os.getenv('MODEL_PATH'), 'wb') as f:
        pickle.dump(best_model, f)  # serialize model
    print(f"Model saved to {os.getenv('MODEL_PATH')}")  # log save path

    return best_model, X_test, y_test, best_name  # return for evaluate use

if __name__ == "__main__":
    train()  # run training