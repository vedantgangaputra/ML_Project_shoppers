# This file handles MLflow experiment tracking for ML models

import mlflow  # mlflow tracking library
import mlflow.sklearn  # mlflow sklearn integration for logging sklearn models
from dotenv import load_dotenv  # load env variables
import os  # access env variables

load_dotenv()  # load .env file

def init_mlflow():
    # set mlflow tracking uri where runs will be stored
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
    # set experiment name — creates new if not exists
    mlflow.set_experiment("shoppers_classification")
    print("MLflow initialized")  # log init

def log_run(model, model_name, acc, pre, rec, f1, auc):
    init_mlflow()  # initialize mlflow

    with mlflow.start_run(run_name=model_name):  # start a new mlflow run
        # log model hyperparameters
        mlflow.log_param("model_type", model_name)  # log model name
        mlflow.log_param("n_estimators", 100)  # log number of trees

        # log evaluation metrics
        mlflow.log_metric("accuracy", acc)    # log accuracy
        mlflow.log_metric("precision", pre)   # log precision
        mlflow.log_metric("recall", rec)      # log recall
        mlflow.log_metric("f1_score", f1)     # log f1 score
        if auc is not None:
            mlflow.log_metric("roc_auc", auc)     # log roc auc
        else:
            print("MLflow: ROC AUC not logged because it is undefined.")

        # log model artifact
        mlflow.sklearn.log_model(model, model_name)  # save model in mlflow
        print(f"MLflow run logged for {model_name}")  # log success

if __name__ == "__main__":
    # test mlflow tracking with dummy values
    from sklearn.ensemble import RandomForestClassifier  # dummy model
    dummy_model = RandomForestClassifier()  # create dummy model
    log_run(dummy_model, "RandomForest", 0.92, 0.90, 0.94, 0.92, 0.97)  # log dummy run