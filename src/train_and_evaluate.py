import os
import argparse
import pandas as pd
from get_data import read_params
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNet
import joblib
import json
import sys
import warnings
from urllib.parse import urlparse
import mlflow

def evaluate_metrics(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual,predicted)
    return rmse,mae,r2


def train_and_evaluate(config_path):
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]

    alpha = config["estimators"]["ElasticNet"]["params"]["alpha"]
    l1_ratio = config["estimators"]["ElasticNet"]["params"]["l1_ratio"]

    target = config["base"]["target_col"]

    train = pd.read_csv(train_data_path,sep=",")
    test = pd.read_csv(test_data_path,sep=',')

    train_y = train[target]
    test_y = test[target]

    train_x = train.drop([target],axis=1)
    test_x = test.drop([target],axis=1)

    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]

    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(mlflow_config["experiment_name"])

    with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlflow_run:

        lr = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            random_state=random_state
        )

    lr.fit(train_x,train_y)
    predicted_qualities = lr.predict(test_x)
    (rmse, mae,r2) = evaluate_metrics(test_y, predicted_qualities)

    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
    if tracking_url_type_store != "file":
        mlflow.sklearn.log_model(lr, "model", registered_model_name=mlflow_config["registered_model_name"])
    else:
        mlflow.sklearn.log_model(lr, "model")

    


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)