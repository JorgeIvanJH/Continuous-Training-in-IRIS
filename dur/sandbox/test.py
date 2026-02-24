import os

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import mlflow

import dotenv
dotenv.load_dotenv()

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"]) # Can also be set using environment variable MLFLOW_TRACKING_URI
mlflow.set_experiment("my-first-experiment") # Can also be set using environment variable MLFLOW_EXPERIMENT_NAME

X, y = datasets.load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": 8888,
}

mlflow.sklearn.autolog() # WITH JUST THIS LINE, MLFLOW WILL SAVE TRAINED MODEL, PARAMETERS, AND METRICS AUTOMATICALLY!

lr = LogisticRegression(**params)
lr.fit(X_train, y_train)