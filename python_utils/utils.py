import os
from functools import wraps
import matplotlib.pyplot as plt
import numpy as np
import mlflow
import pandas as pd
import lightgbm as lgb
import time


def measure_time_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        return result, elapsed_time

    return wrapper


def plot_inference(self, Xtrain, Ytrain, Xtest, Ytest, oldrun, newrun):
    """
    Plots the inference results of the old and new models along with the training and testing data.
    Assumes the path to the models' file is defined in the MODELSPATH parameter in an objectscript class
    Args:
        self: The instance of the class calling this function, used to access parameters.
        Xtrain (pd.DataFrame): Training features.
        Ytrain (pd.Series): Training labels.
        Xtest (pd.DataFrame): Testing features.
        Ytest (pd.Series): Testing labels.
        oldrun (mlflow.entities.Run): The MLflow run object for the old model.
        newrun (mlflow.entities.Run): The MLflow run object for the new model.
    """
    oldrunname = oldrun.get("tags.mlflow.runName")
    newrunname = newrun.get("tags.mlflow.runName")

    oldmodel = mlflow.sklearn.load_model(
        os.path.join(eval("""self._GetParameter("MODELSPATH")"""), oldrun.run_id)
    )
    newmodel = mlflow.sklearn.load_model(
        os.path.join(eval("""self._GetParameter("MODELSPATH")"""), newrun.run_id)
    )

    line_x = np.linspace(Xtest.min(), Xtest.max(), 100).reshape(-1, 1)
    line_y_old = oldmodel.predict(line_x)
    line_y_new = newmodel.predict(line_x)
    plt.figure(figsize=(10, 6))
    if not Xtrain.empty and not Ytrain.empty:
        plt.scatter(Xtrain, Ytrain, color="orange", label="Train Data")
    if not Xtest.empty and not Ytest.empty:
        plt.scatter(Xtest, Ytest, color="blue", label="Test Data")
    plt.plot(line_x, line_y_old, color="red", label=f"Old Model: {oldrunname}")
    plt.plot(line_x, line_y_new, color="green", label=f"New Model: {newrunname}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Model Comparison")
    plt.legend()
    plt.grid()
    plt.savefig(f"/dur/log/model_comparison_{oldrunname}_vs_{newrunname}.png")
    plt.close()
