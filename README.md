# Continuous Training (CT) in IRIS

This repo contains a solid base for Machine Learning Pipeline Automation as part of a proper MLOps (as the Google's standard defined in https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning?hl=en) using the IRIS native components shown in the following graph:


![alt text](images/MLOps_IRIS_level1.png)

Everything should work by only running

```
docker-compose up --build -d
```

The Docker Compose setup creates:

An IRIS container responsible for data management and application processes. A separate MLflow stack (MLflow server, Postgres, and MinIO) dedicated to experiment tracking and UI-based monitoring for Data Science projects.All MLflow-related state (metadata and artifacts such as models and metrics) is stored in the durable host-mounted directory dur/sandbox/mlflow.,Because this directory resides in the host filesystem and is bind-mounted, its contents persist across container restarts and are accessible outside the containers.



WARNING: a .env is in this project uploaded for testing purposes, but it should be added to the .gitignore to avoid sharing credentials in a production environment

If during build, any of the container components for mlflow fail, retry commenting out all services except "iris" to create only the iris container through "docker-compose up --build -d" and manually start server (http://localhost:5000) in the desired folder by running 

```
mlflow server --port 5000
```

then you can Open http://localhost:5000 in your browser to view the UI.

# Experimentation framework (MLflow Tracking)
MLflow Tracking is the component of MLflow for data scientists who train traditional machine learning models. It keeps track of model performance metrics, saves weights in a standard manner, logs hyperparameters, and much more. All this information is saved locally (in this repo), and can be easily consulted just by going to http://localhost:5000 in any web explored.

Codewise only a couple of additional lines of code are added to link training progress to be stored in the platform

only 2 lines added

```python
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
```

Quickstart for Data Scientists: https://mlflow.org/docs/latest/ml/getting-started/quickstart/