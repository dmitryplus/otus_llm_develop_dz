FROM ghcr.io/mlflow/mlflow

WORKDIR /app

CMD mlflow server --host 0.0.0.0