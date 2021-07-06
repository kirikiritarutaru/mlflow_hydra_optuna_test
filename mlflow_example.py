import mlflow

with mlflow.start_run():
    mlflow.log_param('params1', 5)

    mlflow.log_metric('foo', 2, step=1)
    mlflow.log_metric('foo', 4, step=2)
    mlflow.log_metric('foo', 6, step=3)

    with open('mlflow_output.txt', 'w') as f:
        f.write('Hello world!')

    mlflow.log_artifact('mlflow_output.txt')
