import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "test_experiment"

# log onto remote lew server
mlflow.set_tracking_uri('https://mlflow.lewagon.co/')

client = MlflowClient()

try:
    experiment_id = client.create_experiment(EXPERIMENT_NAME)
except BaseException:
    experiment_id = client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id

myname = 'cberg'

for model in ["linear", "Randomforest"]:
    run = client.create_run(experiment_id)
    client.log_metric(run.info.run_id, "rmse", 4.5)
    client.log_param(run.info.run_id, "model", model)
    client.log_param(run.info.run_id, "student_name", myname)
