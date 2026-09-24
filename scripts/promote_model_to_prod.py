import mlflow
import dagshub
import json
from mlflow import MlflowClient
import os

os.environ["MLFLOW_TRACKING_USERNAME"] = "guptatannu538"
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_USER_TOKEN", "")

# Initialize DagsHub and set tracking URI
dagshub.init(
    repo_owner="guptatannu538",
    repo_name="food_delivery_swiggy_time_prediction",
    mlflow=True,
)

mlflow.set_tracking_uri(
    "https://dagshub.com/guptatannu538/food_delivery_swiggy_time_prediction.mlflow"
)

def load_model_information(file_path):
    with open(file_path) as f:
        run_info=json.load(f)
    return run_info

# get the name
model_name=load_model_information('run_information.json')['model_name']
stage='Staging'

# get the latest version from staging to stage
client=MlflowClient()

# get the latest version of model in staging
latest_versions=client.get_latest_versions(name=model_name, stages=[stage])

latest_model_version_staging=latest_versions[0].version

# promotion stage
promotion_stage = 'Production'

client.transition_model_version_stage(
    name=model_name,
    version=latest_model_version_staging,
    stage=promotion_stage,
    archive_existing_versions=True
)



