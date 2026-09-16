import json
import logging
from pathlib import Path

import dagshub
import mlflow
from mlflow.tracking import MlflowClient

# Logger Setup
logger = logging.getLogger("register_model")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
)
logger.addHandler(handler)

# Initialize DagsHub
dagshub.init(
    repo_owner="guptatannu538",
    repo_name="food_delivery_swiggy_time_prediction",
    mlflow=True,
)
mlflow.set_tracking_uri(
    "https://dagshub.com/guptatannu538/food_delivery_swiggy_time_prediction.mlflow"
)


def load_model_information(file_path: Path) -> dict:
    with open(file_path) as f:
        return json.load(f)


if __name__ == "__main__":
    root_path = Path(__file__).parent.parent.parent
    run_info_path = root_path / "run_information.json"

    run_info = load_model_information(run_info_path)
    run_id = run_info["run_id"]
    model_name = run_info["model_name"]
    artifact_path=run_info['artifact_path']

    client = MlflowClient()

    model_uri = f"runs:/{run_id}/{artifact_path}"

# 2. Register the model directly
model_version = mlflow.register_model(
    model_uri=model_uri,
    name=model_name
)

# 3. Transition to Staging using the version number returned
client = MlflowClient()
client.transition_model_version_stage(
    name=model_name,
    version=model_version.version,
    stage="Staging"
)

    
logger.info(
    f"Model '{model_name}' version {model_version} successfully transitioned to Staging"
)
