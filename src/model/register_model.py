import json
import logging
from pathlib import Path

import dagshub
import mlflow
from mlflow import MlflowClient


# ---------------- LOGGER SETUP ----------------
logger = logging.getLogger("register_model")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)


# ---------------- DAGSHUB INIT ----------------
import dagshub
dagshub.init(repo_owner='guptatannu538', repo_name='food_delivery_swiggy_time_prediction', mlflow=True)




# ---------------- UTILS ----------------
def load_model_information(file_path: Path) -> dict:
    if not file_path.exists():
        raise FileNotFoundError(f"Run info file not found: {file_path}")
    with open(file_path, "r") as f:
        return json.load(f)


# ---------------- MAIN ----------------
if __name__ == "__main__":
    root_path = Path(__file__).parent.parent.parent
    run_info_path = root_path / "run_information.json"

    # load run info
    run_info = load_model_information(run_info_path)

    run_id = run_info["run_id"]
    # We'll use the name from the JSON, which I updated to "model"
    model_name = run_info["model_name"]

    logger.info(f"Using run_id: {run_id}")
    logger.info(f"Using model_name: {model_name}")

    tracking_uri = "https://dagshub.com/guptatannu538/food_delivery_swiggy_time_prediction.mlflow"
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    logger.info(f"Checking for artifacts in run {run_id}...")
    artifacts = client.list_artifacts(run_id)
    artifact_paths = [a.path for a in artifacts]
    logger.info(f"Found artifacts: {artifact_paths}")

    if model_name not in artifact_paths:
        # Fallback check for "model" if the specific name isn't found
        if "model" in artifact_paths:
            logger.info("Specific model_name not found, but 'model' folder exists. Using 'model'.")
            model_name = "model"
        else:
            raise RuntimeError(
                f"Model artifact '{model_name}' not found in run {run_id}. "
                f"Found artifacts: {artifact_paths}. "
                f"Evaluation stage may be cached or failed."
            )

    # -------- REGISTER MODEL --------
    model_uri = f"runs:/{run_id}/{model_name}"
    logger.info(f"Registering model from URI: {model_uri}")

    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
    )

    logger.info(
        f"Model registered: name={model_version.name}, "
        f"version={model_version.version}"
    )

    # -------- MOVE TO STAGING --------
    client.transition_model_version_stage(
        name=model_version.name,
        version=model_version.version,
        stage="Staging",
        archive_existing_versions=False,
    )

    logger.info("Model successfully promoted to STAGING")
    
    