import json
import logging
from pathlib import Path
import warnings
import dagshub
import joblib
import mlflow
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")

dagshub.init(
    repo_owner="guptatannu538",
    repo_name="food_delivery_swiggy_time_prediction",
    mlflow=True,
)
mlflow.set_tracking_uri(
    "https://dagshub.com/guptatannu538/food_delivery_swiggy_time_prediction.mlflow"
)
mlflow.set_experiment("DVC Pipeline")

TARGET = "time_taken"

logger = logging.getLogger("evaluation")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)


def load_data(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        logger.error(f"File does not exist at {data_path}")
        raise FileNotFoundError(f"Data file not found: {data_path}")
    return pd.read_csv(data_path)


def make_X_and_y(data: pd.DataFrame, target_column: str):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y


def load_model(model_path: Path):
    return joblib.load(model_path)


def save_model_info(save_json_path, run_id, artifact_path, model_name):
    info_dict = {
        "run_id": run_id,
        "artifact_path": artifact_path,
        "model_name": model_name,
    }
    with open(save_json_path, "w") as f:
        json.dump(info_dict, f, indent=4)


if __name__ == "__main__":
    root_path = Path(__file__).parent.parent.parent
    train_data_path = root_path / "data" / "processed" / "train_trans.csv"
    test_data_path = root_path / "data" / "processed" / "test_trans.csv"
    model_path = root_path / "models" / "model.joblib"

    train_data = load_data(train_data_path)
    test_data = load_data(test_data_path)

    X_train, y_train = make_X_and_y(train_data, TARGET)
    X_test, y_test = make_X_and_y(test_data, TARGET)

    model = load_model(model_path)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    mean_cv_score = -(cv_scores.mean())

    with mlflow.start_run() as run:
        mlflow.set_tag("model", "Food Delivery Time Regressor")
        mlflow.log_params(model.get_params())

        mlflow.log_metric("train_mae", train_mae)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("train_r2", train_r2)
        mlflow.log_metric("test_r2", test_r2)
        mlflow.log_metric("cross_val_score", mean_cv_score)
        mlflow.log_metrics(
            {f"cv{num}": -score for num, score in enumerate(cv_scores)}
        )

        train_data_input = mlflow.data.from_pandas(train_data, targets=TARGET)
        test_data_input = mlflow.data.from_pandas(test_data, targets=TARGET)
        mlflow.log_input(train_data_input, context="training")
        mlflow.log_input(test_data_input, context="validation")

        signature = mlflow.models.infer_signature(
            model_input=X_train.sample(20, random_state=42),
            model_output=model.predict(X_train.sample(20, random_state=42)),
        )

        model_artifact_path = "delivery_time_pred_model"
        model_name = "delivery_time_pred_model"

        # Log model without registering it here
        mlflow.sklearn.log_model(
            sk_model=model,
            name=model_artifact_path,
            signature=signature,
            serialization_format="cloudpickle",
            registered_model_name=model_name,
        )

        mlflow.log_artifact(root_path / "models" / "stacking_regressor.joblib")
        mlflow.log_artifact(root_path / "models" / "power_transformer.joblib")
        mlflow.log_artifact(root_path / "models" / "preprocessor.joblib")

        run_id = run.info.run_id
        save_json_path = root_path / "run_information.json"

        save_model_info(
            save_json_path=save_json_path,
            run_id=run_id,
            artifact_path=model_artifact_path,
            model_name=model_name,
        )
        logger.info("Model evaluation completed and info saved.")