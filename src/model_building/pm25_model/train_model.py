import yaml
import dagshub
import mlflow

from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import r2_score
from xgboost import XGBRegressor


def train_model(x_train, y_train, logger):
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    hp = params["pm25_model"]["hyperparameters"]

    dagshub.init(
        repo_owner="singh-milind",
        repo_name="aero-metrics",
        mlflow=True
    )

    mlflow.set_tracking_uri("https://dagshub.com/singh-milind/aero-metrics.mlflow")
    mlflow.set_experiment("pm25_model")

    param_grid = {
        "n_estimators": [900, 1000, 1100],
        "max_depth": [6, 7, 8, 9],
        "learning_rate": [0.02, 0.03, 0.04],
        "subsample": [0.8, 0.85, 0.9],
        "colsample_bytree": [0.75, 0.8, 0.85],
        "min_child_weight": [6, 7, 8],
        "reg_alpha": [0.4, 0.5, 0.6],
        "reg_lambda": [1, 2, 3]
    }

    # Create validation split from training data
    x_train_split, x_val, y_train_split, y_val = train_test_split(
        x_train,
        y_train,
        test_size=0.2,
        random_state=42
    )

    best_model = None
    best_params = None
    best_score = float("-inf")

    with mlflow.start_run(run_name="TrainValidationSearch"):

        for i, param in enumerate(ParameterGrid(param_grid)):

            model = XGBRegressor(
                **param,
                objective=hp["objective"],
                random_state=hp["random_state"],
                n_jobs=-1,
                enable_categorical=True
            )

            model.fit(x_train_split, y_train_split)

            train_pred = model.predict(x_train_split)
            val_pred = model.predict(x_val)

            train_r2 = r2_score(y_train_split, train_pred)
            val_r2 = r2_score(y_val, val_pred)
            gap = train_r2 - val_r2

            with mlflow.start_run(
                run_name=f"trial_{i+1}",
                nested=True
            ):
                mlflow.log_params(param)

                mlflow.log_metric("train_r2", train_r2)
                mlflow.log_metric("val_r2", val_r2)
                mlflow.log_metric("generalization_gap", gap)

            if val_r2 > best_score:
                best_score = val_r2
                best_params = param
                best_model = model

        mlflow.log_params(best_params)
        mlflow.log_metric("best_val_r2", best_score)

    logger.info(f"Best Parameters: {best_params}")
    logger.info(f"Best Validation R²: {best_score:.4f}")
    logger.info("Hyperparameter search completed successfully.")

    return best_model