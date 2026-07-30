import yaml
import dagshub
import mlflow
from sklearn.model_selection import KFold, GridSearchCV
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
        "n_estimators": [500, 700, 900],
        "max_depth": [6, 8, 10],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.8, 0.9],
        "colsample_bytree": [0.8, 0.9],
        "min_child_weight": [3, 5],
        "reg_alpha": [0.1, 0.3],
        "reg_lambda": [1, 2]
    }

    model = XGBRegressor(
        objective=hp["objective"],
        random_state=hp["random_state"],
        n_jobs=-1,
        enable_categorical=True
    )

    kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="r2",
        cv=kf,
        n_jobs=-1,
        verbose=2,
        refit=True,
        return_train_score=True
    )

    grid_search.fit(x_train, y_train)

    with mlflow.start_run(run_name="GridSearchCV"):

        # Log every parameter combination
        for i, param in enumerate(grid_search.cv_results_["params"]):

            with mlflow.start_run(
                run_name=f"trial_{i+1}",
                nested=True
            ):
                mlflow.log_params(param)

                mlflow.log_metric(
                    "mean_cv_r2",
                    grid_search.cv_results_["mean_test_score"][i]
                )

                mlflow.log_metric(
                    "std_cv_r2",
                    grid_search.cv_results_["std_test_score"][i]
                )

                mlflow.log_metric(
                    "mean_train_r2",
                    grid_search.cv_results_["mean_train_score"][i]
                )

                mlflow.log_metric(
                    "generalization_gap",
                    grid_search.cv_results_["mean_train_score"][i]
                    - grid_search.cv_results_["mean_test_score"][i]
                )

                mlflow.log_metric(
                    "fit_time",
                    grid_search.cv_results_["mean_fit_time"][i]
                )

        # Log best model summary
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("best_cv_r2", grid_search.best_score_)

    logger.info(f"Best Parameters: {grid_search.best_params_}")
    logger.info(f"Best CV R²: {grid_search.best_score_:.4f}")
    logger.info("GridSearchCV completed successfully.")

    return grid_search.best_estimator_