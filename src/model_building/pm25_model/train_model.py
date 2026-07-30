import yaml
import dagshub
import mlflow
from sklearn.model_selection import KFold, RandomizedSearchCV
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

    param_dist = {
        "n_estimators": [300, 500, 700, 900, 1200],
        "max_depth": [4, 6, 8, 10],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5, 8, 10],
        "reg_alpha": [0, 0.1, 0.3, 0.5, 1],
        "reg_lambda": [0.5, 1, 2, 5]
    }

    model = XGBRegressor(
        objective=hp["objective"],
        random_state=hp["random_state"],
        n_jobs=-1,
        enable_categorical=True
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=30,
        scoring="r2",
        cv=kf,
        random_state=42,
        n_jobs=-1,
        verbose=2,
        refit=True,
        return_train_score=True
    )

    random_search.fit(x_train, y_train)

    with mlflow.start_run(run_name="RandomizedSearchCV"):

        # Log every sampled parameter combination
        for i, param in enumerate(random_search.cv_results_["params"]):

            with mlflow.start_run(
                run_name=f"trial_{i+1}",
                nested=True
            ):
                mlflow.log_params(param)

                mlflow.log_metric(
                    "mean_cv_r2",
                    random_search.cv_results_["mean_test_score"][i]
                )
                mlflow.log_metric(
                    "std_cv_r2",
                    random_search.cv_results_["std_test_score"][i]
                )
                mlflow.log_metric(
                    "mean_train_r2",
                    random_search.cv_results_["mean_train_score"][i]
                )
                mlflow.log_metric(
                    "fit_time",
                    random_search.cv_results_["mean_fit_time"][i]
                )

        # Log best model summary
        mlflow.log_params(random_search.best_params_)
        mlflow.log_metric("best_cv_r2", random_search.best_score_)

    logger.info(f"Best Parameters: {random_search.best_params_}")
    logger.info(f"Best CV R²: {random_search.best_score_:.4f}")
    logger.info("RandomizedSearchCV completed successfully.")

    return random_search.best_estimator_