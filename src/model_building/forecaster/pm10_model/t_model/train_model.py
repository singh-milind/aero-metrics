import yaml
import dagshub
import mlflow
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import TimeSeriesSplit, cross_validate
from xgboost import XGBRegressor

from src.utils.plot_importance import plot_feature_importance
from metrics.metrics import make_metrics_dict, dump_metrics_json


def train_model(x_train, y_train, logger):

    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    hp = params["pm10_forecaster_t"]["hyperparameters"]

    import os

    dagshub.init(
        repo_owner=os.getenv("DAGSHUB_USERNAME"),
        repo_name="aero-metrics",
        mlflow=True
    )

    mlflow.set_tracking_uri(
        f"https://dagshub.com/{os.getenv('DAGSHUB_USERNAME')}/aero-metrics.mlflow"
    )

    mlflow.set_experiment("pm10_forecaster_t_ratio_production")

   
    q90 = y_train.quantile(0.90)
    q95 = y_train.quantile(0.95)
    q97 = y_train.quantile(0.97)

    weights = np.select(
        [
            y_train >= q97,
            y_train >= q95,
            y_train >= q90,
        ],
        [
            1.75,
            1.50,
            1.25,
        ],
        default=1.00
    )
    
    model = XGBRegressor(
        objective=hp["objective"],
        random_state=hp["random_state"],
        n_estimators=hp["n_estimators"],
        max_depth=hp["max_depth"],
        learning_rate=hp["learning_rate"],
        subsample=hp["subsample"],
        colsample_bytree=hp["colsample_bytree"],
        min_child_weight=hp["min_child_weight"],
        reg_alpha=hp["reg_alpha"],
        reg_lambda=hp["reg_lambda"],
        enable_categorical=True,
        n_jobs=-1
    )

    cv = TimeSeriesSplit(
        n_splits=5
    )

    scores = cross_validate(
        estimator=model,
        X=x_train,
        y=y_train,
        cv=cv,
        scoring={
            "r2": "r2",
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error"
        },
        params={
            "sample_weight": weights},
        return_train_score=True,
        n_jobs=-1
    )

    model.fit(x_train, y_train, sample_weight=weights)

    train_scores = scores["train_r2"]
    cv_scores = scores["test_r2"]

    train_mae = -scores["train_mae"]
    cv_mae = -scores["test_mae"]

    train_rmse = -scores["train_rmse"]
    cv_rmse = -scores["test_rmse"]

    mean_train_r2 = train_scores.mean()
    mean_train_mae = train_mae.mean()
    mean_train_rmse = train_rmse.mean()

    std_train_r2 = train_scores.std()
    std_train_mae = train_mae.std()
    std_train_rmse = train_rmse.std()

    mean_cv_r2 = cv_scores.mean()
    mean_cv_mae = cv_mae.mean()
    mean_cv_rmse = cv_rmse.mean()

    std_cv_r2 = cv_scores.std()
    std_cv_mae = cv_mae.std()
    std_cv_rmse = cv_rmse.std()

    gap = mean_train_r2 - mean_cv_r2

    fold_results = pd.DataFrame({
        "Fold": range(1, 6),
        "Train_R2": train_scores,
        "CV_R2": cv_scores,
        "Train_MAE": train_mae,
        "CV_MAE": cv_mae,
        "Train_RMSE": train_rmse,
        "CV_RMSE": cv_rmse,
        "Gap": train_scores - cv_scores
    })

    fold_results.to_csv(
        "cv_results.csv",
        index=False
    )

    plot_feature_importance(model)

    metrics = make_metrics_dict(
        mean_train_r2,
        mean_cv_r2,
        mean_train_mae,
        mean_cv_mae,
        mean_train_rmse,
        mean_cv_rmse,
        std_train_r2,
        std_cv_r2,
        std_train_mae,
        std_cv_mae,
        std_train_rmse,
        std_cv_rmse,
        gap
    )

    dump_metrics_json(
        metrics,
        model_type="forecaster",
        model_sub_type="pm10_model",
        model_name="t_model"
    )

    logger.info(
        "Metrics saved to metrics/forecaster/pm10_model/t_model_metrics.json"
    )

    with mlflow.start_run():

        mlflow.log_params(hp)

        mlflow.log_metric(
            "mean_train_r2",
            mean_train_r2
        )
        mlflow.log_metric(
            "mean_cv_r2",
            mean_cv_r2
        )
        mlflow.log_metric(
            "mean_train_mae",
            mean_train_mae
        )
        mlflow.log_metric(
            "mean_cv_mae",
            mean_cv_mae
        )
        mlflow.log_metric(
            "mean_train_rmse",
            mean_train_rmse
        )
        mlflow.log_metric(
            "mean_cv_rmse",
            mean_cv_rmse
        )
        mlflow.log_metric(
            "generalization_gap",
            gap
        )

        mlflow.log_metric(
            "std_train_r2",
            std_train_r2
        )
        mlflow.log_metric(
            "std_train_mae",
            std_train_mae
        )
        mlflow.log_metric(
            "std_train_rmse",
            std_train_rmse
        )
        mlflow.log_metric(
            "std_cv_r2",
            std_cv_r2
        )
        mlflow.log_metric(
            "std_cv_mae",
            std_cv_mae
        )
        mlflow.log_metric(
            "std_cv_rmse",
            std_cv_rmse
        )

        mlflow.log_metric(
            "best_fold_r2",
            cv_scores.max()
        )
        mlflow.log_metric(
            "worst_fold_r2",
            cv_scores.min()
        )

        mlflow.log_metric(
            "best_train_r2",
            train_scores.max()
        )
        mlflow.log_metric(
            "worst_train_r2",
            train_scores.min()
        )

        mlflow.log_artifact(
            "feature_importance.png"
        )

        mlflow.log_artifact(
            "cv_results.csv"
        )

        mlflow.xgboost.log_model(
            model,
            "model"
        )

    logger.info("=" * 65)
    logger.info("       5-FOLD TIME SERIES CROSS VALIDATION")
    logger.info("=" * 65)
    logger.info(
        f"Mean Train R²      : "
        f"{mean_train_r2:.4f} ± {std_train_r2:.4f}"
    )
    logger.info(
        f"Mean CV R²         : "
        f"{mean_cv_r2:.4f} ± {std_cv_r2:.4f}"
    )
    logger.info(
        f"Generalization Gap : {gap:.4f}"
    )
    logger.info("-" * 65)
    logger.info(
        f"Mean Train MAE     : "
        f"{mean_train_mae:.4f} ± {std_train_mae:.4f}"
    )
    logger.info(
        f"Mean CV MAE        : "
        f"{mean_cv_mae:.4f} ± {std_cv_mae:.4f}"
    )
    logger.info(
        f"Mean Train RMSE    : "
        f"{mean_train_rmse:.4f} ± {std_train_rmse:.4f}"
    )
    logger.info(
        f"Mean CV RMSE       : "
        f"{mean_cv_rmse:.4f} ± {std_cv_rmse:.4f}"
    )
    logger.info("-" * 65)
    logger.info(
        f"Best Fold CV R²    : {cv_scores.max():.4f}"
    )
    logger.info(
        f"Worst Fold CV R²   : {cv_scores.min():.4f}"
    )
    logger.info(
        f"Best Train R²      : {train_scores.max():.4f}"
    )
    logger.info(
        f"Worst Train R²     : {train_scores.min():.4f}"
    )
    logger.info("=" * 65)

    return model