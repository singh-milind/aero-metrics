import yaml
import dagshub
import mlflow
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import KFold, cross_validate
from xgboost import XGBRegressor, plot_importance


def train_model(x_train, y_train, logger):

    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    hp = params["pm10_predictor"]["hyperparameters"]

    dagshub.init(
        repo_owner="singh-milind",
        repo_name="aero-metrics",
        mlflow=True
    )

    mlflow.set_tracking_uri(
        "https://dagshub.com/singh-milind/aero-metrics.mlflow"
    )
    mlflow.set_experiment("pm10_predictor_new_data")

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

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=hp["random_state"]
    )

    scores = cross_validate(
        estimator=model,
        X=x_train,
        y=y_train,
        cv=cv,
        scoring={
        "r2": "r2",
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error"},
        return_train_score=True,
        n_jobs=-1
    )

    # Train final model on complete training data
    model.fit(x_train, y_train)

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

    fold_results.to_csv("cv_results.csv", index=False)

    plt.figure(figsize=(10, 8))
    plot_importance(
        model,
        importance_type="gain",
        max_num_features=40,
        height=0.6
    )
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=300)
    plt.close()

    with mlflow.start_run():

        # Hyperparameters
        mlflow.log_params(hp)

        # Mean Metrics
        mlflow.log_metric("mean_train_r2", mean_train_r2)
        mlflow.log_metric("mean_cv_r2", mean_cv_r2)
        mlflow.log_metric("mean_train_mae", mean_train_mae)
        mlflow.log_metric("mean_cv_mae", mean_cv_mae)
        mlflow.log_metric("mean_train_rmse", mean_train_rmse)
        mlflow.log_metric("mean_cv_rmse", mean_cv_rmse)
        mlflow.log_metric("generalization_gap", gap)

        # Standard Deviations
        mlflow.log_metric("std_train_r2", std_train_r2)
        mlflow.log_metric("std_train_mae", std_train_mae)
        mlflow.log_metric("std_train_rmse", std_train_rmse)
        mlflow.log_metric("std_cv_r2", std_cv_r2)
        mlflow.log_metric("std_cv_mae", std_cv_mae)
        mlflow.log_metric("std_cv_rmse", std_cv_rmse)

        # Fold Statistics
        mlflow.log_metric("best_fold_r2", cv_scores.max())
        mlflow.log_metric("worst_fold_r2", cv_scores.min())

        mlflow.log_metric("best_train_r2", train_scores.max())
        mlflow.log_metric("worst_train_r2", train_scores.min())

        # Artifacts
        mlflow.log_artifact("feature_importance.png")
        mlflow.log_artifact("cv_results.csv")

        # Model
        mlflow.xgboost.log_model(model, "model")

    logger.info("=" * 65)
    logger.info("           5-FOLD CROSS VALIDATION SUMMARY")
    logger.info("=" * 65)
    logger.info(f"Mean Train R²      : {mean_train_r2:.4f} ± {std_train_r2:.4f}")
    logger.info(f"Mean CV R²         : {mean_cv_r2:.4f} ± {std_cv_r2:.4f}")
    logger.info(f"Generalization Gap : {gap:.4f}")
    logger.info("-" * 65)
    logger.info(f"Mean Train MAE     : {mean_train_mae:.4f} ± {std_train_mae:.4f}")
    logger.info(f"Mean CV MAE        : {mean_cv_mae:.4f} ± {std_cv_mae:.4f}")
    logger.info(f"Mean Train RMSE    : {mean_train_rmse:.4f} ± {std_train_rmse:.4f}")
    logger.info(f"Mean CV RMSE       : {mean_cv_rmse:.4f} ± {std_cv_rmse:.4f}")
    logger.info("-" * 65)
    logger.info(f"Best Fold CV R²    : {cv_scores.max():.4f}")
    logger.info(f"Worst Fold CV R²   : {cv_scores.min():.4f}")
    logger.info(f"Best Train R²      : {train_scores.max():.4f}")
    logger.info(f"Worst Train R²     : {train_scores.min():.4f}")
    logger.info("=" * 65)

    return model