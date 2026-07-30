import yaml
import dagshub
import mlflow
from sklearn.model_selection import KFold, cross_val_score
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

    model = XGBRegressor(
        n_estimators=hp["n_estimators"],
        learning_rate=hp["learning_rate"],
        max_depth=hp["max_depth"],
        random_state=hp["random_state"],
        subsample=hp["subsample"],
        colsample_bytree=hp["colsample_bytree"],
        min_child_weight=hp["min_child_weight"],
        reg_alpha=hp["reg_alpha"],
        reg_lambda=hp["reg_lambda"],
        objective=hp["objective"],
        n_jobs=-1,
        enable_categorical=True
    )

    # Cross Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_r2 = cross_val_score(
        model,
        x_train,
        y_train,
        cv=kf,
        scoring="r2"
    )

    cv_rmse = -cross_val_score(
        model,
        x_train,
        y_train,
        cv=kf,
        scoring="neg_root_mean_squared_error"
    )

    cv_mae = -cross_val_score(
        model,
        x_train,
        y_train,
        cv=kf,
        scoring="neg_mean_absolute_error"
    )

    # Train final model on complete training data
    model.fit(x_train, y_train)

    # MLflow Logging
    with mlflow.start_run():

        mlflow.log_params(hp)

        mlflow.log_metric("cv_r2_mean", cv_r2.mean())
        mlflow.log_metric("cv_r2_std", cv_r2.std())

        mlflow.log_metric("cv_rmse_mean", cv_rmse.mean())
        mlflow.log_metric("cv_rmse_std", cv_rmse.std())

        mlflow.log_metric("cv_mae_mean", cv_mae.mean())
        mlflow.log_metric("cv_mae_std", cv_mae.std())

    logger.info(
        f"Cross Validation Results | "
        f"MAE: {cv_mae.mean():.4f} ± {cv_mae.std():.4f} | "
        f"RMSE: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f} | "
        f"R2: {cv_r2.mean():.4f} ± {cv_r2.std():.4f}"
    )

    logger.info("Final model trained successfully on the complete training dataset.")

    return model