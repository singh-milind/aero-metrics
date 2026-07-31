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

    hp = params["pm25_model"]["hyperparameters"]

    dagshub.init(
        repo_owner="singh-milind",
        repo_name="aero-metrics",
        mlflow=True
    )

    mlflow.set_tracking_uri(
        "https://dagshub.com/singh-milind/aero-metrics.mlflow"
    )
    mlflow.set_experiment("pm25_model_new_features")

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
        scoring="r2",
        return_train_score=True,
        n_jobs=-1
    )

    # Train final model on complete training data
    model.fit(x_train, y_train)

    train_scores = scores["train_score"]
    cv_scores = scores["test_score"]

    mean_train_r2 = train_scores.mean()
    std_train_r2 = train_scores.std()

    mean_cv_r2 = cv_scores.mean()
    std_cv_r2 = cv_scores.std()

    gap = mean_train_r2 - mean_cv_r2

    fold_results = pd.DataFrame({
        "Fold": range(1, 6),
        "Train_R2": train_scores,
        "CV_R2": cv_scores,
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
        mlflow.log_metric("generalization_gap", gap)

        # Standard Deviations
        mlflow.log_metric("std_train_r2", std_train_r2)
        mlflow.log_metric("std_cv_r2", std_cv_r2)

        # Fold Statistics
        mlflow.log_metric("best_fold_r2", cv_scores.max())
        mlflow.log_metric("worst_fold_r2", cv_scores.min())

        mlflow.log_metric("best_train_r2", train_scores.max())
        mlflow.log_metric("worst_train_r2", train_scores.min())

        for i in range(5):
            mlflow.log_metric(f"fold_{i+1}_train_r2", train_scores[i])
            mlflow.log_metric(f"fold_{i+1}_cv_r2", cv_scores[i])
            mlflow.log_metric(f"fold_{i+1}_gap", train_scores[i] - cv_scores[i])

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
    logger.info(f"Best Fold CV R²    : {cv_scores.max():.4f}")
    logger.info(f"Worst Fold CV R²   : {cv_scores.min():.4f}")
    logger.info(f"Best Train R²      : {train_scores.max():.4f}")
    logger.info(f"Worst Train R²     : {train_scores.min():.4f}")
    logger.info("=" * 65)

    return model