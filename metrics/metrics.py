import json
from pathlib import Path


def make_metrics_dict(
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
    gap,
):
    metrics = {
        "mean_train_r2": round(float(mean_train_r2), 4),
        "mean_cv_r2": round(float(mean_cv_r2), 4),

        "mean_train_mae": round(float(mean_train_mae), 4),
        "mean_cv_mae": round(float(mean_cv_mae), 4),

        "mean_train_rmse": round(float(mean_train_rmse), 4),
        "mean_cv_rmse": round(float(mean_cv_rmse), 4),

        "std_train_r2": round(float(std_train_r2), 4),
        "std_cv_r2": round(float(std_cv_r2), 4),

        "std_train_mae": round(float(std_train_mae), 4),
        "std_cv_mae": round(float(std_cv_mae), 4),

        "std_train_rmse": round(float(std_train_rmse), 4),
        "std_cv_rmse": round(float(std_cv_rmse), 4),

        "generalization_gap": round(float(gap), 4),
    }

    return metrics


def dump_metrics_json(metrics, model_type,model_name):

    root_dir = Path(__file__).resolve().parents[1]

    metrics_path = (
        root_dir
        / "metrics"
        / f"{model_type}"
        / f"{model_name}_metrics.json"
    )

    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)