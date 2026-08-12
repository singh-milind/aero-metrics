def plot_feature_importance(model, output_path="feature_importance.png", max_num_features=40):
    import matplotlib.pyplot as plt

    importance = model.get_booster().get_score(importance_type="gain")

    total_gain = sum(importance.values())

    importance_pct = {
        feature: (gain / total_gain) * 100
        for feature, gain in importance.items()
    }

    importance_pct = dict(
        sorted(
            importance_pct.items(),
            key=lambda x: x[1],
            reverse=True
        )[:max_num_features]
    )

    plt.figure(figsize=(10, 8))

    plt.barh(
        list(importance_pct.keys())[::-1],
        list(importance_pct.values())[::-1],
        height=0.6
    )

    plt.xlabel("Importance (%)")
    plt.ylabel("Features")
    plt.title("Feature Importance")
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()