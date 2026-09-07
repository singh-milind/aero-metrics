import plotly.graph_objects as go
import pandas as pd

def plot_shap_waterfall(shap_result, baseline, target,prediction, top_n=10,):

    df = pd.DataFrame(shap_result)

    # Sort by contribution magnitude
    df = df.sort_values("abs_shap", ascending=False)

    # Keep top N
    top = df.head(top_n).copy()

    # Aggregate remaining features
    other_shap = df.iloc[top_n:]["shap_value"].sum()

    if abs(other_shap) > 0:
        top = pd.concat([
            top,
            pd.DataFrame({
                "feature": ["Other features"],
                "shap_value": [other_shap],
                "abs_shap": [abs(other_shap)]
            })
        ], ignore_index=True)

    # Reverse so largest contribution appears near top
    top = top.iloc[::-1]

    features = top["feature"].tolist()
    contributions = top["shap_value"].tolist()

    # Build cumulative values
    cumulative = baseline
    x_values = [baseline]

    for value in contributions:
        cumulative += value
        x_values.append(cumulative)

    fig = go.Figure()

    # Baseline
    fig.add_trace(go.Bar(
        x=[baseline],
        y=["Baseline"],
        orientation="h",
        text=[f"{baseline:.2f}"],
        textposition="outside",
        name="Baseline"
    ))

    # Contributions
    current = baseline

    for feature, shap_value in zip(features, contributions):

        fig.add_trace(go.Bar(
            x=[shap_value],
            y=[feature],
            orientation="h",
            base=current,
            text=[f"{shap_value:+.2f}"],
            textposition="outside",
            name=feature,
            showlegend=False
        ))

        current += shap_value

    # Prediction
    fig.add_trace(go.Bar(
        x=[prediction],
        y=["Prediction"],
        orientation="h",
        text=[f"{prediction:.2f}"],
        textposition="outside",
        name="Prediction"
    ))

    fig.update_layout(
        title="SHAP Feature Contributions",
        xaxis_title=f"{target} Prediction",
        yaxis_title="",
        barmode="overlay",
        height=650,
        showlegend=False
    )

    return fig