import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go

API_BASE_URL = st.secrets["API_BASE_URL"]
GLOBAL_SHAP_ENDPOINT = f"{API_BASE_URL}/api/explainer/forecaster/global"

st.title("Global SHAP Analysis — Forecaster")
st.write(
    "Global SHAP analysis shows which features have the greatest overall "
    "influence on the forecaster's predictions for a selected target and horizon."
)
st.divider()

HORIZON_LABELS = {
    "t": "Current / T",
    "t12": "12 Hours",
    "t24": "24 Hours",
    "t48": "48 Hours",
}

TARGET_LABELS = {
    "pm25": "PM2.5",
    "pm10": "PM10",
}

@st.cache_data(ttl=300)
def get_global_shap(target, horizon):
    response = requests.get(
        GLOBAL_SHAP_ENDPOINT,
        params={"target": target, "horizon": horizon},
        timeout=120,
    )
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(
            f"API Error {response.status_code}: {detail}"
        )

    result = response.json()

    df = pd.DataFrame(result.get("data", []))

    if not df.empty:
        df = (
            df.sort_values(
                "importance",
                ascending=False,
            )
            .reset_index(drop=True)
        )

    return result, df


# CONTROLS

col1, col2 = st.columns(2)

with col1:
    target = st.selectbox(
        "Select Target",
        ["pm25", "pm10"],
        format_func=lambda x: TARGET_LABELS[x],
    )

with col2:
    horizon = st.selectbox(
        "Select Horizon",
        ["t", "t12", "t24", "t48"],
        format_func=lambda x: HORIZON_LABELS[x],
    )

try:
    result, shap_df = get_global_shap(
        target,
        horizon,
    )
except Exception as e:
    st.error(str(e))
    st.stop()

if shap_df.empty:
    st.warning(
        "No SHAP data available for the selected model."
    )
    st.stop()

target_label = TARGET_LABELS[target]
horizon_label = HORIZON_LABELS[horizon]


# BEESWARM DATA

beeswarm = result.get("beeswarm")

if not beeswarm:
    st.error(
        "Beeswarm data was not returned by the API."
    )
    st.stop()

feature_names = [
    str(feature)
    for feature in beeswarm.get(
        "feature_names",
        [],
    )
]

shap_values = np.asarray(
    beeswarm.get(
        "shap_values",
        [],
    ),
    dtype=np.float32,
)

feature_values = np.asarray(
    beeswarm.get(
        "feature_values",
        [],
    ),
    dtype=np.float32,
)


# VALIDATE BEESWARM DATA

if len(feature_names) == 0:
    st.error(
        "No beeswarm feature names were returned by the API."
    )
    st.stop()

if shap_values.ndim != 2:
    st.error(
        f"Invalid SHAP array shape: {shap_values.shape}"
    )
    st.stop()

if feature_values.ndim != 2:
    st.error(
        f"Invalid feature value array shape: {feature_values.shape}"
    )
    st.stop()

if shap_values.shape != feature_values.shape:
    st.error(
        f"SHAP/features shape mismatch: "
        f"{shap_values.shape} vs {feature_values.shape}"
    )
    st.stop()

if shap_values.shape[1] != len(feature_names):
    st.error(
        f"Feature mismatch: "
        f"{shap_values.shape[1]} SHAP columns vs "
        f"{len(feature_names)} feature names."
    )
    st.stop()


# METRICS

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Target",
        target_label,
    )

with col2:
    st.metric(
        "Forecast Horizon",
        horizon_label,
    )

with col3:
    st.metric(
        "Features Analyzed",
        len(shap_df),
    )

with col4:
    st.metric(
        "SHAP Samples",
        len(shap_values),
    )

st.divider()


# GLOBAL FEATURE IMPORTANCE

st.subheader(
    f"Global Feature Importance — "
    f"{target_label} ({horizon_label})"
)

plot_df = shap_df.copy()

plot_df["feature"] = (
    plot_df["feature"]
    .astype(str)
    .str.replace("_", " ")
    .str.title()
)

plot_df = plot_df.sort_values(
    "importance",
    ascending=True,
)

fig = go.Figure(
    go.Bar(
        x=plot_df["importance"],
        y=plot_df["feature"],
        orientation="h",
        hovertemplate=(
            "<b>%{y}</b>"
            "<br>Mean |SHAP|: %{x:.4f}"
            "<extra></extra>"
        ),
    )
)

fig.update_layout(
    height=max(
        600,
        len(plot_df) * 30,
    ),
    xaxis_title="Mean |SHAP Value|",
    yaxis_title="Feature",
    showlegend=False,
    margin=dict(
        l=20,
        r=20,
        t=30,
        b=20,
    ),
)

st.plotly_chart(
    fig,
    use_container_width=True,
)

st.divider()


# SHAP BEESWARM

st.subheader(
    f"SHAP Beeswarm — "
    f"{target_label} ({horizon_label})"
)

st.caption(
    "Each point represents a sampled observation. "
    "Positive SHAP values increase the prediction, while negative "
    "SHAP values decrease the prediction. Red indicates higher "
    "feature values and blue indicates lower feature values."
)

max_features = min(
    20,
    len(shap_df),
    len(feature_names),
)

selected_features = (
    shap_df["feature"]
    .astype(str)
    .head(max_features)
    .tolist()
)

feature_indices = [
    feature_names.index(feature)
    for feature in selected_features
    if feature in feature_names
]

fig = go.Figure()

rng = np.random.default_rng(42)

for y_pos, feature_idx in enumerate(feature_indices):
    feature_name = feature_names[feature_idx]

    shap_col = shap_values[:, feature_idx]
    raw_feature_values = feature_values[:, feature_idx]

    valid = np.isfinite(shap_col)

    shap_col = shap_col[valid]
    raw_feature_values = raw_feature_values[valid]

    if len(shap_col) == 0:
        continue

    order = np.argsort(shap_col)

    x = shap_col[order]
    values = raw_feature_values[order]

    finite_values = np.isfinite(values)

    if finite_values.any():
        min_value = np.nanmin(
            values[finite_values]
        )
        max_value = np.nanmax(
            values[finite_values]
        )

        if max_value > min_value:
            normalized = np.full(
                len(values),
                0.5,
                dtype=np.float32,
            )

            normalized[finite_values] = (
                (
                    values[finite_values]
                    - min_value
                )
                / (
                    max_value
                    - min_value
                )
            )
        else:
            normalized = np.full(
                len(values),
                0.5,
                dtype=np.float32,
            )
    else:
        normalized = np.full(
            len(values),
            0.5,
            dtype=np.float32,
        )

    jitter = rng.uniform(
        -0.22,
        0.22,
        len(x),
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=(
                np.full(
                    len(x),
                    y_pos,
                )
                + jitter
            ),
            mode="markers",
            marker=dict(
                size=5,
                color=normalized,
                colorscale="RdBu_r",
                cmin=0,
                cmax=1,
                opacity=0.65,
                showscale=(y_pos == 0),
                colorbar=(
                    dict(
                        title="Feature<br>Value",
                    )
                    if y_pos == 0
                    else None
                ),
            ),
            customdata=values,
            hovertemplate=(
                f"<b>"
                f"{feature_name.replace('_', ' ').title()}"
                f"</b>"
                "<br>SHAP Value: %{x:.4f}"
                "<br>Feature Value: %{customdata:.4f}"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )

display_names = [
    feature.replace(
        "_",
        " ",
    ).title()
    for feature in selected_features
    if feature in feature_names
]

fig.add_vline(
    x=0,
    line_width=1,
    line_dash="dash",
)

fig.update_layout(
    height=max(
        600,
        len(feature_indices) * 38,
    ),
    xaxis_title="SHAP Value",
    yaxis_title="Feature",
    yaxis=dict(
        tickmode="array",
        tickvals=list(
            range(
                len(feature_indices)
            )
        ),
        ticktext=display_names,
        autorange="reversed",
    ),
    showlegend=False,
    margin=dict(
        l=20,
        r=20,
        t=20,
        b=20,
    ),
)

st.plotly_chart(
    fig,
    use_container_width=True,
)

st.caption(
    "Red indicates higher feature values and blue indicates lower "
    "feature values. Points to the right increase the prediction; "
    "points to the left decrease it."
)

st.divider()


# FEATURE IMPORTANCE DETAILS

st.subheader(
    "Feature Importance Details"
)

table_df = shap_df.copy()

table_df.insert(
    0,
    "Rank",
    range(
        1,
        len(table_df) + 1,
    ),
)

table_df["feature"] = (
    table_df["feature"]
    .astype(str)
    .str.replace(
        "_",
        " ",
    )
    .str.title()
)

table_df["importance"] = (
    table_df["importance"]
    .astype(float)
    .round(4)
)

st.dataframe(
    table_df,
    use_container_width=True,
    hide_index=True,
)