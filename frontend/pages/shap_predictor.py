import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go

API_BASE_URL = st.secrets["API_BASE_URL"]
GLOBAL_SHAP_ENDPOINT = f"{API_BASE_URL}/api/explainer/predictor/global"

st.title("Global SHAP Analysis")
st.write(
    "Global SHAP analysis shows which features have the greatest overall "
    "influence on the model's predictions."
)
st.divider()

@st.cache_data(ttl=300)
def get_global_shap(target):
    response = requests.get(
        GLOBAL_SHAP_ENDPOINT,
        params={"target": target},
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
    return response.json()

target = st.selectbox(
    "Select Target",
    ["pm25", "pm10"],
    format_func=lambda x: "PM2.5" if x == "pm25" else "PM10",
)

try:
    result = get_global_shap(target)
except Exception as e:
    st.error(str(e))
    st.stop()

# GLOBAL IMPORTANCE DATA
shap_df = pd.DataFrame(result.get("data", []))

if shap_df.empty:
    st.warning("No SHAP importance data was returned by the API.")
    st.stop()

shap_df = (
    shap_df
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)

# BEESWARM DATA
beeswarm = result.get("beeswarm", {})

feature_names = [
    str(feature)
    for feature in beeswarm.get("feature_names", [])
]

shap_values = np.asarray(
    beeswarm.get("shap_values", []),
    dtype=np.float32,
)

feature_values = np.asarray(
    beeswarm.get("feature_values", []),
    dtype=np.float32,
)

# VALIDATE BEESWARM DATA
if len(feature_names) > 0:
    if shap_values.ndim != 2:
        st.error(
            f"Invalid SHAP values returned by API: {shap_values.shape}"
        )
        st.stop()

    if feature_values.ndim != 2:
        st.error(
            f"Invalid feature values returned by API: {feature_values.shape}"
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
            f"Feature count mismatch: "
            f"{shap_values.shape[1]} SHAP columns vs "
            f"{len(feature_names)} feature names."
        )
        st.stop()

# METRICS
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Features Analyzed",
        len(shap_df),
    )

with col2:
    most_important = (
        str(shap_df.iloc[0]["feature"])
        .replace("_", " ")
        .title()
        if len(shap_df) > 0
        else "N/A"
    )

    st.metric(
        "Most Important Feature",
        most_important,
    )

with col3:
    st.metric(
        "SHAP Samples",
        len(shap_values),
    )

st.divider()

# GLOBAL FEATURE IMPORTANCE
st.subheader(
    f"Global Feature Importance — {target.upper()}"
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
    height=max(600, len(plot_df) * 30),
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
    f"SHAP Beeswarm — {target.upper()}"
)

st.caption(
    "Each point represents a sampled observation. "
    "Positive SHAP values push the prediction higher, "
    "while negative values push it lower."
)

if (
    len(feature_names) == 0
    or shap_values.size == 0
    or feature_values.size == 0
):
    st.info("Beeswarm data is not available.")
else:
    # TOP 20 FEATURES
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

    # Match importance features to the columns returned by API
    feature_indices = [
        feature_names.index(feature)
        for feature in selected_features
        if feature in feature_names
    ]

    if len(feature_indices) == 0:
        st.warning(
            "None of the global importance features were found "
            "in the beeswarm feature data."
        )
    else:
        fig = go.Figure()

        rng = np.random.default_rng(42)

        plotted_features = []

        for y_pos, feature_idx in enumerate(feature_indices):
            feature_name = feature_names[feature_idx]

            shap_col = shap_values[:, feature_idx]
            feature_col = feature_values[:, feature_idx]

            # Remove invalid values
            valid = (
                np.isfinite(shap_col)
                & np.isfinite(feature_col)
            )

            shap_col = shap_col[valid]
            feature_col = feature_col[valid]

            if len(shap_col) == 0:
                continue

            plotted_features.append(feature_name)

            # Sort by SHAP value
            order = np.argsort(shap_col)

            x = shap_col[order]
            values = feature_col[order]

            # Normalize feature values to 0-1
            value_min = np.nanmin(values)
            value_max = np.nanmax(values)

            if (
                np.isfinite(value_min)
                and np.isfinite(value_max)
                and value_max > value_min
            ):
                normalized = (
                    (values - value_min)
                    / (value_max - value_min)
                )
            else:
                normalized = np.full(
                    len(values),
                    0.5,
                    dtype=np.float32,
                )

            # Vertical jitter
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
                            dtype=np.float32,
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
            feature.replace("_", " ").title()
            for feature in plotted_features
        ]

        fig.add_vline(
            x=0,
            line_width=1,
            line_dash="dash",
        )

        fig.update_layout(
            height=max(
                600,
                len(plotted_features) * 35,
            ),
            xaxis_title="SHAP Value",
            yaxis_title="Feature",
            yaxis=dict(
                tickmode="array",
                tickvals=list(
                    range(len(plotted_features))
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
            "Red indicates higher feature values and blue indicates "
            "lower feature values. Points to the right increase the "
            "prediction; points to the left decrease it."
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
    .str.replace("_", " ")
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