import os

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import textwrap
from resources.predictor_metadata import get_feature_metadata

st.markdown(
    """
    <style>
    .shap-kicker {
        color: #38b9ff;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

API_BASE_URL = os.getenv("API_BASE_URL") or st.secrets["API_BASE_URL"]
GLOBAL_SHAP_ENDPOINT = f"{API_BASE_URL}/api/explainer/predictor/global"

FEATURE_METADATA = {
    "temperature_2m": {
        "description": "Air temperature at 2 metres above ground level.",
        "unit": "°C",
        "type": "Raw meteorological feature",
    },
    "relative_humidity_2m": {
        "description": "Relative humidity at 2 metres above ground level.",
        "unit": "%",
        "type": "Raw meteorological feature",
    },
    "wind_speed_10m": {
        "description": "Wind speed at 10 metres above ground level.",
        "unit": "m/s",
        "type": "Raw meteorological feature",
    },
    "wind_direction_10m": {
        "description": "Wind direction at 10 metres above ground level.",
        "unit": "degrees",
        "type": "Raw meteorological feature",
    },
    "surface_pressure": {
        "description": "Atmospheric pressure at the surface.",
        "unit": "hPa",
        "type": "Raw meteorological feature",
    },
    "precipitation": {
        "description": "Precipitation amount associated with the observation.",
        "unit": "mm",
        "type": "Raw meteorological feature",
    },
    "city": {
        "description": "City or location category used by the model.",
        "unit": None,
        "type": "Categorical feature",
    },
    "weather_verdict": {
        "description": "Categorical summary of the prevailing weather condition.",
        "unit": None,
        "type": "Categorical feature",
    },
    "time_of_day": {
        "description": "Categorical period: Morning, Afternoon, Evening, or Midnight.",
        "unit": None,
        "type": "Categorical feature",
    },
    "season_region": {
        "description": "Combined feature representing season and geographic region.",
        "unit": None,
        "type": "Categorical engineered feature",
    },
    "temp_humidity": {
        "description": "Interaction between temperature and relative humidity.",
        "unit": "°C·%",
        "type": "Engineered interaction feature",
    },
    "wind_precip": {
        "description": "Interaction between wind speed and precipitation.",
        "unit": "m/s·mm",
        "type": "Engineered interaction feature",
    },
    "pressure_temp": {
        "description": "Interaction between surface pressure and temperature.",
        "unit": "hPa·°C",
        "type": "Engineered interaction feature",
    },
    "wind_dir_sin": {
        "description": "Sine transformation of wind direction.",
        "unit": None,
        "type": "Cyclic engineered feature",
    },
    "wind_dir_cos": {
        "description": "Cosine transformation of wind direction.",
        "unit": None,
        "type": "Cyclic engineered feature",
    },
    "month_sin": {
        "description": "Sine transformation encoding annual seasonality.",
        "unit": None,
        "type": "Cyclic time feature",
    },
    "month_cos": {
        "description": "Cosine transformation encoding annual seasonality.",
        "unit": None,
        "type": "Cyclic time feature",
    },
    "hour_sin": {
        "description": "Sine transformation encoding daily time patterns.",
        "unit": None,
        "type": "Cyclic time feature",
    },
    "hour_cos": {
        "description": "Cosine transformation encoding daily time patterns.",
        "unit": None,
        "type": "Cyclic time feature",
    },
    "dow_sin": {
        "description": "Sine transformation encoding weekly patterns.",
        "unit": None,
        "type": "Cyclic time feature",
    },
    "dow_cos": {
        "description": "Cosine transformation encoding weekly patterns.",
        "unit": None,
        "type": "Cyclic time feature",
    },
    "is_weekend": {
        "description": "Binary indicator for whether the day is a weekend.",
        "unit": None,
        "type": "Binary time feature",
    },
}

FEATURE_METADATA = get_feature_metadata()


def feature_label(feature):
    return str(feature).replace("_", " ").title()


def feature_metadata(feature):
    return FEATURE_METADATA.get(
        str(feature),
        {
            "description": "Metadata is not available for this feature.",
            "unit": None,
            "type": "Unknown feature",
        },
    )

st.markdown('<p class="shap-kicker">Model explanations · predictor</p>', unsafe_allow_html=True)
st.title("Understand predictor influence")
st.write("See which features most influence PM2.5 and PM10 predictions across the full evaluation dataset.")


def compact_description(description, width=58):
    lines = textwrap.wrap(str(description), width=width)
    if len(lines) <= 2:
        return "\n".join(lines)
    second_line = lines[1][: width - 3].rstrip()
    return f"{lines[0]}\n{second_line}..."


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

with st.container(border=True):
    st.markdown("#### Explanation target")
    target = st.selectbox(
        "Select target model",
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
st.markdown("### Explanation overview")
col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    with st.container(border=True):
        st.metric("Features analyzed", len(shap_df))

with col2:
    most_important = (
        str(shap_df.iloc[0]["feature"])
        .replace("_", " ")
        .title()
        if len(shap_df) > 0
        else "N/A"
    )

    with st.container(border=True):
        st.metric("Most important feature", most_important)

with col3:
    with st.container(border=True):
        st.metric("SHAP samples", len(shap_values))

st.markdown("### Global feature importance")
st.caption("Mean absolute SHAP values show the overall influence of each feature on the selected model.")

with st.expander("Feature guide"):
    st.caption("Feature meanings and categories come from the predictor metadata used by the explanation service.")
    metadata_df = pd.DataFrame(
        [
            {
                "Feature": feature_label(feature),
                "Description": compact_description(metadata["description"]),
                "Unit": metadata["unit"] or "—",
                "Type": metadata["type"],
            }
            for feature in shap_df["feature"].astype(str)
            for metadata in [feature_metadata(feature)]
        ]
    )
    st.dataframe(
        metadata_df,
        column_config={
            "Feature": st.column_config.TextColumn("Feature", width="medium"),
            "Description": st.column_config.TextColumn(
                "Description",
                width="large",
                help="Compact two-line feature description.",
            ),
            "Unit": st.column_config.TextColumn("Unit", width="small"),
            "Type": st.column_config.TextColumn("Type", width="medium"),
        },
        use_container_width=True,
        hide_index=True,
    )

plot_df = shap_df.copy()

plot_df["feature"] = (
    plot_df["feature"]
    .astype(str)
    .map(feature_label)
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

with st.container(border=True):
    st.plotly_chart(fig, use_container_width=True)

st.markdown("### SHAP beeswarm")
st.caption("Each point is a sampled observation. Points to the right increase the prediction; points to the left decrease it.")

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

        display_names = [feature_label(feature) for feature in plotted_features]

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

        with st.container(border=True):
            st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Red indicates higher feature values and blue indicates "
            "lower feature values. Points to the right increase the "
            "prediction; points to the left decrease it."
        )

st.markdown("### Feature importance details")
st.caption("Ranked values behind the global importance chart.")

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
    .map(feature_label)
)

table_df["importance"] = (
    table_df["importance"]
    .astype(float)
    .round(4)
)

table_df.insert(
    2,
    "Description",
    [
        compact_description(feature_metadata(feature)["description"])
        for feature in shap_df["feature"].astype(str)
    ],
)
table_df.insert(
    3,
    "Unit",
    [
        feature_metadata(feature)["unit"] or "—"
        for feature in shap_df["feature"].astype(str)
    ],
)
table_df.insert(
    4,
    "Type",
    [
        feature_metadata(feature)["type"]
        for feature in shap_df["feature"].astype(str)
    ],
)

with st.container(border=True):
    st.dataframe(
        table_df,
        column_config={
            "Description": st.column_config.TextColumn(
                "Description",
                width="large",
                help="Compact two-line feature description.",
            ),
        },
        use_container_width=True,
        hide_index=True,
    )