import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go

API_BASE_URL = st.secrets["API_BASE_URL"]
GLOBAL_SHAP_ENDPOINT = f"{API_BASE_URL}/api/explainer/predictor/global"

st.title("Global SHAP Analysis")
st.write("Global SHAP analysis shows which features have the greatest overall influence on the model's predictions.")
st.divider()

@st.cache_data(ttl=300)
def get_global_shap(target):
    response = requests.get(
        GLOBAL_SHAP_ENDPOINT,
        params={"target": target},
        timeout=60,
    )
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"API Error {response.status_code}: {detail}")
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

shap_df = pd.DataFrame(result["data"])
shap_df = shap_df.sort_values("importance", ascending=False).reset_index(drop=True)

beeswarm = result["beeswarm"]
feature_names = beeswarm["feature_names"]
shap_values = np.asarray(beeswarm["shap_values"], dtype=float)
feature_values = pd.DataFrame(beeswarm["feature_values"])

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Features Analyzed", len(shap_df))

with col2:
    st.metric(
        "Most Important Feature",
        str(shap_df.iloc[0]["feature"]).replace("_", " ").title(),
    )

with col3:
    st.metric("SHAP Samples", len(shap_values))

st.divider()

# GLOBAL FEATURE IMPORTANCE

st.subheader(f"Global Feature Importance — {target.upper()}")

plot_df = shap_df.copy()
plot_df["feature"] = plot_df["feature"].str.replace("_", " ").str.title()
plot_df = plot_df.sort_values("importance", ascending=True)

fig = go.Figure(
    go.Bar(
        x=plot_df["importance"],
        y=plot_df["feature"],
        orientation="h",
        hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>",
    )
)

fig.update_layout(
    height=max(600, len(plot_df) * 30),
    xaxis_title="Mean |SHAP Value|",
    yaxis_title="Feature",
    showlegend=False,
    margin=dict(l=20, r=20, t=30, b=20),
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# SHAP BEESWARM

st.subheader(f"SHAP Beeswarm — {target.upper()}")

st.caption(
    "Each point represents a sampled observation. "
    "Positive SHAP values push the prediction higher, while negative values push it lower."
)

# Number of features shown in beeswarm
max_features = min(20, len(feature_names))

selected_features = shap_df["feature"].head(max_features).tolist()

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
    feature_col = pd.to_numeric(
        feature_values[feature_name],
        errors="coerce",
    ).to_numpy()

    valid = np.isfinite(shap_col)

    shap_col = shap_col[valid]
    feature_col = feature_col[valid]

    if len(shap_col) == 0:
        continue

    # Horizontal jitter to create beeswarm-like distribution
    order = np.argsort(shap_col)

    x = shap_col[order]
    values = feature_col[order]

    # Normalize feature values for marker coloring
    if len(values) > 1 and np.nanmax(values) != np.nanmin(values):
        normalized = (
            (values - np.nanmin(values))
            / (np.nanmax(values) - np.nanmin(values))
        )
    else:
        normalized = np.full(len(values), 0.5)

    # Vertical jitter
    jitter = rng.uniform(-0.22, 0.22, len(x))

    fig.add_trace(
        go.Scatter(
            x=x,
            y=np.full(len(x), y_pos) + jitter,
            mode="markers",
            marker=dict(
                size=5,
                color=normalized,
                colorscale="RdBu_r",
                showscale=(y_pos == 0),
                colorbar=dict(
                    title="Feature<br>Value",
                ) if y_pos == 0 else None,
                opacity=0.65,
            ),
            customdata=np.column_stack([values]),
            hovertemplate=(
                f"<b>{feature_name.replace('_', ' ').title()}</b>"
                "<br>SHAP: %{x:.4f}"
                "<br>Feature Value: %{customdata[0]:.4f}"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )

display_names = [
    feature.replace("_", " ").title()
    for feature in selected_features
]

fig.add_vline(
    x=0,
    line_width=1,
    line_dash="dash",
)

fig.update_layout(
    height=max(600, len(feature_indices) * 35),
    xaxis_title="SHAP Value",
    yaxis=dict(
        tickmode="array",
        tickvals=list(range(len(feature_indices))),
        ticktext=display_names,
        autorange="reversed",
    ),
    showlegend=False,
    margin=dict(l=20, r=20, t=20, b=20),
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Red indicates higher feature values and blue indicates lower feature values. "
    "Points to the right increase the prediction; points to the left decrease it."
)

st.divider()

# FEATURE IMPORTANCE DETAILS

st.subheader("Feature Importance Details")

table_df = shap_df.copy()
table_df.insert(0, "Rank", range(1, len(table_df) + 1))
table_df["feature"] = table_df["feature"].str.replace("_", " ").str.title()
table_df["importance"] = table_df["importance"].round(4)

st.dataframe(
    table_df,
    use_container_width=True,
    hide_index=True,
)