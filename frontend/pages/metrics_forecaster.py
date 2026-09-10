import os

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

st.markdown(
    """
    <style>
    .metrics-kicker {
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
METRICS_ENDPOINT = f"{API_BASE_URL}/api/metrics/forecaster"

HORIZONS = ["t", "t12", "t24", "t48"]

@st.cache_data
def get_metrics(model, horizon):
    response = requests.post(
        METRICS_ENDPOINT,
        params={"model": model, "horizon": horizon},
        timeout=30,
    )
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"API Error {response.status_code}: {detail}")
    return response.json()

st.markdown('<p class="metrics-kicker">Model evaluation · forecaster</p>', unsafe_allow_html=True)
st.title("Forecaster model metrics")
st.write("Compare performance, error, stability, and generalization across PM2.5 and PM10 forecast horizons.")

try:
    metrics = {
        "pm25": {h: get_metrics("pm25", h) for h in HORIZONS},
        "pm10": {h: get_metrics("pm10", h) for h in HORIZONS},
    }
except Exception as e:
    st.error(str(e))
    st.stop()

st.markdown("### PM2.5 forecaster")
st.caption("Performance across the t, t+12, t+24, and t+48 forecast horizons.")

pm25_df = pd.DataFrame([
    {
        "Horizon": h.upper(),
        "Train R²": metrics["pm25"][h]["mean_train_r2"],
        "CV R²": metrics["pm25"][h]["mean_cv_r2"],
        "Train MAE": metrics["pm25"][h]["mean_train_mae"],
        "CV MAE": metrics["pm25"][h]["mean_cv_mae"],
        "Train RMSE": metrics["pm25"][h]["mean_train_rmse"],
        "CV RMSE": metrics["pm25"][h]["mean_cv_rmse"],
        "Generalization Gap": metrics["pm25"][h]["generalization_gap"],
    }
    for h in HORIZONS
])

with st.container(border=True):
    st.dataframe(pm25_df.round(4), use_container_width=True, hide_index=True)

st.markdown("#### PM2.5 R² across horizons")

fig = go.Figure()
fig.add_trace(go.Bar(
    x=pm25_df["Horizon"],
    y=pm25_df["Train R²"],
    name="Train R²",
))
fig.add_trace(go.Bar(
    x=pm25_df["Horizon"],
    y=pm25_df["CV R²"],
    name="CV R²",
))
fig.update_layout(
    title="PM2.5 Train vs Cross-Validation R²",
    xaxis_title="Forecast Horizon",
    yaxis_title="R² Score",
    yaxis=dict(range=[0, 1]),
    barmode="group",
    height=450,
)
fig.update_traces(texttemplate="%{y:.3f}", textposition="auto")
st.plotly_chart(fig, use_container_width=True)

st.markdown("#### PM2.5 errors across horizons")

col1, col2 = st.columns(2)

with col1:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=pm25_df["Horizon"],
        y=pm25_df["Train MAE"],
        name="Train MAE",
    ))
    fig.add_trace(go.Bar(
        x=pm25_df["Horizon"],
        y=pm25_df["CV MAE"],
        name="CV MAE",
    ))
    fig.update_layout(
        title="PM2.5 MAE",
        xaxis_title="Forecast Horizon",
        yaxis_title="MAE",
        barmode="group",
        height=400,
    )
    fig.update_traces(texttemplate="%{y:.3f}", textposition="auto")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=pm25_df["Horizon"],
        y=pm25_df["Train RMSE"],
        name="Train RMSE",
    ))
    fig.add_trace(go.Bar(
        x=pm25_df["Horizon"],
        y=pm25_df["CV RMSE"],
        name="CV RMSE",
    ))
    fig.update_layout(
        title="PM2.5 RMSE",
        xaxis_title="Forecast Horizon",
        yaxis_title="RMSE",
        barmode="group",
        height=400,
    )
    fig.update_traces(texttemplate="%{y:.3f}", textposition="auto")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("### PM10 forecaster")
st.caption("Performance across the t, t+12, t+24, and t+48 forecast horizons.")

pm10_df = pd.DataFrame([
    {
        "Horizon": h.upper(),
        "Train R²": metrics["pm10"][h]["mean_train_r2"],
        "CV R²": metrics["pm10"][h]["mean_cv_r2"],
        "Train MAE": metrics["pm10"][h]["mean_train_mae"],
        "CV MAE": metrics["pm10"][h]["mean_cv_mae"],
        "Train RMSE": metrics["pm10"][h]["mean_train_rmse"],
        "CV RMSE": metrics["pm10"][h]["mean_cv_rmse"],
        "Generalization Gap": metrics["pm10"][h]["generalization_gap"],
    }
    for h in HORIZONS
])

with st.container(border=True):
    st.dataframe(pm10_df.round(4), use_container_width=True, hide_index=True)

st.markdown("#### PM10 R² across horizons")

fig = go.Figure()
fig.add_trace(go.Bar(
    x=pm10_df["Horizon"],
    y=pm10_df["Train R²"],
    name="Train R²",
))
fig.add_trace(go.Bar(
    x=pm10_df["Horizon"],
    y=pm10_df["CV R²"],
    name="CV R²",
))
fig.update_layout(
    title="PM10 Train vs Cross-Validation R²",
    xaxis_title="Forecast Horizon",
    yaxis_title="R² Score",
    yaxis=dict(range=[0, 1]),
    barmode="group",
    height=450,
)
fig.update_traces(texttemplate="%{y:.3f}", textposition="auto")
st.plotly_chart(fig, use_container_width=True)

st.markdown("#### PM10 errors across horizons")

col1, col2 = st.columns(2)

with col1:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=pm10_df["Horizon"],
        y=pm10_df["Train MAE"],
        name="Train MAE",
    ))
    fig.add_trace(go.Bar(
        x=pm10_df["Horizon"],
        y=pm10_df["CV MAE"],
        name="CV MAE",
    ))
    fig.update_layout(
        title="PM10 MAE",
        xaxis_title="Forecast Horizon",
        yaxis_title="MAE",
        barmode="group",
        height=400,
    )
    fig.update_traces(texttemplate="%{y:.3f}", textposition="auto")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=pm10_df["Horizon"],
        y=pm10_df["Train RMSE"],
        name="Train RMSE",
    ))
    fig.add_trace(go.Bar(
        x=pm10_df["Horizon"],
        y=pm10_df["CV RMSE"],
        name="CV RMSE",
    ))
    fig.update_layout(
        title="PM10 RMSE",
        xaxis_title="Forecast Horizon",
        yaxis_title="RMSE",
        barmode="group",
        height=400,
    )
    fig.update_traces(texttemplate="%{y:.3f}", textposition="auto")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("### Generalization gap")
st.write("Lower generalization gap indicates a smaller difference between training and cross-validation performance.")

gap_df = pd.DataFrame({
    "Horizon": [h.upper() for h in HORIZONS],
    "PM2.5 Gap": [metrics["pm25"][h]["generalization_gap"] for h in HORIZONS],
    "PM10 Gap": [metrics["pm10"][h]["generalization_gap"] for h in HORIZONS],
})

fig = go.Figure()
fig.add_trace(go.Bar(
    x=gap_df["Horizon"],
    y=gap_df["PM2.5 Gap"],
    name="PM2.5",
))
fig.add_trace(go.Bar(
    x=gap_df["Horizon"],
    y=gap_df["PM10 Gap"],
    name="PM10",
))
fig.update_layout(
    title="Generalization Gap Across Forecast Horizons",
    xaxis_title="Forecast Horizon",
    yaxis_title="Generalization Gap",
    barmode="group",
    height=450,
)
fig.update_traces(texttemplate="%{y:.3f}", textposition="auto")
st.plotly_chart(fig, use_container_width=True)

st.markdown("### Cross-validation stability")

stability_df = pd.DataFrame([
    {
        "Model": "PM2.5",
        "Horizon": h.upper(),
        "CV R² Std": metrics["pm25"][h]["std_cv_r2"],
        "CV MAE Std": metrics["pm25"][h]["std_cv_mae"],
        "CV RMSE Std": metrics["pm25"][h]["std_cv_rmse"],
    }
    for h in HORIZONS
] + [
    {
        "Model": "PM10",
        "Horizon": h.upper(),
        "CV R² Std": metrics["pm10"][h]["std_cv_r2"],
        "CV MAE Std": metrics["pm10"][h]["std_cv_mae"],
        "CV RMSE Std": metrics["pm10"][h]["std_cv_rmse"],
    }
    for h in HORIZONS
])

with st.container(border=True):
    st.dataframe(stability_df.round(4), use_container_width=True, hide_index=True)

st.markdown("### Complete metrics")

complete_df = pd.DataFrame([
    {
        "Model": model.upper(),
        "Horizon": h.upper(),
        "Train R²": metrics[model][h]["mean_train_r2"],
        "CV R²": metrics[model][h]["mean_cv_r2"],
        "Train MAE": metrics[model][h]["mean_train_mae"],
        "CV MAE": metrics[model][h]["mean_cv_mae"],
        "Train RMSE": metrics[model][h]["mean_train_rmse"],
        "CV RMSE": metrics[model][h]["mean_cv_rmse"],
        "Generalization Gap": metrics[model][h]["generalization_gap"],
    }
    for model in ["pm25", "pm10"]
    for h in HORIZONS
])

with st.container(border=True):
    st.dataframe(complete_df.round(4), use_container_width=True, hide_index=True)
with st.expander("About these metrics"):
    st.write("These metrics are derived from training results and cross-validation. Lower standard deviations indicate more stable performance across folds.")
if st.button(
        "Open forecaster model workflow",
        use_container_width=True,
        type="secondary",
    ):
        st.switch_page("pages/working_forecaster.py")
