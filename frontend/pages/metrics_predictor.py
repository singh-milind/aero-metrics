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
METRICS_ENDPOINT = f"{API_BASE_URL}/api/metrics/predcitor"

@st.cache_data
def get_metrics(model):
    response = requests.post(
        METRICS_ENDPOINT,
        params={"model": model},
        timeout=30,
    )
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"API Error {response.status_code}: {detail}")
    return response.json()

st.markdown('<p class="metrics-kicker">Model evaluation · predictor</p>', unsafe_allow_html=True)
st.title("Predictor model metrics")
st.write("Review performance, error, stability, and generalization for the PM2.5 and PM10 prediction models.")

try:
    pm25 = get_metrics("pm25")
    pm10 = get_metrics("pm10")
except Exception as e:
    st.error(str(e))
    st.stop()
import streamlit as st

with st.expander("How the predictor model works"):
    st.write("Model used: XGBoost Regressor")
    st.markdown("""
    1. **PM2.5 model** predicts particulate concentration.
    2. **Ratio model** estimates the PM10/PM2.5 relationship.
    3. **Final PM10** is calculated as PM2.5 × ratio.
    4. **AQI** is calculated using CPCB and WHO formulas.
    """)
    st.info("PM2.5 and ratio models are trained independently to avoid leakage.")

st.markdown("### Performance overview")
st.caption("Training metrics show fit on known data; cross-validation metrics indicate how the models generalize.")

with st.container(border=True):
    st.markdown("#### PM2.5 model")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Train R²", f"{pm25['mean_train_r2']:.2f}")
    with c2:
        st.metric("Train MAE", f"{pm25['mean_train_mae']:.2f}")
    with c3:
        st.metric("Train RMSE", f"{pm25['mean_train_rmse']:.2f}")
    c4, c5, c6, c7 = st.columns(4)
    with c4:
        st.metric("CV R²", f"{pm25['mean_cv_r2']:.2f}")
    with c5:
        st.metric("CV MAE", f"{pm25['mean_cv_mae']:.2f}")
    with c6:
        st.metric("CV RMSE", f"{pm25['mean_cv_rmse']:.2f}")
    with c7:
        st.metric("Generalization Gap", f"{pm25['generalization_gap']:.2f}")

with st.container(border=True):
    st.markdown("#### PM10 model")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Train R²", f"{pm10['mean_train_r2']:.2f}")
    with c2:
        st.metric("Train MAE", f"{pm10['mean_train_mae']:.2f}")
    with c3:
        st.metric("Train RMSE", f"{pm10['mean_train_rmse']:.2f}")
    c4, c5, c6, c7 = st.columns(4)
    with c4:
        st.metric("CV R²", f"{pm10['mean_cv_r2']:.2f}")
    with c5:
        st.metric("CV MAE", f"{pm10['mean_cv_mae']:.2f}")
    with c6:
        st.metric("CV RMSE", f"{pm10['mean_cv_rmse']:.2f}")
    with c7:
        st.metric("Generalization Gap", f"{pm10['generalization_gap']:.2f}")

st.markdown("### Model comparisons")
st.caption("Higher R² is better. Lower MAE and RMSE indicate smaller prediction errors.")

r2_df = pd.DataFrame({
    "Model": ["PM2.5", "PM10"],
    "Train R²": [pm25["mean_train_r2"], pm10["mean_train_r2"]],
    "CV R²": [pm25["mean_cv_r2"], pm10["mean_cv_r2"]],
})

with st.container(border=True):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=r2_df["Model"],
        y=r2_df["Train R²"],
        name="Train R²",
    ))
    fig.add_trace(go.Bar(
        x=r2_df["Model"],
        y=r2_df["CV R²"],
        name="CV R²",
    ))
    fig.update_layout(
        title="Train vs Cross-Validation R²",
        yaxis_title="R² Score",
        yaxis=dict(range=[0, 1]),
        barmode="group",
        height=450,
    )
    fig.update_traces(texttemplate="%{y:.3f}", textposition="auto")
    st.plotly_chart(fig, use_container_width=True)

error_df = pd.DataFrame({
    "Model": ["PM2.5", "PM10"],
    "Train MAE": [pm25["mean_train_mae"], pm10["mean_train_mae"]],
    "CV MAE": [pm25["mean_cv_mae"], pm10["mean_cv_mae"]],
    "Train RMSE": [pm25["mean_train_rmse"], pm10["mean_train_rmse"]],
    "CV RMSE": [pm25["mean_cv_rmse"], pm10["mean_cv_rmse"]],
})

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["PM2.5", "PM10"],
            y=error_df["Train MAE"],
            name="Train MAE",
        ))
        fig.add_trace(go.Bar(
            x=["PM2.5", "PM10"],
            y=error_df["CV MAE"],
            name="CV MAE",
        ))
        fig.update_layout(
            title="Mean Absolute Error",
            yaxis_title="MAE",
            barmode="group",
            height=400,
        )
        fig.update_traces(texttemplate="%{y:.3f}", textposition="auto")
        st.plotly_chart(fig, use_container_width=True)

with col2:
    with st.container(border=True):
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["PM2.5", "PM10"],
            y=error_df["Train RMSE"],
            name="Train RMSE",
        ))
        fig.add_trace(go.Bar(
            x=["PM2.5", "PM10"],
            y=error_df["CV RMSE"],
            name="CV RMSE",
        ))
        fig.update_layout(
            title="Root Mean Squared Error",
            yaxis_title="RMSE",
            barmode="group",
            height=400,
        )
        fig.update_traces(texttemplate="%{y:.3f}", textposition="auto")
        st.plotly_chart(fig, use_container_width=True)



st.markdown("### Complete metrics")

comparison_df = pd.DataFrame({
    "Metric": [
        "Mean Train R²",
        "Mean CV R²",
        "Mean Train MAE",
        "Mean CV MAE",
        "Mean Train RMSE",
        "Mean CV RMSE",
        "Std Train R²",
        "Std CV R²",
        "Std Train MAE",
        "Std CV MAE",
        "Std Train RMSE",
        "Std CV RMSE",
        "Generalization Gap",
    ],
    "PM2.5": [
        pm25["mean_train_r2"],
        pm25["mean_cv_r2"],
        pm25["mean_train_mae"],
        pm25["mean_cv_mae"],
        pm25["mean_train_rmse"],
        pm25["mean_cv_rmse"],
        pm25["std_train_r2"],
        pm25["std_cv_r2"],
        pm25["std_train_mae"],
        pm25["std_cv_mae"],
        pm25["std_train_rmse"],
        pm25["std_cv_rmse"],
        pm25["generalization_gap"],
    ],
    "PM10": [
        pm10["mean_train_r2"],
        pm10["mean_cv_r2"],
        pm10["mean_train_mae"],
        pm10["mean_cv_mae"],
        pm10["mean_train_rmse"],
        pm10["mean_cv_rmse"],
        pm10["std_train_r2"],
        pm10["std_cv_r2"],
        pm10["std_train_mae"],
        pm10["std_cv_mae"],
        pm10["std_train_rmse"],
        pm10["std_cv_rmse"],
        pm10["generalization_gap"],
    ],
})

st.dataframe(
    comparison_df.round(4),
    use_container_width=True,
    hide_index=True,
)
with st.expander("About these metrics"):
    st.write("These metrics are derived from training results and cross-validation. Lower standard deviations indicate more stable performance across folds.")
if st.button(
        "Open predictor model workflow",
        use_container_width=True,
        type="secondary",
    ):
        st.switch_page("pages/working_predictor.py")
