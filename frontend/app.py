import streamlit as st

st.set_page_config(
    page_title="Aero Metrics",
    layout="wide",
    initial_sidebar_state="auto",
)

API_BASE_URL = st.secrets["API_BASE_URL"]


pages = {
    "": [
        st.Page("pages/home.py", title="Home", default=True),
        st.Page("pages/simulator.py", title="Simulator"),
        st.Page("pages/connect.py", title="Connect with Me"),
    ],

    "Predictor": [
        st.Page("pages/predictor_single.py", title="Single City"),
        st.Page("pages/predictor_multi.py", title="Multi City"),
    ],

    "Forecaster": [
        st.Page("pages/forecaster_single.py", title="Single City"),
        st.Page("pages/forecaster_multi.py", title="Multi City"),
    ],

    "SHAP Analysis": [
        st.Page("pages/shap_predictor.py", title="Predictor"),
        st.Page("pages/shap_forecaster.py", title="Forecaster"),
    ],

    "Data Analytics": [
        st.Page("pages/analytics.py", title="Data Analytics"),
    ],

    "How It Works": [
        st.Page("pages/working_predictor.py", title="Predictor"),
        st.Page("pages/working_forecaster.py", title="Forecaster"),
    ],

    "Metrics": [
        st.Page("pages/metrics_predictor.py", title="Predictor"),
        st.Page("pages/metrics_forecaster.py", title="Forecaster"),
    ],
}


pg = st.navigation(
    pages,
    position="sidebar",
)

pg.run()