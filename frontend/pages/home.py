
import streamlit as st
from resources.city_info import city_info


st.markdown(
    """
    <div style="max-width: 820px; padding: 1.5rem 0 2rem;">
        <p style="color: #38b9ff; font-size: 0.78rem; font-weight: 700;
                  letter-spacing: 0.2em; text-transform: uppercase; margin-bottom: 1rem;">
            Environmental intelligence platform
        </p>
        <h1 style="margin-bottom: 1.25rem;">AERO<br><span style="color: #38b9ff;">METRICS</span></h1>
        <p style="font-size: 1.25rem; color: #d7e6fb; max-width: 640px;">
            Clearer air-quality insights for a cleaner, smarter India.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="padding: 1.25rem 1.5rem; border: 1px solid rgba(56,185,255,0.2);
                border-radius: 14px; background: rgba(17,20,23,0.66);">
        <p style="margin: 0; color: #71839a;">
            <strong style="color: #f2f5f7;">Predict</strong> current conditions
            <span style="color: #38b9ff;"> / </span>
            <strong style="color: #f2f5f7;">Forecast</strong> what comes next
            <span style="color: #38b9ff;"> / </span>
            <strong style="color: #f2f5f7;">Simulate</strong> possible outcomes
            <span style="color: #38b9ff;"> / </span>
            <strong style="color: #f2f5f7;">Understand</strong> the why
        </p>
    </div>
    """
    ,
    unsafe_allow_html=True,
)

st.write(
    """
    Aero Metrics is an intelligent air-quality analytics platform that combines
    environmental data and machine learning to predict, forecast, simulate, and
    understand air pollution across Indian cities.
    """
)

st.divider()


# ABOUT

st.header("About Aero Metrics")

st.write(
    """
    Air quality is influenced by multiple interacting factors, including weather
    conditions, location, seasonality, and recent pollution levels. Aero Metrics
    brings these factors together to provide a data-driven view of air-quality
    conditions.

    The platform provides tools for estimating current pollution levels, forecasting
    future conditions, exploring the effect of environmental changes, analyzing
    historical data, and understanding how machine-learning models arrive at their
    predictions.
    """
)

st.divider()


# WHY AERO METRICS

st.header("Why Aero Metrics?")

st.write(
    """
    A single air-quality measurement tells you what is happening now, but not
    necessarily why it is happening or what may happen next.

    Aero Metrics approaches air-quality analysis from multiple perspectives:
    """
)

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Predict")
    st.write(
        "Estimate current air-quality conditions from environmental and contextual data."
    )

with col2:
    st.subheader("Forecast")
    st.write(
        "Estimate future pollution levels using recent trends and weather information."
    )

with col3:
    st.subheader("Understand")
    st.write(
        "Analyze data and model behavior to understand the factors influencing predictions."
    )

st.divider()


# CITIES

st.header("Cities Covered")

st.write(
    """
    Aero Metrics currently supports air-quality analysis across the following
    Indian cities, organized by region.
    """
)

region_cities = {}

for city, info in city_info.items():
    region = info["region"]

    if region not in region_cities:
        region_cities[region] = []

    region_cities[region].append(city)

for region, cities in region_cities.items():
    with st.expander(region):
        st.write(" • ".join(cities))

st.divider()


# MODULES

st.header("What Can You Explore?")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Predictor")
    st.write(
        """
        Estimate current air-quality conditions using environmental, temporal,
        and contextual features. Analyze individual cities or compare multiple
        cities simultaneously.
        """
    )

    st.subheader("Simulator")
    st.write(
        """
        Experiment with environmental conditions and observe how changes in
        input features affect the model's predicted air-quality value.
        """
    )

    st.subheader("Data Analytics")
    st.write(
        """
        Explore the underlying training data through distributions, trends,
        statistical patterns, and relationships between environmental variables.
        """
    )

with col2:
    st.subheader("Forecaster")
    st.write(
        """
        Forecast future pollution levels using recent pollution history,
        meteorological conditions, and temporal information for individual
        or multiple cities.
        """
    )

    st.subheader("SHAP Analysis")
    st.write(
        """
        Examine how individual features contribute to model predictions and
        understand which factors have the greatest influence on model output.
        """
    )

    st.subheader("Metrics")
    st.write(
        """
        Evaluate and compare model performance using regression metrics for
        prediction and forecasting tasks.
        """
    )

st.divider()


# HOW IT WORKS

st.header("How It Works")

st.write(
    """
    Explore the underlying workflow of the machine-learning models, from input
    data and feature processing to prediction and interpretation.
    """
)

col1, col2 = st.columns(2)

with col1:
    if st.button(
        "How Predictor Works →",
        use_container_width=True,
    ):
        st.switch_page("pages/working_predictor.py")

with col2:
    if st.button(
        "How Forecaster Works →",
        use_container_width=True,
    ):
        st.switch_page("pages/working_forecaster.py")

st.divider()


# QUICK ACCESS

st.header("Explore Aero Metrics")

st.write(
    """
    Jump directly to any of the major analysis workflows.
    """
)

col1, col2 = st.columns(2)

with col1:
    if st.button(
        "Single City Predictor",
        use_container_width=True,
        type="secondary",
    ):
        st.switch_page("pages/predictor_single.py")

    if st.button(
        "Multi City Predictor",
        use_container_width=True,
        type="secondary",
    ):
        st.switch_page("pages/predictor_multi.py")

    if st.button(
        "SHAP Analysis — Predictor",
        use_container_width=True,
        type="secondary",
    ):
        st.switch_page("pages/shap_predictor.py")

    if st.button(
        "Model Metrics — Predictor",
        use_container_width=True,
        type="secondary",
    ):
        st.switch_page("pages/metrics_predictor.py")


with col2:
    if st.button(
        "Single City Forecaster",
        use_container_width=True,
        type="secondary",
    ):
        st.switch_page("pages/forecaster_single.py")

    if st.button(
        "Multi City Forecaster",
        use_container_width=True,
        type="secondary",
    ):
        st.switch_page("pages/forecaster_multi.py")

    if st.button(
        "SHAP Analysis — Forecaster",
        use_container_width=True,
        type="secondary",
    ):
        st.switch_page("pages/shap_forecaster.py")

    if st.button(
        "Model Metrics — Forecaster",
        use_container_width=True,
        type="secondary",
    ):
        st.switch_page("pages/metrics_forecaster.py")


st.divider()


# DATA ANALYTICS

st.header("Analyze Training Data")

st.write(
    """
    Explore the dataset used for model development and investigate its
    statistical and environmental patterns.
    """
)

if st.button(
    "Explore Training Data →",
    use_container_width=True,
):
    st.switch_page("pages/analytics.py")

st.divider()


# CONNECT

st.header("Connect with Me")

st.write(
    """
    Interested in the project, its implementation, or the machine-learning
    approach behind Aero Metrics?
    """
)

if st.button(
    "View Developer Profile →",
    use_container_width=True,
):
    st.switch_page("pages/connect.py")


st.divider()

st.caption(
    "Built with Python • Machine Learning • FastAPI • Streamlit • Docker • Azure"
)
