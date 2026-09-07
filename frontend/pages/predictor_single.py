import streamlit as st
import requests

from resources.city_info import city_info
from resources.pm_to_aqi import calculate_aqi
from resources.plot_waterfall import plot_shap_waterfall
# Configuration

API_BASE_URL = st.secrets["API_BASE_URL"]
PREDICT_PM25_ENDPOINT = f"{API_BASE_URL}/api/predict_pm25"
PREDICT_PM10_ENDPOINT = f"{API_BASE_URL}/api/predict_pm10"
EXPLAINER_PM25_ENDPOINT = f"{API_BASE_URL}/api/explainer/predictor/local/pm25"
EXPLAINER_PM10_ENDPOINT = f"{API_BASE_URL}/api/explainer/predictor/local/pm10"
REASONING_ENDPOINT = f"{API_BASE_URL}/api/explainer/predictor/local/reasoning"


# Page

st.title("Single City Predictor")

st.write(
    """
    Predict the PM2.5 concentration for a single city using current
    environmental and contextual conditions.
    """
)

st.divider()


# Input Section

st.header("Prediction Inputs")

colA, colB, colC, colD = st.columns([0.5, 1, 1, 0.5])

with colB:

    # City

    cities = sorted(city_info.keys())

    city = st.selectbox(
        "City",
        cities,
    )

    # Temperature

    temperature = st.slider(
        "Temperature (°C)",
        min_value=-10.0,
        max_value=50.0,
        value=25.0,
        step=0.5,
    )

    # Humidity

    humidity = st.slider(
        "Relative Humidity (%)",
        min_value=0,
        max_value=100,
        value=60,
        step=1,
    )

    # Wind Speed

    wind_speed = st.slider(
        "Wind Speed (m/s)",
        min_value=0.0,
        max_value=20.0,
        value=5.0,
        step=0.5,
    )

    # Wind Direction

    wind_direction = st.slider(
        "Wind Direction (°)",
        min_value=0,
        max_value=360,
        value=180,
        step=1,
        help="0° = North, 90° = East, 180° = South, 270° = West",
    )



with colC:

    # Surface Pressure

    surface_pressure = st.slider(
        "Surface Pressure (hPa)",
        min_value=900.0,
        max_value=1050.0,
        value=1000.0,
        step=5.0,
    )

    # Precipitation

    precipitation = st.slider(
        "Precipitation (mm)",
        min_value=0.0,
        max_value=15.0,
        value=0.0,
        step=0.2,
    )

    # Month

    month_names = {
        1: "January",
        2: "February",
        3: "March",
        4: "April",
        5: "May",
        6: "June",
        7: "July",
        8: "August",
        9: "September",
        10: "October",
        11: "November",
        12: "December",
    }

    month_name = st.selectbox(
        "Month",
        options=list(month_names.values()),
    )

    month = list(month_names.keys())[
        list(month_names.values()).index(month_name)
    ]

    # Day of Week

    days = {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday",
    }

    day_name = st.selectbox(
        "Day of Week",
        options=list(days.values()),
    )

    day_of_week = list(days.keys())[
        list(days.values()).index(day_name)
    ]

    # Time of Day

    time_of_day = st.selectbox(
        "Time of Day",
        [
            "Morning",
            "Afternoon",
            "Evening",
            "Midnight",
        ],
    )


st.divider()


# Prediction
# Prediction
if st.button("Predict AQI", type="primary", use_container_width=True):
    payload = {
        "temperature_2m": temperature,
        "relative_humidity_2m": humidity,
        "wind_speed_10m": wind_speed,
        "wind_direction_10m": wind_direction,
        "surface_pressure": surface_pressure,
        "precipitation": precipitation,
        "month": month,
        "day_of_week": day_of_week,
        "time_of_day": time_of_day,
        "city": city,
    }
    with st.spinner("Generating prediction..."):
        try:
            # PM2.5 SHAP
            shap_pm25 = requests.post(
                EXPLAINER_PM25_ENDPOINT,
                json=payload,
                timeout=30,
            )
            shap_pm25.raise_for_status()

            # PM10 SHAP
            shap_pm10 = requests.post(
                EXPLAINER_PM10_ENDPOINT,
                json=payload,
                timeout=30,
            )
            shap_pm10.raise_for_status()

            # Parse responses
            shap_pm25_result = shap_pm25.json()
            shap_pm10_result = shap_pm10.json()

            # PM2.5
            pm25 = float(shap_pm25_result["prediction"])
            baseline_pm25 = float(shap_pm25_result["base_value"])
            shap_pm25_values = shap_pm25_result["data"]

            # PM10 ratio
            ratio_pm10 = float(shap_pm10_result["ratio_prediction"])
            baseline_pm10 = float(shap_pm10_result["base_value"])
            shap_pm10_values = shap_pm10_result["data"]

            # Calculate PM10 + AQI
            pm10 = pm25 * ratio_pm10
            aqi = calculate_aqi({
                "pm2_5": pm25,
                "pm10": pm10,
            })

            # SHAP plots
            fig_pm25 = plot_shap_waterfall(
                shap_result=shap_pm25_values,
                baseline=baseline_pm25,
                prediction=pm25,
                target="PM2.5",
            )
            fig_pm10 = plot_shap_waterfall(
                shap_result=shap_pm10_values,
                baseline=baseline_pm10,
                prediction=ratio_pm10,
                target="PM10 Ratio",
            )

            # Store results
            st.session_state["prediction_done"] = True
            st.session_state["ratio_pm10"] = ratio_pm10
            st.session_state["pm25"] = pm25
            st.session_state["pm10"] = pm10
            st.session_state["aqi"] = aqi
            st.session_state["shap_pm25_result"] = shap_pm25_result
            st.session_state["shap_pm10_result"] = shap_pm10_result
            st.session_state["fig_pm25"] = fig_pm25
            st.session_state["fig_pm10"] = fig_pm10

            # Clear previous AI reasoning
            st.session_state.pop("pm25_reasoning_result", None)
            st.session_state.pop("pm10_reasoning_result", None)

        except requests.exceptions.Timeout:
            st.error("The SHAP service took too long to respond.")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the SHAP service.")
        except KeyError as e:
            st.error(f"Unexpected API response. Missing field: {e}")
        except (TypeError, ValueError) as e:
            st.error(f"Invalid data returned by the SHAP API: {e}")
        except requests.exceptions.RequestException as e:
            st.error(f"SHAP API request failed: {e}")

# Display results
if st.session_state.get("prediction_done", False):
    st.success("Prediction generated successfully.")
    st.divider()
    st.subheader("Prediction Results")

    pm25 = st.session_state["pm25"]
    pm10 = st.session_state["pm10"]
    aqi = st.session_state["aqi"]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="PM2.5",
            value=f"{round(pm25)} µg/m³",
        )

    with col2:
        st.metric(
            label="PM10",
            value=f"{round(pm10)} µg/m³",
        )

    with col3:
        st.metric(
            label="AQI",
            value=f"{round(aqi)}",
        )

    st.divider()

    # SHAP Analysis
    fig_pm25 = st.session_state["fig_pm25"]
    fig_pm10 = st.session_state["fig_pm10"]
    shap_pm25_result = st.session_state["shap_pm25_result"]
    shap_pm10_result = st.session_state["shap_pm10_result"]

    col4, col5 = st.columns(2)

    # PM2.5
    with col4:
        st.subheader("PM2.5 SHAP Analysis")
        st.plotly_chart(
            fig_pm25,
            use_container_width=True,
        )

        if st.button(
            "Get AI Reasoning",
            key="pm25_reasoning_button",
            use_container_width=True,
        ):
            with st.spinner("Generating AI reasoning..."):
                try:
                    reasoning_payload = {
                        "target": shap_pm25_result["target"],
                        "prediction": float(shap_pm25_result["prediction"]),
                        "base_value": float(shap_pm25_result["base_value"]),
                        "data": shap_pm25_result["data"],
                    }
                    response = requests.post(
                        REASONING_ENDPOINT,
                        json=reasoning_payload,
                        timeout=30,
                    )
                    if response.status_code == 422:
                        st.error("Reasoning API validation error:")
                        st.json(response.json())
                        st.stop()
                    response.raise_for_status()
                    st.session_state["pm25_reasoning_result"] = response.json()
                except requests.exceptions.RequestException as e:
                    st.error(f"AI reasoning request failed: {e}")

        if "pm25_reasoning_result" in st.session_state:
            st.markdown("### AI Reasoning")
            st.write(st.session_state["pm25_reasoning_result"])

    # PM10
    with col5:
        st.subheader("PM10 SHAP Analysis")
        st.plotly_chart(
            fig_pm10,
            use_container_width=True,
        )

        if st.button(
            "Get AI Reasoning",
            key="pm10_reasoning_button",
            use_container_width=True,
        ):
            with st.spinner("Generating AI reasoning..."):
                try:
                    ratio_pm10 = st.session_state["ratio_pm10"]
                    reasoning_payload = {
                        "target": "pm10_ratio",
                        "prediction": float(ratio_pm10),
                        "base_value": float(shap_pm10_result["base_value"]),
                        "data": shap_pm10_result["data"],
                    }
                    response = requests.post(
                        REASONING_ENDPOINT,
                        json=reasoning_payload,
                        timeout=30,
                    )
                    if response.status_code == 422:
                        st.error("Reasoning API validation error:")
                        st.json(response.json())
                        st.stop()
                    response.raise_for_status()
                    st.session_state["pm10_reasoning_result"] = response.json()
                except requests.exceptions.RequestException as e:
                    st.error(f"AI reasoning request failed: {e}")

        if "pm10_reasoning_result" in st.session_state:
            st.markdown("### AI Reasoning")
            st.write(st.session_state["pm10_reasoning_result"])