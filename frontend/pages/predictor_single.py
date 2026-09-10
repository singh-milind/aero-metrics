import streamlit as st
import requests
import pandas as pd
import pydeck as pdk
import re

from resources.city_info import city_info
from resources.pm_to_aqi import calculate_aqi
from resources.plot_waterfall import plot_shap_waterfall

st.markdown(
    """
    <style>
    .st-key-predictor-city,
    .st-key-predictor-x,
    .st-key-predictor-y,
    .st-key-predictor-z,
    .st-key-predictor-results {
        min-height: 100%;
    }

    .predictor-kicker {
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

# Configuration

API_BASE_URL = st.secrets["API_BASE_URL"]
PREDICT_PM25_ENDPOINT = f"{API_BASE_URL}/api/predict_pm25"
PREDICT_PM10_ENDPOINT = f"{API_BASE_URL}/api/predict_pm10"
EXPLAINER_PM25_ENDPOINT = f"{API_BASE_URL}/api/explainer/predictor/local/pm25"
EXPLAINER_PM10_ENDPOINT = f"{API_BASE_URL}/api/explainer/predictor/local/pm10"
REASONING_ENDPOINT = f"{API_BASE_URL}/api/explainer/predictor/local/reasoning"


def reasoning_text(result):
    if isinstance(result, dict):
        for key in ("reasoning", "explanation", "response", "text"):
            value = result.get(key)
            if isinstance(value, str):
                result = value
                break
        else:
            result = str(result)

    text = str(result).replace("\r\n", "\n").strip()
    if "\n" not in text:
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
        text = "\n\n".join(
            " ".join(sentences[index:index + 3])
            for index in range(0, len(sentences), 3)
        )
    return text


# Page

st.markdown('<p class="predictor-kicker">Current conditions · single city</p>', unsafe_allow_html=True)
st.title("Predict air quality")

st.write(
    """
    Estimate PM2.5, PM10, and AQI for one city using environmental and
    contextual conditions.
    """
)

# Input Section
st.markdown("### Prediction setup")
st.caption("Set the location, environmental conditions, and time context used by the model.")

with st.container(border=True, key="predictor-city"):
   st.markdown("#### City")
   city_layout = st.columns([1, 1], gap="large")

   with city_layout[0]:
       cities = sorted(city_info.keys())
       city = st.selectbox("Select a city", cities, label_visibility="collapsed")

   with city_layout[1]:
       selected_city = city_info[city]
       selected_city_point = pd.DataFrame(
           [
               {
                   "city": city,
                   "region": selected_city["region"],
                   "latitude": selected_city["lat"],
                   "longitude": selected_city["lon"],
               }
           ]
       )
       st.pydeck_chart(
           pdk.Deck(
               map_style=None,
               initial_view_state=pdk.ViewState(
                   latitude=selected_city["lat"],
                   longitude=selected_city["lon"],
                   zoom=3.4,
                   min_zoom=3.2,
                   max_zoom=7,
               ),
               layers=[
                   pdk.Layer(
                       "ScatterplotLayer",
                       data=selected_city_point,
                       get_position="[longitude, latitude]",
                       get_radius=26000,
                       get_fill_color=[56, 185, 255, 220],
                       get_line_color=[242, 245, 247, 230],
                       line_width_min_pixels=2,
                       pickable=True,
                   )
               ],
               tooltip={
                   "html": "<b>{city}</b><br/>{region} region",
                   "style": {"color": "#f2f5f7"},
               },
           ),
           height=220,
           use_container_width=True,
       )

input_columns = st.columns(3, gap="medium")

with input_columns[0]:
   with st.container(border=True, key="predictor-x"):
       st.markdown("#### · Atmosphere")
       temperature = st.slider("Temperature (°C)", -10.0, 50.0, 25.0, 0.5)
       humidity = st.slider("Relative Humidity (%)", 0, 100, 60, 1)
       surface_pressure = st.slider("Surface Pressure (hPa)", 900.0, 1050.0, 1000.0, 5.0)

with input_columns[1]:
   with st.container(border=True, key="predictor-y"):
       st.markdown("#### · Wind and rain")
       wind_speed = st.slider("Wind Speed (m/s)", 0.0, 20.0, 5.0, 0.5)
       wind_direction = st.slider(
           "Wind Direction (°)",
           0,
           360,
           180,
           1,
           help="0° = North, 90° = East, 180° = South, 270° = West",
       )
       precipitation = st.slider("Precipitation(mm) in 6h", 0.0, 15.0, 0.0, 0.2)

with input_columns[2]:
   with st.container(border=True, key="predictor-z"):
       st.markdown("#### · Time context")
       month_names = {
           1: "January", 2: "February", 3: "March", 4: "April",
           5: "May", 6: "June", 7: "July", 8: "August",
           9: "September", 10: "October", 11: "November", 12: "December",
       }
       month_name = st.selectbox("Month", list(month_names.values()))
       month = list(month_names.keys())[list(month_names.values()).index(month_name)]

       days = {
           0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
           4: "Friday", 5: "Saturday", 6: "Sunday",
       }
       day_name = st.selectbox("Day of Week", list(days.values()))
       day_of_week = list(days.keys())[list(days.values()).index(day_name)]
       time_of_day = st.selectbox(
           "Time of Day",
           ["Morning", "Afternoon", "Evening", "Midnight"],
       )

st.caption("The model combines these inputs to estimate particulate concentration and the resulting AQI.")

st.markdown("### Ready to estimate?")
st.caption(f"Prediction target: **{city}**")

# Prediction
# Prediction
if st.button("Generate air-quality prediction", type="primary", use_container_width=True):
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
    st.markdown("### Prediction summary")
    st.caption("Estimated pollutant concentrations and AQI for the selected conditions.")

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

    st.markdown("### Explain this prediction")
    st.caption(
        "Waterfall charts show how each input moved the estimate away from the model baseline."
    )

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
            "Explain PM2.5 result",
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
            with st.container(border=True):
                st.markdown("### PM2.5 reasoning")
                st.caption("How the model arrived at this estimate.")
                st.markdown(reasoning_text(st.session_state["pm25_reasoning_result"]))

    # PM10
    with col5:
        st.subheader("PM10 SHAP Analysis")
        st.plotly_chart(
            fig_pm10,
            use_container_width=True,
        )

        if st.button(
            "Explain PM10 result",
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
            with st.container(border=True):
                st.markdown("### PM10 reasoning")
                st.caption("How the model arrived at this estimate.")
                st.markdown(reasoning_text(st.session_state["pm10_reasoning_result"]))