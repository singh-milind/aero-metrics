import streamlit as st
import requests
import re
import pandas as pd
import pydeck as pdk
from resources.city_info import city_info
from resources.pm_to_aqi import calculate_aqi
from resources.plot_waterfall import plot_shap_waterfall

st.markdown(
    """
    <style>
    .multi-predictor-kicker {
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


# Configuration
API_BASE_URL = st.secrets["API_BASE_URL"]
EXPLAINER_PM25_ENDPOINT = f"{API_BASE_URL}/api/explainer/predictor/local/pm25"
EXPLAINER_PM10_ENDPOINT = f"{API_BASE_URL}/api/explainer/predictor/local/pm10"
REASONING_ENDPOINT = f"{API_BASE_URL}/api/explainer/predictor/local/reasoning"

st.markdown('<p class="multi-predictor-kicker">Current conditions · city comparison</p>', unsafe_allow_html=True)
st.title("Compare air quality across cities")
st.write("Apply one set of environmental conditions to up to four cities and compare their PM2.5, PM10, and AQI estimates.")

st.markdown("### Prediction setup")
st.caption("Select the cities first, then define the shared conditions used for every prediction.")

with st.container(border=True):
    st.markdown("#### Cities to compare")
    cities = sorted(city_info.keys())
    selected_cities = st.multiselect(
        "Select up to four cities",
        options=cities,
        max_selections=4,
        placeholder="Select up to 4 cities",
    )
    if selected_cities:
        selected_city_points = pd.DataFrame(
            [
                {
                    "city": city,
                    "region": city_info[city]["region"],
                    "latitude": city_info[city]["lat"],
                    "longitude": city_info[city]["lon"],
                }
                for city in selected_cities
            ]
        )
        st.pydeck_chart(
            pdk.Deck(
                map_style=None,
                initial_view_state=pdk.ViewState(
                    latitude=22.7,
                    longitude=79.2,
                    zoom=3.7,
                    min_zoom=3.2,
                    max_zoom=7,
                ),
                layers=[
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=selected_city_points,
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
            height=400,
            use_container_width=True,
        )
    else:
        st.caption("Select at least one city to show its location on the map.")

input_columns = st.columns(3, gap="medium")

with input_columns[0]:
    with st.container(border=True):
        st.markdown("#### X · Atmosphere")
        temperature = st.slider("Temperature (°C)", -10.0, 50.0, 25.0, 0.5)
        humidity = st.slider("Relative Humidity (%)", 0, 100, 60, 1)
        surface_pressure = st.slider("Surface Pressure (hPa)", 900.0, 1050.0, 1000.0, 5.0)

with input_columns[1]:
    with st.container(border=True):
        st.markdown("#### Y · Wind and rain")
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
    with st.container(border=True):
        st.markdown("#### Z · Time context")
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
        time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Midnight"])

st.caption("The same X, Y, and Z conditions are applied consistently across every selected city.")

# Prediction
if st.button("Generate city comparison", type="primary", use_container_width=True):
    if not selected_cities:
        st.warning("Please select at least one city.")
        st.stop()

    payload_base = {
        "temperature_2m": temperature,
        "relative_humidity_2m": humidity,
        "wind_speed_10m": wind_speed,
        "wind_direction_10m": wind_direction,
        "surface_pressure": surface_pressure,
        "precipitation": precipitation,
        "month": month,
        "day_of_week": day_of_week,
        "time_of_day": time_of_day,
    }
    results = {}

    with st.spinner("Generating predictions..."):
        try:
            for city in selected_cities:
                payload = {**payload_base, "city": city}

                shap_pm25 = requests.post(EXPLAINER_PM25_ENDPOINT, json=payload, timeout=30)
                shap_pm25.raise_for_status()

                shap_pm10 = requests.post(EXPLAINER_PM10_ENDPOINT, json=payload, timeout=30)
                shap_pm10.raise_for_status()

                shap_pm25_result = shap_pm25.json()
                shap_pm10_result = shap_pm10.json()

                pm25 = float(shap_pm25_result["prediction"])
                baseline_pm25 = float(shap_pm25_result["base_value"])
                shap_pm25_values = shap_pm25_result["data"]

                ratio_pm10 = float(shap_pm10_result["ratio_prediction"])
                baseline_pm10 = float(shap_pm10_result["base_value"])
                shap_pm10_values = shap_pm10_result["data"]

                pm10 = pm25 * ratio_pm10
                aqi = calculate_aqi({"pm2_5": pm25, "pm10": pm10})

                fig_pm25 = plot_shap_waterfall(
                    shap_result=shap_pm25_values,
                    baseline=baseline_pm25,
                    prediction=pm25,
                    target="PM2.5",
                    top_n=5,
                )
                fig_pm10 = plot_shap_waterfall(
                    shap_result=shap_pm10_values,
                    baseline=baseline_pm10,
                    prediction=ratio_pm10,
                    target="PM10 Ratio",
                    top_n=5,
                )

                results[city] = {
                    "payload": payload,
                    "pm25": pm25,
                    "pm10": pm10,
                    "aqi": aqi,
                    "ratio_pm10": ratio_pm10,
                    "shap_pm25_result": shap_pm25_result,
                    "shap_pm10_result": shap_pm10_result,
                    "fig_pm25": fig_pm25,
                    "fig_pm10": fig_pm10,
                }

            st.session_state["multicity_prediction_done"] = True
            st.session_state["multicity_results"] = results

            for city in selected_cities:
                st.session_state.pop(f"{city}_pm25_reasoning_result", None)
                st.session_state.pop(f"{city}_pm10_reasoning_result", None)
            st.divider()
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

# Display Results
if st.session_state.get("multicity_prediction_done", False):
    results = st.session_state["multicity_results"]
    st.success("Predictions generated successfully.")
    st.markdown("### Comparison summary")
    st.caption("Review the estimated pollution levels for each selected city.")

    for city, result in results.items():
        with st.container(border=True):
            st.markdown(f"#### {city}")
            st.caption(f"{city_info[city]['region']} region")
            metric_columns = st.columns(3, gap="medium")
            with metric_columns[0]:
                st.metric("PM2.5", f"{round(result['pm25'])} µg/m³")
            with metric_columns[1]:
                st.metric("PM10", f"{round(result['pm10'])} µg/m³")
            with metric_columns[2]:
                st.metric("AQI", f"{round(result['aqi'])}")

    st.markdown("### Explain the comparison")
    st.caption("Inspect how the shared conditions influenced each city's prediction.")

    for city, result in results.items():
        with st.expander(f"{city} · model explanations"):
            col4, col5 = st.columns(2)

            with col4:
                st.markdown("##### PM2.5 explanation")
                st.plotly_chart(result["fig_pm25"], use_container_width=True)

                if st.button("Explain PM2.5 result", key=f"{city}_pm25_reasoning_button", use_container_width=True):
                    with st.spinner("Generating AI reasoning..."):
                        try:
                            shap_result = result["shap_pm25_result"]
                            reasoning_payload = {
                                "target": shap_result["target"],
                                "prediction": float(shap_result["prediction"]),
                                "base_value": float(shap_result["base_value"]),
                                "data": shap_result["data"],
                            }
                            response = requests.post(REASONING_ENDPOINT, json=reasoning_payload, timeout=30)
                            if response.status_code == 422:
                                st.error("Reasoning API validation error:")
                                st.json(response.json())
                                st.stop()
                            response.raise_for_status()
                            st.session_state[f"{city}_pm25_reasoning_result"] = response.json()
                        except requests.exceptions.RequestException as e:
                            st.error(f"AI reasoning request failed: {e}")

                if f"{city}_pm25_reasoning_result" in st.session_state:
                    with st.container(border=True):
                        st.markdown("##### PM2.5 reasoning")
                        st.markdown(reasoning_text(st.session_state[f"{city}_pm25_reasoning_result"]))

            with col5:
                st.markdown("##### PM10 explanation")
                st.plotly_chart(result["fig_pm10"], use_container_width=True)

                if st.button("Explain PM10 result", key=f"{city}_pm10_reasoning_button", use_container_width=True):
                    with st.spinner("Generating AI reasoning..."):
                        try:
                            shap_result = result["shap_pm10_result"]
                            ratio_pm10 = result["ratio_pm10"]
                            reasoning_payload = {
                                "target": "pm10_ratio",
                                "prediction": float(ratio_pm10),
                                "base_value": float(shap_result["base_value"]),
                                "data": shap_result["data"],
                            }
                            response = requests.post(REASONING_ENDPOINT, json=reasoning_payload, timeout=30)
                            if response.status_code == 422:
                                st.error("Reasoning API validation error:")
                                st.json(response.json())
                                st.stop()
                            response.raise_for_status()
                            st.session_state[f"{city}_pm10_reasoning_result"] = response.json()
                        except requests.exceptions.RequestException as e:
                            st.error(f"AI reasoning request failed: {e}")

                if f"{city}_pm10_reasoning_result" in st.session_state:
                    with st.container(border=True):
                        st.markdown("##### PM10 reasoning")
                        st.markdown(reasoning_text(st.session_state[f"{city}_pm10_reasoning_result"]))