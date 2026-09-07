import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from resources.city_info import city_info
from resources.pm_to_aqi import calculate_aqi
from resources.plot_waterfall import plot_shap_waterfall

API_BASE_URL = st.secrets["API_BASE_URL"]
EXPLAINER_PM25_ENDPOINT = f"{API_BASE_URL}/api/explainer/predictor/local/pm25"
EXPLAINER_PM10_ENDPOINT = f"{API_BASE_URL}/api/explainer/predictor/local/pm10"

st.title("Air Quality Simulator")
st.write("Change environmental conditions and compare their impact on PM2.5, PM10 and AQI.")
st.divider()

st.header("Simulation Setup")
cities = sorted(city_info.keys())
city = st.selectbox("City", cities)

month_names = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}
days = {
    0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
    4: "Friday", 5: "Saturday", 6: "Sunday"
}
st.divider()

colX, base_col, modified_col, colY = st.columns([0.3,1,1,0.3])

with base_col:
    st.subheader("Base Conditions")
    base_temperature = st.slider("Temperature (°C)", -10.0, 50.0, 25.0, 0.5, key="base_temperature")
    base_humidity = st.slider("Relative Humidity (%)", 0, 100, 60, 1, key="base_humidity")
    base_wind_speed = st.slider("Wind Speed (m/s)", 0.0, 20.0, 5.0, 0.5, key="base_wind_speed")
    base_wind_direction = st.slider(
        "Wind Direction (°)", 0, 360, 180, 1,
        help="0° = North, 90° = East, 180° = South, 270° = West",
        key="base_wind_direction"
    )
    base_pressure = st.slider("Surface Pressure (hPa)", 900.0, 1050.0, 1000.0, 5.0, key="base_pressure")
    base_precipitation = st.slider("Precipitation (mm)", 0.0, 15.0, 0.0, 0.2, key="base_precipitation")
    base_month_name = st.selectbox("Month", list(month_names.values()), key="base_month")
    base_month = list(month_names.keys())[list(month_names.values()).index(base_month_name)]
    base_day_name = st.selectbox("Day of Week", list(days.values()), key="base_day")
    base_day = list(days.keys())[list(days.values()).index(base_day_name)]
    base_time = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Midnight"], key="base_time")

with modified_col:
    st.subheader("Modified Conditions")
    modified_temperature = st.slider("Temperature (°C)", -10.0, 50.0, 30.0, 0.5, key="modified_temperature")
    modified_humidity = st.slider("Relative Humidity (%)", 0, 100, 70, 1, key="modified_humidity")
    modified_wind_speed = st.slider("Wind Speed (m/s)", 0.0, 20.0, 3.0, 0.5, key="modified_wind_speed")
    modified_wind_direction = st.slider(
        "Wind Direction (°)", 0, 360, 180, 1,
        help="0° = North, 90° = East, 180° = South, 270° = West",
        key="modified_wind_direction"
    )
    modified_pressure = st.slider("Surface Pressure (hPa)", 900.0, 1050.0, 1000.0, 5.0, key="modified_pressure")
    modified_precipitation = st.slider("Precipitation (mm)", 0.0, 15.0, 2.0, 0.2, key="modified_precipitation")
    modified_month_name = st.selectbox("Month", list(month_names.values()), key="modified_month")
    modified_month = list(month_names.keys())[list(month_names.values()).index(modified_month_name)]
    modified_day_name = st.selectbox("Day of Week", list(days.values()), key="modified_day")
    modified_day = list(days.keys())[list(days.values()).index(modified_day_name)]
    modified_time = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Midnight"], key="modified_time")

st.divider()

def predict_scenario(city, temperature, humidity, wind_speed, wind_direction, pressure, precipitation, month, day, time_of_day):
    payload = {
        "city": city,
        "temperature_2m": temperature,
        "relative_humidity_2m": humidity,
        "wind_speed_10m": wind_speed,
        "wind_direction_10m": wind_direction,
        "surface_pressure": pressure,
        "precipitation": precipitation,
        "month": month,
        "day_of_week": day,
        "time_of_day": time_of_day,
    }
    shap_pm25_response = requests.post(EXPLAINER_PM25_ENDPOINT, json=payload, timeout=30)
    shap_pm25_response.raise_for_status()
    shap_pm10_response = requests.post(EXPLAINER_PM10_ENDPOINT, json=payload, timeout=30)
    shap_pm10_response.raise_for_status()
    shap_pm25_result = shap_pm25_response.json()
    shap_pm10_result = shap_pm10_response.json()
    pm25 = float(shap_pm25_result["prediction"])
    ratio_pm10 = float(shap_pm10_result["ratio_prediction"])
    pm10 = pm25 * ratio_pm10
    aqi = calculate_aqi({"pm2_5": pm25, "pm10": pm10})
    fig_pm25 = plot_shap_waterfall(
        shap_result=shap_pm25_result["data"],
        baseline=float(shap_pm25_result["base_value"]),
        prediction=pm25,
        target="PM2.5",
    )
    fig_pm10 = plot_shap_waterfall(
        shap_result=shap_pm10_result["data"],
        baseline=float(shap_pm10_result["base_value"]),
        prediction=ratio_pm10,
        target="PM10 Ratio",
    )
    return {
        "payload": payload,
        "pm25": pm25,
        "ratio_pm10": ratio_pm10,
        "pm10": pm10,
        "aqi": aqi,
        "shap_pm25_result": shap_pm25_result,
        "shap_pm10_result": shap_pm10_result,
        "fig_pm25": fig_pm25,
        "fig_pm10": fig_pm10,
    }

if st.button("Run Simulation", type="primary", use_container_width=True):
    with st.spinner("Running base and modified predictions..."):
        try:
            base_result = predict_scenario(
                city,
                base_temperature,
                base_humidity,
                base_wind_speed,
                base_wind_direction,
                base_pressure,
                base_precipitation,
                base_month,
                base_day,
                base_time,
            )
            modified_result = predict_scenario(
                city,
                modified_temperature,
                modified_humidity,
                modified_wind_speed,
                modified_wind_direction,
                modified_pressure,
                modified_precipitation,
                modified_month,
                modified_day,
                modified_time,
            )
            st.session_state["simulator_done"] = True
            st.session_state["simulator_base"] = base_result
            st.session_state["simulator_modified"] = modified_result
        except requests.exceptions.Timeout:
            st.error("The prediction service took too long to respond.")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the prediction service.")
        except KeyError as e:
            st.error(f"Unexpected API response. Missing field: {e}")
        except (TypeError, ValueError) as e:
            st.error(f"Invalid data returned by the API: {e}")
        except requests.exceptions.RequestException as e:
            st.error(f"Prediction API request failed: {e}")

if st.session_state.get("simulator_done", False):
    base = st.session_state["simulator_base"]
    modified = st.session_state["simulator_modified"]

    st.divider()
    st.header("Simulation Results")
    st.caption(f"City: {city}")

    def change_value(base_value, modified_value):
        change = modified_value - base_value
        percentage = (change / base_value * 100) if base_value != 0 else 0
        return change, percentage

    pm25_change, pm25_pct = change_value(base["pm25"], modified["pm25"])
    pm10_change, pm10_pct = change_value(base["pm10"], modified["pm10"])
    aqi_change, aqi_pct = change_value(base["aqi"], modified["aqi"])

    st.subheader("Output Comparison")
    comparison = pd.DataFrame({
        "Metric": ["PM2.5 (µg/m³)", "PM10 (µg/m³)", "AQI"],
        "Base": [base["pm25"], base["pm10"], base["aqi"]],
        "Modified": [modified["pm25"], modified["pm10"], modified["aqi"]],
        "Change": [pm25_change, pm10_change, aqi_change],
        "Change (%)": [pm25_pct, pm10_pct, aqi_pct],
    })
    st.dataframe(
        comparison.style.format({
            "Base": "{:.2f}",
            "Modified": "{:.2f}",
            "Change": "{:+.2f}",
            "Change (%)": "{:+.2f}%",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Impact on Air Quality")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "PM2.5",
            f"{modified['pm25']:.2f} µg/m³",
            f"{pm25_change:+.2f} ({pm25_pct:+.1f}%)",
        )
    with col2:
        st.metric(
            "PM10",
            f"{modified['pm10']:.2f} µg/m³",
            f"{pm10_change:+.2f} ({pm10_pct:+.1f}%)",
        )
    with col3:
        st.metric(
            "AQI",
            f"{modified['aqi']:.0f}",
            f"{aqi_change:+.0f} ({aqi_pct:+.1f}%)",
        )

    st.divider()
    st.subheader("Base vs Modified")

    chart = go.Figure()
    chart.add_trace(go.Bar(
        name="Base",
        x=["PM2.5", "PM10", "AQI"],
        y=[base["pm25"], base["pm10"], base["aqi"]],
    ))
    chart.add_trace(go.Bar(
        name="Modified",
        x=["PM2.5", "PM10", "AQI"],
        y=[modified["pm25"], modified["pm10"], modified["aqi"]],
    ))
    chart.update_layout(
        barmode="group",
        xaxis_title="Metric",
        yaxis_title="Value",
        height=500,
    )
    st.plotly_chart(chart, use_container_width=True)

    st.divider()
    st.header("SHAP Analysis")

    shap_col1, shap_col2 = st.columns(2)

    with shap_col1:
        st.subheader("Base Conditions")
        st.plotly_chart(base["fig_pm25"], use_container_width=True)
        st.plotly_chart(base["fig_pm10"], use_container_width=True)

    with shap_col2:
        st.subheader("Modified Conditions")
        st.plotly_chart(modified["fig_pm25"], use_container_width=True)
        st.plotly_chart(modified["fig_pm10"], use_container_width=True)