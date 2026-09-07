import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from resources.city_info import city_info
from resources.pm_to_aqi import calculate_aqi
from resources.plot_waterfall import plot_shap_waterfall

API_BASE_URL = st.secrets["API_BASE_URL"].rstrip("/")
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
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday"
}

st.divider()

colX, base_col, modified_col, colY = st.columns([0.3, 1, 1, 0.3])

with base_col:
    st.subheader("Base Conditions")

    base_temperature = st.slider(
        "Temperature (°C)",
        -10.0, 50.0, 25.0, 0.5,
        key="base_temperature"
    )

    base_humidity = st.slider(
        "Relative Humidity (%)",
        0, 100, 60, 1,
        key="base_humidity"
    )

    base_wind_speed = st.slider(
        "Wind Speed (m/s)",
        0.0, 20.0, 5.0, 0.5,
        key="base_wind_speed"
    )

    base_wind_direction = st.slider(
        "Wind Direction (°)",
        0, 360, 180, 1,
        help="0° = North, 90° = East, 180° = South, 270° = West",
        key="base_wind_direction"
    )

    base_pressure = st.slider(
        "Surface Pressure (hPa)",
        900.0, 1050.0, 1000.0, 5.0,
        key="base_pressure"
    )

    base_precipitation = st.slider(
        "Precipitation (mm)",
        0.0, 15.0, 0.0, 0.2,
        key="base_precipitation"
    )

    base_month_name = st.selectbox(
        "Month",
        list(month_names.values()),
        key="base_month"
    )
    base_month = list(month_names.keys())[list(month_names.values()).index(base_month_name)]

    base_day_name = st.selectbox(
        "Day of Week",
        list(days.values()),
        key="base_day"
    )
    base_day = list(days.keys())[list(days.values()).index(base_day_name)]

    base_time = st.selectbox(
        "Time of Day",
        ["Morning", "Afternoon", "Evening", "Midnight"],
        key="base_time"
    )

with modified_col:
    st.subheader("Modified Conditions")

    modified_temperature = st.slider(
        "Temperature (°C)",
        -10.0, 50.0, 30.0, 0.5,
        key="modified_temperature"
    )

    modified_humidity = st.slider(
        "Relative Humidity (%)",
        0, 100, 70, 1,
        key="modified_humidity"
    )

    modified_wind_speed = st.slider(
        "Wind Speed (m/s)",
        0.0, 20.0, 3.0, 0.5,
        key="modified_wind_speed"
    )

    modified_wind_direction = st.slider(
        "Wind Direction (°)",
        0, 360, 180, 1,
        help="0° = North, 90° = East, 180° = South, 270° = West",
        key="modified_wind_direction"
    )

    modified_pressure = st.slider(
        "Surface Pressure (hPa)",
        900.0, 1050.0, 1000.0, 5.0,
        key="modified_pressure"
    )

    modified_precipitation = st.slider(
        "Precipitation (mm)",
        0.0, 15.0, 2.0, 0.2,
        key="modified_precipitation"
    )

    modified_month_name = st.selectbox(
        "Month",
        list(month_names.values()),
        key="modified_month"
    )
    modified_month = list(month_names.keys())[list(month_names.values()).index(modified_month_name)]

    modified_day_name = st.selectbox(
        "Day of Week",
        list(days.values()),
        key="modified_day"
    )
    modified_day = list(days.keys())[list(days.values()).index(modified_day_name)]

    modified_time = st.selectbox(
        "Time of Day",
        ["Morning", "Afternoon", "Evening", "Midnight"],
        key="modified_time"
    )

st.divider()


def format_api_error(response, endpoint_name):
    try:
        error_data = response.json()
    except ValueError:
        return f"{endpoint_name} API returned HTTP {response.status_code}: {response.text}"

    if response.status_code == 422:
        details = error_data.get("detail", error_data)

        if isinstance(details, list):
            messages = []
            for error in details:
                location = " → ".join(str(x) for x in error.get("loc", []))
                message = error.get("msg", "Validation error")
                messages.append(f"{location}: {message}")
            return f"{endpoint_name} validation error:\n\n" + "\n".join(messages)

        return f"{endpoint_name} validation error:\n\n{details}"

    return f"{endpoint_name} API error ({response.status_code}): {error_data}"


def call_explainer(endpoint, payload, endpoint_name):
    try:
        response = requests.post(
            endpoint,
            json=payload,
            timeout=30
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(f"{endpoint_name} API timed out after 30 seconds.")
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Could not connect to {endpoint_name} API.\n\nEndpoint: {endpoint}"
        )
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"{endpoint_name} request failed: {e}")

    if response.status_code != 200:
        raise RuntimeError(format_api_error(response, endpoint_name))

    try:
        result = response.json()
    except ValueError:
        raise RuntimeError(
            f"{endpoint_name} API returned invalid JSON:\n{response.text}"
        )

    if not isinstance(result, dict):
        raise RuntimeError(
            f"{endpoint_name} API returned an unexpected response format."
        )

    return result


def predict_scenario(
    city,
    temperature,
    humidity,
    wind_speed,
    wind_direction,
    pressure,
    precipitation,
    month,
    day,
    time_of_day
):
    payload = {
        "city": str(city),
        "temperature_2m": float(temperature),
        "relative_humidity_2m": float(humidity),
        "wind_speed_10m": float(wind_speed),
        "wind_direction_10m": float(wind_direction),
        "surface_pressure": float(pressure),
        "precipitation": float(precipitation),
        "month": int(month),
        "day_of_week": int(day),
        "time_of_day": str(time_of_day)
    }

    shap_pm25_result = call_explainer(
        EXPLAINER_PM25_ENDPOINT,
        payload,
        "PM2.5"
    )

    shap_pm10_result = call_explainer(
        EXPLAINER_PM10_ENDPOINT,
        payload,
        "PM10"
    )

    if "prediction" not in shap_pm25_result:
        raise KeyError("PM2.5 response is missing 'prediction'.")

    if "ratio_prediction" not in shap_pm10_result:
        raise KeyError("PM10 response is missing 'ratio_prediction'.")

    if "data" not in shap_pm25_result:
        raise KeyError("PM2.5 response is missing 'data'.")

    if "base_value" not in shap_pm25_result:
        raise KeyError("PM2.5 response is missing 'base_value'.")

    if "data" not in shap_pm10_result:
        raise KeyError("PM10 response is missing 'data'.")

    if "base_value" not in shap_pm10_result:
        raise KeyError("PM10 response is missing 'base_value'.")

    pm25 = float(shap_pm25_result["prediction"])
    ratio_pm10 = float(shap_pm10_result["ratio_prediction"])
    pm10 = pm25 * ratio_pm10

    aqi = calculate_aqi({
        "pm2_5": pm25,
        "pm10": pm10
    })

    fig_pm25 = plot_shap_waterfall(
        shap_result=shap_pm25_result["data"],
        baseline=float(shap_pm25_result["base_value"]),
        prediction=pm25,
        target="PM2.5"
    )

    fig_pm10 = plot_shap_waterfall(
        shap_result=shap_pm10_result["data"],
        baseline=float(shap_pm10_result["base_value"]),
        prediction=ratio_pm10,
        target="PM10 Ratio"
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
        "fig_pm10": fig_pm10
    }


if st.button(
    "Run Simulation",
    type="primary",
    use_container_width=True
):
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
                base_time
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
                modified_time
            )

            st.session_state["simulator_done"] = True
            st.session_state["simulator_base"] = base_result
            st.session_state["simulator_modified"] = modified_result

        except RuntimeError as e:
            st.session_state["simulator_done"] = False
            st.error(str(e))

        except KeyError as e:
            st.session_state["simulator_done"] = False
            st.error(f"Unexpected API response. Missing field: {e}")

        except (TypeError, ValueError) as e:
            st.session_state["simulator_done"] = False
            st.error(f"Invalid data returned by the API: {e}")

        except Exception as e:
            st.session_state["simulator_done"] = False
            st.error(f"Simulation failed: {e}")


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

    pm25_change, pm25_pct = change_value(
        base["pm25"],
        modified["pm25"]
    )

    pm10_change, pm10_pct = change_value(
        base["pm10"],
        modified["pm10"]
    )

    aqi_change, aqi_pct = change_value(
        base["aqi"],
        modified["aqi"]
    )

    st.subheader("Output Comparison")

    comparison = pd.DataFrame({
        "Metric": [
            "PM2.5 (µg/m³)",
            "PM10 (µg/m³)",
            "AQI"
        ],
        "Base": [
            base["pm25"],
            base["pm10"],
            base["aqi"]
        ],
        "Modified": [
            modified["pm25"],
            modified["pm10"],
            modified["aqi"]
        ],
        "Change": [
            pm25_change,
            pm10_change,
            aqi_change
        ],
        "Change (%)": [
            pm25_pct,
            pm10_pct,
            aqi_pct
        ]
    })

    st.dataframe(
        comparison.style.format({
            "Base": "{:.2f}",
            "Modified": "{:.2f}",
            "Change": "{:+.2f}",
            "Change (%)": "{:+.2f}%"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Impact on Air Quality")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "PM2.5",
            f"{modified['pm25']:.2f} µg/m³",
            f"{pm25_change:+.2f} ({pm25_pct:+.1f}%)"
        )

    with col2:
        st.metric(
            "PM10",
            f"{modified['pm10']:.2f} µg/m³",
            f"{pm10_change:+.2f} ({pm10_pct:+.1f}%)"
        )

    with col3:
        st.metric(
            "AQI",
            f"{modified['aqi']:.0f}",
            f"{aqi_change:+.0f} ({aqi_pct:+.1f}%)"
        )

    st.divider()
    st.subheader("Base vs Modified")

    chart = go.Figure()

    chart.add_trace(
        go.Bar(
            name="Base",
            x=["PM2.5", "PM10", "AQI"],
            y=[
                base["pm25"],
                base["pm10"],
                base["aqi"]
            ]
        )
    )

    chart.add_trace(
        go.Bar(
            name="Modified",
            x=["PM2.5", "PM10", "AQI"],
            y=[
                modified["pm25"],
                modified["pm10"],
                modified["aqi"]
            ]
        )
    )

    chart.update_layout(
        barmode="group",
        xaxis_title="Metric",
        yaxis_title="Value",
        height=500
    )

    st.plotly_chart(
        chart,
        use_container_width=True
    )

    st.divider()
    st.header("SHAP Analysis")

    shap_col1, shap_col2 = st.columns(2)

    with shap_col1:
        st.subheader("Base Conditions")

        st.plotly_chart(
            base["fig_pm25"],
            use_container_width=True
        )

        st.plotly_chart(
            base["fig_pm10"],
            use_container_width=True
        )

    with shap_col2:
        st.subheader("Modified Conditions")

        st.plotly_chart(
            modified["fig_pm25"],
            use_container_width=True
        )

        st.plotly_chart(
            modified["fig_pm10"],
            use_container_width=True
        )