import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from resources.city_info import city_info
from resources.pm_to_aqi import calculate_aqi

API_BASE_URL = st.secrets["API_BASE_URL"]
PM25_ENDPOINT = f"{API_BASE_URL}/api/forecast_pm25_auto"
PM10_ENDPOINT = f"{API_BASE_URL}/api/forecast_pm10_auto"

st.title("Multi City Forecaster")
st.write("Compare PM2.5, PM10 and AQI forecasts across up to four cities.")
st.divider()

st.header("Forecast Inputs")
st.caption("The target date and time is the starting point for the forecast. The model will generate forecasts for t, t+12h, t+24h, and t+48h horizons from this point.")
st.caption("NOTE: Enter target time more than current time to get future forecasts.")
st.caption("NOTE: You can enter target dates only till 2 more days in the future.")
st.caption("OTHERWISE, API will return error. Please enter target date and time accordingly.")
st.divider()
col1, col2 = st.columns(2)

with col1:
    cities = sorted(city_info.keys())
    selected_cities = st.multiselect(
        "Select Cities",
        cities,
        max_selections=4,
        placeholder="Select up to 4 cities",
    )

with col2:
    target_date = st.date_input("Target Date")
    target_time = st.time_input(
        "Target Time",
        step=timedelta(hours=6),
    )

target_datetime = datetime.combine(target_date, target_time)
target_time_iso = target_datetime.isoformat()

st.caption(
    f"Forecast starting point: {target_datetime.strftime('%d %b %Y, %H:%M')}"
)

st.divider()

if st.button("Generate Multi-City Forecast", type="primary", use_container_width=True):
    if not selected_cities:
        st.warning("Please select at least one city.")
        st.stop()

    horizon_hours = {
        "t": 0,
        "t12": 12,
        "t24": 24,
        "t48": 48,
    }

    results = []

    with st.spinner("Generating forecasts..."):
        try:
            progress = st.progress(0)

            for city_index, city in enumerate(selected_cities):
                payload = {
                    "target_time": target_time_iso,
                    "city": city,
                }

                pm25_response = requests.post(
                    PM25_ENDPOINT,
                    json=payload,
                    timeout=60,
                )

                if not pm25_response.ok:
                    try:
                        detail = pm25_response.json().get(
                            "detail",
                            pm25_response.text,
                        )
                    except ValueError:
                        detail = pm25_response.text
                    raise RuntimeError(
                        f"PM2.5 API Error for {city}: {detail}"
                    )

                pm25_result = pm25_response.json()

                pm10_response = requests.post(
                    PM10_ENDPOINT,
                    json=payload,
                    timeout=60,
                )

                if not pm10_response.ok:
                    try:
                        detail = pm10_response.json().get(
                            "detail",
                            pm10_response.text,
                        )
                    except ValueError:
                        detail = pm10_response.text
                    raise RuntimeError(
                        f"PM10 API Error for {city}: {detail}"
                    )

                pm10_result = pm10_response.json()

                for horizon, hours in horizon_hours.items():
                    pm25 = float(pm25_result["predictions"][horizon])
                    pm10 = float(pm10_result["predictions"][horizon])

                    aqi = calculate_aqi({
                        "pm2_5": pm25,
                        "pm10": pm10,
                    })

                    forecast_time = target_datetime + pd.Timedelta(
                        hours=hours
                    )

                    results.append({
                        "city": city,
                        "horizon": horizon,
                        "hours": hours,
                        "time": forecast_time,
                        "pm25": pm25,
                        "pm10": pm10,
                        "aqi": aqi,
                    })

                progress.progress(
                    (city_index + 1) / len(selected_cities)
                )

            progress.empty()

            st.session_state["multicity_forecaster_done"] = True
            st.session_state["multicity_forecaster_results"] = results
            st.session_state["multicity_forecaster_cities"] = selected_cities
            st.session_state["multicity_forecaster_target_time"] = target_datetime

        except requests.exceptions.Timeout:
            progress.empty()
            st.error("The forecasting service took too long to respond.")
        except requests.exceptions.ConnectionError:
            progress.empty()
            st.error("Could not connect to the forecasting service.")
        except requests.exceptions.RequestException as e:
            progress.empty()
            st.error(f"Forecasting API request failed: {e}")
        except (KeyError, TypeError, ValueError) as e:
            progress.empty()
            st.error(f"Invalid API response: {e}")
        except RuntimeError as e:
            progress.empty()
            st.error(str(e))

if st.session_state.get("multicity_forecaster_done", False):
    results = st.session_state["multicity_forecaster_results"]
    forecast_cities = st.session_state["multicity_forecaster_cities"]
    forecast_start = st.session_state["multicity_forecaster_target_time"]

    df = pd.DataFrame(results)

    st.divider()
    st.header("Multi-City Forecast Results")
    st.caption(
        f"{' • '.join(forecast_cities)} • Starting {forecast_start.strftime('%d %b %Y, %H:%M')}"
    )

    st.subheader("Forecast Summary")

    display_df = df[
        ["city", "horizon", "time", "pm25", "pm10", "aqi"]
    ].copy()

    display_df["horizon"] = display_df["horizon"].replace({
        "t": "T",
        "t12": "T+12",
        "t24": "T+24",
        "t48": "T+48",
    })

    display_df.columns = [
        "City",
        "Horizon",
        "Forecast Time",
        "PM2.5",
        "PM10",
        "AQI",
    ]

    display_df["Forecast Time"] = display_df["Forecast Time"].dt.strftime(
        "%d %b %Y, %H:%M"
    )

    st.dataframe(
        display_df.style.format({
            "PM2.5": "{:.2f}",
            "PM10": "{:.2f}",
            "AQI": "{:.0f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.subheader("PM2.5 Comparison")

    fig = go.Figure()

    for city in forecast_cities:
        city_df = df[df["city"] == city].sort_values("hours")

        fig.add_trace(
            go.Scatter(
                x=city_df["time"],
                y=city_df["pm25"],
                mode="lines+markers",
                name=city,
                text=city_df["horizon"].str.upper(),
                hovertemplate=(
                    f"<b>{city}</b><br>"
                    "%{text}<br>"
                    "%{x|%d %b %H:%M}<br>"
                    "PM2.5: %{y:.2f} µg/m³"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="PM2.5 Forecast Comparison",
        xaxis_title="Forecast Time",
        yaxis_title="PM2.5 (µg/m³)",
        height=500,
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("PM10 Comparison")

    fig = go.Figure()

    for city in forecast_cities:
        city_df = df[df["city"] == city].sort_values("hours")

        fig.add_trace(
            go.Scatter(
                x=city_df["time"],
                y=city_df["pm10"],
                mode="lines+markers",
                name=city,
                text=city_df["horizon"].str.upper(),
                hovertemplate=(
                    f"<b>{city}</b><br>"
                    "%{text}<br>"
                    "%{x|%d %b %H:%M}<br>"
                    "PM10: %{y:.2f} µg/m³"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="PM10 Forecast Comparison",
        xaxis_title="Forecast Time",
        yaxis_title="PM10 (µg/m³)",
        height=500,
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("AQI Comparison")

    fig = go.Figure()

    for city in forecast_cities:
        city_df = df[df["city"] == city].sort_values("hours")

        fig.add_trace(
            go.Scatter(
                x=city_df["time"],
                y=city_df["aqi"],
                mode="lines+markers",
                name=city,
                text=city_df["horizon"].str.upper(),
                hovertemplate=(
                    f"<b>{city}</b><br>"
                    "%{text}<br>"
                    "%{x|%d %b %H:%M}<br>"
                    "AQI: %{y:.0f}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="AQI Forecast Comparison",
        xaxis_title="Forecast Time",
        yaxis_title="AQI",
        height=500,
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Horizon-wise Comparison")

    horizon = st.selectbox(
        "Select Forecast Horizon",
        ["T", "T+12", "T+24", "T+48"],
    )

    horizon_map = {
        "T": "t",
        "T+12": "t12",
        "T+24": "t24",
        "T+48": "t48",
    }

    horizon_df = df[
        df["horizon"] == horizon_map[horizon]
    ][["city", "pm25", "pm10", "aqi"]].copy()

    horizon_df.columns = [
        "City",
        "PM2.5",
        "PM10",
        "AQI",
    ]

    st.dataframe(
        horizon_df.style.format({
            "PM2.5": "{:.2f}",
            "PM10": "{:.2f}",
            "AQI": "{:.0f}",
        }),
        use_container_width=True,
        hide_index=True,
    )