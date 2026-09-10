import os

import streamlit as st
import requests
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.graph_objects as go
from datetime import datetime, timedelta
from resources.city_info import city_info
from resources.pm_to_aqi import calculate_aqi

st.markdown(
    """
    <style>
    .multiforecaster-kicker {
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
PM25_ENDPOINT = f"{API_BASE_URL}/api/forecast_pm25_auto"
PM10_ENDPOINT = f"{API_BASE_URL}/api/forecast_pm10_auto"

st.markdown('<p class="multiforecaster-kicker">Future conditions · multi city</p>', unsafe_allow_html=True)
st.title("Compare air quality forecasts")
st.write("Compare PM2.5, PM10, and AQI forecasts across up to four cities.")

st.markdown("### Forecast setup")
st.caption("Choose the cities and starting point. Each selected city will be forecast at t, t+12h, t+24h, and t+48h.")

with st.container(border=True):
    st.markdown("#### Cities and forecast target")
    target_layout = st.columns([1.2, 1], gap="large")

    with target_layout[0]:
        cities = sorted(city_info.keys())
        selected_cities = st.multiselect(
            "Select up to four cities",
            cities,
            max_selections=4,
            placeholder="Select cities to compare",
        )
        if selected_cities:
            selected_points = pd.DataFrame(
                [
                    {
                        "city": selected_city,
                        "region": city_info[selected_city]["region"],
                        "latitude": city_info[selected_city]["lat"],
                        "longitude": city_info[selected_city]["lon"],
                    }
                    for selected_city in selected_cities
                ]
            )
            st.caption(f"{len(selected_cities)} of 4 cities selected")
        else:
            selected_points = pd.DataFrame(
                columns=["city", "region", "latitude", "longitude"]
            )
            st.caption("Select at least one city to place it on the map.")
        target_columns = st.columns(2, gap="medium")
        with target_columns[0]:
            target_date = st.date_input("Target Date")
        with target_columns[1]:
            target_time = st.time_input("Target Time", step=timedelta(hours=6))
            st.write("Please select a time in 6-hour increments (00:00, 06:00, 12:00, 18:00).")
            
    with target_layout[1]:
        st.pydeck_chart(
            pdk.Deck(
                map_style=None,
                initial_view_state=pdk.ViewState(
                    latitude=np.mean(selected_points["latitude"]) if not selected_points.empty else 22.7,
                    longitude=np.mean(selected_points["longitude"]) if not selected_points.empty else 79.2,
                    zoom=3.7,
                    min_zoom=3.2,
                    max_zoom=7,
                ),
                layers=[
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=selected_points,
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
            height=300,
            use_container_width=True,
        )

target_datetime = datetime.combine(target_date, target_time)
target_time_iso = target_datetime.isoformat()

st.caption(f"Forecast starting point: {target_datetime.strftime('%d %b %Y, %H:%M')}")
st.caption("Target dates can be entered up to two days ahead, and the target time should be later than the current time.")

st.markdown("### Ready to compare?")
st.caption(f"{len(selected_cities)} selected {'city' if len(selected_cities) == 1 else 'cities'} · Four forecast horizons")

if st.button("Generate multi-city forecast", type="primary", use_container_width=True):
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

    st.markdown("### Multi-city forecast results")
    st.caption(
        f"{' • '.join(forecast_cities)} • Starting {forecast_start.strftime('%d %b %Y, %H:%M')}"
    )

    st.markdown("#### Forecast summary")

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

    st.markdown("#### PM2.5 comparison")

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

    st.markdown("#### PM10 comparison")

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

    st.markdown("#### AQI comparison")

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

    st.markdown("#### Horizon-wise comparison")

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