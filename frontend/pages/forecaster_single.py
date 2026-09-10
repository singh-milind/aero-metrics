import os

import streamlit as st
import requests
import pandas as pd
import pydeck as pdk
import plotly.graph_objects as go
from datetime import datetime, timedelta
from resources.city_info import city_info
from resources.pm_to_aqi import calculate_aqi
from resources.plot_waterfall import plot_shap_waterfall

st.markdown(
    """
    <style>
    .forecaster-kicker {
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
PM25_AUTO_ENDPOINT = f"{API_BASE_URL}/api/explainer/forecaster/local/pm25/auto"
PM10_AUTO_ENDPOINT = f"{API_BASE_URL}/api/explainer/forecaster/local/pm10/auto"
PM25_MANUAL_ENDPOINT = f"{API_BASE_URL}/api/explainer/forecaster/local/pm25/manual"
PM10_MANUAL_ENDPOINT = f"{API_BASE_URL}/api/explainer/forecaster/local/pm10/manual"
REASONING_ENDPOINT = f"{API_BASE_URL}/api/explainer/forecaster/local/reasoning"
def create_local_waterfall(shap_result, target, top_n=10):
    base_value = float(shap_result["base_value"])
    prediction = float(shap_result["prediction"])
    shap_df = pd.DataFrame(shap_result["data"]).copy()
    shap_df["shap_value"] = pd.to_numeric(shap_df["shap_value"])
    shap_df["abs_shap"] = shap_df["shap_value"].abs()
    shap_df = shap_df.sort_values("abs_shap", ascending=False).head(top_n)
    shap_df = shap_df.sort_values("shap_value")
    labels = ["Base Value"] + shap_df["feature"].tolist() + ["Prediction"]
    y_values = [base_value] + shap_df["shap_value"].tolist() + [prediction]
    measures = ["absolute"] + ["relative"] * len(shap_df) + ["total"]
    hover = [f"Base Value: {base_value:.3f}<extra></extra>"]
    for _, row in shap_df.iterrows():
        hover.append(f"<b>{row['feature']}</b><br>Feature Value: {row['value']}<br>SHAP Value: {row['shap_value']:+.3f}<br>Impact: {row['impact']}<extra></extra>")
    hover.append(f"Prediction: {prediction:.3f}<extra></extra>")
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=measures,
        x=labels,
        y=y_values,
        text=[f"{base_value:.2f}"] + [f"{v:+.2f}" for v in shap_df["shap_value"]] + [f"{prediction:.2f}"],
        textposition="outside",
        hovertemplate=hover,
        connector={"line": {"width": 1}},
    ))
    fig.update_layout(
        title=f"{target} — Local SHAP Explanation",
        xaxis_title="Features",
        yaxis_title="Prediction",
        height=600,
        showlegend=False,
    )
    return fig
def get_ai_reasoning(target, horizon, shap_result):
    payload = {
        "target": target,
        "prediction": float(shap_result["prediction"]),
        "base_value": float(shap_result["base_value"]),
        "data": shap_result["data"],
    }
    response = requests.post(
        REASONING_ENDPOINT,
        params={"horizon": horizon},
        json=payload,
        timeout=60,
    )
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"AI Reasoning API Error {response.status_code}: {detail}")
    return response.json()
st.markdown('<p class="forecaster-kicker">Future conditions · single city</p>', unsafe_allow_html=True)
st.title("Forecast air quality")
st.write("Generate PM2.5, PM10, and AQI forecasts for the current moment and the next 48 hours.")
st.caption("Eight models run behind the scenes. Forecast generation may take a few seconds.")

st.markdown("### Forecast setup")
mode = st.radio(
    "Forecast Mode",
    ["Automatic", "Manual"],
    horizontal=True,
    help="Automatic mode uses weather and PM data from the backend. Manual mode lets you provide the input features."
)

with st.container(border=True):
    st.markdown("#### Forecast target")
    st.caption("Choose a city and starting point. The model will forecast t, t+12h, t+24h, and t+48h.")
    target_layout = st.columns([1.7, 1], gap="large")

    with target_layout[0]:
        target_columns = st.columns(3, gap="medium")

        with target_columns[0]:
            cities = sorted(city_info.keys())
            city = st.selectbox("City", cities)

        with target_columns[1]:
            target_date = st.date_input("Target Date")

        with target_columns[2]:
            target_time = st.time_input("Target Time", step=timedelta(hours=6))
            st.write("Please select a time in 6-hour increments (00:00, 06:00, 12:00, 18:00).")

    with target_layout[1]:
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

target_datetime = datetime.combine(target_date, target_time)
target_time_iso = target_datetime.isoformat()

st.caption(f"Forecast starting point: {target_datetime.strftime('%d %b %Y, %H:%M')}")
st.caption("Target dates can be entered up to two days ahead, and the target time should be later than the current time.")

if mode == "Manual":
    st.markdown("#### Manual inputs")
    st.caption("Provide the weather conditions and recent pollution values used by the manual forecasting models.")
    manual_columns = st.columns(2, gap="medium")

    with manual_columns[0]:
        with st.container(border=True):
            st.markdown("##### Weather conditions")
            temperature_2m = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0, step=0.5)
            relative_humidity_2m = st.number_input("Relative Humidity (%)", min_value=0.0, max_value=100.0, value=60.0, step=1.0)
            wind_speed_10m = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=20.0,value=10.0, step=0.2)
            wind_direction_10m = st.number_input("Wind Direction (°)", min_value=0.0, max_value=360.0, value=180.0, step=1.0)
            surface_pressure = st.number_input("Surface Pressure (hPa)", min_value=900,max_value=1050,value=1013.0, step=10.0)
            precipitation = st.number_input("Precipitation(mm) in 6h", min_value=0.0, value=0.0, step=0.2,max_value=15.0)

    with manual_columns[1]:
        with st.container(border=True):
            st.markdown("##### Recent pollution lags")
            st.caption("Use the latest observed values available before the forecast target.")
            lag_columns = st.columns(2, gap="small")
            with lag_columns[0]:
                pm2_5_lag_12h = st.number_input("PM2.5 Lag 12h", min_value=0.0, value=0.0, step=5.0)
                pm2_5_lag_24h = st.number_input("PM2.5 Lag 24h", min_value=0.0, value=0.0, step=5.0)
                pm2_5_lag_48h = st.number_input("PM2.5 Lag 48h", min_value=0.0, value=0.0, step=5.0)
            with lag_columns[1]:
                pm10_lag_12h = st.number_input("PM10 Lag 12h", min_value=0.0, value=0.0, step=5.0)
                pm10_lag_24h = st.number_input("PM10 Lag 24h", min_value=0.0, value=0.0, step=5.0)
                pm10_lag_48h = st.number_input("PM10 Lag 48h", min_value=0.0, value=0.0, step=5.0)

    manual_payload = {
        "target_time": target_time_iso,
        "city": city,
        "temperature_2m": temperature_2m,
        "relative_humidity_2m": relative_humidity_2m,
        "wind_speed_10m": wind_speed_10m,
        "wind_direction_10m": wind_direction_10m,
        "surface_pressure": surface_pressure,
        "precipitation": precipitation,
        "pm_2_5_lag_12h": pm2_5_lag_12h,
        "pm_10_lag_12h": pm10_lag_12h,
        "pm_2_5_lag_24h": pm2_5_lag_24h,
        "pm_10_lag_24h": pm10_lag_24h,
        "pm_2_5_lag_48h": pm2_5_lag_48h,
        "pm_10_lag_48h": pm10_lag_48h,
    }
else:
    manual_payload = None

st.markdown("### Ready to forecast?")
st.caption(f"Forecast target: **{city}** · Mode: **{mode}**")

if st.button("Generate 48-hour forecast", type="primary", use_container_width=True):
    horizons = ["t", "t12", "t24", "t48"]
    results = []

    with st.spinner("Generating forecast..."):
        try:
            for horizon in horizons:
                if mode == "Automatic":
                    pm25_response = requests.post(
                        PM25_AUTO_ENDPOINT,
                        params={"horizon": horizon},
                        json={
                            "target_time": target_time_iso,
                            "city": city,
                        },
                        timeout=60,
                    )
                else:
                    pm25_response = requests.post(
                        PM25_MANUAL_ENDPOINT,
                        params={"horizon": horizon},
                        json=manual_payload,
                        timeout=60,
                    )

                if not pm25_response.ok:
                    try:
                        detail = pm25_response.json().get("detail", pm25_response.text)
                    except ValueError:
                        detail = pm25_response.text
                    raise RuntimeError(
                        f"PM2.5 API Error {pm25_response.status_code}: {detail}"
                    )

                pm25_result = pm25_response.json()

                if mode == "Automatic":
                    pm10_response = requests.post(
                        PM10_AUTO_ENDPOINT,
                        params={"horizon": horizon},
                        json={
                            "target_time": target_time_iso,
                            "city": city,
                        },
                        timeout=60,
                    )
                else:
                    pm10_response = requests.post(
                        PM10_MANUAL_ENDPOINT,
                        params={"horizon": horizon},
                        json=manual_payload,
                        timeout=60,
                    )

                if not pm10_response.ok:
                    try:
                        detail = pm10_response.json().get("detail", pm10_response.text)
                    except ValueError:
                        detail = pm10_response.text
                    raise RuntimeError(
                        f"PM10 API Error {pm10_response.status_code}: {detail}"
                    )

                pm10_result = pm10_response.json()

                pm25 = float(pm25_result["prediction"])
                ratio_pm10 = float(pm10_result["prediction"])
                pm10 = pm25 * ratio_pm10
                aqi = calculate_aqi({
                    "pm2_5": pm25,
                    "pm10": pm10,
                })

                horizon_hours = {
                    "t": 0,
                    "t12": 12,
                    "t24": 24,
                    "t48": 48,
                }[horizon]

                forecast_time = target_datetime + pd.Timedelta(hours=horizon_hours)

                results.append({
                    "horizon": horizon,
                    "hours": horizon_hours,
                    "time": forecast_time,
                    "pm25": pm25,
                    "ratio_pm10": ratio_pm10,
                    "pm10": pm10,
                    "aqi": aqi,
                    "pm25_result": pm25_result,
                    "pm10_result": pm10_result,
                })

            st.session_state["forecaster_auto_done"] = True
            st.session_state["forecaster_auto_results"] = results
            st.session_state["forecaster_auto_city"] = city
            st.session_state["forecaster_auto_target_time"] = target_datetime
            st.session_state["forecaster_auto_mode"] = mode

        except requests.exceptions.Timeout:
            st.error("The forecasting service took too long to respond.")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the forecasting service.")
        except KeyError as e:
            st.error(f"Unexpected API response. Missing field: {e}")
        except (TypeError, ValueError) as e:
            st.error(f"Invalid data returned by the forecasting API: {e}")
        except requests.exceptions.RequestException as e:
            st.error(f"Forecasting API request failed: {e}")
        except RuntimeError as e:
            st.error(str(e))

if st.session_state.get("forecaster_auto_done", False):
    results = st.session_state["forecaster_auto_results"]
    forecast_city = st.session_state["forecaster_auto_city"]
    forecast_start = st.session_state["forecaster_auto_target_time"]
    forecast_mode = st.session_state.get("forecaster_auto_mode", "Automatic")

    st.markdown("### Forecast results")
    st.caption(
        f"{forecast_city} • {forecast_mode} Mode • Starting {forecast_start.strftime('%d %b %Y, %H:%M')}"
    )

    df = pd.DataFrame(results)

    st.markdown("#### Forecast timeline")

    col1, col2, col3, col4 = st.columns(4)
    horizon_labels = {
        "t": "T",
        "t12": "T+12",
        "t24": "T+24",
        "t48": "T+48",
    }

    for col, result in zip([col1, col2, col3, col4], results):
        with col:
            st.markdown(
                f"""
                <div style="
                    border: 1px solid rgba(148, 163, 184, 0.22);
                    border-radius: 18px;
                    background: rgba(15, 23, 42, 0.42);
                    min-height: 128px;
                    padding: 18px 20px 14px;
                ">
                    <div style="color: #8294ad; font-size: 0.85rem; margin-bottom: 8px;">
                        {horizon_labels.get(result["horizon"], result["horizon"].upper())}
                    </div>
                    <div style="color: #8294ad; font-size: 2.35rem; line-height: 1.1; font-weight: 500;">
                        {result["aqi"]:.2f}
                    </div>
                    <div style="color: #53647c; font-size: 0.8rem; margin-top: 12px;">
                        AQI · {result["time"].strftime("%d %b %H:%M")}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("#### Forecast values")

    display_df = df[["horizon", "time", "pm25", "pm10", "aqi"]].copy()
    display_df["horizon"] = display_df["horizon"].map(horizon_labels)
    display_df.columns = ["Horizon", "Forecast Time", "PM2.5", "PM10", "AQI"]
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

    st.markdown("#### PM2.5 forecast")

    pm25_fig = go.Figure()
    pm25_fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["pm25"],
            mode="lines+markers",
            name="PM2.5",
            text=df["horizon"].str.upper(),
            hovertemplate="%{text}<br>%{x|%d %b %H:%M}<br>PM2.5: %{y:.2f} µg/m³<extra></extra>",
        )
    )
    pm25_fig.update_layout(
        xaxis_title="Forecast Time",
        yaxis_title="PM2.5 (µg/m³)",
        height=450,
        hovermode="x unified",
    )
    st.plotly_chart(pm25_fig, use_container_width=True)

    st.markdown("#### PM10 forecast")

    pm10_fig = go.Figure()
    pm10_fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["pm10"],
            mode="lines+markers",
            name="PM10",
            text=df["horizon"].str.upper(),
            hovertemplate="%{text}<br>%{x|%d %b %H:%M}<br>PM10: %{y:.2f} µg/m³<extra></extra>",
        )
    )
    pm10_fig.update_layout(
        xaxis_title="Forecast Time",
        yaxis_title="PM10 (µg/m³)",
        height=450,
        hovermode="x unified",
    )
    st.plotly_chart(pm10_fig, use_container_width=True)

    st.markdown("#### AQI forecast")

    aqi_fig = go.Figure()
    aqi_fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["aqi"],
            mode="lines+markers",
            name="AQI",
            text=df["horizon"].str.upper(),
            hovertemplate="%{text}<br>%{x|%d %b %H:%M}<br>AQI: %{y:.0f}<extra></extra>",
        )
    )
    aqi_fig.update_layout(
        xaxis_title="Forecast Time",
        yaxis_title="AQI",
        height=450,
        hovermode="x unified",
    )
    st.plotly_chart(aqi_fig, use_container_width=True)
    st.markdown("### Explain each forecast")
    st.caption("Open a horizon to inspect the features that contributed to its PM2.5 and PM10 estimates.")

    horizon_tabs = st.tabs(["T", "T+12", "T+24", "T+48"])

    for tab, result in zip(horizon_tabs, results):
        with tab:
            horizon = result["horizon"]

            st.subheader("PM2.5")
            pm25_shap = result["pm25_result"]

            if "data" in pm25_shap and "base_value" in pm25_shap:
                st.plotly_chart(
                    create_local_waterfall(
                        pm25_shap,
                        "PM2.5",
                    ),
                    use_container_width=True,
                )

                if st.button(
                    "Explain PM2.5 forecast",
                    key=f"pm25_reasoning_{horizon}",
                    use_container_width=True,
                ):
                    with st.spinner("Generating AI reasoning..."):
                        try:
                            reasoning = get_ai_reasoning(
                                "pm25",
                                horizon,
                                pm25_shap,
                            )
                            st.session_state[f"pm25_reasoning_{horizon}"] = reasoning
                        except Exception as e:
                            st.error(str(e))

                reasoning = st.session_state.get(f"pm25_reasoning_{horizon}")

                if reasoning:
                    st.markdown("#### PM2.5 reasoning")
                    if isinstance(reasoning, dict):
                        st.info(
                            reasoning.get(
                                "reasoning",
                                reasoning.get("response", str(reasoning)),
                            )
                        )
                    else:
                        st.info(reasoning)
            else:
                st.warning("Local SHAP data not available for PM2.5.")

            st.divider()

            st.subheader("PM10 Ratio")
            pm10_shap = result["pm10_result"]

            if "data" in pm10_shap and "base_value" in pm10_shap:
                st.plotly_chart(
                    create_local_waterfall(
                        pm10_shap,
                        "PM10 Ratio",
                    ),
                    use_container_width=True,
                )

                if st.button(
                    "Explain PM10 forecast",
                    key=f"pm10_reasoning_{horizon}",
                    use_container_width=True,
                ):
                    with st.spinner("Generating AI reasoning..."):
                        try:
                            reasoning = get_ai_reasoning(
                                "pm10_ratio",
                                horizon,
                                pm10_shap,
                            )
                            st.session_state[f"pm10_reasoning_{horizon}"] = reasoning
                        except Exception as e:
                            st.error(str(e))

                reasoning = st.session_state.get(f"pm10_reasoning_{horizon}")

                if reasoning:
                    st.markdown("#### PM10 reasoning")
                    if isinstance(reasoning, dict):
                        st.info(
                            reasoning.get(
                                "reasoning",
                                reasoning.get("response", str(reasoning)),
                            )
                        )
                    else:
                        st.info(reasoning)
            else:
                st.warning("Local SHAP data not available for PM10.")