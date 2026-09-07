import streamlit as st
import requests
import pandas as pd
from datetime import datetime, time
from resources.charts import create_dual_axis_chart, create_line_chart
from resources.city_info import city_info

API_BASE_URL = st.secrets["API_BASE_URL"]
TRENDS_ENDPOINT = f"{API_BASE_URL}/api/analytics/trends"

CATEGORICAL_FEATURES = [
    "city",
    "region",
    "regional_season",
    "season",
    "month",
    "year",
    "time_of_day",
    "day_of_week",
    "is_weekend",
    "weather_verdict",
]

NUMERICAL_FEATURES = [
    "aqi",
    "pm2_5",
    "pm10",
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "surface_pressure",
]

def get_trend(x_axis, y_axis, filters):
    response = requests.post(
        TRENDS_ENDPOINT,
        params={"x_axis": x_axis, "y_axis": y_axis},
        json=filters,
        timeout=30,
    )
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"API Error {response.status_code}: {detail}")
    return pd.DataFrame(response.json())

def create_filters(city, region, year, season, time_of_day, weather_verdict, weekend, start_date, end_date):
    return {
        "city": None if city == "All" else city,
        "region": None if region == "All" else region,
        "year": None if year == "All" else year,
        "season": None if season == "All" else season,
        "time_of_day": None if time_of_day == "All" else time_of_day,
        "is_weekend": None if weekend == "All" else weekend == "Yes",
        "weather_verdict": None if weather_verdict == "All" else weather_verdict,
        "start_date": datetime.combine(start_date, time.min).isoformat() if start_date else None,
        "end_date": datetime.combine(end_date, time.max).isoformat() if end_date else None,
    }

st.title("Data Analytics")
st.write("Explore the relationships between various factors and air quality metrics (AQI, PM2.5, PM10, etc.) using interactive charts.")
st.caption("Open sidebar to apply filters to the data.")
st.divider()
st.warning("If dashboards appear empty, it means such filtered data is not available in the training dataset. Please try different filters.")

if "analytics_filters" not in st.session_state:
    st.session_state["analytics_filters"] = {
        "city": None,
        "region": None,
        "year": None,
        "season": None,
        "time_of_day": None,
        "is_weekend": None,
        "weather_verdict": None,
        "start_date": None,
        "end_date": None,
    }

with st.sidebar:
    st.header("Filters")
    st.write("Select the filters to apply to the data.")
    with st.form("analytics_filter_form"):
        filter_y_axis = st.selectbox("Y-Axis", ["aqi", "pm2_5", "pm10"])
        filter_city = st.selectbox("City", ["All"] + sorted(city_info.keys()))
        filter_region = st.selectbox("Region", ["All", "North", "NCR", "South", "East", "West", "Central", "Northeast", "UT"])
        filter_year = st.selectbox("Year", ["All"] + list(range(2020, datetime.now().year + 1)))
        filter_season = st.selectbox("Season", ["All", "Winter", "Summer", "Southwest Monsoon", "Post-Monsoon", "Pre-Monsoon", "Extended Monsoon", "Retreating Monsoon"])
        filter_time_of_day = st.selectbox("Time of Day", ["All", "Morning", "Afternoon", "Evening", "Midnight"])
        filter_weather = st.selectbox("Weather", ["All", "Pleasant / Normal", "Hot & Humid", "Windy & Clear", "Rainy / Washed", "Cold & Stagnant", "Hot & Dry"])
        filter_weekend = st.selectbox("Weekend", ["All", "Yes", "No"])
        filter_dates = st.checkbox("Date Range")
        if filter_dates:
            filter_start_date = st.date_input("Start Date")
            filter_end_date = st.date_input("End Date")
        else:
            filter_start_date = None
            filter_end_date = None
        apply_filters = st.form_submit_button("Apply Filters", type="primary", use_container_width=True)

    if apply_filters:
        st.session_state["analytics_filters"] = create_filters(
            filter_city,
            filter_region,
            filter_year,
            filter_season,
            filter_time_of_day,
            filter_weather,
            filter_weekend,
            filter_start_date,
            filter_end_date,
        )
        st.session_state["analytics_y_axis"] = filter_y_axis

if "analytics_y_axis" not in st.session_state:
    st.session_state["analytics_y_axis"] = "aqi"

y_axis = st.session_state["analytics_y_axis"]
filters = st.session_state["analytics_filters"]

try:
    weather_data = get_trend("weather_verdict", y_axis, filters)
    weather_wind = get_trend("weather_verdict", "wind_speed_10m", filters)
    season_data = get_trend("regional_season", y_axis, filters)
    month_data = get_trend("month", y_axis, filters)
    month_rain = get_trend("month", "precipitation", filters)
except Exception as e:
    st.error(str(e))
    st.stop()

col1, col2 = st.columns(2)

with col1:
    if not weather_data.empty and not weather_wind.empty:
        fig = create_dual_axis_chart(
            weather_data,
            weather_wind,
            "weather_verdict",
            y_axis,
            "wind_speed_10m",
            f"{y_axis.replace('_', ' ').upper()} vs Wind Speed",
            f"Avg {y_axis.replace('_', ' ').upper()}",
            "Avg Wind Speed",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available for this filter combination.")

with col2:
    if not season_data.empty:
        fig = create_line_chart(
            season_data,
            "regional_season",
            y_axis,
            f"{y_axis.replace('_', ' ').upper()} vs Regional Season",
            f"Avg {y_axis.replace('_', ' ').upper()}",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available for this filter combination.")

st.divider()

if not month_data.empty and not month_rain.empty:
    fig = create_dual_axis_chart(
        month_data,
        month_rain,
        "month",
        y_axis,
        "precipitation",
        f"{y_axis.replace('_', ' ').upper()} vs Precipitation by Month",
        f"Avg {y_axis.replace('_', ' ').upper()}",
        "Avg Precipitation",
        " mm",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No monthly data available for this filter combination.")

st.divider()
st.header("Want to create your own custom analysis?")
st.write("Pick your own X, Y axes, chart type, and apply the selected filters to explore relationships between different variables.")

if "custom_chart" not in st.session_state:
    st.session_state["custom_chart"] = None

with st.form("custom_chart_form"):
    colX, colY, colZ = st.columns(3)
    with colX:
        chart_type = st.selectbox("Chart Type", ["Line Chart", "Dual Axis Chart"])
    with colY:
        x_axis = st.selectbox(
            "X-Axis",
            CATEGORICAL_FEATURES,
            format_func=lambda x: x.replace("_", " ").title(),
        )
    with colZ:
        y_axis_1 = st.selectbox(
            "Y-Axis 1",
            NUMERICAL_FEATURES,
            format_func=lambda x: x.replace("_", " ").title(),
        )
        y_axis_2 = st.selectbox(
            "Y-Axis 2",
            NUMERICAL_FEATURES,
            format_func=lambda x: x.replace("_", " ").title(),
        )
        if chart_type == "Line Chart":
            y_axis_2 = None
    generate_chart = st.form_submit_button("Generate Chart", type="primary", use_container_width=True)

if generate_chart:
    try:
        y1 = get_trend(x_axis, y_axis_1, filters)
        if y1.empty:
            st.session_state["custom_chart"] = None
            st.warning("No data available for the selected filters.")
        elif chart_type == "Dual Axis Chart":
            y2 = get_trend(x_axis, y_axis_2, filters)
            if y2.empty:
                st.session_state["custom_chart"] = None
                st.warning("No data available for the selected filters.")
            else:
                st.session_state["custom_chart"] = create_dual_axis_chart(
                    y1,
                    y2,
                    x_axis,
                    y_axis_1,
                    y_axis_2,
                    f"{y_axis_1.replace('_', ' ').title()} vs {y_axis_2.replace('_', ' ').title()} by {x_axis.replace('_', ' ').title()}",
                    f"Avg {y_axis_1.replace('_', ' ').title()}",
                    f"Avg {y_axis_2.replace('_', ' ').title()}",
                )
        else:
            st.session_state["custom_chart"] = create_line_chart(
                y1,
                x_axis,
                y_axis_1,
                f"{y_axis_1.replace('_', ' ').title()} by {x_axis.replace('_', ' ').title()}",
                f"Avg {y_axis_1.replace('_', ' ').title()}",
            )
    except Exception as e:
        st.session_state["custom_chart"] = None
        st.error(str(e))

if st.session_state["custom_chart"] is not None:
    st.plotly_chart(st.session_state["custom_chart"], use_container_width=True)