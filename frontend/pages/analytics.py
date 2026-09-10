import streamlit as st
import os
import requests
import pandas as pd
from datetime import datetime, time
from resources.charts import create_dual_axis_chart, create_line_chart
from resources.city_info import city_info

st.markdown(
    """
    <style>
    .analytics-kicker {
        color: #38b9ff;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    .analytics-context {
        border-left: 3px solid #38b9ff;
        padding: 0.15rem 0 0.15rem 0.8rem;
        color: #9aa8bb;
        margin: 0.75rem 0 1.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


API_BASE_URL = os.getenv("API_BASE_URL") or st.secrets["API_BASE_URL"]
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

st.markdown('<p class="analytics-kicker">Patterns · relationships · air quality</p>', unsafe_allow_html=True)
st.title("Explore air quality data")
st.write("Understand how weather, seasonality, and time relate to AQI, PM2.5, and PM10.")
st.markdown(
    '<div class="analytics-context">Define the dataset below, then read the comparisons that follow.</div>',
    unsafe_allow_html=True,
)
st.warning("If a chart is empty, the selected combination is not available in the training dataset. Try broadening the filters.\n\nPlease note that the analytics data is based on the training dataset and may not reflect real-time conditions.\n\nPlease wait a little while for the charts to load, as the analytics endpoint can take a few seconds to respond.")

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

st.markdown("### Analysis filters")
st.caption("Choose the slice of data used by every chart on this page.")

with st.container(border=True):
    with st.form("analytics_filter_form"):
        filter_row_one = st.columns(4, gap="medium")
        with filter_row_one[0]:
            filter_y_axis = st.selectbox("Primary metric", ["aqi", "pm2_5", "pm10"])
        with filter_row_one[1]:
            filter_city = st.selectbox("City", ["All"] + sorted(city_info.keys()))
        with filter_row_one[2]:
            filter_region = st.selectbox("Region", ["All", "North", "NCR", "South", "East", "West", "Central", "Northeast", "UT"])
        with filter_row_one[3]:
            filter_year = st.selectbox("Year", ["All"] + list(range(2020, datetime.now().year + 1)))

        filter_row_two = st.columns(4, gap="medium")
        with filter_row_two[0]:
            filter_season = st.selectbox("Season", ["All", "Winter", "Summer", "Southwest Monsoon", "Post-Monsoon", "Pre-Monsoon", "Extended Monsoon", "Retreating Monsoon"])
        with filter_row_two[1]:
            filter_time_of_day = st.selectbox("Time of Day", ["All", "Morning", "Afternoon", "Evening", "Midnight"])
        with filter_row_two[2]:
            filter_weather = st.selectbox("Weather", ["All", "Pleasant / Normal", "Hot & Humid", "Windy & Clear", "Rainy / Washed", "Cold & Stagnant", "Hot & Dry"])
        with filter_row_two[3]:
            filter_weekend = st.selectbox("Weekend", ["All", "Yes", "No"])

        filter_dates = st.checkbox("Limit to a date range")
        st.caption("Training data is available between 2023-06-30 and 2026-06-30. Leave this unchecked to use all dates.")
        if filter_dates:
            date_columns = st.columns(2, gap="medium")
            with date_columns[0]:
                filter_start_date = st.date_input("Start Date")
            with date_columns[1]:
                filter_end_date = st.date_input("End Date")
        else:
            filter_start_date = None
            filter_end_date = None
        apply_filters = st.form_submit_button("Apply filters", type="primary", use_container_width=True)

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
    weather_data = get_trend("weather_verdict", y_axis, filters).round(2)
    weather_wind = get_trend("weather_verdict", "wind_speed_10m", filters).round(2)
    season_data = get_trend("regional_season", y_axis, filters).round(2)
    month_data = get_trend("month", y_axis, filters).round(2)
    month_rain = get_trend("month", "precipitation", filters).round(2)
    city_data = get_trend("city", y_axis, filters).round(2)
    time_data = get_trend("time_of_day", y_axis, filters).round(2)
    region_data = get_trend("region", y_axis, filters).round(2)
    city_pm25 = get_trend("city", "pm2_5", filters).round(2)
    city_pm10 = get_trend("city", "pm10", filters).round(2)
    region_aqi = get_trend("region", "aqi", filters).round(2)
    region_pm25 = get_trend("region", "pm2_5", filters).round(2)
    region_pm10 = get_trend("region", "pm10", filters).round(2)
    season_pm25 = get_trend("regional_season", "pm2_5", filters).round(2)
    season_pm10 = get_trend("regional_season", "pm10", filters).round(2)
except Exception as e:
    st.error(str(e))
    st.stop()

def readable_axis(value):
    return value.replace("_", " ").upper()

st.markdown("### Filtered insights")
st.caption(f"Primary metric: **{readable_axis(y_axis)}** · Results update when you apply filters in the sidebar.")

insight_columns = st.columns(2, gap="medium")

with insight_columns[0]:
    with st.container(border=True):
        st.markdown("#### Weather conditions")
        st.caption(f"{readable_axis(y_axis)} and wind speed by weather verdict")
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

with insight_columns[1]:
    with st.container(border=True):
        st.markdown("#### Seasonal conditions")
        st.caption(f"{readable_axis(y_axis)} by regional season")
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

st.markdown("### Monthly patterns")
st.caption(f"{readable_axis(y_axis)} and precipitation across the year")
with st.container(border=True):
    if not month_data.empty and not month_rain.empty:
        fig = create_dual_axis_chart(
            month_data,
            month_rain,
            "month",
            y_axis,
            "precipitation",
            f"{y_axis.replace('_', ' ').upper()} vs Precipitation by Month",
            f"Avg {y_axis.replace('_', ' ').upper()}",
            "Avg Precipitation in 6h",
            " mm",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No monthly data available for this filter combination.")

st.markdown("### Distribution by place and time")
st.caption(f"Compare {readable_axis(y_axis)} across cities, regions, and times of day.")

distribution_columns = st.columns(2, gap="medium")

with distribution_columns[0]:
    with st.container(border=True):
        st.markdown("#### Time of day")
        if not time_data.empty:
            time_order = ["Morning", "Afternoon", "Evening", "Midnight"]
            time_data = time_data.copy()
            time_data["time_of_day"] = pd.Categorical(
                time_data["time_of_day"],
                categories=time_order,
                ordered=True,
            )
            time_data = time_data.sort_values("time_of_day")
            fig = create_line_chart(
                time_data,
                "time_of_day",
                y_axis,
                f"Average {y_axis.replace('_', ' ').upper()} by Time of Day",
                f"Avg {y_axis.replace('_', ' ').upper()}",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No time-of-day data available for this filter combination.")

with distribution_columns[1]:
    with st.container(border=True):
        st.markdown("#### Regional comparison")
        if not region_data.empty:
            fig = create_line_chart(
                region_data,
                "region",
                y_axis,
                f"Average {y_axis.replace('_', ' ').upper()} by Region",
                f"Avg {y_axis.replace('_', ' ').upper()}",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No regional data available for this filter combination.")

with st.container(border=True):
    st.markdown("#### City comparison")
    st.caption("City-level values are shown without persistent labels; hover over a point for details.")
    if not city_data.empty:
        fig = create_line_chart(
            city_data,
            "city",
            y_axis,
            f"Average {y_axis.replace('_', ' ').upper()} by City",
            f"Avg {y_axis.replace('_', ' ').upper()}",
            show_text=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No city data available for this filter combination.")

st.markdown("### Metric comparisons")
st.caption("These comparisons use the same filtered aggregate data supplied by the analytics endpoint.")

comparison_columns = st.columns(2, gap="medium")

with comparison_columns[0]:
    with st.container(border=True):
        st.markdown("#### AQI vs PM2.5 by region")
        if not region_aqi.empty and not region_pm25.empty:
            fig = create_dual_axis_chart(
                region_aqi,
                region_pm25,
                "region",
                "aqi",
                "pm2_5",
                "AQI vs PM2.5 by Region",
                "Avg AQI",
                "Avg PM2.5",
                " µg/m³",
                show_text=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No regional comparison data available.")

with comparison_columns[1]:
    with st.container(border=True):
        st.markdown("#### PM2.5 vs PM10 by season")
        if not season_pm25.empty and not season_pm10.empty:
            fig = create_dual_axis_chart(
                season_pm10,
                season_pm25,
                "regional_season",
                "pm10",
                "pm2_5",
                "PM2.5 vs PM10 by Season",
                "Avg PM10",
                "Avg PM2.5",
                " µg/m³",
                show_text=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No seasonal comparison data available.")

with st.container(border=True):
    st.markdown("#### PM2.5 vs PM10 by city")
    st.caption("City-level values are shown without persistent labels; hover over a bar or point for details.")
    if not city_pm25.empty and not city_pm10.empty:
        fig = create_dual_axis_chart(
            city_pm10,
            city_pm25,
            "city",
            "pm10",
            "pm2_5",
            "PM2.5 vs PM10 by City",
            "Avg PM10",
            "Avg PM2.5",
            " µg/m³",
            show_text=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No city comparison data available.")

st.markdown("### Build a custom analysis")
st.caption("Choose your own dimensions and metrics while keeping the active sidebar filters.")

if "custom_chart" not in st.session_state:
    st.session_state["custom_chart"] = None

with st.container(border=True):
    with st.form("custom_chart_form"):
        colX, colY, colZ = st.columns(3, gap="medium")
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
        generate_chart = st.form_submit_button("Generate custom chart", type="primary", use_container_width=True)

if generate_chart:
    try:
        y1 = get_trend(x_axis, y_axis_1, filters).round(2)
        if y1.empty:
            st.session_state["custom_chart"] = None
            st.warning("No data available for the selected filters.")
        elif chart_type == "Dual Axis Chart":
            y2 = get_trend(x_axis, y_axis_2, filters).round(2)
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
    st.markdown("#### Custom result")
    with st.container(border=True):
        st.plotly_chart(st.session_state["custom_chart"], use_container_width=True)