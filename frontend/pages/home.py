import pandas as pd
import pydeck as pdk
import streamlit as st

from resources.city_info import city_info


st.markdown(
    """
    <style>
    .st-key-home-hero-primary,
    .st-key-home-hero-context {
        min-height: 188px;
    }

    .st-key-home-workflow-predict,
    .st-key-home-workflow-forecast,
    .st-key-home-workflow-simulate,
    .st-key-home-workflow-understand {
        min-height: 310px;
    }

    .st-key-home-evidence-workflows,
    .st-key-home-evidence-shap,
    .st-key-home-evidence-metrics {
        min-height: 220px;
    }

    @media (max-width: 768px) {
        .st-key-home-hero-primary,
        .st-key-home-hero-context,
        .st-key-home-workflow-predict,
        .st-key-home-workflow-forecast,
        .st-key-home-workflow-simulate,
        .st-key-home-workflow-understand,
        .st-key-home-evidence-workflows,
        .st-key-home-evidence-shap,
        .st-key-home-evidence-metrics {
            min-height: 0;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def navigate(label, page, button_type="secondary"):
    key = f"home-nav-{page.replace('/', '-').replace('.', '-')}-{label.lower().replace(' ', '-')}"
    if st.button(label, use_container_width=True, type=button_type, key=key):
        st.switch_page(page)


def section_heading(eyebrow, title, description):
    st.markdown(
        f"""
        <div style="margin: 3rem 0 1.2rem;">
            <p style="margin: 0 0 0.45rem; color: #38b9ff; font-size: 0.72rem;
                      font-weight: 700; letter-spacing: 0.18em; text-transform: uppercase;">
                {eyebrow}
            </p>
            <h2 style="margin: 0 0 0.5rem;">{title}</h2>
            <p style="max-width: 680px; margin: 0;">{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <div style="padding: 2rem 0 1.25rem;">
        <p style="color: #38b9ff; font-size: 0.75rem; font-weight: 700;
                  letter-spacing: 0.2em; text-transform: uppercase; margin: 0 0 1rem;">
            Environmental intelligence platform
        </p>
        <h1 style="margin: 0 0 1rem;">AERO<br><span style="color: #38b9ff;">METRICS</span></h1>
        <p style="max-width: 680px; margin: 0; color: #d7e6fb; font-size: 1.2rem;">
            Understand air quality today, anticipate what comes next, and see the
            environmental factors behind every result.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

primary_col, context_col = st.columns([1.25, 1], gap="large")

with primary_col:
    with st.container(border=True, key="home-hero-primary"):
        st.markdown("#### Start with a prediction")
        st.caption(
            "Use current environmental conditions to estimate PM2.5, PM10, and AQI."
        )
        start_col, compare_col = st.columns(2, gap="small")
        with start_col:
            navigate("Predict one city", "pages/predictor_single.py", "primary")
        with compare_col:
            navigate("Compare cities", "pages/predictor_multi.py")

with context_col:
    with st.container(border=True, key="home-hero-context"):
        st.markdown("#### Or look ahead")
        st.caption(
            "Generate forecasts at current, 12-hour, 24-hour, and 48-hour horizons."
        )
        navigate("Open forecaster", "pages/forecaster_single.py", "primary")

section_heading(
    "Platform overview",
    "One question, one focused workflow",
    "Choose the tool that matches the kind of answer you need. Every module uses the same environmental context and AQI calculation pipeline.",
)

workflow_columns = st.columns(2, gap="medium")
workflows = [
    (
        "01",
        "Predict",
        "Estimate current pollution levels for one city or compare up to four cities.",
        "Single city",
        "pages/predictor_single.py",
        "Multi city",
        "pages/predictor_multi.py",
    ),
    (
        "02",
        "Forecast",
        "Project PM2.5, PM10, and AQI across four future horizons.",
        "Single city",
        "pages/forecaster_single.py",
        "Multi city",
        "pages/forecaster_multi.py",
    ),
    (
        "03",
        "Simulate",
        "Change weather and temporal inputs to compare two possible conditions.",
        "Run a simulation",
        "pages/simulator.py",
        None,
        None,
    ),
    (
        "04",
        "Understand",
        "Inspect data patterns, feature influence, and model performance.",
        "Explore analytics",
        "pages/analytics.py",
        "View explanations",
        "pages/shap_predictor.py",
    ),
]

workflow_keys = [
    "home-workflow-predict",
    "home-workflow-forecast",
    "home-workflow-simulate",
    "home-workflow-understand",
]

for index, workflow in enumerate(workflows):
    number, title, description, primary_label, primary_page, secondary_label, secondary_page = workflow
    column = workflow_columns[index % 2]
    with column:
        with st.container(border=True, key=workflow_keys[index]):
            st.caption(number)
            st.markdown(f"#### {title}")
            st.caption(description)
            navigate(primary_label, primary_page, "primary" if title in {"Predict", "Forecast"} else "secondary")
            if title == "Understand":
                with st.expander("View explanations"):
                    st.caption(
                        "Choose an explanation view for predictor or forecaster outputs."
                    )
                    navigate("Predictor SHAP", "pages/shap_predictor.py")
                    navigate("Forecaster SHAP", "pages/shap_forecaster.py")
            elif secondary_page:
                navigate(secondary_label, secondary_page)

section_heading(
    "Coverage",
    "A live view of the supported cities",
    f"Explore the {len(city_info)} Indian cities currently available for prediction, forecasting, and simulation.",
)

city_points = pd.DataFrame(
    [
        {
            "city": city,
            "region": info["region"],
            "latitude": info["lat"],
            "longitude": info["lon"],
        }
        for city, info in city_info.items()
    ]
)

city_layer = pdk.Layer(
    "ScatterplotLayer",
    data=city_points,
    get_position="[longitude, latitude]",
    get_radius=18000,
    get_fill_color=[56, 185, 255, 190],
    get_line_color=[242, 245, 247, 220],
    line_width_min_pixels=1,
    pickable=True,
)

city_view = pdk.ViewState(
    latitude=22.7,
    longitude=79.2,
    zoom=3.7,
    min_zoom=3.2,
    max_zoom=7,
)

st.pydeck_chart(
    pdk.Deck(
        map_style=None,
        initial_view_state=city_view,
        layers=[city_layer],
        tooltip={"html": "<b>{city}</b><br/>{region} region", "style": {"color": "#f2f5f7"}},
    ),
    use_container_width=True,
)

region_cities = {}
for city, info in city_info.items():
    region_cities.setdefault(info["region"], []).append(city)

with st.expander("Browse cities by region", expanded=True):
    region_columns = st.columns(3, gap="medium")
    for index, (region, cities) in enumerate(sorted(region_cities.items())):
        with region_columns[index % 3]:
            st.markdown(f"**{region} · {len(cities)}**")
            st.caption(" · ".join(sorted(cities)))

section_heading(
    "Validation",
    "Validate the results you use",
    "Trace how the models work and review their performance before relying on an estimate or forecast.",
)

evidence_columns = st.columns(2, gap="medium")
evidence = [
    (
        "Model workflows",
        "Follow the current prediction pipeline from features to AQI.",
        "Predictor workflow",
        "pages/working_predictor.py",
        "Forecaster workflow",
        "pages/working_forecaster.py",
    ),
    (
        "Model performance",
        "Review training, cross-validation, error, and generalization metrics.",
        "Predictor metrics",
        "pages/metrics_predictor.py",
        "Forecaster metrics",
        "pages/metrics_forecaster.py",
    ),
]
evidence_keys = [
    "home-evidence-workflows",
    "home-evidence-metrics",
]

for index, item in enumerate(evidence):
    title, description, primary_label, primary_page, secondary_label, secondary_page = item
    column = evidence_columns[index % 2]
    with column:
        with st.container(border=True, key=evidence_keys[index]):
            st.markdown(f"#### {title}")
            st.caption(description)
            navigate(primary_label, primary_page)
            navigate(secondary_label, secondary_page)

st.divider()

footer_left, footer_right = st.columns([2, 1], gap="large")
with footer_left:
    st.caption("Built with Python · Machine Learning · FastAPI · Streamlit · Docker · Azure")
with footer_right:
    navigate("Connect with the developer", "pages/connect.py")
