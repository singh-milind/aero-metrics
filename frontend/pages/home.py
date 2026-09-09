import streamlit as st

from resources.city_info import city_info


def navigate(label, page, button_type="secondary"):
    key = f"home-nav-{page.replace('/', '-').replace('.', '-')}-{label.lower().replace(' ', '-')}"
    if st.button(label, use_container_width=True, type=button_type, key=key):
        st.switch_page(page)


def section_heading(eyebrow, title, description):
    st.markdown(
        f"""
        <div style="margin: 3.5rem 0 1.25rem;">
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
    <div style="padding: 2rem 0 1rem;">
        <p style="color: #38b9ff; font-size: 0.75rem; font-weight: 700;
                  letter-spacing: 0.2em; text-transform: uppercase; margin: 0 0 1rem;">
            Environmental intelligence platform
        </p>
        <h1 style="margin: 0 0 1rem;">AERO<br><span style="color: #38b9ff;">METRICS</span></h1>
        <p style="max-width: 620px; margin: 0; color: #d7e6fb; font-size: 1.2rem;">
            Clearer air-quality insights for a cleaner, smarter India.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

hero_actions, hero_context = st.columns([1.25, 1], gap="large")

with hero_actions:
    with st.container(border=True):
        st.markdown("#### Start with an air-quality question")
        st.caption(
            "Estimate current conditions, compare cities, or look ahead with a forecast."
        )
        action_col1, action_col2 = st.columns(2, gap="small")
        with action_col1:
            navigate("Predict current air quality", "pages/predictor_single.py", "primary")
        with action_col2:
            navigate("Forecast future conditions", "pages/forecaster_single.py")

with hero_context:
    with st.container(border=True):
        st.markdown("#### One platform, four lenses")
        st.markdown(
            """
            <p style="margin: 0; line-height: 2;">
                <strong style="color: #f2f5f7;">Predict</strong> what is happening now<br>
                <strong style="color: #f2f5f7;">Forecast</strong> what comes next<br>
                <strong style="color: #f2f5f7;">Simulate</strong> possible changes<br>
                <strong style="color: #f2f5f7;">Understand</strong> the factors behind it
            </p>
            """,
            unsafe_allow_html=True,
        )

section_heading(
    "Explore the platform",
    "Choose the right lens for your analysis",
    "Each workflow answers a different part of the air-quality question, from a quick estimate to model interpretation.",
)

capabilities = [
    (
        "Predictor",
        "Estimate current conditions from environmental and contextual data.",
        "Single city",
        "pages/predictor_single.py",
        "Compare cities",
        "pages/predictor_multi.py",
    ),
    (
        "Forecaster",
        "Look ahead using recent pollution history and meteorological conditions.",
        "Single city",
        "pages/forecaster_single.py",
        "Compare cities",
        "pages/forecaster_multi.py",
    ),
    (
        "Simulator",
        "Change environmental inputs and observe how the predicted result responds.",
        "Open simulator",
        "pages/simulator.py",
        None,
        None,
    ),
    (
        "Understand",
        "Inspect feature influence, training patterns, and model performance.",
        "SHAP analysis",
        "pages/shap_predictor.py",
        "Data analytics",
        "pages/analytics.py",
    ),
]

capability_columns = st.columns(4, gap="medium")
for column, capability in zip(capability_columns, capabilities):
    title, description, primary_label, primary_page, secondary_label, secondary_page = capability
    with column:
        with st.container(border=True):
            st.markdown(f"#### {title}")
            st.caption(description)
            navigate(primary_label, primary_page, "primary" if title in {"Predictor", "Forecaster"} else "secondary")
            if secondary_page:
                navigate(secondary_label, secondary_page)

section_heading(
    "Coverage",
    "Air-quality context across Indian cities",
    "Browse the cities currently supported by the platform, organized by region.",
)

region_cities = {}
for city, info in city_info.items():
    region_cities.setdefault(info["region"], []).append(city)

region_columns = st.columns(3, gap="medium")
for index, (region, cities) in enumerate(sorted(region_cities.items())):
    with region_columns[index % 3]:
        with st.expander(f"{region}  ·  {len(cities)} cities"):
            st.write(" · ".join(sorted(cities)))

section_heading(
    "How it works",
    "From environmental inputs to insight",
    "Follow the model workflow to see how data becomes a prediction and how that prediction can be interpreted.",
)

workflow_columns = st.columns(2, gap="large")
with workflow_columns[0]:
    with st.container(border=True):
        st.markdown("#### Predictor workflow")
        st.caption("See how current air-quality estimates are produced from input features.")
        navigate("View predictor workflow", "pages/working_predictor.py")

with workflow_columns[1]:
    with st.container(border=True):
        st.markdown("#### Forecaster workflow")
        st.caption("See how recent history and weather information shape future estimates.")
        navigate("View forecaster workflow", "pages/working_forecaster.py")

section_heading(
    "Go deeper",
    "Inspect the evidence behind the models",
    "Move from a result to the data, feature contributions, and performance metrics behind it.",
)

deep_dive_columns = st.columns(3, gap="medium")
deep_dive_items = [
    ("Training data", "Explore distributions, trends, and relationships.", "Explore analytics", "pages/analytics.py"),
    ("Model explanations", "See which features influence each prediction.", "Open SHAP analysis", "pages/shap_predictor.py"),
    ("Model metrics", "Compare prediction and forecasting performance.", "View metrics", "pages/metrics_predictor.py"),
]

for column, (title, description, label, page) in zip(deep_dive_columns, deep_dive_items):
    with column:
        with st.container(border=True):
            st.markdown(f"#### {title}")
            st.caption(description)
            navigate(label, page)

st.divider()

footer_left, footer_right = st.columns([2, 1], gap="large")
with footer_left:
    st.caption("Built with Python · Machine Learning · FastAPI · Streamlit · Docker · Azure")
with footer_right:
    navigate("Connect with the developer", "pages/connect.py")
