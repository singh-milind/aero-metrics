import streamlit as st

st.set_page_config(
    page_title="Aero Metrics",
    layout="wide",
    initial_sidebar_state="auto",
)

API_BASE_URL = st.secrets["API_BASE_URL"]

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --ink: #f2f5f7;
        --muted: #71839a;
        --line: rgba(145, 158, 178, 0.2);
        --glass: rgba(17, 20, 23, 0.72);
        --glass-strong: rgba(22, 25, 29, 0.9);
        --cyan: #38b9ff;
        --blue: #8b92ff;
        --red: #ff464d;
        --orange: #ff9d00;
    }

    .stApp {
        color: var(--ink);
        background:
            radial-gradient(circle at 8% 42%, rgba(0, 105, 142, 0.2), transparent 26rem),
            radial-gradient(circle at 86% 19%, rgba(116, 37, 112, 0.2), transparent 28rem),
            linear-gradient(135deg, #030506 0%, #060809 52%, #0a080c 100%);
        font-family: 'DM Sans', sans-serif;
    }

    .stApp::before {
        content: '';
        position: fixed;
        inset: 0;
        pointer-events: none;
        opacity: 0.12;
        background-image: radial-gradient(rgba(255,255,255,0.18) 0.6px, transparent 0.6px);
        background-size: 5px 5px;
        mask-image: linear-gradient(to bottom, black, transparent 65%);
    }

    [data-testid="stHeader"] {
        background: rgba(3, 5, 6, 0.72);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(8, 12, 16, 0.94), rgba(5, 8, 10, 0.82));
        border-right: 1px solid var(--line);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }

    [data-testid="stSidebar"]::before {
        content: 'AERO METRICS';
        display: block;
        padding: 0.25rem 1.2rem 1.35rem;
        color: var(--ink);
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: 0.16em;
    }

    [data-testid="stSidebarNav"] span {
        color: var(--muted);
        font-weight: 500;
    }

    [data-testid="stSidebarNav"] a {
        border-radius: 12px;
        margin: 0.2rem 0.65rem;
        transition: background 160ms ease, color 160ms ease, transform 160ms ease;
    }

    [data-testid="stSidebarNav"] a:hover {
        background: rgba(56, 185, 255, 0.1);
        transform: translateX(3px);
    }

    [data-testid="stSidebarNav"] a[aria-current="page"] {
        background: linear-gradient(90deg, rgba(56, 185, 255, 0.16), rgba(139, 146, 255, 0.08));
        box-shadow: inset 3px 0 0 var(--cyan);
    }

    [data-testid="stSidebarNav"] a[aria-current="page"] span {
        color: var(--ink);
    }

    .block-container {
        max-width: 1120px;
        padding: 2.5rem 2rem 5rem;
    }

    h1, h2, h3 {
        color: var(--cyan);
        font-family: 'Space Grotesk', sans-serif;
        letter-spacing: -0.035em;
    }

    h1 {
        font-size: clamp(2.4rem, 5vw, 4.8rem) !important;
        line-height: 1.02 !important;
    }

    h2 {
        margin-top: 2.25rem !important;
    }

    p, li, label, [data-testid="stCaptionContainer"] {
        color: var(--muted);
        line-height: 1.65;
    }

    hr {
        border-color: var(--line);
        margin: 2.5rem 0;
    }

    [data-testid="stVerticalBlockBorderWrapper"],
    [data-testid="stMetric"],
    [data-testid="stExpander"],
    [data-testid="stForm"] {
        border: 1px solid var(--line);
        border-radius: 20px;
        background: linear-gradient(145deg, rgba(24, 28, 32, 0.76), rgba(10, 12, 14, 0.72));
        box-shadow: 0 18px 50px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255,255,255,0.05);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
    }

    [data-testid="stMetric"] {
        padding: 1.15rem 1.25rem;
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted);
    }

    [data-testid="stMetricValue"] {
        color: var(--ink);
        font-family: 'Space Grotesk', sans-serif;
    }

    [data-testid="stExpander"] {
        overflow: hidden;
    }

    [data-testid="stExpander"] summary:hover {
        color: var(--blue);
    }

    .stButton > button,
    .stLinkButton > a,
    [data-testid="stFormSubmitButton"] > button {
        min-height: 2.7rem;
        border: 1px solid rgba(255, 70, 77, 0.62);
        border-radius: 999px;
        color: var(--ink);
        background: linear-gradient(90deg, rgba(32, 39, 43, 0.82), rgba(46, 26, 32, 0.78));
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.16);
        transition: transform 160ms ease, border-color 160ms ease, background 160ms ease;
    }

    .stButton > button:hover,
    .stLinkButton > a:hover,
    [data-testid="stFormSubmitButton"] > button:hover {
        border-color: var(--red);
        background: linear-gradient(90deg, rgba(48, 45, 45, 0.9), rgba(77, 29, 35, 0.84));
        transform: translateY(-2px);
    }

    .stButton > button[kind="primary"],
    [data-testid="stFormSubmitButton"] > button {
        border-color: var(--red);
    }

    .stTextInput input,
    .stNumberInput input,
    [data-baseweb="select"] > div,
    [data-testid="stDateInput"] input {
        border: 1px solid rgba(145, 158, 178, 0.24);
        border-radius: 8px;
        color: var(--ink);
        background: rgba(7, 9, 11, 0.82);
    }

    [data-testid="stSlider"] [role="slider"] {
        background: var(--red);
        border-color: var(--red);
    }

    [data-testid="stAlert"] {
        border: 1px solid rgba(56, 185, 255, 0.24);
        border-radius: 16px;
        background: rgba(16, 32, 39, 0.52);
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 16px;
        overflow: hidden;
    }

    footer { visibility: hidden; }

    @media (max-width: 768px) {
        .block-container {
            padding: 2rem 1rem 4rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

pages = {
    "": [
        st.Page("pages/home.py", title="Home", default=True),
        st.Page("pages/simulator.py", title="Simulator"),
        st.Page("pages/connect.py", title="Connect with Me"),
    ],

    "Predictor": [
        st.Page("pages/predictor_single.py", title="Single City"),
        st.Page("pages/predictor_multi.py", title="Multi City"),
    ],

    "Forecaster": [
        st.Page("pages/forecaster_single.py", title="Single City"),
        st.Page("pages/forecaster_multi.py", title="Multi City"),
    ],

    "SHAP Analysis": [
        st.Page("pages/shap_predictor.py", title="Predictor"),
        st.Page("pages/shap_forecaster.py", title="Forecaster"),
    ],

    "Data Analytics": [
        st.Page("pages/analytics.py", title="Data Analytics"),
    ],

    "How It Works": [
        st.Page("pages/working_predictor.py", title="Predictor"),
        st.Page("pages/working_forecaster.py", title="Forecaster"),
    ],

    "Metrics": [
        st.Page("pages/metrics_predictor.py", title="Predictor"),
        st.Page("pages/metrics_forecaster.py", title="Forecaster"),
    ],
}


pg = st.navigation(
    pages,
    position="sidebar",
)

pg.run()