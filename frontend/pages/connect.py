import streamlit as st


st.markdown(
    """
    <style>
    .connect-kicker {
        color: #38b9ff;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    .connect-card {
        min-height: 100%;
    }
    .connect-role {
        color: #9aa8bb;
        font-size: 1rem;
        margin-top: -0.45rem;
    }
    .connect-link {
        display: block;
        padding: 0.65rem 0.8rem;
        margin: 0.45rem 0;
        border: 1px solid rgba(145, 158, 178, 0.2);
        border-radius: 10px;
        color: #dce8f2 !important;
        background: rgba(17, 20, 23, 0.48);
        text-decoration: none !important;
        transition: border-color 150ms ease, background 150ms ease;
    }
    .connect-link:hover {
        border-color: rgba(56, 185, 255, 0.7);
        background: rgba(56, 185, 255, 0.1);
    }
    .connect-link small {
        display: block;
        color: #71839a;
        margin-top: 0.15rem;
    }
    .connect-muted {
        display: block;
        padding: 0.65rem 0.8rem;
        margin: 0.45rem 0;
        border: 1px dashed rgba(145, 158, 178, 0.18);
        border-radius: 10px;
        color: #71839a;
    }
    .stack-list {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin-top: 0.8rem;
    }
    .stack-badge {
        padding: 0.35rem 0.6rem;
        border: 1px solid rgba(56, 185, 255, 0.22);
        border-radius: 999px;
        color: #b9d8ec;
        background: rgba(56, 185, 255, 0.07);
        font-size: 0.78rem;
    }
    .connect-disclaimer {
        border-left: 3px solid #ff9d00;
        padding: 0.8rem 1rem;
        color: #b8c3d0;
        background: rgba(255, 157, 0, 0.06);
        border-radius: 0 10px 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

API_BASE_URL = st.secrets["API_BASE_URL"].rstrip("/")
GITHUB_URL = "https://github.com/singh-milind"
PROJECT_GITHUB_URL = "https://github.com/singh-milind/aero-metrics"
LINKEDIN_URL = "https://www.linkedin.com/in/milind-singh-880a41314/"
DAGSHUB_URL = "https://dagshub.com/singh-milind/aero-metrics"
API_DOCS_URL = f"{API_BASE_URL}/docs"

st.markdown(
    '<p class="connect-kicker">People · project · technology</p>',
    unsafe_allow_html=True,
)
st.title("Connect with Me")
st.write(
    "Learn more about the developer behind Aero Metrics and the systems "
    "that power the platform."
)

developer_col, project_col = st.columns(2, gap="medium")

with developer_col:
    with st.container(border=True):
        st.markdown("### Milind Singh")
        st.markdown(
            '<p class="connect-role">Student · Aspiring Data Scientist & ML Engineer</p>',
            unsafe_allow_html=True,
        )
        st.write(
            "I am a student building practical machine-learning systems and "
            "developing Aero Metrics as a resume project. I am aspiring to "
            "work in data science, machine learning engineering, and "
            "applied AI roles."
        )

        st.markdown("#### Connect with me")
        st.markdown(
            f'<a class="connect-link" href="{GITHUB_URL}" target="_blank">'
            "GitHub<small>View projects and source contributions</small></a>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<a class="connect-link" href="{LINKEDIN_URL}" target="_blank">'
            "LinkedIn<small>Connect professionally</small></a>",
            unsafe_allow_html=True,
        )
        # st.markdown(
        #     '<span class="connect-muted">Portfolio<small>URL not configured</small></span>',
        #     unsafe_allow_html=True,
        # )
        st.markdown(
            '<span class="connect-link">Email<small>   singhmilind604@gmail.com</small></span>',
            unsafe_allow_html=True,
        )

with project_col:
    with st.container(border=True):
        st.markdown("### Aero Metrics")
        st.write(
            "A full-stack air-quality intelligence platform that collects "
            "weather and PM2.5/PM10 data, prepares city and time-based "
            "features, and serves machine-learning predictions and "
            "multi-horizon forecasts through a FastAPI backend. The "
            "Streamlit application brings together interactive prediction, "
            "forecasting, analytics, model metrics, and SHAP explanations "
            "for exploring air-quality patterns across cities. Containerized "
            "data and model jobs run through Azure container infrastructure, "
            "while the FastAPI service is hosted in Azure Container Apps and "
            "the application data is stored in Azure Database for PostgreSQL."
        )

        st.markdown("#### Project links")
        st.markdown(
            f'<a class="connect-link" href="{PROJECT_GITHUB_URL}" target="_blank">'
            "View source code<small>Open the Aero Metrics repository</small></a>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<a class="connect-link" href="{API_DOCS_URL}" target="_blank">'
            "API documentation<small>Explore the configured FastAPI docs</small></a>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<a class="connect-link" href="{DAGSHUB_URL}" target="_blank">'
            "Visualized DVC pipeline <small>Sroll down after clicking the link</small></a>",
            unsafe_allow_html=True,
        )

        st.markdown("#### Technology stack")
        st.markdown(
            """
            <div class="stack-list">
                <span class="stack-badge">Python</span>
                <span class="stack-badge">FastAPI</span>
                <span class="stack-badge">Streamlit</span>
                <span class="stack-badge">Pandas · NumPy</span>
                <span class="stack-badge">Scikit-learn · XGBoost</span>
                <span class="stack-badge">SHAP</span>
                <span class="stack-badge">Plotly</span>
                <span class="stack-badge">Azure PostgreSQL · SQLAlchemy</span>
                <span class="stack-badge">Docker</span>
                <span class="stack-badge">DVC · DAGsHub</span>
                <span class="stack-badge">MLflow</span>
                <span class="stack-badge">Azure Blob Storage</span>
                <span class="stack-badge">Azure Container Apps · API hosting</span>
                <span class="stack-badge">Azure Container Apps · Job hosting</span>
                <span class="stack-badge">GitHub Actions</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("### Responsible use")
st.markdown(
    '<div class="connect-disclaimer"><strong>Disclaimer:</strong> '
    "Forecasts are model-generated estimates and should not be treated as "
    "official air-quality measurements or medical advice.</div>",
    unsafe_allow_html=True,
)
