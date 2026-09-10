import streamlit as st


st.markdown(
    """
    <style>
    .working-kicker {
        color: #38b9ff;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    .working-intro {
        max-width: 800px;
        color: #d7e6fb;
        font-size: 1.08rem;
        line-height: 1.7;
    }
    .stage-number {
        color: #ff9d00;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.16em;
    }
    .stage-arrow {
        color: #38b9ff;
        font-size: 1.2rem;
        text-align: center;
        padding-top: 4.7rem;
    }
    .working-note {
        border-left: 3px solid #38b9ff;
        padding: 0.85rem 1rem;
        color: #b8c3d0;
        background: rgba(56, 185, 255, 0.06);
        border-radius: 0 10px 10px 0;
    }
    .dvc-flowchart {
        display: grid;
        gap: 0.7rem;
        padding: 1rem;
        border: 1px solid rgba(145, 158, 178, 0.2);
        border-radius: 16px;
        background: rgba(8, 11, 14, 0.42);
    }
    .dvc-flow-row {
        display: grid;
        grid-template-columns: 1fr auto 1fr auto 1fr;
        align-items: center;
        gap: 0.7rem;
    }
    .dvc-branch {
        display: grid;
        grid-template-columns: 1fr auto 1fr;
        align-items: center;
        gap: 0.7rem;
    }
    .dvc-path-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.9rem;
    }
    .dvc-path {
        display: grid;
        gap: 0.55rem;
        padding: 0.75rem;
        border: 1px solid rgba(139, 146, 255, 0.22);
        border-radius: 14px;
        background: rgba(21, 19, 33, 0.34);
    }
    .dvc-path-title {
        color: #b9b9ff;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }
    .dvc-node {
        min-height: 92px;
        padding: 0.85rem 0.95rem;
        border: 1px solid rgba(56, 185, 255, 0.28);
        border-radius: 12px;
        background: linear-gradient(145deg, rgba(24, 35, 42, 0.82), rgba(10, 13, 16, 0.78));
    }
    .dvc-node strong {
        display: block;
        color: #f2f5f7;
        font-family: 'Space Grotesk', sans-serif;
        margin-bottom: 0.35rem;
    }
    .dvc-node small {
        display: block;
        color: #9aa8bb;
        line-height: 1.45;
    }
    .dvc-node.model {
        border-color: rgba(139, 146, 255, 0.4);
        background: linear-gradient(145deg, rgba(37, 32, 58, 0.8), rgba(14, 12, 22, 0.78));
    }
    .dvc-node.output {
        border-color: rgba(255, 157, 0, 0.38);
        background: linear-gradient(145deg, rgba(54, 38, 20, 0.72), rgba(18, 13, 9, 0.78));
    }
    .dvc-arrow {
        color: #38b9ff;
        font-size: 1.2rem;
        text-align: center;
    }
    .dvc-branch-label {
        color: #71839a;
        font-size: 0.76rem;
        letter-spacing: 0.08em;
        text-align: center;
        text-transform: uppercase;
    }
    .feature-list {
        margin: 0.6rem 0 0;
        padding-left: 1.1rem;
    }
    .feature-list li {
        margin: 0.25rem 0;
    }
    @media (max-width: 768px) {
        .dvc-flow-row,
        .dvc-branch,
        .dvc-path-grid {
            grid-template-columns: 1fr;
        }
        .dvc-arrow {
            transform: rotate(90deg);
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="working-kicker">DVC pipeline · forecaster</p>',
    unsafe_allow_html=True,
)
st.title("How the forecaster is built")
st.markdown(
    '<p class="working-intro">Aero Metrics forecasts future particulate levels '
    "from recent city history, changing weather, and environmental context. "
    "The DVC pipeline trains separate PM2.5 and PM10 models for the current, "
    "12-hour, 24-hour, and 48-hour horizons.</p>",
    unsafe_allow_html=True,
)

st.markdown("### Forecaster DVC flow")
st.markdown(
    """
    <div class="dvc-flowchart">
        <div class="dvc-flow-row">
            <div class="dvc-node">
                <strong>Tracked source data</strong>
                <small>data/processed/engineered_features.csv</small>
            </div>
            <div class="dvc-arrow">→</div>
            <div class="dvc-node">
                <strong>Chronological preparation</strong>
                <small>Sort each city by time and build history-aware training rows</small>
            </div>
            <div class="dvc-arrow">→</div>
            <div class="dvc-node">
                <strong>Forecast features</strong>
                <small>Lags, rolling means, changes, interactions, and temporal context</small>
            </div>
        </div>
        <div class="dvc-branch">
            <div class="dvc-arrow">↙</div>
            <div class="dvc-branch-label">same history-aware features · two pollutant pathways</div>
            <div class="dvc-arrow">↘</div>
        </div>
        <div class="dvc-path-grid">
            <div class="dvc-path">
                <div class="dvc-path-title">PM2.5 pathway</div>
                <div class="dvc-node model">
                    <strong>Four horizon stages</strong>
                    <small>forecaster_train_pm25_model_t · t12 · t24 · t48</small>
                </div>
                <div class="dvc-arrow">↓</div>
                <div class="dvc-node output">
                    <strong>PM2.5 forecast artifacts</strong>
                    <small>Four models, metrics, TreeExplainers, and global SHAP outputs</small>
                </div>
                <div class="dvc-arrow">↓</div>
                <div class="dvc-node">
                    <strong>PM2.5 forecast API</strong>
                    <small>Returns current, 12-hour, 24-hour, and 48-hour predictions</small>
                </div>
            </div>
            <div class="dvc-path">
                <div class="dvc-path-title">PM10 pathway</div>
                <div class="dvc-node model">
                    <strong>Four ratio stages</strong>
                    <small>forecaster_train_pm10_model_t · t12 · t24 · t48</small>
                </div>
                <div class="dvc-arrow">↓</div>
                <div class="dvc-node output">
                    <strong>PM10 forecast artifacts</strong>
                    <small>Four ratio models, metrics, TreeExplainers, and global SHAP outputs</small>
                </div>
                <div class="dvc-arrow">↓</div>
                <div class="dvc-node">
                    <strong>PM10 forecast API</strong>
                    <small>Each ratio is multiplied by its matching PM2.5 horizon prediction</small>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### 01 · Build history-aware features")
feature_columns = st.columns(3, gap="medium")

with feature_columns[0]:
    with st.container(border=True):
        st.markdown("#### Pollution history")
        st.write(
            "Rows are sorted by city and timestamp before the forecaster "
            "creates only past-looking pollution features."
        )
        st.markdown(
            """
            <ul class="feature-list">
                <li>12h, 24h, and 48h lags</li>
                <li>24h and 48h rolling means</li>
                <li>Recent change, acceleration, max, and mean</li>
            </ul>
            """,
            unsafe_allow_html=True,
        )

with feature_columns[1]:
    with st.container(border=True):
        st.markdown("#### Weather movement")
        st.write(
            "Weather levels are paired with their recent movement so the model "
            "can distinguish a stable condition from a changing one."
        )
        st.markdown(
            """
            <ul class="feature-list">
                <li>6h and 12h changes for PM2.5</li>
                <li>6h, 12h, 24h, and 48h changes for PM10 ratio</li>
                <li>Temperature, humidity, wind, pressure, and rain</li>
            </ul>
            """,
            unsafe_allow_html=True,
        )

with feature_columns[2]:
    with st.container(border=True):
        st.markdown("#### Context and interactions")
        st.write(
            "The shared pipeline adds regional season, weather conditions, "
            "cyclic time signals, and interaction features to the history."
        )
        st.markdown(
            """
            <ul class="feature-list">
                <li>Stagnation and dry-stagnation indicators</li>
                <li>No-rain flag and weather verdict</li>
                <li>Wind direction and environmental interactions</li>
            </ul>
            """,
            unsafe_allow_html=True,
        )

st.markdown("### 02 · Separate targets without future leakage")
target_columns = st.columns(2, gap="medium")

with target_columns[0]:
    with st.container(border=True):
        st.markdown("#### PM2.5 target")
        st.write(
            "The PM2.5 forecaster predicts future PM2.5 directly. The current "
            "target is not passed as an input; only historical lags and known "
            "context are used for each horizon."
        )
        st.caption("One direct PM2.5 model for each horizon: T, T+12, T+24, T+48.")

with target_columns[1]:
    with st.container(border=True):
        st.markdown("#### PM10 ratio target")
        st.write(
            "The PM10 forecaster learns the PM10-to-PM2.5 ratio. Current "
            "pollutant targets are excluded, and the ratio is clipped for "
            "extreme training outliers before horizon-specific learning."
        )
        st.caption("One PM10 ratio model for each horizon, paired with PM2.5 output.")

st.markdown("### 03 · Horizon-specific training")
horizon_columns = st.columns(4, gap="small")
horizons = [
    ("T", "Current step", "Closest-horizon model using the latest available history."),
    ("T+12", "12 hours", "Predicts two six-hour steps ahead."),
    ("T+24", "24 hours", "Uses the same feature contract for a full day ahead."),
    ("T+48", "48 hours", "Extends the forecast to two days ahead."),
]

for column, (label, title, description) in zip(horizon_columns, horizons):
    with column:
        with st.container(border=True):
            st.markdown(f"#### {label}")
            st.markdown(f"**{title}**")
            st.write(description)

st.markdown("### 04 · Time-aware validation and artifacts")
validation_columns = st.columns(3, gap="medium")

validation_cards = [
    (
        validation_columns[0],
        "Chronological split",
        "Each forecaster keeps the first 80% of ordered rows for training and "
        "the final 20% for testing, rather than shuffling future rows backward.",
    ),
    (
        validation_columns[1],
        "TimeSeriesSplit",
        "Every horizon uses five expanding time-series folds to evaluate "
        "generalization while preserving temporal order.",
    ),
    (
        validation_columns[2],
        "Explainable outputs",
        "Each of the eight stages stores a model, metrics, TreeExplainer, "
        "global SHAP importance, SHAP values, and feature values.",
    ),
]

for column, title, description in validation_cards:
    with column:
        with st.container(border=True):
            st.markdown(f"#### {title}")
            st.write(description)

st.markdown("### 05 · Production runtime")
st.caption(
    "The deployed forecaster does not retrain during a request. It keeps the "
    "runtime database fresh, reads the latest city history, and loads the "
    "versioned model artifacts already published for inference."
)

runtime_flow = st.columns([1, 0.13, 1, 0.13, 1, 0.13, 1], gap="small")
runtime_steps = [
    (
        0,
        "01",
        "Refresh runtime data",
        "An Azure Container Job runs runtime_data_fetch on the configured "
        "schedule: 0 1,13 * * *. It refreshes the last two days of PM2.5/PM10 "
        "data plus observed and seven-day forecast weather data.",
    ),
    (
        2,
        "02",
        "Store in Azure PostgreSQL",
        "PM data is ingested into pm_data. Weather rows are written to "
        "weather_data with observed or forecast data types at the shared "
        "six-hour cadence.",
    ),
    (
        4,
        "03",
        "Prepare a request",
        "The API validates the city and limits the target to now through two "
        "days ahead, then floors the target to the nearest six-hour boundary.",
    ),
    (
        6,
        "04",
        "Run and return",
        "The API reads the selected city's database history, builds each "
        "horizon's features, runs the matching models, and returns the "
        "forecast series.",
    ),
]

for column_index, number, title, description in runtime_steps:
    with runtime_flow[column_index]:
        with st.container(border=True, key=f"working-forecaster-runtime-{number}"):
            st.markdown(f'<span class="stage-number">{number}</span>', unsafe_allow_html=True)
            st.markdown(f"#### {title}")
            st.write(description)

    if column_index < 6:
        with runtime_flow[column_index + 1]:
            st.markdown('<div class="stage-arrow">→</div>', unsafe_allow_html=True)

runtime_columns = st.columns(3, gap="medium")

with runtime_columns[0]:
    with st.container(border=True):
        st.markdown("#### Automatic mode")
        st.write(
            "The client submits a city and target time. The API reads the "
            "latest PM history and weather forecast from Azure PostgreSQL, "
            "then constructs PM2.5 and PM10-ratio inputs without requiring "
            "manual lag values."
        )
        st.caption("Routes: /api/forecast_pm25_auto and /api/forecast_pm10_auto")

with runtime_columns[1]:
    with st.container(border=True):
        st.markdown("#### Manual mode")
        st.write(
            "The client supplies current weather and PM2.5/PM10 lag values. "
            "The API combines those values with database weather history to "
            "build the same four-horizon model inputs."
        )
        st.caption("Route: /api/forecast_manual")

with runtime_columns[2]:
    with st.container(border=True):
        st.markdown("#### Horizon execution")
        st.write(
            "Each request creates inputs for T, T+12, T+24, and T+48. PM2.5 "
            "runs its direct models; PM10 runs ratio models and multiplies "
            "each ratio by the matching PM2.5 prediction."
        )
        st.caption("Outputs: PM2.5, PM10, and application-level AQI views")

st.markdown("#### Runtime data refresh cadence")
cadence_columns = st.columns(3, gap="medium")

with cadence_columns[0]:
    with st.container(border=True):
        st.markdown("##### Container cron")
        st.code("0 1,13 * * *", language="text")
        st.caption("The runtime fetch job is scheduled at 01:00 and 13:00.")

with cadence_columns[1]:
    with st.container(border=True):
        st.markdown("##### Refresh gap")
        st.metric("Scheduled gap", "12 hours")
        st.caption("Two scheduled refreshes run during each 24-hour period.")

with cadence_columns[2]:
    with st.container(border=True):
        st.markdown("##### Data coverage")
        st.write(
            "Each run fetches a rolling two-day PM window and refreshes "
            "weather observations plus a seven-day weather forecast, keeping "
            "the database ready between scheduled updates."
        )

st.markdown(
    '<div class="working-note"><strong>Production boundary:</strong> Azure '
    "Container Jobs keep the runtime data current, while the Azure-hosted "
    "FastAPI service performs inference. Model training and artifact creation "
    "remain separate from request-time prediction, so user requests stay "
    "focused on feature preparation and model execution.</div>",
    unsafe_allow_html=True,
)

st.markdown("### The 15-day refresh cycle")
cycle_columns = st.columns([1, 0.13, 1, 0.13, 1, 0.13, 1], gap="small")
cycle_steps = [
    (
        0,
        "01",
        "Schedule",
        "An Azure Container Job triggers the refresh workflow after 15 days.",
    ),
    (
        2,
        "02",
        "Refresh history",
        "New air-quality and weather observations are collected and aligned.",
    ),
    (
        4,
        "03",
        "Rebuild eight stages",
        "DVC reproduces the feature pipeline and retrains both pollutants "
        "across all four horizons.",
    ),
    (
        6,
        "04",
        "Publish artifacts",
        "Updated models and SHAP artifacts become available to forecast and "
        "explanation endpoints.",
    ),
]

for column_index, number, title, description in cycle_steps:
    with cycle_columns[column_index]:
        with st.container(border=True, key=f"working-forecaster-cycle-{number}"):
            st.markdown(f'<span class="stage-number">{number}</span>', unsafe_allow_html=True)
            st.markdown(f"#### {title}")
            st.write(description)

    if column_index < 6:
        with cycle_columns[column_index + 1]:
            st.markdown('<div class="stage-arrow">→</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="working-note"><strong>Why this matters:</strong> Forecasting '
    "depends on recent pollution and weather movement. Re-running the pipeline "
    "every 15 days lets the eight horizon models keep learning from the latest "
    "patterns while preserving DVC lineage and time-aware validation.</div>",
    unsafe_allow_html=True,
)

st.markdown("### Explore the runtime workflow")
st.caption(
    "The deployed API prepares the requested target time, builds matching "
    "horizon features, loads the appropriate artifacts, and returns PM2.5, "
    "PM10, and AQI forecasts."
)
action_columns = st.columns(2, gap="medium")

with action_columns[0]:
    if st.button(
        "Open single-city forecaster",
        type="primary",
        use_container_width=True,
        key="working-forecaster-open-single",
    ):
        st.switch_page("pages/forecaster_single.py")

with action_columns[1]:
    if st.button(
        "Open forecaster explanations",
        use_container_width=True,
        key="working-forecaster-open-shap",
    ):
        st.switch_page("pages/shap_forecaster.py")
