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
        max-width: 780px;
        color: #d7e6fb;
        font-size: 1.08rem;
        line-height: 1.7;
    }
    .working-stage {
        min-height: 255px;
    }
    .stage-number {
        color: #ff9d00;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.16em;
    }
    .stage-arrow {
        color: #71839a;
        font-size: 1.15rem;
        text-align: center;
        padding-top: 5rem;
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
    .dvc-flow-branch {
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
        min-height: 100px;
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
    @media (max-width: 768px) {
        .dvc-flow-row,
        .dvc-flow-branch,
        .dvc-path-grid {
            grid-template-columns: 1fr;
        }
        .dvc-arrow {
            transform: rotate(90deg);
        }
    }
    .pipeline-pill-list {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin-top: 0.75rem;
    }
    .pipeline-pill {
        padding: 0.38rem 0.62rem;
        border: 1px solid rgba(56, 185, 255, 0.22);
        border-radius: 999px;
        color: #b9d8ec;
        background: rgba(56, 185, 255, 0.07);
        font-size: 0.78rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="working-kicker">DVC pipeline · predictor</p>',
    unsafe_allow_html=True,
)
st.title("How the predictor is built")
st.markdown(
    '<p class="working-intro">The predictor is trained from aligned air-quality '
    "and weather observations. DVC moves the data through reproducible "
    "preprocessing and feature-engineering stages before training separate "
    "PM2.5 and PM10 models with cross-validation and SHAP artifacts.</p>",
    unsafe_allow_html=True,
)

st.markdown("### Predictor DVC flow")
st.markdown(
    """
    <div class="dvc-flowchart">
        <div class="dvc-flow-row">
            <div class="dvc-node">
                <strong>Raw inputs</strong>
                <small>data/raw/aqi_data.csv<br>data/raw/weather_data.csv</small>
            </div>
            <div class="dvc-arrow">→</div>
            <div class="dvc-node">
                <strong>preprocess_data</strong>
                <small>Inner merge on city and time</small>
            </div>
            <div class="dvc-arrow">→</div>
            <div class="dvc-node">
                <strong>feature_engineering</strong>
                <small>Temporal, regional, weather, and interaction features</small>
            </div>
        </div>
        <div class="dvc-branch">
            <div class="dvc-arrow">↙</div>
            <div class="dvc-branch-label">same engineered dataset · two independent targets</div>
            <div class="dvc-arrow">↘</div>
        </div>
        <div class="dvc-path-grid">
            <div class="dvc-path">
                <div class="dvc-path-title">PM2.5 pathway</div>
                <div class="dvc-node model">
                    <strong>predictor_train_pm25_model</strong>
                    <small>Train against PM2.5 without PM2.5 or PM10 inputs</small>
                </div>
                <div class="dvc-arrow">↓</div>
                <div class="dvc-node output">
                    <strong>PM2.5 artifacts</strong>
                    <small>Model · metrics · TreeExplainer · global SHAP outputs</small>
                </div>
                <div class="dvc-arrow">↓</div>
                <div class="dvc-node">
                    <strong>PM2.5 API and SHAP flow</strong>
                    <small>DVC-tracked artifacts are loaded by prediction and explanation endpoints</small>
                </div>
            </div>
            <div class="dvc-path">
                <div class="dvc-path-title">PM10 pathway</div>
                <div class="dvc-node model">
                    <strong>predictor_train_pm10_model</strong>
                    <small>Train the PM10 / PM2.5 ratio without pollutant inputs</small>
                </div>
                <div class="dvc-arrow">↓</div>
                <div class="dvc-node output">
                    <strong>PM10 artifacts</strong>
                    <small>Model · metrics · TreeExplainer · global SHAP outputs</small>
                </div>
                <div class="dvc-arrow">↓</div>
                <div class="dvc-node">
                    <strong>PM10 API and SHAP flow</strong>
                    <small>The ratio is applied to predicted PM2.5 for the final PM10 result</small>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### The reproducible pipeline")
pipeline_columns = st.columns([1, 0.13, 1, 0.13, 1, 0.13, 1], gap="small")

stages = [
    (
        0,
        "01",
        "Gather data",
        "Jobs collect three years of six-hour observations for the configured "
        "cities from Open-Meteo's air-quality and weather APIs.",
    ),
    (
        2,
        "02",
        "Merge datasets",
        "AQI and weather tables are joined with an inner merge on city and "
        "timestamp, producing the interim training dataset.",
    ),
    (
        4,
        "03",
        "Engineer features",
        "Time, region, season, weather verdict, interaction, direction, and "
        "cyclic features are added to the merged observations.",
    ),
    (
        6,
        "04",
        "Train models",
        "The engineered table feeds independent PM2.5 and PM10 DVC stages, "
        "each producing a model, metrics, and explainability artifacts.",
    ),
]

for column_index, number, title, description in stages:
    with pipeline_columns[column_index]:
        with st.container(border=True, key=f"working-predictor-stage-{number}"):
            st.markdown(f'<span class="stage-number">{number}</span>', unsafe_allow_html=True)
            st.markdown(f"#### {title}")
            st.write(description)

    if column_index < 6:
        with pipeline_columns[column_index + 1]:
            st.markdown('<div class="stage-arrow">→</div>', unsafe_allow_html=True)

st.markdown("### 01 · Data collection and alignment")
collection_columns = st.columns(2, gap="medium")

with collection_columns[0]:
    with st.container(border=True):
        st.markdown("#### Air-quality observations")
        st.write(
            "The AQI job calls the Open-Meteo air-quality endpoint for PM2.5 "
            "and PM10. Hourly values are resampled to six-hour means, labeled "
            "with the city, and missing rows are removed."
        )
        st.markdown(
            """
            <div class="pipeline-pill-list">
                <span class="pipeline-pill">PM2.5</span>
                <span class="pipeline-pill">PM10</span>
                <span class="pipeline-pill">6-hour mean</span>
                <span class="pipeline-pill">City + timestamp</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

with collection_columns[1]:
    with st.container(border=True):
        st.markdown("#### Weather observations")
        st.write(
            "The weather job uses the Open-Meteo archive endpoint and applies "
            "the same six-hour cadence. It aggregates temperature, humidity, "
            "wind, precipitation, and surface pressure."
        )
        st.markdown(
            """
            <div class="pipeline-pill-list">
                <span class="pipeline-pill">Temperature</span>
                <span class="pipeline-pill">Humidity</span>
                <span class="pipeline-pill">Wind speed / direction</span>
                <span class="pipeline-pill">Precipitation</span>
                <span class="pipeline-pill">Pressure</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("### 02 · Feature engineering")
feature_columns = st.columns(3, gap="medium")

feature_cards = [
    (
        feature_columns[0],
        "Temporal context",
        "The timestamp becomes month, hour, weekday, time of day, and a "
        "weekend flag. Month, weekday, and hour are represented with sine "
        "and cosine transformations.",
        "Month · weekday · hour · weekend",
    ),
    (
        feature_columns[1],
        "Regional context",
        "Cities are mapped to regions, then paired with region-specific "
        "season rules. Weather thresholds add a readable condition category.",
        "Region · seasonal label · weather verdict",
    ),
    (
        feature_columns[2],
        "Model-ready signals",
        "The pipeline creates temperature-humidity, wind-precipitation, and "
        "pressure-temperature interactions. Wind direction becomes sine/cosine "
        "components and categories remain XGBoost categorical features.",
        "Interactions · direction encoding · categories",
    ),
]

for column, title, description, pills in feature_cards:
    with column:
        with st.container(border=True):
            st.markdown(f"#### {title}")
            st.write(description)
            st.caption(pills)

st.markdown("### 03 · Independent predictor stages")
model_columns = st.columns(2, gap="medium")

with model_columns[0]:
    with st.container(border=True):
        st.markdown("#### PM2.5 model")
        st.write(
            "The PM2.5 stage removes pm2_5, pm10, and time from the feature "
            "table, then learns directly against the pm2_5 target."
        )
        st.caption(
            "Leakage control: the PM2.5 target and the observed PM10 value "
            "are removed before training. The model must estimate PM2.5 from "
            "environmental, location, and time features only."
        )

with model_columns[1]:
    with st.container(border=True):
        st.markdown("#### PM10 model")
        st.write(
            "The PM10 stage uses the same inputs but learns the pm10 / pm2_5 "
            "ratio. At runtime, the ratio is multiplied by the PM2.5 estimate "
            "to reconstruct PM10."
        )
        st.caption(
            "Leakage control: both observed pollutant columns are removed "
            "before training the PM10-to-PM2.5 ratio. At runtime, the learned "
            "ratio is applied to the independently predicted PM2.5 value."
        )

st.markdown("### 04 · Validation and experiment tracking")
validation_columns = st.columns(3, gap="medium")

validation_cards = [
    (
        validation_columns[0],
        "Data split",
        "Each model uses an 80/20 train-test split with random_state 42.",
    ),
    (
        validation_columns[1],
        "Cross-validation",
        "Training uses shuffled 5-fold KFold validation and records R², MAE, "
        "RMSE, fold variation, and the train-to-CV generalization gap.",
    ),
    (
        validation_columns[2],
        "Tracking",
        "Hyperparameters, metrics, feature importance, fold results, and the "
        "trained XGBoost model are logged to MLflow through DAGsHub.",
    ),
]

for column, title, description in validation_cards:
    with column:
        with st.container(border=True):
            st.markdown(f"#### {title}")
            st.write(description)

st.markdown("### The 15-day model refresh cycle")
st.caption(
    "The training workflow runs on a recurring schedule so the predictors "
    "can learn from the latest available air-quality and weather patterns."
)

cycle_columns = st.columns([1, 0.13, 1, 0.13, 1, 0.13, 1], gap="small")
cycle_steps = [
    (
        0,
        "01",
        "Schedule",
        "After 15 days, an Azure Container Job triggers the refresh workflow "
        "again.",
    ),
    (
        2,
        "02",
        "Collect latest data",
        "New air-quality and weather observations are fetched for the "
        "configured cities, Later stored in Azure Blob Storage alongside Azure PostgreSQL.",
    ),
    (
        4,
        "03",
        "Rebuild and retrain",
        "The Azure Container Job runs DVC to reproduce preprocessing, feature "
        "engineering, and both predictor training stages with refreshed data.\n\n All model and SHAP artifacts are updated in Azure Blob Storage.",

    ),
    (
        6,
        "04",
        "Refresh deployment",
        "The FastAPI service is restarted to load the new artifacts \n\n New model and SHAP artifacts are uploaded and made available to the "
        "API and explanation workflows.",
    ),
]

for column_index, number, title, description in cycle_steps:
    with cycle_columns[column_index]:
        with st.container(border=True, key=f"working-predictor-cycle-{number}"):
            st.markdown(f'<span class="stage-number">{number}</span>', unsafe_allow_html=True)
            st.markdown(f"#### {title}")
            st.write(description)

    if column_index < 6:
        with cycle_columns[column_index + 1]:
            st.markdown('<div class="stage-arrow">→</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="working-note"><strong>Why this matters:</strong> The models '
    "are not treated as static artifacts. Repeating the pipeline every 15 days "
    "helps the deployed predictors keep the latest seasonal, weather, and "
    "pollution patterns in their training data while preserving the same "
    "reproducible DVC workflow.</div>",
    unsafe_allow_html=True,
)

st.markdown("### Explore the runtime workflow")
st.caption(
    "The deployed API applies the same feature construction logic to submitted "
    "conditions before loading the corresponding predictor artifacts."
)
action_columns = st.columns(2, gap="medium")

with action_columns[0]:
    if st.button(
        "Open single-city predictor",
        type="primary",
        use_container_width=True,
        key="working-predictor-open-single",
    ):
        st.switch_page("pages/predictor_single.py")

with action_columns[1]:
    if st.button(
        "Open predictor explanations",
        use_container_width=True,
        key="working-predictor-open-shap",
    ):
        st.switch_page("pages/shap_predictor.py")
