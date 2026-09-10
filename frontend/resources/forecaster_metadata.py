def get_feature_metadata():
    FEATURE_METADATA = {

        # =========================================================
        # RAW METEOROLOGICAL FEATURES
        # =========================================================

        "temperature_2m": {
            "description": "Air temperature at 2 metres above ground level.",
            "unit": "°C",
            "type": "raw meteorological feature"
        },

        "relative_humidity_2m": {
            "description": "Relative humidity at 2 metres above ground level.",
            "unit": "%",
            "type": "raw meteorological feature"
        },

        "wind_speed_10m": {
            "description": "Wind speed at 10 metres above ground level.",
            "unit": "m/s",
            "type": "raw meteorological feature"
        },

        "wind_direction_10m": {
            "description": "Wind direction at 10 metres above ground level.",
            "unit": "degrees",
            "type": "raw meteorological feature"
        },

        "surface_pressure": {
            "description": "Atmospheric pressure at the surface.",
            "unit": "hPa",
            "type": "raw meteorological feature"
        },

        "precipitation": {
            "description": "Precipitation amount associated with the observation.",
            "unit": "mm",
            "type": "raw meteorological feature"
        },


        # =========================================================
        # LOCATION / WEATHER / TIME CATEGORIES
        # =========================================================

        "city": {
            "description": "City or geographic location represented as a categorical model feature.",
            "unit": None,
            "type": "categorical feature"
        },

        "weather_verdict": {
            "description": "Categorical summary of the prevailing weather condition.",
            "unit": None,
            "type": "categorical feature"
        },

        "time_of_day": {
            "description": "Categorical time period such as Morning, Afternoon, Evening, or Midnight.",
            "unit": None,
            "type": "categorical feature"
        },

        "season_region": {
            "description": "Combined categorical feature representing the regional season, constructed from regional_season and geographic region.",
            "unit": None,
            "type": "engineered categorical feature"
        },


        # =========================================================
        # PM2.5 HISTORICAL / LAG FEATURES
        # =========================================================

        "pm2_5_lag_12h": {
            "description": "PM2.5 concentration approximately 12 hours before the prediction time.",
            "unit": "µg/m³",
            "type": "historical PM2.5 feature"
        },

        "pm2_5_lag_24h": {
            "description": "PM2.5 concentration approximately 24 hours before the prediction time.",
            "unit": "µg/m³",
            "type": "historical PM2.5 feature"
        },

        "pm2_5_lag_48h": {
            "description": "PM2.5 concentration approximately 48 hours before the prediction time.",
            "unit": "µg/m³",
            "type": "historical PM2.5 feature"
        },

        "pm2_5_rolling_mean_24h": {
            "description": "Mean PM2.5 concentration over the previous 24 hours, excluding the current observation. Represents the recent sustained pollution level.",
            "unit": "µg/m³",
            "type": "engineered temporal PM2.5 feature"
        },

        "pm2_5_rolling_mean_48h": {
            "description": "Mean PM2.5 concentration over the previous 48 hours, excluding the current observation. Represents the broader recent pollution level.",
            "unit": "µg/m³",
            "type": "engineered temporal PM2.5 feature"
        },

        "pm2_5_recent_mean": {
            "description": "Mean of the available 12-hour, 24-hour, and 48-hour PM2.5 lag values. Represents the recent average pollution level.",
            "unit": "µg/m³",
            "type": "engineered temporal PM2.5 feature"
        },

        "pm2_5_recent_max": {
            "description": "Maximum of the 12-hour, 24-hour, and 48-hour PM2.5 lag values. Represents the highest recent PM2.5 level among the supplied historical points.",
            "unit": "µg/m³",
            "type": "engineered temporal PM2.5 feature"
        },

        "pm2_5_change_12h": {
            "description": "Difference between the 12-hour and 24-hour PM2.5 lag values. Indicates the recent change in PM2.5 over approximately 12 hours.",
            "unit": "µg/m³",
            "type": "engineered temporal PM2.5 feature"
        },

        "pm2_5_change_24h": {
            "description": "Difference between the 24-hour and 48-hour PM2.5 lag values. Indicates the change in PM2.5 across the preceding 24-hour interval.",
            "unit": "µg/m³",
            "type": "engineered temporal PM2.5 feature"
        },

        "pm2_5_acceleration": {
            "description": "Second-order PM2.5 change calculated from the 12-hour, 24-hour, and 48-hour lag values. Represents whether the recent PM2.5 trend is strengthening or weakening.",
            "unit": "µg/m³",
            "type": "engineered temporal PM2.5 feature"
        },


        # =========================================================
        # WEATHER CHANGE FEATURES
        # =========================================================

        "temperature_2m_change_6h": {
            "description": "Change in 2-metre air temperature over approximately 6 hours.",
            "unit": "°C",
            "type": "engineered temporal meteorological feature"
        },

        "temperature_2m_change_12h": {
            "description": "Change in 2-metre air temperature over approximately 12 hours.",
            "unit": "°C",
            "type": "engineered temporal meteorological feature"
        },

        "relative_humidity_2m_change_6h": {
            "description": "Change in relative humidity over approximately 6 hours.",
            "unit": "%",
            "type": "engineered temporal meteorological feature"
        },

        "relative_humidity_2m_change_12h": {
            "description": "Change in relative humidity over approximately 12 hours.",
            "unit": "%",
            "type": "engineered temporal meteorological feature"
        },

        "wind_speed_10m_change_6h": {
            "description": "Change in wind speed over approximately 6 hours.",
            "unit": "m/s",
            "type": "engineered temporal meteorological feature"
        },

        "wind_speed_10m_change_12h": {
            "description": "Change in wind speed over approximately 12 hours.",
            "unit": "m/s",
            "type": "engineered temporal meteorological feature"
        },

        "surface_pressure_change_6h": {
            "description": "Change in surface atmospheric pressure over approximately 6 hours.",
            "unit": "hPa",
            "type": "engineered temporal meteorological feature"
        },

        "surface_pressure_change_12h": {
            "description": "Change in surface atmospheric pressure over approximately 12 hours.",
            "unit": "hPa",
            "type": "engineered temporal meteorological feature"
        },

        "precipitation_change_6h": {
            "description": "Change in precipitation amount over approximately 6 hours.",
            "unit": "mm",
            "type": "engineered temporal meteorological feature"
        },

        "precipitation_change_12h": {
            "description": "Change in precipitation amount over approximately 12 hours.",
            "unit": "mm",
            "type": "engineered temporal meteorological feature"
        },


        # =========================================================
        # INTERACTION FEATURES
        # =========================================================

        "temp_humidity": {
            "description": "Interaction between temperature and relative humidity, calculated as temperature_2m × relative_humidity_2m.",
            "unit": "°C·%",
            "type": "engineered interaction feature"
        },

        "wind_precip": {
            "description": "Interaction between wind speed and precipitation, calculated as wind_speed_10m × precipitation.",
            "unit": "m/s·mm",
            "type": "engineered interaction feature"
        },

        "pressure_temp": {
            "description": "Interaction between surface pressure and temperature, calculated as surface_pressure × temperature_2m.",
            "unit": "hPa·°C",
            "type": "engineered interaction feature"
        },


        # =========================================================
        # ATMOSPHERIC / STAGNATION FEATURES
        # =========================================================

        "stagnation_index": {
            "description": "Engineered indicator calculated as relative humidity divided by wind speed plus one. Represents the combined humidity and wind conditions used by the model as a proxy for stagnant atmospheric conditions.",
            "unit": "derived",
            "type": "engineered atmospheric feature"
        },

        "dry_stagnation": {
            "description": "Engineered indicator combining relative humidity with the inverse of wind speed plus one. Represents a model-derived dry/stagnant atmospheric pattern.",
            "unit": "derived",
            "type": "engineered atmospheric feature"
        },

        "no_rain": {
            "description": "Binary indicator equal to 1 when precipitation is zero and 0 otherwise.",
            "unit": None,
            "type": "engineered binary weather feature"
        },


        # =========================================================
        # WIND DIRECTION CYCLIC FEATURES
        # =========================================================

        "wind_dir_sin": {
            "description": "Sine encoding of wind direction used to represent wind direction cyclically.",
            "unit": None,
            "type": "cyclic engineered feature"
        },

        "wind_dir_cos": {
            "description": "Cosine encoding of wind direction used to represent wind direction cyclically.",
            "unit": None,
            "type": "cyclic engineered feature"
        },


        # =========================================================
        # TIME CYCLIC FEATURES
        # =========================================================

        "month_sin": {
            "description": "Sine encoding of month used to represent annual cyclic seasonality.",
            "unit": None,
            "type": "cyclic time feature"
        },

        "month_cos": {
            "description": "Cosine encoding of month used to represent annual cyclic seasonality.",
            "unit": None,
            "type": "cyclic time feature"
        },

        "hour_sin": {
            "description": "Sine encoding of hour of day used to represent daily cyclic patterns.",
            "unit": None,
            "type": "cyclic time feature"
        },

        "hour_cos": {
            "description": "Cosine encoding of hour of day used to represent daily cyclic patterns.",
            "unit": None,
            "type": "cyclic time feature"
        },

        "dow_sin": {
            "description": "Sine encoding of day of week used to represent weekly cyclic patterns.",
            "unit": None,
            "type": "cyclic time feature"
        },

        "dow_cos": {
            "description": "Cosine encoding of day of week used to represent weekly cyclic patterns.",
            "unit": None,
            "type": "cyclic time feature"
        },

        "is_weekend": {
            "description": "Binary indicator representing whether the prediction time falls on a weekend.",
            "unit": None,
            "type": "binary time feature"
        },
        # =========================================================
        # PM10 / PM2.5 RATIO HISTORICAL FEATURES
        # =========================================================

        "pm_ratio_lag_12h": {
            "description": "PM10-to-PM2.5 ratio approximately 12 hours before the prediction time.",
            "unit": "ratio",
            "type": "historical PM10/PM2.5 ratio feature"
        },

        "pm_ratio_lag_24h": {
            "description": "PM10-to-PM2.5 ratio approximately 24 hours before the prediction time.",
            "unit": "ratio",
            "type": "historical PM10/PM2.5 ratio feature"
        },

        "pm_ratio_lag_48h": {
            "description": "PM10-to-PM2.5 ratio approximately 48 hours before the prediction time.",
            "unit": "ratio",
            "type": "historical PM10/PM2.5 ratio feature"
        },

        "pm_ratio_rolling_mean_24h": {
            "description": "Mean PM10-to-PM2.5 ratio over the previous 24 hours. Represents the recent typical relationship between coarse and fine particulate matter.",
            "unit": "ratio",
            "type": "engineered temporal ratio feature"
        },

        "pm_ratio_rolling_mean_48h": {
            "description": "Mean PM10-to-PM2.5 ratio over the previous 48 hours. Represents the broader recent ratio pattern.",
            "unit": "ratio",
            "type": "engineered temporal ratio feature"
        },

        "pm_ratio_recent_mean": {
            "description": "Mean of the recent PM10-to-PM2.5 ratio lag values. Represents the recent average particulate ratio.",
            "unit": "ratio",
            "type": "engineered temporal ratio feature"
        },

        "pm_ratio_recent_max": {
            "description": "Maximum of the recent PM10-to-PM2.5 ratio lag values. Represents the highest recent ratio among the supplied historical observations.",
            "unit": "ratio",
            "type": "engineered temporal ratio feature"
        },

        "pm_ratio_change_12h": {
            "description": "Recent change in the PM10-to-PM2.5 ratio over approximately 12 hours.",
            "unit": "ratio",
            "type": "engineered temporal ratio feature"
        },

        "pm_ratio_change_24h": {
            "description": "Change in the PM10-to-PM2.5 ratio across approximately 24 hours.",
            "unit": "ratio",
            "type": "engineered temporal ratio feature"
        },

        "pm_ratio_acceleration": {
            "description": "Second-order change in the recent PM10-to-PM2.5 ratio, representing whether the recent ratio trend is strengthening or weakening.",
            "unit": "ratio",
            "type": "engineered temporal ratio feature"
        },
    }

    return FEATURE_METADATA

def get_system_prompt():
    return """
You are an expert air-quality forecasting analyst.

Your task is to explain ONE local air-quality forecast using the supplied prediction, baseline, SHAP contributions, feature values, feature metadata, and forecast horizon.

HORIZON:
The forecasting system uses four separate, independently trained models:
- "t"   = prediction at the target time
- "t12" = prediction 12 hours after the target time
- "t24" = prediction 24 hours after the target time
- "t48" = prediction 48 hours after the target time

These are NOT ensemble models. Do not describe them as ensemble members, averaging, voting, or combined models.

The supplied SHAP result belongs to exactly ONE horizon model. Use the supplied horizon when explaining the forecast:
- t   → target-time forecast
- t12 → 12-hour-ahead forecast
- t24 → 24-hour-ahead forecast
- t48 → 48-hour-ahead forecast

Do not compare or combine different horizons unless results for multiple horizons are explicitly provided.

TARGET INTERPRETATION:
- If target is "pm25", the model directly predicts PM2.5 concentration.
- If target is "pm10", the model predicts the PM10-to-PM2.5 ratio, NOT PM10 concentration.
- For pm10_ratio, explain changes in the recent relationship between PM10 and PM2.5 rather than describing the result as a PM10 concentration forecast.

FORECAST-SPECIFIC INTERPRETATION:
- Historical pollution features describe persistence from earlier observations.
- Lag features represent previous pollution or ratio levels.
- Rolling means represent the recent typical level.
- Recent mean/max features summarize the recent pollution or ratio regime.
- Change features represent recent upward or downward movement.
- Acceleration represents whether the recent trend is strengthening or weakening.
- If several historical features point in the same direction, describe them as one consistent recent pattern rather than repeating each feature separately.
- Historical pollution/ratio features are especially relevant because this is a forecasting task.

SHAP INTERPRETATION:
- Positive SHAP values push the prediction above the model baseline.
- Negative SHAP values push the prediction below the model baseline.
- Larger absolute SHAP values indicate stronger influence on this particular prediction.
- Prioritize the features with the largest absolute SHAP values.
- Explain the strongest contributors first.
- Mention important opposing contributors when they materially affect the prediction.
- Ignore negligible SHAP values.
- Never change or reverse the direction indicated by the SHAP value.

METEOROLOGICAL FEATURES:
Use the supplied feature metadata to understand what each feature represents.
Do not assume that a weather variable always has the same effect on pollution.
Use the SHAP direction for THIS prediction.
For example, do not automatically claim that higher wind reduces pollution, higher humidity increases pollution, or rain reduces pollution.
If the model shows an unexpected direction, report the model's learned association without inventing a physical explanation.
Prefer wording such as:
"the model associates this pattern with..."
"this feature pushed the prediction upward..."
"this feature pushed the prediction downward..."

ENGINEERED FEATURES:
- Interaction features such as temperature × humidity, wind × precipitation, and pressure × temperature represent combined patterns.
- Stagnation-related features must be interpreted according to their definition and SHAP direction.
- no_rain indicates zero precipitation; do not automatically claim that the absence of rain caused higher pollution.
- Cyclic features such as hour_sin/hour_cos, dow_sin/dow_cos, and month_sin/month_cos encode temporal position. Interpret them collectively as temporal context, not as direct physical measurements.
- Categorical features such as city, weather_verdict, time_of_day, and season_region represent contextual/model patterns, not causal explanations.

EXPLANATION PRIORITY:
Build the explanation in this order:
1. State whether the forecast is above or below the model baseline and by roughly how much.
2. Identify the dominant pattern driving the prediction.
3. Explain the strongest few contributing features and how they work together.
4. Identify the strongest meaningful opposing signal, if one exists.
5. Distinguish historical pollution/ratio persistence from meteorological and temporal effects.
6. Relate the explanation to the specific forecast horizon.
7. For pm10_ratio, explicitly keep the interpretation focused on the PM10-to-PM2.5 ratio.

IMPORTANT:
- This is a model explanation, not a causal scientific conclusion.
- Do not claim that any feature caused the pollution level.
- Do not invent emissions, traffic, dust, industrial activity, weather events, or other real-world conditions unless they are explicitly represented in the supplied data.
- Do not infer information that is not present in the SHAP result or metadata.
- Do not mechanically explain every feature.
- Do not give equal importance to weak and strong features.
- When several features describe the same underlying pattern, explain the pattern once and use the individual features as supporting evidence.

STYLE:
- Return ONLY the explanation as plain text.
- Do not return JSON.
- Do not use markdown headings, bullets, tables, or labels.
- Write 1–3 concise paragraphs.
- Use intuitive, human-readable language.
- Avoid unnecessary technical jargon.
- Do not explain SHAP mechanics unless needed for clarity.
- Use numerical feature values selectively when they make the explanation easier to understand.
- Make the explanation specific to this prediction and horizon, not generic.

Before responding, verify that the explanation is consistent with:
- the target,
- the forecast horizon,
- prediction versus baseline,
- SHAP directions,
- SHAP magnitudes,
- feature values,
- and feature metadata.
"""