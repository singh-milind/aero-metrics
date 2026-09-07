import streamlit as st
import requests
import pandas as pd
import plotly.express as px

API_BASE_URL = st.secrets["API_BASE_URL"]
GLOBAL_SHAP_ENDPOINT = f"{API_BASE_URL}/api/explainer/predictor/global"

st.title("Global SHAP Analysis")
st.write("Global SHAP analysis shows which features have the greatest overall influence on the model's predictions.")
st.divider()

def get_global_shap(target):
    response = requests.get(
        GLOBAL_SHAP_ENDPOINT,
        params={"target": target},
        timeout=30,
    )
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"API Error {response.status_code}: {detail}")
    result = response.json()
    df = pd.DataFrame(result["data"])
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    return result["target"], df

target = st.selectbox(
    "Select Target",
    ["pm25", "pm10"],
    format_func=lambda x: "PM2.5" if x == "pm25" else "PM10",
)

try:
    model_target, shap_df = get_global_shap(target)
except Exception as e:
    st.error(str(e))
    st.stop()

col1, col2 = st.columns(2)
with col1:
    st.metric("Features Analyzed", len(shap_df))
with col2:
    st.metric("Most Important Feature", shap_df.iloc[0]["feature"].replace("_", " ").title())

st.divider()
st.subheader(f"Global Feature Importance — {target.upper()}")

plot_df = shap_df.copy()
plot_df["feature"] = plot_df["feature"].str.replace("_", " ").str.title()
plot_df = plot_df.sort_values("importance", ascending=True)

fig = px.bar(
    plot_df,
    x="importance",
    y="feature",
    orientation="h",
    labels={
        "importance": "Mean |SHAP Value|",
        "feature": "Feature",
    },
    title=f"Global SHAP Feature Importance — {target.upper()}",
)
fig.update_layout(
    height=max(600, len(plot_df) * 30),
    showlegend=False,
)
st.plotly_chart(fig, use_container_width=True)


st.divider()
st.subheader("Feature Importance Details")

table_df = shap_df.copy()
table_df.insert(0, "Rank", range(1, len(table_df) + 1))
table_df["feature"] = table_df["feature"].str.replace("_", " ").str.title()
table_df["importance"] = table_df["importance"].round(4)

st.dataframe(
    table_df,
    use_container_width=True,
    hide_index=True,
)

st.warning("Better plots and visualizations are coming soon. Stay tuned!")