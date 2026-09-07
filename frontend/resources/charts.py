import plotly.graph_objects as go
def create_dual_axis_chart(df1, df2, x, y1, y2, title, y1_label, y2_label, y2_suffix=""):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df1[x],
        y=df1[y1],
        name=y1_label,
        text=df1[y1].round(1),
        textposition="inside",
        yaxis="y"
    ))
    fig.add_trace(go.Scatter(
        x=df2[x],
        y=df2[y2],
        name=y2_label,
        mode="lines+markers+text",
        text=df2[y2].round(1).astype(str) + y2_suffix,
        textposition="top center",
        yaxis="y2"
    ))
    fig.update_layout(
        title=title,
        height=450,
        xaxis_title=x.replace("_", " ").title(),
        yaxis=dict(title=y1_label),
        yaxis2=dict(title=y2_label, overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.08),
        margin=dict(l=60, r=70, t=80, b=80),
        hovermode="x unified"
    )
    return fig

def create_line_chart(df, x, y, title, y_label):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[x],
        y=df[y],
        name=y_label,
        mode="lines+markers+text",
        text=df[y].round(1),
        textposition="top center"
    ))
    fig.update_layout(
        title=title,
        height=450,
        xaxis_title=x.replace("_", " ").title(),
        yaxis_title=y_label,
        margin=dict(l=60, r=40, t=80, b=80),
        hovermode="x unified"
    )
    return fig