import plotly.graph_objects as go
def create_dual_axis_chart(
    df1,
    df2,
    x,
    y1,
    y2,
    title,
    y1_label,
    y2_label,
    y2_suffix="",
    show_text=False,
):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df1[x],
        y=df1[y1],
        name=y1_label,
        text=df1[y1].round(1) if show_text else None,
        textposition="inside" if show_text else None,
        yaxis="y"
    ))
    fig.add_trace(go.Scatter(
        x=df2[x],
        y=df2[y2],
        name=y2_label,
        mode="lines+markers+text" if show_text else "lines+markers",
        text=df2[y2].round(1).astype(str) + y2_suffix if show_text else None,
        textposition="top center" if show_text else None,
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

def create_line_chart(df, x, y, title, y_label, show_text=False):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[x],
        y=df[y],
        name=y_label,
        mode="lines+markers+text" if show_text else "lines+markers",
        text=df[y].round(1) if show_text else None,
        textposition="top center" if show_text else None
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