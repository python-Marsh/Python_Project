import pandas as pd
import plotly.graph_objects as go


def plot_cumulative_returns(start_month, end_month, asset_columns, df, style="default"):
    """
    Parameters
    ----------
    start_month : str | datetime-like
        Inclusive start of the plotting window, e.g. "2018-01".
    end_month : str | datetime-like
        Inclusive end of the plotting window, e.g. "2018-12".
    asset_columns : list[str]
        Column names in ``df`` to plot.
    df : pandas.DataFrame
        Time series data at monthly (or higher) frequency with a datetime index.
    style : str, optional
        Visual style. Include ``"dark"`` or ``"light"`` for alternative colour
        palettes. If ``"box"`` is present (e.g. ``"dark-box"``), a labelled
        box showing the final value of each trace is added.

    Returns
    -------
    plotly.graph_objects.Figure
    """

    # --------------------------- config ---------------------------
    lead_name = "RDGFF" if "RDGFF" in asset_columns else (asset_columns[0] if asset_columns else None)
    palette = {
        "RDGFF": "#2F2F2F",
        "MSCI CHINA": "#D8C3A5",
        "MSCI WORLD": "#B8AEA0",
    }

    add_final_box = "box" in style
    if "dark" in style:
        palette = {"RDGFF": "#B58B80", "MSCI CHINA": "#DACEBF", "MSCI WORLD": "#C1AE94"}
    elif "light" in style:
        palette = {"RDGFF": "#E0E0E0", "MSCI CHINA": "#A67C52", "MSCI WORLD": "#6A5ACD"}

    default_color = "#888888"
    title_text = "Cumulative Return Comparison"

    if start_month and end_month:
        try:
            yr = pd.to_datetime(end_month).year
            title_text = f"{yr} Market Correction"
        except Exception:
            pass

    # --------------------------- data wrangling ---------------------------
    df_local = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_local.index):
        for cand in ("date", "Date", "Month", "month"):
            if cand in df_local.columns:
                df_local[cand] = pd.to_datetime(df_local[cand])
                df_local = df_local.set_index(cand)
                break
        if not pd.api.types.is_datetime64_any_dtype(df_local.index):
            raise ValueError("`df` must have a datetime index or a 'date'/'Date'/'Month' column.")

    df_local = df_local.sort_index()
    start_dt = pd.to_datetime(start_month)
    end_dt = pd.to_datetime(end_month)

    t_minus_1 = start_dt - pd.DateOffset(months=1)
    t_plus_1 = end_dt + pd.DateOffset(months=1)
    df_win = df_local.loc[(df_local.index >= t_minus_1) & (df_local.index <= t_plus_1), asset_columns]
    df_win[:1] = 0.0

    if df_win.empty:
        raise ValueError("No data in the requested window. Check dates and asset columns.")

    months = df_win.index

    # helper
    def fmt_pct(x: float) -> str:
        return f"{x:+.2%}"

    # --------------------------- build figure ---------------------------
    fig = go.Figure()

    for asset in asset_columns:
        if asset not in df_win.columns:
            continue
        monthly_ret = df_win[asset].apply(lambda x: float(str(x)) if pd.notnull(x) else 0.0)
        cumulative_ret = (1 + monthly_ret).cumprod()
        cumulative_ret = cumulative_ret / cumulative_ret.iloc[0] - 1
        color = palette.get(asset, default_color)
        is_lead = (asset == lead_name)

        fig.add_trace(
            go.Scatter(
                x=months,
                y=cumulative_ret,
                name=asset,
                mode="lines+markers",
                line=dict(width=3 if is_lead else 2, color=color, shape="spline", smoothing=0.6),
                marker=dict(size=8 if is_lead else 6, line=dict(width=1, color="white")),
                hovertemplate=f"{asset}<br>%{{x|%b %Y}}<br>%{{y:.2%}}<extra></extra>",
            )
        )

        final_value = cumulative_ret.iloc[-1]
        fig.add_trace(
            go.Scatter(
                x=[months[-1]],
                y=[final_value],
                mode="markers",
                marker=dict(size=12 if is_lead else 9, color=color, line=dict(width=2, color="white")),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        if add_final_box:
            fig.add_annotation(
                x=months[-1],
                y=final_value,
                text=fmt_pct(final_value),
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                xshift=8,
                font=dict(color=color),
                bgcolor="white",
                bordercolor=color,
                borderwidth=1,
                borderpad=4,
            )

    # --------------------------- layout ---------------------------
    fig.update_layout(
        title=dict(text=f"<b>{title_text}</b>", font=dict(size=32), x=0.5, xanchor="center", y=0.95),
        template="plotly_white",
        font=dict(family="Roboto, MontSerrat Semibold, sans-serif", size=14, color="#2f2f2f"),
        margin=dict(l=70, r=20, t=70, b=110),
        xaxis=dict(
            showgrid=True,
            gridcolor="#E9E9E9",
            tickformat="%b %Y",
            tickvals=list(months)[::max(1, len(months)//5)],
            ticktext=[dt.strftime("%b %Y") for dt in list(months)[::max(1, len(months)//8)]],
        ),
        yaxis=dict(title="Cumulative Return (%)", tickformat=".0%", zeroline=True),
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="left", x=0.0),
        hovermode="x unified",
    )

    # summary annotation
    finals = {a: float(df_win[a].iloc[-1]) for a in asset_columns if a in df_win.columns}
    summary_lines = []
    if lead_name in finals:
        summary_lines.append(f"<b>{lead_name}: {fmt_pct(finals[lead_name])}</b>")
    others = [a for a in asset_columns if a in finals and a != lead_name]
    if others:
        summary_lines.append(" &nbsp; &nbsp; ".join([f"{a}: <b>{fmt_pct(finals[a])}</b>" for a in others]))

    if summary_lines:
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.0, y=-0.40,
            showarrow=False,
            align="left",
            text="<br>".join(summary_lines)
        )

    fig.show()
    return fig
