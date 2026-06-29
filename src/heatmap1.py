import plotly.express as px
def create_heatmap(
        data,
        parameter,
        plane):

    if plane == "XY":

        fig = px.density_heatmap(
            data,
            x="X",
            y="Y",
            z=parameter,
            histfunc="avg",
            color_continuous_scale="Turbo"
        )

        fig.update_layout(
            title=f"{parameter} Heat Map (Top View)"
        )

    elif plane == "XZ":

        fig = px.density_heatmap(
            data,
            x="X",
            y="Z",
            z=parameter,
            histfunc="avg",
            color_continuous_scale="Turbo"
        )

        fig.update_layout(
            title=f"{parameter} Heat Map (Front View)"
        )

    else:

        fig = px.density_heatmap(
            data,
            x="Y",
            y="Z",
            z=parameter,
            histfunc="avg",
            color_continuous_scale="Turbo"
        )

        fig.update_layout(
            title=f"{parameter} Heat Map (Side View)"
        )

    fig.update_layout(
        template="plotly_dark",
        height=700
    )

    return fig
