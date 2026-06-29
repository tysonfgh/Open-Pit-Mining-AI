import plotly.express as px

def create_3d_map(
    data,
    color_by="Ore_Grade (%)",
    point_size=3,
    opacity=0.8
):

    fig = px.scatter_3d(
        data,
        x="X",
        y="Y",
        z="Z",
        color=color_by,
        color_continuous_scale="Turbo",
        title="⛏️ Interactive 3D Ore Grade Map",
        opacity=opacity,
        hover_data=[
            "Rock_Type",
            "Tonnage",
            "Mining_Cost_Yen",
            "Processing_Cost_Yen",
            "Net_profit_Rupees"
        ],
        size="Tonnage"
    )

    fig.update_traces(
        marker=dict(size=point_size)
    )

    fig.update_layout(
        template="plotly_dark",
        scene_camera=dict(
            eye=dict(x=1.8, y=1.8, z=1.3)
        ),
        scene=dict(
            xaxis_title="X Coordinate",
            yaxis_title="Y Coordinate",
            zaxis_title="Depth"
        ),
        height=750,
        margin=dict(l=0, r=0, t=50, b=0)
    )

    return fig