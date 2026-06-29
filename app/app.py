import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
import numpy as np
import pandas as pd
import joblib as jb
import plotly.express as px
from src.heatmap1 import create_heatmap
from src.visualization import create_3d_map 
predicted_grade=0
@st.cache_resource
def load_model():
    return jb.load("../models/predictor_model_RandomForest.pkl")


model = load_model()
enc=jb.load("../models/Rock encoder.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("../data/processed/processed_data.csv")

data = load_data()
total_revenue = (data["Gross"]*0.6).sum()
total_profit = data["Net_profit_Rupees"].sum()
avg_grade = data["Ore_Grade (%)"].mean()
blocks = len(data)
st.set_page_config(
    page_title="AI Mining Analytics Dashboard",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)



st.markdown("""
<style>

.main{
    background-color:#050505;
}

.block-container{
    padding-top:1rem;
    padding-bottom:1rem;
}

h1{
    color:#0B3D91;
}

div[data-testid="stMetric"]{
    background-color:white;
    border-radius:12px;
    padding:18px;
    border-left:6px solid #1f77b4;
    box-shadow:0px 2px 8px rgba(0,0,0,0.08);
}

section[data-testid="stSidebar"]{
    background-color:#0f172a;
}

st.sidebar.subheader("🤖 AI Model")

model=st.sidebar.selectbox(
    "Select Model",
    [
        "Random Forest",
        "XGBoost"
    ]
)

st.sidebar.subheader("💱 Currency")

currency=st.sidebar.selectbox(
    "Currency",
    [
        "INR",
        "USD",
        "JPY"
    ]
)

st.sidebar.subheader("⛏ Rock Type")

rock=st.sidebar.multiselect(
    "Choose Rock",
    [
        "Hematite",
        "Magnetite",
        "Waste"
    ],
    default=[
        "Hematite",
        "Magnetite",
        "Waste"
    ]
)
section[data-testid="stSidebar"] *{
    color:white;
}

</style>
""",unsafe_allow_html=True)
st.markdown("""
<style>

div[data-testid="stMetric"] * {
    color: black !important;
}

</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
div[data-testid="stMetric"]{
    transition: all 0.3s ease;
}

div[data-testid="stMetric"]:hover{
    transform: translateY(-8px);
    box-shadow: 0 12px 25px rgba(0,0,0,0.35);
}
</style>
""", unsafe_allow_html=True)
st.sidebar.title("⛏️ Dashboard")

page=st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview",
        "🌍 3D Viewer",
        "🔥 Heat Map",
        "📈 Analytics",
        "🤖 Prediction"
    ]
)

st.sidebar.divider()
st.sidebar.subheader("3D Viewer Settings")

color_by = st.sidebar.selectbox(
    "Filter By",
    [
        "Ore_Grade (%)",
        "Net_profit_Rupees",
        "Gross",
        "Rock_Type",
        "Tonnage"
    ]
)

point_size = st.sidebar.slider(
    "Point Size",
    2,
    10,
    3
)

opacity = st.sidebar.slider(
    "Opacity",
    0.2,
    1.0,
    0.8
)


left,right=st.columns([8,2])

with left:

    st.title("⛏️ AI Mining Analytics Dashboard")

    st.caption(
        "AI Powered Decision Support System for Open Pit Mining Operations"
    )

with right:

    st.success("🟢 System Online")

st.divider()





st.sidebar.subheader("📊 Ore Grade")

grade=st.sidebar.slider(
    "Minimum Grade %",
    0,
    70,
    30
)

st.sidebar.subheader("📍 Depth")

depth=st.sidebar.slider(
    "Maximum Depth",
    0,
    100,
    100
)

st.sidebar.subheader("💰 Profit")

profit=st.sidebar.slider(
    "Minimum Profit",
    0,
    1000000,
    0
)
st.sidebar.markdown("---")
st.sidebar.subheader("📍 X Slice")

x_min, x_max = st.sidebar.slider(
    "X Range",
    min_value=int(data["X"].min()),
    max_value=int(data["X"].max()),
    value=(
        int(data["X"].min()),
        int(data["X"].max())
    )
)

st.sidebar.markdown("---")
st.sidebar.subheader("📍 Y Slice")

y_min, y_max = st.sidebar.slider(
    "Y Range",
    min_value=int(data["Y"].min()),
    max_value=int(data["Y"].max()),
    value=(
        int(data["Y"].min()),
        int(data["Y"].max())
    )
)

st.sidebar.markdown("---")
st.sidebar.subheader("📍 Z Slice")

z_min, z_max = st.sidebar.slider(
    "Z Range",
    min_value=int(data["Z"].min()),
    max_value=int(data["Z"].max()),
    value=(
        int(data["Z"].min()),
        int(data["Z"].max())
    )

)
st.sidebar.subheader("⛏ Rock Type")
rock=st.sidebar.multiselect(
    "Choose Rock",
    [
        "Hematite",
        "Magnetite",
        "Waste"
    ],
    default=[
        "Hematite",
        "Magnetite",
        "Waste"
    ]
)





c1,c2,c3,c4=st.columns(4)
col1, col2, col3, col4 = st.columns(4)

col1, col2, col3, col4 = st.columns(4)
c1.metric(
    "💰 Revenue",
    f"₹ {total_revenue/1e7:.2f} Cr"
)

c2.metric(
    "📈 Profit",
    f"₹ {total_profit/1e7:.2f} Cr"
)

c3.metric(
    "⛏ Avg Grade",
    f"{avg_grade:.2f}%"
)

c4.metric(
    "📦 Blocks",
    f"{blocks:,}"
)

tab1,tab2,tab3,tab4,tab5=st.tabs(
[
"🏠 Overview",
"🌍 3D Viewer",
"🔥 Heat Map",
"📈 Analytics",
"🤖 Prediction"
]
)


with tab1:

    st.subheader("📋 Dataset Overview")

    col1,col2=st.columns([2,1])

    with col1:

        st.info(
        """
        Welcome to the AI Mining Analytics Platform.

        This dashboard provides:

        • Ore Grade Prediction

        • 3D Mine Visualization

        • Revenue Analysis

        • Profit Analysis

        • High Value Zone Detection

        • AI Decision Support
        """
        )

    with col2:

        st.metric("Total Records","75,000")

        st.metric("Rock Types","3")

        st.metric("Models","2")

with tab2:

    st.subheader("🌍 Interactive 3D Mine Viewer")
    filtered_data = data.copy()

    filtered_data = filtered_data[
     filtered_data["Rock_Type"].isin(rock)
]


    filtered_data = filtered_data[
     filtered_data["Ore_Grade (%)"] >=grade
]


    filtered_data = filtered_data[
     filtered_data["Z"] <=depth
]


    filtered_data = filtered_data[
     filtered_data["Net_profit_Rupees"] >=profit
]


    filtered_data = filtered_data[
    (filtered_data["X"] >= x_min) &
    (filtered_data["X"] <= x_max)
]


    filtered_data = filtered_data[
    (filtered_data["Y"] >= y_min) &
    (filtered_data["Y"] <= y_max)
]
    filtered_data = filtered_data[
    (filtered_data["Z"] >= z_min) &
    (filtered_data["Z"] <= z_max)
]



    fig = create_3d_map(
        filtered_data,
        color_by=color_by,
        point_size=point_size,
        opacity=opacity
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
with tab3:

    st.subheader("🔥 Ore Body Heat Map")

    col1, col2 = st.columns(2)

    with col1:

        parameter = st.selectbox(
            "Parameter",
            [
                "Ore_Grade (%)",
                "Net_profit_Rupees",
                "Revenue",
                "Tonnage",
                "Mining_Cost_Yen",
                "Processing_Cost_Yen"
            ]
        )

    with col2:

        plane = st.radio(
            "Plane",
            [
                "XY",
                "XZ",
                "YZ"
            ]
        )

    fig = create_heatmap(
        filtered_data,
        parameter,
        plane
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )
    st.divider()

    st.subheader("📊 Statistics")

    c1,c2,c3,c4 = st.columns(4)

    c1.metric("Maximum",f"{filtered_data[parameter].max():,.2f}")

    c2.metric("Minimum",f"{filtered_data[parameter].min():,.2f}")

    c3.metric("Average",f"{filtered_data[parameter].mean():,.2f}")

    c4.metric("Std Dev",f"{filtered_data[parameter].std():,.2f}")
    st.divider()

    st.subheader("🏆 Top Richest Blocks")

    top = filtered_data.sort_values(
    "Net_profit_Rupees",
    ascending=False
).head(20)

    st.dataframe(
    top[
        [
            "Block_ID",
            "Rock_Type",
            "Ore_Grade (%)",
            "Gross",
            "Net_profit_Rupees"
        ]
    ],
    width="stretch"
)

   

with tab4:
    st.subheader("🤖 Model Performance")
    features=["X","Y","Z","Rock_Type","Tonnage","Mining_Cost_Yen","Processing_Cost_Yen"]




    c1, c2, c3 = st.columns(3)

    c1.metric("R² Score", "0.971")
    c2.metric("MAE", "3.13")
    c3.metric("Model", "Random Forest")
    importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

    importance = importance.sort_values(
    "Importance",
    ascending=False
)
    fig = px.bar(
    importance,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Feature Importance"
)
    st.plotly_chart(fig, use_container_width=True)
    corr = data.corr(numeric_only=True)

    fig = px.imshow(
    corr,
    text_auto=True,
    color_continuous_scale="RdBu_r",
    title="Correlation Matrix"
)


    st.plotly_chart(fig, use_container_width=True)
    st.subheader("📈 Dataset Statistics")

    st.dataframe(
    data.describe(),
    use_container_width=True
)
    st.subheader("📋 Feature Ranking")

    st.dataframe(
    importance,
    use_container_width=True
)
with tab5:

    st.subheader("🤖 AI Ore Grade Predictor")

    c1,c2,c3=st.columns(3)

    with c1:
        x=st.number_input("X Coordinate")

        y=st.number_input("Y Coordinate")

        z=st.number_input("Z Coordinate")

    with c2:

        rock=st.selectbox(
            "Rock Type",
            [
                "Hematite",
                "Magnetite",
                "Waste"
            ]
        )

        tonnage=st.number_input("Tonnage")

    with c3:

        mining_cost=st.number_input("Mining Cost")

        processing_cost=st.number_input("Processing Cost")
    rock_encoded=enc.transform([rock])[0]

    if st.button(
    "🚀 Predict Ore Grade",
    use_container_width=True
):
   

     input_data = pd.DataFrame({
        "X": [x],
        "Y": [y],
        "Z": [z],
        "Rock_Type": [rock_encoded],
        "Tonnage": [tonnage],
        "Mining_Cost_Yen": [mining_cost],
        "Processing_Cost_Yen": [processing_cost]
    })

     predicted_grade1 = model.predict(input_data)[0]
     predicted_grade=predicted_grade1
    Recovery = 0.92
    dilution = 0.05
    smelter_charge = 0.03
    royalty = 0.05

    ore_value_per_tonne_yen = 25000      # Use the same value as your notebook

    effective_grade = predicted_grade * (1 - dilution)

    contained = tonnage * (effective_grade / 100)

    recovered = contained * Recovery

    gross = recovered * ore_value_per_tonne_yen

    smelter_fee = gross * smelter_charge

    royalty_fee = gross * royalty

    operating_cost = mining_cost + processing_cost

    net_profit_yen = gross - smelter_fee - royalty_fee - operating_cost

    net_profit_rupees = net_profit_yen * 0.6
    gross_rupees = gross * 0.6   
    c1, c2 = st.columns(2)

    with c1:
     st.metric("Gross Revenue (₹)", f"₹{gross_rupees:,.0f}")
     st.metric("Operating Cost (₹)", f"₹{operating_cost*0.6:,.0f}")
     st.metric("Smelter Fee (₹)", f"₹{smelter_fee*0.6:,.0f}")

    with c2:
     st.metric("Royalty (₹)", f"₹{royalty_fee*0.6:,.0f}")
     st.metric("Net Profit (₹)", f"₹{net_profit_rupees:,.0f}")
     st.metric("Ore Grade (%)",f"{predicted_grade:,.0f}%")
    if net_profit_rupees > 0:
     st.success("🟢 Recommendation: Mine this block")
    else:
     st.error("🔴 Recommendation: Do Not Mine")

st.caption(
"© 2026 AI Powered Open Pit Mining Analytics Dashboard | Built using Streamlit • Plotly • Scikit-Learn • Random Forest • XGBoost"
)