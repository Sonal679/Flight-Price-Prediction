import streamlit as st
import pandas as pd
import joblib


# Page Configuration

st.set_page_config(
    page_title="Flight Price Prediction",
    page_icon="✈️",
    layout="wide"
)


# Custom CSS
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}

.title {
    text-align:center;
    color:#0E76A8;
    font-size:45px;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:gray;
    font-size:18px;
}

.result{
    background:#d4edda;
    padding:25px;
    border-radius:15px;
    text-align:center;
    font-size:28px;
    color:#155724;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)


# Load Model

model = joblib.load("model/flight_price_model.pkl")


# Heading
st.markdown('<p class="title"> Flight Price Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predict Airline Ticket Prices using Machine Learning</p>', unsafe_allow_html=True)

st.divider()


# Sidebar

st.sidebar.header("Flight Details")

airline = st.sidebar.selectbox(
    "Airline",
    [
        "IndiGo",
        "Air India",
        "Jet Airways",
        "SpiceJet",
        "Vistara",
        "GoAir",
        "Multiple carriers",
        "Multiple carriers Premium economy",
        "Jet Airways Business",
        "Trujet",
        "Vistara Premium economy"
    ]
)

source = st.sidebar.selectbox(
    "Source",
    ["Delhi","Kolkata","Mumbai","Chennai"]
)

destination = st.sidebar.selectbox(
    "Destination",
    ["Cochin","Delhi","New Delhi","Hyderabad","Kolkata"]
)

total_stops = st.sidebar.slider("Total Stops",0,4,1)

journey_day = st.sidebar.slider("Journey Day",1,31,15)

journey_month = st.sidebar.slider("Journey Month",1,12,7)

dep_hour = st.sidebar.slider("Departure Hour",0,23,10)

dep_min = st.sidebar.slider("Departure Minute",0,59,30)

arrival_hour = st.sidebar.slider("Arrival Hour",0,23,13)

arrival_min = st.sidebar.slider("Arrival Minute",0,59,45)

duration_hours = st.sidebar.slider("Duration Hours",0,20,3)

duration_mins = st.sidebar.slider("Duration Minutes",0,59,15)


# Prediction Button

if st.button("Predict Flight Price 🚀"):

    input_df = pd.DataFrame([{
        "Airline": airline,
        "Source": source,
        "Destination": destination,
        "Total_Stops": total_stops,
        "Journey_Day": journey_day,
        "Journey_Month": journey_month,
        "Dep_Hour": dep_hour,
        "Dep_Min": dep_min,
        "Arrival_Hour": arrival_hour,
        "Arrival_Min": arrival_min,
        "Duration_Hours": duration_hours,
        "Duration_Mins": duration_mins
    }])

    prediction = model.predict(input_df)[0]

    st.markdown(
        f'<div class="result">💰 Predicted Flight Price<br><br>₹ {prediction:,.2f}</div>',
        unsafe_allow_html=True
    )

st.divider()

st.caption("Developed using Python • Scikit-learn • FastAPI • Streamlit")