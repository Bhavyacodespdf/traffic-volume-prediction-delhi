import streamlit as st
import pandas as pd
import joblib
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Traffic Speed Prediction – Delhi",
    page_icon="🚦",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    padding-top: 2rem;
}
h1 {
    font-weight: 700;
}
.card {
    background-color: #111827;
    padding: 1.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
}
.metric {
    font-size: 32px;
    font-weight: 700;
    color: #22c55e;
}
.sub {
    color: #9ca3af;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🚦 Traffic Speed Prediction – Delhi")
st.caption("Predict traffic speed using **time, road type, and weather conditions**")

# ---------------- ROAD TYPES ----------------
HIGHWAY_MAP = {
    "Motorway / Expressway": 0,
    "Primary Road": 1,
    "Secondary Road": 2,
    "Tertiary Road": 3,
    "Residential Street": 4,
    "Service Road": 5
}

# ---------------- INPUT SECTIONS ----------------
st.markdown("### 🕒 Time Information")
col1, col2 = st.columns(2)
with col1:
    hour = st.slider("Hour of Day", 0, 23, 9)
with col2:
    day = st.selectbox(
        "Day of Week",
        options=[0,1,2,3,4,5,6],
        format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x]
    )

is_rush_hour = 1 if hour in [8,9,10,17,18,19] else 0
is_weekend = 1 if day in [5,6] else 0

# ---------------- ROAD INFO ----------------
st.markdown("### 🛣 Road Information")
col3, col4 = st.columns(2)
with col3:
    road_type = st.selectbox("Road Category", list(HIGHWAY_MAP.keys()))
with col4:
    lanes = st.number_input("Number of Lanes", 1, 6, 2)

highway = HIGHWAY_MAP[road_type]
maxspeed = st.number_input("Speed Limit (km/h)", 20, 120, 50)

# ---------------- WEATHER ----------------
st.markdown("### 🌦 Weather Conditions")
col5, col6 = st.columns(2)
with col5:
    temp = st.number_input("Temperature (°C)", 0.0, 50.0, 30.0)
    humidity = st.number_input("Humidity (%)", 0, 100, 60)
with col6:
    pressure = st.number_input("Pressure (hPa)", 950, 1050, 1012)
    wind_speed = st.number_input("Wind Speed (m/s)", 0.0, 20.0, 3.0)

confidence = st.slider("Traffic Data Confidence", 0.0, 1.0, 1.0)

# ---------------- MODEL ----------------
FEATURES = [
    'confidence','lanes','maxspeed','temp','humidity','pressure',
    'wind_speed','hour','day','highway','is_rush_hour','is_weekend'
]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "final_speed_model.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# ---------------- INPUT DF ----------------
input_df = pd.DataFrame([[
    confidence, lanes, maxspeed, temp, humidity,
    pressure, wind_speed, hour, day,
    highway, is_rush_hour, is_weekend
]], columns=FEATURES)

# ---------------- PREDICTION ----------------
st.markdown("### 🚗 Prediction")
if st.button("Predict Traffic Speed", use_container_width=True):
    speed = model.predict(input_df)[0]

    st.markdown(f"""
    <div class="card">
        <div class="metric">{speed:.2f} km/h</div>
        <div class="sub">Predicted Traffic Speed</div>
    </div>
    """, unsafe_allow_html=True)

    if speed < 20:
        st.error("🔴 Heavy Congestion")
    elif speed < 35:
        st.warning("🟡 Moderate Traffic")
    else:
        st.success("🟢 Free Flow Traffic")
