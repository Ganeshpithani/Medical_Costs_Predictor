import streamlit as st
import pandas as pd
import pickle
import numpy
import joblib
import base64
import os




# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="Medical Costs Predictor.pkl",
    initial_sidebar_state="collapsed"
)

# Session state
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False

# -------------------------------------------------
# CUSTOM STYLE (UI ONLY – FUNCTIONALITY UNCHANGED)
# -------------------------------------------------
def set_custom_style(image_file):
    if not os.path.exists(image_file):
        st.error(f"Background image not found: {image_file}")
        return

    with open(image_file, "rb") as f:
        data = f.read()

    bin_str = base64.b64encode(data).decode()

    style = f"""
    <style>
    .stApp {{
        background: linear-gradient(
            rgba(0, 20, 40, 0.75),
            rgba(0, 20, 40, 0.75)
        ), url("data:image/jpg;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """

    st.markdown(style, unsafe_allow_html=True)

set_custom_style("pexels-karola-g-4386183.jpg")





style = f"""
<style>

/* -----------------------------
   BACKGROUND
------------------------------*/

/* -----------------------------
   RESULT EMPHASIS CARD
------------------------------*/
/* -----------------------------
   RESULT EMPHASIS CARD
------------------------------*/
.result-card {{
    background: linear-gradient(
        rgba(0, 119, 182, 0.25),
        rgba(0, 245, 212, 0.15)
    );
    border: 1px solid rgba(144, 224, 239, 0.6);
    border-radius: 22px;
    padding: 2.5rem;
    margin-top: 30px;
    box-shadow: 0 0 30px rgba(0, 245, 212, 0.25);
    animation: fadeInUp 0.6s ease;
}}

/* Smooth entry animation */
@keyframes fadeInUp {{
    from {{
        opacity: 0;
        transform: translateY(12px);
    }}
    to {{
        opacity: 1;
        transform: translateY(0);
    }}
}}


/* -----------------------------
   LOADING SPINNER
------------------------------*/
@keyframes spin {{
    0% {{ transform: rotate(0deg); }}
    100% {{ transform: rotate(360deg); }}
}}

.button-loading::after {{
    content: "";
    width: 20px;
    height: 20px;
    border: 3px solid rgba(255,255,255,0.4);
    border-top-color: white;
    border-radius: 50%;
    display: inline-block;
    animation: spin 0.8s linear infinite;
    margin-left: 12px;
}}

/* -----------------------------
   MAIN WIDTH
------------------------------*/
.stMainBlockContainer {{
    max-width: 80% !important;
    padding-top: 2rem !important;
}}

/* -----------------------------
   HELPER TEXT
------------------------------*/
.helper-text {{
    font-size: 0.8rem;
    color: #bde0fe;
    margin-top: -4px;      /* reduced */
    margin-bottom: 10px;   /* reduced */
    padding-left: 4px;
    opacity: 0.85;

/* -----------------------------
   GLASS CARD
------------------------------*/
.glass-card {{
    background: rgba(255, 255, 255, 0.10);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border-radius: 28px;
    padding: 2.8rem;
    margin: 2.5rem auto;
    box-shadow: 0 20px 50px rgba(0,0,0,0.6);
    border: 1px solid rgba(255,255,255,0.2);
}}

/* -----------------------------
   TITLES
------------------------------*/
h1 {{
    font-size: 3.2rem;
    font-weight: 800;
    text-align: center;
    color: #ffffff;
}}

h2, h3 {{
    color: #caf0f8;
    text-align: center;
    letter-spacing: 1px;
}}

/* -----------------------------
   LABELS
------------------------------*/
label {{
    color: #e0f7fa !important;
    font-size: 15px !important;
    font-weight: 600;
    margin-bottom: 6px !important;
}}

/* -----------------------------
   INPUTS
------------------------------*/
.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] {{
    background: rgba(0, 0, 0, 0.55) !important;
    color: white !important;
        height: 52px !important;
    padding-top: 2px !important;
    border-radius: 14px !important;
    margin-bottom: 6px !important;
    border: 1px solid rgba(144, 224, 239, 0.8) !important;
    padding: 12px;
    transition: all 0.25s ease-in-out;
}}

.stTextInput input:focus,
.stNumberInput input:focus {{
    border-color: #00e5ff !important;
    box-shadow: 0 0 10px rgba(0,229,255,0.6);
}}

/* -----------------------------
   INPUT ALIGNMENT
------------------------------*/
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] {{
    height: 52px !important;
    font-size: 16px !important;
}}

.stNumberInput,
.stSelectbox {{
    width: 100% !important;
}}

div[data-testid="stVerticalBlock"] > div {{
    margin-bottom: 18px;
}}

/* -----------------------------
   PREDICT BUTTON (MEDIUM)
------------------------------*/
/* Center the button */
div.stButton {{
    display: flex;
    justify-content: center;
    margin-top: 25px;
}}
/* Medium-size Predict button */
div.stButton > button {{
    background: linear-gradient(135deg, #00f5d4, #0077b6);
    color: #001219;
    width: 200px;
    height: 3.0em;
    font-size: 1.1rem;
    font-weight: 700;
    padding: 0 26px;
    border-radius: 10px;
    border: none;
    letter-spacing: 0.8px;
    transition: all 0.25s ease;
    box-shadow: 0 10px 24px rgba(0,0,0,0.5);
}}
/* Hover effect */
div.stButton > button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 14px 32px rgba(0,0,0,0.7);
    background: linear-gradient(135deg, #00bbf9, #03045e);
    color: white;
}}

/* -----------------------------
   RESULT METRIC
------------------------------*/
[data-testid="stMetricValue"] {{
    font-size: 4.5rem;
    font-weight: 800;
    color: #90dbf4;
    text-shadow: 2px 4px 15px rgba(0,0,0,0.9);
}}

[data-testid="stMetricLabel"] {{
    font-size: 1.3rem;
    color: #e0f7fa;
    letter-spacing: 1px;
}}

/* -----------------------------
   VALIDATION BOX
------------------------------*/
.validation-box {{
    background: rgba(255, 99, 71, 0.12);
    border-left: 6px solid #ff6b6b;
    border-radius: 14px;
    padding: 18px 22px;
    margin-top: 20px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.4);
}}

.validation-title {{
    color: #ffb3b3;
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 10px;
}}

.validation-item {{
    color: #ffd6d6;
    font-size: 0.95rem;
    margin-left: 10px;
    line-height: 1.6;
}}

/* Info text (helper, subtitles) */
.info-text {{
    color: #caf0f8;
}}

/* Success / prediction highlight */
.success-text {{
    color: #00f5d4;
    font-weight: 700;
}}

/* Error text already handled by validation-box */


</style>
"""
st.markdown(style, unsafe_allow_html=True)


# ---------------------------------------------------------backend------------------------
# -------------------------------------------------
# MODEL LOADING
# -------------------------------------------------


with open("DecisionTree_model.pkl", "rb") as f:
    model = pickle.load(f)


# -------------------------------------------------
# TITLE
# -------------------------------------------------
st.markdown("""
<div class="glass-card">
    <h1>Medical Insurance Cost Predictor</h1>
    <p style="text-align:center; color:#caf0f8;">
        Predict estimated medical insurance charges using patient details
    </p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# INPUT SECTION
# -------------------------------------------------


col_empty_left, col1, col2, col_empty_right = st.columns([1, 2, 2, 1])

# LEFT COLUMN
with col1:

    age = st.number_input("Age", min_value=18, max_value=65, value=18)
    st.markdown(
        "<div class='helper-text'>Insurance data supports ages between 18 and 65</div>",
        unsafe_allow_html=True
    )

    bmi = st.number_input("BMI", min_value=15.0, max_value=40.0, value=15.0)
    st.markdown(
        "<div class='helper-text'>Higher BMI may increase medical insurance cost</div>",
        unsafe_allow_html=True
    )

    children = st.number_input("Children", min_value=0, max_value=5, value=0)
    st.markdown(
        "<div class='helper-text'>Number of dependents covered under insurance</div>",
        unsafe_allow_html=True
    )


# RIGHT COLUMN
with col2:
    

    sex = st.selectbox("Sex", ["Select", "female", "male"])
    st.markdown(
        "<div class='helper-text'>Medical expenses vary statistically by gender</div>",
        unsafe_allow_html=True
    )

    smoker = st.selectbox("Smoker", ["Select", "yes", "no"])
    st.markdown(
        "<div class='helper-text'>Smoking significantly increases health risks</div>",
        unsafe_allow_html=True
    )

    region = st.selectbox(
        "Region",
        ["Select", "southeast", "southwest", "northeast", "northwest"]
    )
    st.markdown(
        "<div class='helper-text'>Medical costs differ by geographical region</div>",
        unsafe_allow_html=True
    )


st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------
# BUTTON ACTION
# -------------------------------------------------
if st.button("Predict Charges"):
    st.session_state.prediction_made = False

    with st.spinner("Analyzing medical data..."):
        # your existing validation + prediction logic stays SAME
        pass


    # -------------------------------
    # INPUT VALIDATION
    # -------------------------------
    error_messages = []

    if age <= 18:
        error_messages.append("Age must be greater than 18")

    if bmi <= 15:
        error_messages.append("BMI must be greater than 15")

    if children < 0:
        error_messages.append("Children count cannot be negative")

    if sex == "Select":
        error_messages.append("Please select Sex")

    if smoker == "Select":
        error_messages.append("Please select Smoking status")

    if region == "Select":
        error_messages.append("Please select Region")

    # ❌ Stop if errors exist
    if error_messages:
        st.markdown(
            f"""
            <div class="validation-box">
                <div class="validation-title">⚠️ Please fix the following issues</div>
                {''.join([f'<div class="validation-item">• {msg}</div>' for msg in error_messages])}
            </div>
            """,
            unsafe_allow_html=True
        )
        st.stop()

    # -------------------------------
    # PREDICTION
    # -------------------------------
    st.session_state.prediction_made = True

    input_data = pd.DataFrame(
        [[age, sex, bmi, children, smoker, region]],
        columns=['Age', 'Sex', 'BMI', 'Children', 'Smoker', 'Region']
    )

    prediction = model_pipeline.predict(input_data)
    st.session_state.last_prediction = prediction[0] * EXCHANGE_RATE_USD_TO_INR

# -------------------------------------------------
# RESULT SECTION
# -------------------------------------------------
if st.session_state.prediction_made:
   
   st.markdown(
    "<p class='info-text' style='text-align:center;'>Predict estimated medical insurance charges using patient details</p>",
    unsafe_allow_html=True)
   st.metric(
        label="Estimated Medical Cost (INR)",
        value=f"₹{st.session_state.last_prediction:,.2f}"
    )
   st.markdown('</div>', unsafe_allow_html=True)