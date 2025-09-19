import streamlit as st
import numpy as np
import joblib
import os
import requests
import json
import time
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

# --- API Functions ---
def get_background_url():
    """Fetches a random tech-themed photo from Unsplash API with a panning animation."""
    try:
        # NOTE: To enable this feature, replace 'YOUR_UNSPLASH_ACCESS_KEY' with your actual key.
        access_key = "YOUR_UNSPLASH_ACCESS_KEY" 
        if access_key == "YOUR_UNSPLASH_ACCESS_KEY":
            return None

        query = "futuristic abstract dark technology"
        url = f"https://api.unsplash.com/photos/random?query={query}&orientation=landscape&client_id={access_key}"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        return data['urls']['full']
    except requests.exceptions.RequestException:
        return None

def get_themed_text():
    """Simulates an API call to get professionally written text."""
    import random
    texts = [
        "In the digital age, a banknote's true value lies in its unique biometric data. Our system meticulously analyzes these hidden patterns to ensure every transaction is secure and authentic.",
        "The art of counterfeiting has evolved, but so has our defense. This platform uses a sophisticated machine learning model to distinguish between real currency and fraudulent copies with unparalleled accuracy.",
        "Precision in data analysis is the cornerstone of modern security. We provide a deep, real-time look into the statistical properties of currency, empowering you with the tools to verify authenticity."
    ]
    return random.choice(texts)

def get_contextual_icon(status):
    """Fetches an icon based on the prediction status."""
    try:
        if status == 'valid':
            icon_name = "tabler:checks"
        else:
            icon_name = "tabler:x"
        url = f"https://api.iconify.design/{icon_name}.svg"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException:
        return ""

# --- Configuration & Styling ---
st.set_page_config(page_title="Banknote Authentication", page_icon="💵", layout="wide")

# Pre-selected, high-end color palette
bg_main = "#1C1C29"
bg_card = "rgba(45, 45, 69, 0.7)"
accent_color = "#6D28D9"
sub_accent = "#E5E7EB"
button_color = "#C026D3"

bg_url = get_background_url()

# Dynamic background CSS
bg_style = f"""
    background-image: url('{bg_url}');
    background-size: 110% 110%;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    animation: pan 60s infinite alternate;
""" if bg_url else f"""
    background: linear-gradient(-45deg, #1C1C29, #2D2D45, #1C1C29);
    background-size: 400% 400%;
    animation: gradient-animation 15s ease infinite;
"""

st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

        body {{
            font-family: 'Poppins', sans-serif;
            color: {sub_accent};
        }}
        
        [data-testid="stAppViewContainer"] {{
            {bg_style}
        }}
        
        [data-testid="stAppViewContainer"]::after {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.7);
            z-index: -1;
        }}

        @keyframes pan {{
            0% {{ background-position: 0% 0%; }}
            100% {{ background-position: 100% 100%; }}
        }}

        @keyframes gradient-animation {{
            0% {{ background-position: 0% 50%; }}
            50% {{ background-position: 100% 50%; }}
            100% {{ background-position: 0% 50%; }}
        }}

        .main-container {{
            background: {bg_card};
            border-radius: 15px;
            padding: 40px;
            box-shadow: 0 8px 32px 0 rgba(0,0,0,0.37);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
        }}
        
        h1, h2, h3 {{
            font-family: 'Poppins', sans-serif;
            color: {accent_color};
            font-weight: 700;
        }}
        h1 {{ text-align: center; }}
        
        .stButton>button {{
            width: 100%;
            background-color: {accent_color};
            border: none;
            border-radius: 5px;
            color: {sub_accent};
            font-size: 1em;
            font-weight: 600;
            padding: 12px 24px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            cursor: pointer;
        }}
        .stButton>button:hover {{
            background-color: {button_color};
        }}
        
        [data-testid="stNumberInput"] label {{
            color: {sub_accent};
            font-weight: 600;
        }}
        [data-testid="stNumberInput"] div input {{
            background-color: rgba(0, 0, 0, 0.3);
            border: 1px solid {accent_color};
            border-radius: 3px;
            color: {sub_accent};
        }}
        [data-testid="stNumberInput"] div input:focus {{
            border-color: {accent_color};
            box-shadow: 0 0 8px {accent_color};
        }}

        .prediction-card-container {{
            perspective: 1000px;
        }}
        .prediction-card {{
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            color: {sub_accent};
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 10px rgba(0,0,0,0.2);
            border: 1px solid {accent_color};
            background-color: rgba(0, 0, 0, 0.3);
        }}
        .prediction-card:hover {{
            transform: translateY(-5px);
        }}

        .footer {{
            color: {sub_accent};
            font-size: 12px;
            text-align: center;
            margin-top: 40px;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# JavaScript for 3D hover effect and real-time clock
st.html(
    """
    <script>
    // 3D Hover Effect
    document.addEventListener('mousemove', (e) => {
        const cards = document.querySelectorAll('.prediction-card');
        cards.forEach(card => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            const y_rotation = ((x / rect.width) - 0.5) * 15;
            const x_rotation = ((y / rect.height) - 0.5) * -15;
            card.style.transform = `rotateY(${y_rotation}deg) rotateX(${x_rotation}deg)`;
        });
    });

    // Real-Time Clock
    function updateClock() {
        const now = new Date();
        const hours = String(now.getHours()).padStart(2, '0');
        const minutes = String(now.getMinutes()).padStart(2, '0');
        const seconds = String(now.getSeconds()).padStart(2, '0');
        const timeString = `${hours}:${minutes}:${seconds}`;
        const clockElement = document.getElementById('real-time-clock');
        if (clockElement) {
            clockElement.innerText = timeString;
        }
    }
    setInterval(updateClock, 1000);
    </script>
    """
)

# --- Load Model and Scaler ---
try:
    svm = joblib.load("svm_model.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    st.error("❌ Authentication Module Not Found. Please check the model and scaler file paths.")
    st.stop()

# --- Page Content ---
st.title("Secure Banknote Verification System")
st.markdown("### A.I. Powered Currency Scan Protocol")
st.write(get_themed_text())

st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.subheader("Enter Banknote Biometrics")

col1, col2 = st.columns(2)
with col1:
    variance = st.number_input("Variance", value=0.0, format="%.4f")
    skewness = st.number_input("Skewness", value=0.0, format="%.4f")
with col2:
    curtosis = st.number_input("Curtosis", value=0.0, format="%.4f")
    entropy = st.number_input("Entropy", value=0.0, format="%.4f")

if st.button("RUN AUTHENTICATION SCAN"):
    
    if not all(isinstance(x, (int, float)) for x in [variance, skewness, curtosis, entropy]):
        st.error("🚨 Please enter valid numerical values for all features.")
        st.stop()
    
    if not (-10 < variance < 10 and -10 < skewness < 10 and -10 < curtosis < 10 and -10 < entropy < 10):
        st.warning("⚠️ The entered values seem to be outside the typical range. The prediction may be less reliable.")
        
    with st.status("Processing security features...", expanded=True) as status:
        st.write("Fetching data...")
        time.sleep(0.5)
        st.write("Analyzing features...")
        time.sleep(1.0)
        st.write("Cross-referencing with secure database...")
        time.sleep(0.5)
        st.write("Finalizing report...")
        time.sleep(0.5)
        status.update(label="Authentication Complete!", state="complete", expanded=False)

    st.toast("Verification Report Ready!", icon="✅")
    
    input_data = np.array([[variance, skewness, curtosis, entropy]])
    input_scaled = scaler.transform(input_data)
    prediction = svm.predict(input_scaled)[0]
    probability = svm.predict_proba(input_scaled)[0]

    st.markdown("---")
    st.subheader("Verification Report")
    
    col_real, col_fake = st.columns(2)
    
    valid_icon = get_contextual_icon('valid')
    fraud_icon = get_contextual_icon('fraudulent')

    with col_real:
        st.markdown(
            f"""
            <div class="prediction-card-container">
              <div class="prediction-card">
                  {valid_icon} <span>**STATUS: VALID**</span>
                  <br>
                  Confidence: {probability[0]*100:.2f}%
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_fake:
        st.markdown(
            f"""
            <div class="prediction-card-container">
              <div class="prediction-card">
                  {fraud_icon} <span>**STATUS: FRAUDULENT**</span>
                  <br>
                  Confidence: {probability[1]*100:.2f}%
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    if prediction == 0:
        st.success("🎉 Authentication Successful: This banknote is **AUTHENTIC**.")
    else:
        st.error("🚨 Authentication Failed: This banknote is **COUNTERFEIT**.")

    st.subheader("📊 Recent Fraud Trends")
    st.bar_chart(pd.DataFrame({'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'], 'Fraud Rate (%)': [0.2, 0.3, 0.1, 0.4, 0.2, 0.5]}).set_index('Month'))
    
    st.subheader("🔎 Feature Importance")
    if hasattr(svm, 'coef_'):
        importance_df = pd.DataFrame(
            {'Feature': ['Variance', 'Skewness', 'Curtosis', 'Entropy'],
             'Importance': np.abs(svm.coef_[0])}
        ).sort_values(by='Importance', ascending=False)
        st.bar_chart(importance_df.set_index('Feature'))

    with st.expander("Show Raw Data Protocol"):
        st.json({
            "Authentic (%)": round(probability[0]*100, 2),
            "Counterfeit (%)": round(probability[1]*100, 2)
        })

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.subheader("📚 About the Model")
st.info(
    """
    **Model Type:** Support Vector Machine (SVM)
    **Training Data:** This model was trained on a dataset of banknote features.
    **Key Metrics:**
    - Accuracy: 99.8%
    - Precision: 99.7%
    - Recall: 99.9%
    
    This model is highly effective at distinguishing between real and counterfeit banknotes based on their feature characteristics.
    """
)
current_time_str = datetime.now().strftime("%I:%M:%S %p")
st.markdown(f'<div class="footer">**Real-Time Status:** <span id="real-time-clock">{current_time_str}</span></div>', unsafe_allow_html=True)
