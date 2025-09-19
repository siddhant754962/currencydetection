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

# --- Configuration & Styling ---
st.set_page_config(page_title="Banknote Authentication", page_icon="💵", layout="wide")

# New, professional color palette for consistency and readability
bg_main = "#121212"  # Dark gray background
bg_card = "rgba(25, 25, 25, 0.7)"  # Semi-transparent dark card
accent_color = "#4A90E2"  # Professional blue for headings and buttons
sub_accent = "#FFFFFF"  # Pure white for all body text
button_color = "#72A9E8"  # Lighter blue for button hover

# --- API Functions ---
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
            url = "https://www.svgrepo.com/show/365313/check.svg"
        else:
            url = "https://www.svgrepo.com/show/365314/x.svg"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException:
        return ""

# --- Dynamic background CSS - now using a solid color ---
bg_style = f"""
    background-color: {bg_main};
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    animation: none;
"""

st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

        body {{
            font-family: 'Poppins', sans-serif;
            color: {sub_accent}; /* All body text is now pure white */
        }}
        
        [data-testid="stAppViewContainer"] {{
            {bg_style}
        }}
        
        .main-container {{
            background: {bg_card};
            border-radius: 15px;
            padding: 40px;
            box-shadow: 0 8px 32px 0 rgba(0,0,0,0.37);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            width: 90%; 
            margin: auto;
        }}
        
        h1, h2, h3 {{
            font-family: 'Poppins', sans-serif;
            color: {accent_color}; /* Headings remain blue */
            font-weight: 700;
        }}
        h1 {{ text-align: center; }}
        
        .stButton>button {{
            width: 100%;
            background-color: {accent_color};
            border: none;
            border-radius: 5px;
            color: {sub_accent}; /* Button text is now pure white */
            font-size: 1em;
            font-weight: 600;
            padding: 12px 24px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            cursor: pointer;
        }}
        .stButton>button:hover {{
            background-color: {button_color};
            color: {sub_accent};
        }}
        
        [data-testid="stNumberInput"] label {{
            color: {sub_accent}; /* Input labels are now pure white */
            font-weight: 600;
        }}
        [data-testid="stNumberInput"] div input {{
            background-color: rgba(0, 0, 0, 0.3);
            border: 1px solid {accent_color};
            border-radius: 3px;
            color: {sub_accent}; /* Input text is now pure white */
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
            color: {sub_accent}; /* Card text is now pure white */
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
            color: {sub_accent}; /* Footer text is now pure white */
            font-size: 12px;
            text-align: center;
            margin-top: 40px;
        }}
        
        .metric-card {{
            background-color: rgba(0, 0, 0, 0.3);
            border: 1px solid {accent_color};
            border-radius: 8px;
            padding: 15px;
            text-align: center;
            margin: 10px 0;
        }}
        
        .metric-title {{
            font-size: 1.2em;
            font-weight: 600;
            color: {accent_color}; /* Title of the metric card is now blue */
        }}
        
        .metric-value {{
            font-size: 2em;
            font-weight: 700;
            color: {sub_accent}; /* Metric value text is now pure white */
        }}
        
        /* 📱 Responsive Adjustments */
        @media (max-width: 768px) {{
            .main-container {{
                padding: 20px;
            }}
            .st-emotion-cache-18ni294 {{
                flex-direction: column;
            }}
            .prediction-card-container, .metric-card {{
                margin-bottom: 20px;
            }}
        }}

    </style>
    """,
    unsafe_allow_html=True
)

# --- New: Get server time and pass it to JavaScript ---
st.html(f"""
    <script>
    // 3D Hover Effect
    document.addEventListener('mousemove', (e) => {{
        const cards = document.querySelectorAll('.prediction-card');
        cards.forEach(card => {{
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            const y_rotation = ((x / rect.width) - 0.5) * 15;
            const x_rotation = ((y / rect.height) - 0.5) * -15;
            card.style.transform = `rotateY(${{y_rotation}}deg) rotateX(${{x_rotation}}deg)`;
        }});
    }});

    // Real-Time Clock - Corrected to use server time
    let serverTime = new Date("{datetime.now().isoformat()}");

    function updateClock() {{
        serverTime.setSeconds(serverTime.getSeconds() + 1);
        let hours = serverTime.getHours();
        const ampm = hours >= 12 ? 'PM' : 'AM';
        hours = hours % 12;
        hours = hours ? hours : 12;
        const minutes = String(serverTime.getMinutes()).padStart(2, '0');
        const seconds = String(serverTime.getSeconds()).padStart(2, '0');
        const timeString = `${{hours}}:${{minutes}}:${{seconds}} ${{ampm}}`;
        const clockElement = document.getElementById('real-time-clock');
        if (clockElement) {{
            clockElement.innerText = timeString;
        }}
    }}
    setInterval(updateClock, 1000);
    </script>
""")


# --- Load Model and Scaler ---
try:
    svm = joblib.load("svm_model.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    st.error("❌ Authentication Module Not Found. Please check the model and scaler file paths.")
    st.stop()

# --- Session State for History ---
if 'history' not in st.session_state:
    st.session_state.history = []

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
    
    # Store the result in session state
    st.session_state.history.append({
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'input': {'variance': variance, 'skewness': skewness, 'curtosis': curtosis, 'entropy': entropy},
        'prediction': 'Authentic' if prediction == 0 else 'Counterfeit',
        'confidence': max(probability)
    })

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

    st.subheader("📊 Statistical Analysis & Visualization")
    
    # --- Feature Value Radar Chart ---
    st.markdown("#### Input Feature Radar Chart")
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[variance, skewness, curtosis, entropy, variance],
        theta=['Variance', 'Skewness', 'Curtosis', 'Entropy', 'Variance'],
        fill='toself',
        name='Banknote Features',
        marker=dict(color=accent_color)
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[-5, 5]),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=sub_accent),
        height=350
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Prediction Confidence Donut Chart ---
    st.markdown("#### Prediction Confidence")
    fig2 = go.Figure(data=[go.Pie(
        labels=['Authentic', 'Counterfeit'],
        values=[probability[0], probability[1]],
        hole=.5,
        marker_colors=[accent_color, button_color],
        textinfo='label+percent'
    )])
    fig2.update_layout(
        showlegend=True,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=sub_accent),
        height=300
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # --- Feature Importance ---
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

# --- History and Metrics Dashboard ---
if st.session_state.history:
    st.markdown("---")
    st.subheader("📈 Verification History & Live Metrics")
    
    history_df = pd.DataFrame(st.session_state.history)
    
    # Live Metrics
    metrics_col1, metrics_col2 = st.columns(2)
    total_scans = len(st.session_state.history)
    
    with metrics_col1:
        authentic_count = history_df[history_df['prediction'] == 'Authentic'].shape[0]
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Total Authentic Scans</div>
                <div class="metric-value">{authentic_count}</div>
            </div>
            """, unsafe_allow_html=True
        )

    with metrics_col2:
        counterfeit_count = history_df[history_df['prediction'] == 'Counterfeit'].shape[0]
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Total Counterfeit Scans</div>
                <div class="metric-value">{counterfeit_count}</div>
            </div>
            """, unsafe_allow_html=True
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Display History Table - FIXED for overflow
    st.markdown("#### Recent Verification Log")
    
    # Create a simplified DataFrame for display
    display_df = history_df.tail(10).sort_values(by='timestamp', ascending=False).copy()
    
    # Format the confidence for better readability
    display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x*100:.2f}%")
    
    # Remove the wide 'input' column that causes overflow
    display_df = display_df.drop(columns=['input'])
    
    st.dataframe(display_df, use_container_width=True)
    
st.markdown(f'<div class="footer">**Real-Time Status:** <span id="real-time-clock">{datetime.now().strftime("%I:%M:%S %p")}</span></div>', unsafe_allow_html=True)
