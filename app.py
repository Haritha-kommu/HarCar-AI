"""
app.py
HarCar AI — Intelligent Used-Car Price Prediction
Tagline: Know Your Car's Worth.

Features:
1. Price Estimator
2. My Valuation
3. Compare Cars
"""

import os
import base64
import joblib
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st

# ==============================================================================
# 1. CORE VALUATION & VALIDATION LOGIC (Reusable & Testable)
# ==============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "used_car_price_model.joblib")
CANDIDATE_MODEL_PATH = os.path.join(BASE_DIR, "models", "candidate_model.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "outputs", "model_metrics.txt")
CINEMATIC_BG_PATH = os.path.join(BASE_DIR, "assets", "cinematic_bg.jpg")
SHOWROOM_BG_PATH = os.path.join(BASE_DIR, "assets", "showroom_bg.jpg")

# Historical dataset boundary constants
DATASET_MIN_PRESENT_PRICE = 0.32   # Lakhs INR
DATASET_MAX_PRESENT_PRICE = 92.60  # Lakhs INR
DATASET_MIN_AGE = 2                # Years
DATASET_MAX_AGE = 17               # Years
DATASET_MAX_KMS = 500000           # Kilometers


def validate_inputs(present_price: float, car_age: int, kms_driven: int, owner: int):
    """
    Validates user inputs against physical constraints and checks advisory bounds.
    Returns: (is_valid: bool, errors: list[str], advisories: list[str])
    """
    errors = []
    advisories = []

    if present_price <= 0:
        errors.append("Original Ex-Showroom Price must be strictly greater than 0 Lakhs INR.")
    if car_age < 0:
        errors.append("Vehicle Age cannot be negative.")
    if kms_driven < 0:
        errors.append("Distance traveled cannot be negative.")
    if owner < 0:
        errors.append("Previous owner count cannot be negative.")

    if present_price > 0 and (present_price < DATASET_MIN_PRESENT_PRICE or present_price > 50.0):
        advisories.append(
            f"Ex-showroom price (₹{present_price:.2f}L) falls outside primary dataset concentration "
            f"(₹{DATASET_MIN_PRESENT_PRICE:.2f}L – ₹50.00L). Outlier estimates may carry higher variance."
        )

    if car_age > DATASET_MAX_AGE:
        advisories.append(
            f"Vehicle age ({car_age} years) exceeds standard reference ceiling ({DATASET_MAX_AGE} years)."
        )

    if kms_driven > 300000:
        advisories.append(
            f"Odometer reading ({kms_driven:,} km) represents a high-mileage vehicle."
        )

    return len(errors) == 0, errors, advisories


def calculate_valuation(predicted_price_lakh: float, original_price_lakh: float) -> dict:
    """
    Computes exact percentage retained value and depreciation using unrounded
    numerical values, and detects when prediction exceeds original cost.
    """
    if original_price_lakh <= 0:
        raise ValueError("Original ex-showroom price must be strictly greater than 0.")
    if predicted_price_lakh < 0:
        raise ValueError("Predicted resale price cannot be negative.")

    retained_pct = (predicted_price_lakh / original_price_lakh) * 100.0
    depreciation_pct = 100.0 - retained_pct
    exceeds_original = predicted_price_lakh > original_price_lakh
    value_diff = predicted_price_lakh - original_price_lakh
    depreciation_amount_lakh = original_price_lakh - predicted_price_lakh
    depreciation_amount_rupees = depreciation_amount_lakh * 100000.0

    return {
        "retained_pct": retained_pct,
        "depreciation_pct": depreciation_pct,
        "exceeds_original": exceeds_original,
        "value_difference_lakh": value_diff,
        "depreciation_amount_lakh": depreciation_amount_lakh,
        "depreciation_amount_rupees": depreciation_amount_rupees,
    }


def format_inr(number: float) -> str:
    """
    Formats a number into standard Indian Rupee notation (e.g. ₹3,62,475).
    """
    val = int(round(number))
    is_negative = val < 0
    val_str = str(abs(val))

    if len(val_str) <= 3:
        formatted = val_str
    else:
        last3 = val_str[-3:]
        remaining = val_str[:-3]
        groups = []
        while len(remaining) > 2:
            groups.insert(0, remaining[-2:])
            remaining = remaining[:-2]
        if remaining:
            groups.insert(0, remaining)
        formatted = ",".join(groups) + "," + last3

    return f"-₹{formatted}" if is_negative else f"₹{formatted}"


@st.cache_resource
def load_trained_pipeline():
    """Loads and caches the serialized scikit-learn Pipeline from disk."""
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def get_base64_image(image_path: str) -> str:
    """Reads a local image file and returns base64 encoded string."""
    if os.path.exists(image_path):
        try:
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode("utf-8")
        except Exception:
            return ""
    return ""


# ==============================================================================
# 2. STREAMLIT APPLICATION CONTROLLER & UI
# ==============================================================================

def main():
    st.set_page_config(
        page_title="HarCar AI — Intelligent Used-Car Price Prediction",
        page_icon="🚘",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Session State Initialization for Valuations History
    if "valuation_history" not in st.session_state:
        st.session_state["valuation_history"] = []

    # Resolve cinematic background image
    bg_base64 = get_base64_image(CINEMATIC_BG_PATH)
    if not bg_base64:
        bg_base64 = get_base64_image(SHOWROOM_BG_PATH)

    bg_css_rule = ""
    if bg_base64:
        bg_css_rule = f"""
            background-image: 
                linear-gradient(180deg, rgba(11, 15, 20, 0.88) 0%, rgba(11, 15, 20, 0.95) 65%, rgba(11, 15, 20, 0.98) 100%),
                radial-gradient(circle at 85% 15%, rgba(255, 107, 0, 0.14) 0%, rgba(11, 15, 20, 0) 55%),
                url("data:image/jpeg;base64,{bg_base64}") !important;
            background-position: center top !important;
            background-size: cover !important;
            background-attachment: fixed !important;
            background-repeat: no-repeat !important;
        """
    else:
        bg_css_rule = """
            background-image: 
                linear-gradient(180deg, #0B0F14 0%, #111620 50%, #0B0F14 100%),
                radial-gradient(circle at 85% 15%, rgba(255, 107, 0, 0.12) 0%, transparent 50%) !important;
        """

    # Comprehensive HarCar AI Styling
    st.markdown(f"""
    <style>
        [data-testid="stSidebar"],
        [data-testid="collapsedControl"],
        #MainMenu,
        header[data-testid="stHeader"] {{
            display: none !important;
        }}

        .stApp {{
            background-color: #0B0F14 !important;
            {bg_css_rule}
            color: #F5F7FA !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
        }}

        .block-container {{
            max-width: 95% !important;
            padding-top: 1.2rem !important;
            padding-bottom: 3.5rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }}

        /* HarCar AI Brand Header */
        .top-navbar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0 16px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            margin-bottom: 20px;
        }}
        .navbar-brand {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .navbar-logo-icon {{
            font-size: 1.85rem;
            filter: drop-shadow(0 2px 10px rgba(255, 107, 0, 0.5));
        }}
        .navbar-title {{
            font-size: 1.85rem;
            font-weight: 800;
            color: #FFFFFF;
            letter-spacing: -0.5px;
            margin: 0;
            line-height: 1;
        }}
        .navbar-title span {{
            color: #FF6B00;
        }}
        .navbar-tagline {{
            font-size: 0.92rem;
            color: #B8C0CC;
            letter-spacing: 0.3px;
        }}
        .navbar-tagline b {{
            color: #FF8A34;
        }}

        /* Navigation Tabs */
        div[data-baseweb="tab-list"] {{
            gap: 8px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
            margin-bottom: 24px;
            background: transparent !important;
            flex-wrap: wrap;
        }}
        button[data-baseweb="tab"] {{
            background: transparent !important;
            color: #94A3B8 !important;
            font-weight: 600 !important;
            font-size: 1.02rem !important;
            padding: 10px 20px !important;
            border-radius: 8px 8px 0 0 !important;
            border: none !important;
            transition: all 0.2s ease-in-out;
        }}
        button[data-baseweb="tab"]:hover {{
            color: #F5F7FA !important;
            background: rgba(255, 255, 255, 0.04) !important;
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{
            color: #FF6B00 !important;
            border-bottom: 2px solid #FF6B00 !important;
            background: rgba(255, 107, 0, 0.09) !important;
        }}

        /* Hero Banner */
        .hero-heading {{
            font-size: 2.3rem;
            font-weight: 800;
            color: #FFFFFF;
            letter-spacing: -0.8px;
            line-height: 1.15;
            margin: 0 0 10px 0;
        }}
        .hero-subtext {{
            font-size: 1.08rem;
            color: #B8C0CC;
            line-height: 1.55;
            margin: 0 0 20px 0;
        }}
        .hero-cta-btn {{
            display: inline-block;
            background: linear-gradient(135deg, #FF6B00 0%, #E05E00 100%);
            color: #FFFFFF !important;
            font-weight: 700;
            font-size: 0.98rem;
            padding: 12px 26px;
            border-radius: 8px;
            text-decoration: none;
            transition: all 0.2s ease-in-out;
            box-shadow: 0 4px 16px rgba(255, 107, 0, 0.38);
        }}
        .hero-cta-btn:hover {{
            background: linear-gradient(135deg, #FF801E 0%, #FF6B00 100%);
            box-shadow: 0 6px 20px rgba(255, 107, 0, 0.5);
            text-decoration: none;
        }}

        /* Animated Toy Car Track */
        .road-track-box {{
            background: rgba(13, 17, 23, 0.9);
            border: 1px solid rgba(255, 107, 0, 0.28);
            border-radius: 12px;
            padding: 16px 20px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
            position: relative;
            overflow: hidden;
        }}
        .track-meta {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }}
        .track-tag {{
            font-size: 0.75rem;
            font-weight: 800;
            color: #FF8A34;
            letter-spacing: 1.2px;
            text-transform: uppercase;
        }}
        .track-status {{
            font-size: 0.78rem;
            color: #94A3B8;
        }}
        .road-surface {{
            height: 70px;
            background: linear-gradient(180deg, #10141D 0%, #18202C 50%, #121721 100%);
            border-radius: 8px;
            border-top: 1px solid rgba(255, 255, 255, 0.08);
            border-bottom: 2px solid #FF6B00;
            position: relative;
            overflow: hidden;
            box-shadow: inset 0 2px 8px rgba(0,0,0,0.65);
        }}
        .road-lane-lines {{
            position: absolute;
            top: 50%;
            left: 0;
            width: 100%;
            height: 2px;
            background: repeating-linear-gradient(90deg, rgba(255,255,255,0.25) 0, rgba(255,255,255,0.25) 18px, transparent 18px, transparent 36px);
            transform: translateY(-50%);
        }}
        .toy-car-wrapper {{
            position: absolute;
            bottom: 4px;
            width: 110px;
            height: 50px;
            animation: drive-across 8.5s cubic-bezier(0.2, 0.8, 0.4, 1) infinite;
        }}
        @keyframes drive-across {{
            0% {{ left: -120px; opacity: 0; }}
            4% {{ opacity: 1; }}
            92% {{ opacity: 1; }}
            100% {{ left: calc(100% + 30px); opacity: 0; }}
        }}
        .wheel-spin {{
            animation: spin-wheel 0.5s linear infinite;
            transform-origin: center;
            transform-box: fill-box;
        }}
        @keyframes spin-wheel {{
            from {{ transform: rotate(0deg); }}
            to {{ transform: rotate(360deg); }}
        }}
        .car-chassis-bounce {{
            animation: chassis-bounce 0.45s ease-in-out infinite;
        }}
        @keyframes chassis-bounce {{
            0%, 100% {{ transform: translateY(0); }}
            50% {{ transform: translateY(-1.2px); }}
        }}

        @media (prefers-reduced-motion: reduce) {{
            .toy-car-wrapper {{
                animation: none !important;
                left: 42% !important;
                opacity: 1 !important;
            }}
            .wheel-spin, .car-chassis-bounce {{
                animation: none !important;
            }}
        }}

        /* Dark Charcoal Input Field Theming */
        div[data-testid="stNumberInput"] div[data-baseweb="input"],
        div[data-testid="stTextInput"] div[data-baseweb="input"],
        div[data-baseweb="input"] {{
            background-color: #171C25 !important;
            border: 1px solid #3B4655 !important;
            border-radius: 8px !important;
            transition: all 0.2s ease-in-out !important;
            overflow: hidden !important;
        }}
        div[data-baseweb="input"] > div,
        div[data-testid="stNumberInput"] div[data-baseweb="input"] > div {{
            background-color: #171C25 !important;
        }}
        div[data-baseweb="input"]:hover,
        div[data-testid="stNumberInput"] div[data-baseweb="input"]:hover {{
            border-color: #55657B !important;
        }}
        div[data-baseweb="input"]:focus-within,
        div[data-testid="stNumberInput"] div[data-baseweb="input"]:focus-within {{
            border-color: #FF6B00 !important;
            box-shadow: 0 0 0 1px #FF6B00 !important;
        }}
        input[type="number"],
        input[type="text"],
        div[data-baseweb="input"] input,
        div[data-testid="stNumberInput"] input,
        div[data-testid="stTextInput"] input {{
            background-color: #171C25 !important;
            color: #F5F7FA !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            border: none !important;
            -webkit-text-fill-color: #F5F7FA !important;
            padding: 8px 12px !important;
        }}
        input::placeholder {{
            color: #64748B !important;
            -webkit-text-fill-color: #64748B !important;
        }}
        div[data-testid="stNumberInput"] button {{
            background-color: #1E2532 !important;
            color: #F5F7FA !important;
            border: none !important;
            border-left: 1px solid #3B4655 !important;
            border-radius: 0 !important;
            transition: all 0.15s ease-in-out !important;
        }}
        div[data-testid="stNumberInput"] button:hover {{
            background-color: #FF6B00 !important;
            color: #FFFFFF !important;
        }}
        div[data-testid="stNumberInput"] button svg {{
            fill: #F5F7FA !important;
        }}

        /* Dropdowns */
        div[data-baseweb="select"] > div {{
            background-color: #171C25 !important;
            border: 1px solid #3B4655 !important;
            border-radius: 8px !important;
            color: #F5F7FA !important;
            transition: all 0.2s ease-in-out !important;
        }}
        div[data-baseweb="select"] > div:hover {{
            border-color: #55657B !important;
        }}
        div[data-baseweb="select"] > div:focus-within {{
            border-color: #FF6B00 !important;
            box-shadow: 0 0 0 1px #FF6B00 !important;
        }}
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] div {{
            color: #F5F7FA !important;
            font-weight: 600 !important;
        }}
        div[data-baseweb="popover"],
        div[data-baseweb="menu"],
        ul[data-baseweb="menu"] {{
            background-color: #171C25 !important;
            border: 1px solid #3B4655 !important;
            border-radius: 8px !important;
            box-shadow: 0 10px 28px rgba(0, 0, 0, 0.65) !important;
        }}
        li[role="option"] {{
            background-color: #171C25 !important;
            color: #F5F7FA !important;
            font-weight: 500 !important;
            padding: 9px 14px !important;
        }}
        li[role="option"]:hover,
        li[role="option"][aria-selected="true"] {{
            background-color: rgba(255, 107, 0, 0.22) !important;
            color: #FF8A34 !important;
        }}

        div[data-testid="stWidgetLabel"] label,
        div[data-testid="stWidgetLabel"] p {{
            color: #E2E8F0 !important;
            font-weight: 600 !important;
            font-size: 0.94rem !important;
        }}

        /* High-contrast typography: make all normal UI text readable on the dark theme */
        .stApp [data-testid="stMarkdownContainer"],
        .stApp [data-testid="stMarkdownContainer"] p,
        .stApp [data-testid="stMarkdownContainer"] li,
        .stApp [data-testid="stMarkdownContainer"] strong,
        .stApp [data-testid="stMarkdownContainer"] em,
        .stApp [data-testid="stCaptionContainer"],
        .stApp [data-testid="stCaptionContainer"] p,
        .stApp [data-testid="stWidgetLabel"],
        .stApp [data-testid="stWidgetLabel"] *,
        .stApp .stRadio label,
        .stApp .stCheckbox label,
        .stApp [data-testid="stExpander"] summary,
        .stApp [data-testid="stExpander"] summary p {{
            color: #F1F5F9 !important;
        }}
        .stApp [data-testid="stMarkdownContainer"] p,
        .stApp [data-testid="stCaptionContainer"] p,
        .stApp [data-testid="stWidgetLabel"] label,
        .stApp [data-testid="stWidgetLabel"] p {{
            font-weight: 600 !important;
        }}
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {{
            color: #FFFFFF !important;
            font-weight: 800 !important;
        }}
        .stApp input, .stApp textarea,
        .stApp [data-baseweb="select"] span,
        .stApp [data-baseweb="select"] div {{
            color: #FFFFFF !important;
            -webkit-text-fill-color: #FFFFFF !important;
            font-weight: 600 !important;
        }}
        .stApp input::placeholder, .stApp textarea::placeholder {{
            color: #C5CEDA !important;
            -webkit-text-fill-color: #C5CEDA !important;
            opacity: 1 !important;
        }}
        .stApp [data-testid="stCaptionContainer"] *,
        .stApp small {{
            color: #D1D9E6 !important;
        }}

        /* Card Styles */
        .valuation-result-box {{
            background: linear-gradient(145deg, rgba(20, 27, 38, 0.92) 0%, rgba(26, 36, 50, 0.92) 100%);
            backdrop-filter: blur(14px);
            border: 1px solid rgba(255, 107, 0, 0.48);
            border-radius: 16px;
            padding: 30px 24px;
            text-align: center;
            box-shadow: 0 10px 32px rgba(0, 0, 0, 0.5);
            margin-bottom: 18px;
        }}
        .metric-card-box {{
            background: rgba(18, 24, 32, 0.85);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 10px;
            padding: 16px 14px;
            text-align: center;
            backdrop-filter: blur(8px);
        }}
        .metric-card-title {{
            font-size: 0.75rem;
            text-transform: uppercase;
            font-weight: 700;
            letter-spacing: 0.6px;
            color: #94A3B8;
            margin-bottom: 5px;
        }}
        .metric-card-val {{
            font-size: 1.5rem;
            font-weight: 800;
            color: #FFFFFF;
        }}

        /* Buyer Vehicle Card */
        .buyer-car-card {{
            background: rgba(18, 24, 32, 0.88);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 18px;
            margin-bottom: 16px;
            transition: all 0.2s ease-in-out;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
        }}
        .buyer-car-card:hover {{
            border-color: rgba(255, 107, 0, 0.4);
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
        }}
        .badge-verified {{
            background: rgba(16, 185, 129, 0.15);
            color: #10B981;
            font-size: 0.72rem;
            font-weight: 700;
            padding: 3px 8px;
            border-radius: 4px;
            border: 1px solid rgba(16, 185, 129, 0.3);
            text-transform: uppercase;
        }}
        .badge-demo {{
            background: rgba(255, 107, 0, 0.15);
            color: #FF8A34;
            font-size: 0.72rem;
            font-weight: 700;
            padding: 3px 8px;
            border-radius: 4px;
            border: 1px solid rgba(255, 107, 0, 0.3);
            text-transform: uppercase;
        }}
        .badge-city {{
            background: rgba(56, 189, 248, 0.15);
            color: #38BDF8;
            font-size: 0.72rem;
            font-weight: 700;
            padding: 3px 8px;
            border-radius: 4px;
        }}

        /* Tables & Charts */
        .dark-table-card {{
            background: rgba(18, 24, 32, 0.85);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            overflow: hidden;
        }}
        .dark-data-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.92rem;
            color: #F5F7FA;
        }}
        .dark-data-table th {{
            background-color: #121822;
            color: #94A3B8;
            padding: 12px 16px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            font-size: 0.85rem;
            text-transform: uppercase;
        }}
        .dark-data-table td {{
            padding: 12px 16px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }}
        .dark-data-table tr.row-alt {{
            background-color: rgba(255, 255, 255, 0.02);
        }}

        /* Primary Orange Buttons */
        div.stButton > button:first-child,
        div[data-testid="stForm"] button[kind="primary"] {{
            background: linear-gradient(135deg, #FF6B00 0%, #E05E00 100%) !important;
            color: #FFFFFF !important;
            font-weight: 700 !important;
            font-size: 1.02rem !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 11px 22px !important;
            transition: all 0.2s ease-in-out !important;
            box-shadow: 0 4px 16px rgba(255, 107, 0, 0.35) !important;
        }}
        div.stButton > button:first-child:hover {{
            background: linear-gradient(135deg, #FF801E 0%, #FF6B00 100%) !important;
            box-shadow: 0 6px 20px rgba(255, 107, 0, 0.48) !important;
            color: #FFFFFF !important;
        }}

        /* Footer */
        .app-clean-footer {{
            text-align: center;
            padding: 26px 0 10px 0;
            border-top: 1px solid rgba(255, 255, 255, 0.06);
            margin-top: 40px;
            color: #94A3B8;
            font-size: 0.88rem;
        }}
    </style>
    """, unsafe_allow_html=True)

    # A. Brand Navigation Header (HarCar AI)
    st.markdown("""
    <div class="top-navbar">
        <div class="navbar-brand">
            <span class="navbar-logo-icon">🚘</span>
            <div>
                <h1 class="navbar-title">HarCar <span>AI</span></h1>
            </div>
        </div>
        <div class="navbar-tagline">
            Intelligent Used-Car Price Prediction • <b>Know Your Car's Worth.</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Load Model Pipeline
    pipeline = load_trained_pipeline()
    if pipeline is None:
        st.error(f"Trained valuation model could not be loaded from `{MODEL_PATH}`. Please run `py -3.13 train_model.py`.")
        st.stop()

    # Core Navigation Tabs (Focused Pure Valuation Platform)
    tab_estimate, tab_valuation, tab_compare = st.tabs([
        "🚗 Price Estimator",
        "📊 My Valuation",
        "⚖️ Compare Cars"
    ])

    # ==========================================================================
    # SECTION 1: ESTIMATE PRICE — HOME PAGE & VALUATION WORKFLOW
    # ==========================================================================
    with tab_estimate:
        h_col1, h_col2 = st.columns([1.1, 0.9], gap="large")

        with h_col1:
            st.markdown("""
            <div style="padding: 14px 0;">
                <h2 class="hero-heading">Know What Your Car Is Worth</h2>
                <p class="hero-subtext">Get an estimated fair-market resale value for your car in seconds with HarCar AI.</p>
                <a href="#valuation-form-anchor" class="hero-cta-btn">Estimate My Car ↓</a>
            </div>
            """, unsafe_allow_html=True)

        with h_col2:
            st.markdown("""
            <div class="road-track-box">
                <div class="track-meta">
                    <span class="track-tag">⚡ HarCar AI Valuation Engine</span>
                    <span class="track-status">Continuous Market Appraisal</span>
                </div>
                <div class="road-surface">
                    <div class="road-lane-lines"></div>
                    <div class="toy-car-wrapper">
                        <svg viewBox="0 0 110 50" fill="none" xmlns="http://www.w3.org/2000/svg" style="width: 100%; height: 100%;">
                            <defs>
                                <linearGradient id="headlight-beam" x1="0%" y1="0%" x2="100%" y2="0%">
                                    <stop offset="0%" stop-color="#FFA834" stop-opacity="0.65"/>
                                    <stop offset="60%" stop-color="#FFA834" stop-opacity="0.2"/>
                                    <stop offset="100%" stop-color="#FFA834" stop-opacity="0"/>
                                </linearGradient>
                                <linearGradient id="car-paint" x1="0%" y1="0%" x2="100%" y2="0%">
                                    <stop offset="0%" stop-color="#E05E00"/>
                                    <stop offset="35%" stop-color="#FF6B00"/>
                                    <stop offset="80%" stop-color="#FF8A34"/>
                                    <stop offset="100%" stop-color="#FFA25B"/>
                                </linearGradient>
                                <linearGradient id="window-glass" x1="0%" y1="0%" x2="0%" y2="100%">
                                    <stop offset="0%" stop-color="#38BDF8" stop-opacity="0.75"/>
                                    <stop offset="100%" stop-color="#0F172A" stop-opacity="0.95"/>
                                </linearGradient>
                            </defs>
                            <polygon points="98,28 140,21 140,39" fill="url(#headlight-beam)"/>
                            <g class="car-chassis-bounce">
                                <ellipse cx="50" cy="42" rx="42" ry="4" fill="rgba(0,0,0,0.55)"/>
                                <path d="M 12 36 L 10 32 C 10 28, 14 26, 18 26 L 28 26 L 40 14 C 42 12, 46 11, 52 11 L 70 11 C 74 11, 78 13, 82 17 L 90 26 L 98 27 C 102 28, 104 31, 104 35 L 103 37 C 103 38, 101 39, 99 39 L 90 39 C 90 34, 84 30, 78 30 C 72 30, 66 34, 66 39 L 34 39 C 34 34, 28 30, 22 30 C 16 30, 10 34, 10 39 Z" fill="url(#car-paint)" stroke="#C84800" stroke-width="0.8"/>
                                <path d="M 42 15 L 53 14 L 53 25 L 32 25 Z" fill="url(#window-glass)"/>
                                <path d="M 56 14 L 69 14 C 72 14, 75 16, 78 19 L 83 25 L 56 25 Z" fill="url(#window-glass)"/>
                                <path d="M 44 16 L 70 16" stroke="rgba(255,255,255,0.6)" stroke-width="1" stroke-linecap="round"/>
                                <circle cx="101" cy="31" r="2.5" fill="#FFFBEB"/>
                                <circle cx="101" cy="31" r="4.5" fill="#FFAA00" opacity="0.6"/>
                                <rect x="8" y="29" width="3" height="5" rx="1.5" fill="#EF4444"/>
                            </g>
                            <g transform="translate(22, 38)">
                                <circle cx="0" cy="0" r="7.5" fill="#181B20" stroke="#2D333B" stroke-width="1.2"/>
                                <circle cx="0" cy="0" r="4.5" fill="#475569"/>
                                <g class="wheel-spin">
                                    <line x1="-4" y1="0" x2="4" y2="0" stroke="#E2E8F0" stroke-width="1.2"/>
                                    <line x1="0" y1="-4" x2="0" y2="4" stroke="#E2E8F0" stroke-width="1.2"/>
                                </g>
                                <circle cx="0" cy="0" r="1.8" fill="#FF6B00"/>
                            </g>
                            <g transform="translate(78, 38)">
                                <circle cx="0" cy="0" r="7.5" fill="#181B20" stroke="#2D333B" stroke-width="1.2"/>
                                <circle cx="0" cy="0" r="4.5" fill="#475569"/>
                                <g class="wheel-spin">
                                    <line x1="-4" y1="0" x2="4" y2="0" stroke="#E2E8F0" stroke-width="1.2"/>
                                    <line x1="0" y1="-4" x2="0" y2="4" stroke="#E2E8F0" stroke-width="1.2"/>
                                </g>
                                <circle cx="0" cy="0" r="1.8" fill="#FF6B00"/>
                            </g>
                        </svg>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div id='valuation-form-anchor' style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

        c_form, c_result = st.columns([1.08, 0.92], gap="large")

        with c_form:
            with st.form("main_car_valuation_form"):
                st.markdown("<h3 style='color: #FFFFFF; margin-bottom: 14px;'>Tell Us About Your Car</h3>", unsafe_allow_html=True)

                st.markdown("<div style='font-size: 0.92rem; font-weight: 700; color: #FF8A34; text-transform: uppercase;'>1. Financial & Mileage Profile</div>", unsafe_allow_html=True)
                r1_c1, r1_c2 = st.columns(2)
                with r1_c1:
                    present_price = st.number_input(
                        "Original Ex-Showroom Price (Lakhs INR)*",
                        min_value=0.10,
                        max_value=100.0,
                        value=5.59,
                        step=0.25,
                        help="Showroom purchase price when brand new in Lakhs (e.g. 5.59 = ₹5,59,000)."
                    )
                    car_age = st.number_input(
                        "Car Age (Years)*",
                        min_value=0,
                        max_value=30,
                        value=6,
                        step=1,
                        help="Years elapsed since original registration."
                    )
                with r1_c2:
                    kms_driven = st.number_input(
                        "Kilometres Driven*",
                        min_value=0,
                        max_value=1000000,
                        value=27000,
                        step=2500,
                        help="Cumulative odometer mileage reading."
                    )
                    owner = st.selectbox(
                        "Previous Registered Owners*",
                        options=[0, 1, 2, 3],
                        index=0,
                        format_func=lambda x: "0 (First Owner / Brand New)" if x == 0 else f"{x} Prior Owner(s)"
                    )

                st.markdown("<div style='font-size: 0.92rem; font-weight: 700; color: #FF8A34; text-transform: uppercase; margin-top: 14px;'>2. Vehicle Specifications</div>", unsafe_allow_html=True)
                r2_c1, r2_c2, r2_c3 = st.columns(3)
                with r2_c1:
                    fuel_type = st.selectbox("Fuel Type*", options=["Petrol", "Diesel", "CNG"], index=0)
                with r2_c2:
                    transmission = st.selectbox("Transmission*", options=["Manual", "Automatic"], index=0)
                with r2_c3:
                    seller_type = st.selectbox("Seller Type*", options=["Dealer", "Individual"], index=0)

                calculate_btn = st.form_submit_button("Calculate My Car's Value", type="primary", use_container_width=True)

            if calculate_btn:
                is_valid, errors, advisories = validate_inputs(present_price, car_age, kms_driven, owner)

                if not is_valid:
                    for err in errors:
                        st.error(f"❌ {err}")
                else:
                    with st.spinner("Analyzing market data & evaluating vehicle condition..."):
                        input_df = pd.DataFrame([{
                            'Present_Price': float(present_price),
                            'Kms_Driven': int(kms_driven),
                            'Owner': int(owner),
                            'Car_Age': int(car_age),
                            'Fuel_Type': str(fuel_type),
                            'Seller_Type': str(seller_type),
                            'Transmission': str(transmission)
                        }])

                        try:
                            raw_pred = pipeline.predict(input_df)[0]
                            pred_lakh = max(0.05, float(raw_pred))
                            val = calculate_valuation(pred_lakh, float(present_price))

                            st.session_state['active_valuation'] = {
                                'present_price': float(present_price),
                                'car_age': int(car_age),
                                'kms_driven': int(kms_driven),
                                'owner': int(owner),
                                'fuel_type': str(fuel_type),
                                'seller_type': str(seller_type),
                                'transmission': str(transmission),
                                'predicted_lakhs': pred_lakh,
                                'predicted_rupees': pred_lakh * 100000.0,
                                'retained_pct': val['retained_pct'],
                                'depreciation_pct': val['depreciation_pct'],
                                'depreciation_amount_lakh': val['depreciation_amount_lakh'],
                                'depreciation_amount_rupees': val['depreciation_amount_rupees'],
                                'exceeds_original': val['exceeds_original'],
                                'value_diff_lakh': val['value_difference_lakh'],
                                'advisories': advisories
                            }
                        except Exception as exc:
                            st.error(f"Valuation Error: {str(exc)}")

        with c_result:
            if 'active_valuation' in st.session_state and st.session_state['active_valuation'] is not None:
                res = st.session_state['active_valuation']

                for adv in res['advisories']:
                    st.info(f"ℹ️ {adv}")

                if res['exceeds_original']:
                    st.markdown(f"""
                    <div class="appreciation-notice">
                        ⚠️ <b>Appreciation Notice:</b> Estimated resale value (<b>₹{res['predicted_lakhs']:.2f}L</b>) 
                        exceeds original showroom purchase price (<b>₹{res['present_price']:.2f}L</b>).
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="valuation-result-box">
                    <div style="font-size: 0.78rem; font-weight: 800; text-transform: uppercase; color: #FF8A34; margin-bottom: 6px;">Estimated Resale Value</div>
                    <div style="font-size: 3.3rem; font-weight: 900; color: #FFFFFF; line-height: 1.05;">{format_inr(res['predicted_rupees'])}</div>
                    <div style="font-size: 1.2rem; color: #CBD5E1; font-weight: 600; margin-top: 6px;">≈ ₹{res['predicted_lakhs']:.2f} Lakhs</div>
                    <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 14px;">Estimate powered by HarCar AI machine-learning valuation intelligence. Actual trade-in offers may vary.</div>
                </div>
                """, unsafe_allow_html=True)

                m1, m2, m3 = st.columns(3)
                with m1:
                    ret_color = "#10B981" if res['retained_pct'] <= 100 else "#FF6B00"
                    st.markdown(f"""
                    <div class="metric-card-box">
                        <div class="metric-card-title">Retained Value</div>
                        <div class="metric-card-val" style="color: {ret_color};">{res['retained_pct']:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                with m2:
                    if res['depreciation_pct'] >= 0:
                        dep_val_str = format_inr(res['depreciation_amount_rupees'])
                        dep_color = "#EF4444"
                        dep_label = "Depreciation Amount"
                    else:
                        dep_val_str = f"+{format_inr(abs(res['depreciation_amount_rupees']))}"
                        dep_color = "#FF6B00"
                        dep_label = "Appreciation Gain"

                    st.markdown(f"""
                    <div class="metric-card-box">
                        <div class="metric-card-title">{dep_label}</div>
                        <div class="metric-card-val" style="color: {dep_color};">{dep_val_str}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with m3:
                    if res['depreciation_pct'] >= 0:
                        pct_str = f"{res['depreciation_pct']:.1f}%"
                        rate_color = "#EF4444"
                        rate_label = "Depreciation Rate"
                    else:
                        pct_str = f"+{abs(res['depreciation_pct']):.1f}%"
                        rate_color = "#FF6B00"
                        rate_label = "Appreciation Rate"

                    st.markdown(f"""
                    <div class="metric-card-box">
                        <div class="metric-card-title">{rate_label}</div>
                        <div class="metric-card-val" style="color: {rate_color};">{pct_str}</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
                if st.button("↺ Appraise Another Vehicle", use_container_width=True):
                    st.session_state['active_valuation'] = None
                    st.rerun()

            else:
                st.markdown("""
                <div class="prompt-card" style="background: rgba(18, 24, 32, 0.85); border: 1px dashed rgba(255, 255, 255, 0.15); border-radius: 14px; padding: 48px 30px; text-align: center;">
                    <div style="font-size: 2.4rem; margin-bottom: 10px;">🚘</div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: #FFFFFF; margin-bottom: 8px;">Ready to Appraise Your Car</div>
                    <div style="font-size: 0.92rem; max-width: 440px; margin: 0 auto; line-height: 1.55; color: #B8C0CC;">
                        Enter your vehicle's specifications on the left and click <b>Calculate My Car's Value</b> 
                        to generate your instant AI fair-market valuation.
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ==========================================================================
    # SECTION 2: MY VALUATION — SUMMARY & AUDIT
    # ==========================================================================
    with tab_valuation:
        if 'active_valuation' not in st.session_state or st.session_state['active_valuation'] is None:
            st.info("ℹ️ No active valuation record found. Please enter vehicle details and click **Calculate My Car's Value** in the **Estimate Price** tab.")
        else:
            v = st.session_state['active_valuation']

            st.markdown("<h2 style='font-size: 1.8rem; font-weight: 800; color: #FFFFFF; margin-bottom: 6px;'>My Valuation Report</h2>", unsafe_allow_html=True)
            st.markdown("<p style='color: #B8C0CC; font-size: 0.95rem; margin-bottom: 20px;'>Comprehensive appraisal breakdown for your appraised vehicle.</p>", unsafe_allow_html=True)

            b1, b2, b3, b4 = st.columns(4)
            with b1:
                st.markdown(f"""
                <div class="metric-card-box">
                    <div class="metric-card-title">Original Showroom Price</div>
                    <div class="metric-card-val" style="color: #FFFFFF;">₹{v['present_price']:.2f}L</div>
                    <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 2px;">{format_inr(v['present_price']*100000)}</div>
                </div>
                """, unsafe_allow_html=True)
            with b2:
                st.markdown(f"""
                <div class="metric-card-box" style="border-color: rgba(255, 107, 0, 0.4);">
                    <div class="metric-card-title">Estimated Resale Price</div>
                    <div class="metric-card-val" style="color: #FF8A34;">₹{v['predicted_lakhs']:.2f}L</div>
                    <div style="font-size: 0.78rem; color: #CBD5E1; margin-top: 2px;">{format_inr(v['predicted_rupees'])}</div>
                </div>
                """, unsafe_allow_html=True)
            with b3:
                ret_c = "#10B981" if v['retained_pct'] <= 100 else "#FF8A34"
                st.markdown(f"""
                <div class="metric-card-box">
                    <div class="metric-card-title">Retained Market Value</div>
                    <div class="metric-card-val" style="color: {ret_c};">{v['retained_pct']:.1f}%</div>
                    <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 2px;">Of original cost</div>
                </div>
                """, unsafe_allow_html=True)
            with b4:
                if v['depreciation_pct'] >= 0:
                    dep_loss_title = "Depreciation Loss"
                    dep_loss_val = f"{v['depreciation_pct']:.1f}%"
                    dep_loss_sub = f"-{format_inr(v['depreciation_amount_rupees'])}"
                    dep_c = "#EF4444"
                else:
                    dep_loss_title = "Appreciation Gain"
                    dep_loss_val = f"+{abs(v['depreciation_pct']):.1f}%"
                    dep_loss_sub = f"+{format_inr(abs(v['depreciation_amount_rupees']))}"
                    dep_c = "#FF8A34"

                st.markdown(f"""
                <div class="metric-card-box">
                    <div class="metric-card-title">{dep_loss_title}</div>
                    <div class="metric-card-val" style="color: {dep_c};">{dep_loss_val}</div>
                    <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 2px;">{dep_loss_sub}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

            col_det, col_chart = st.columns([1, 1], gap="large")

            with col_det:
                st.markdown("<div style='font-size: 1.1rem; font-weight: 700; color: #FFFFFF; margin-bottom: 10px;'>Vehicle Profile & Specification</div>", unsafe_allow_html=True)
                owner_display = "0 (First Owner / Brand New)" if v['owner'] == 0 else f"{v['owner']} Prior Owner(s)"
                profile_rows = [
                    ("Original Showroom Cost", f"₹{v['present_price']:.2f} Lakhs ({format_inr(v['present_price']*100000)})"),
                    ("Vehicle Chronological Age", f"{v['car_age']} Year(s)"),
                    ("Odometer Distance", f"{v['kms_driven']:,} km"),
                    ("Previous Registered Owners", owner_display),
                    ("Powertrain Fuel Type", v['fuel_type']),
                    ("Transmission Mechanism", v['transmission']),
                    ("Distribution Channel", v['seller_type']),
                ]
                
                table_html = ['<div class="dark-table-card"><table class="dark-data-table"><tbody>']
                for idx, (param, val) in enumerate(profile_rows):
                    alt_class = "row-alt" if idx % 2 == 1 else ""
                    table_html.append(f'<tr class="{alt_class}"><td style="color: #B8C0CC; font-weight: 500;">{param}</td><td style="color: #FFFFFF; font-weight: 600;">{val}</td></tr>')
                table_html.append('</tbody></table></div>')
                st.markdown("".join(table_html), unsafe_allow_html=True)

            with col_chart:
                st.markdown("<div style='font-size: 1.1rem; font-weight: 700; color: #FFFFFF; margin-bottom: 10px;'>Value Retention & Depreciation Share</div>", unsafe_allow_html=True)
                if v['retained_pct'] <= 100 and v['depreciation_pct'] >= 0:
                    donut_df = pd.DataFrame({
                        "Category": ["Retained Value", "Depreciation Loss"],
                        "Percentage": [v['retained_pct'], v['depreciation_pct']],
                        "Color": ["#10B981", "#EF4444"]
                    })
                    donut_chart = alt.Chart(donut_df).mark_arc(innerRadius=50, stroke="#0B0F14", strokeWidth=2).encode(
                        theta=alt.Theta("Percentage:Q"),
                        color=alt.Color("Color:N", scale=None),
                        tooltip=["Category:N", alt.Tooltip("Percentage:Q", format=".1f")]
                    ).properties(height=240).configure_view(
                        stroke=None, fill='transparent'
                    ).configure(
                        background='transparent'
                    ).configure_legend(
                        labelColor='#B8C0CC',
                        titleColor='#F5F7FA'
                    )
                    st.altair_chart(donut_chart, use_container_width=True)
                else:
                    bar_df = pd.DataFrame({
                        "Metric": ["Retained %", "Depreciation %"],
                        "Value": [v['retained_pct'], v['depreciation_pct']],
                        "Color": ["#FF8A34", "#EF4444"]
                    })
                    bar_chart = alt.Chart(bar_df).mark_bar(cornerRadius=6, size=40).encode(
                        x=alt.X("Metric:N", title=None, axis=alt.Axis(labelAngle=0, labelColor="#F5F7FA")),
                        y=alt.Y("Value:Q", title="Percentage (%)", axis=alt.Axis(labelColor="#B8C0CC", gridColor="rgba(255,255,255,0.06)")),
                        color=alt.Color("Color:N", scale=None),
                        tooltip=["Metric:N", alt.Tooltip("Value:Q", format=".1f")]
                    ).properties(height=240).configure_view(
                        stroke=None, fill='transparent'
                    ).configure(
                        background='transparent'
                    )
                    st.altair_chart(bar_chart, use_container_width=True)

            st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
            summary_txt = (
                "================================================================================\n"
                "HARCAR AI — VEHICLE VALUATION APPRAISAL SUMMARY\n"
                "Tagline: Know Your Car's Worth.\n"
                "================================================================================\n\n"
                "VEHICLE SPECIFICATIONS:\n"
                f"- Original Showroom Price : ₹{v['present_price']:.2f} Lakhs ({format_inr(v['present_price']*100000)})\n"
                f"- Chronological Age       : {v['car_age']} Year(s)\n"
                f"- Odometer Mileage        : {v['kms_driven']:,} km\n"
                f"- Prior Registered Owners : {v['owner']}\n"
                f"- Powertrain Fuel Type    : {v['fuel_type']}\n"
                f"- Transmission Mechanism  : {v['transmission']}\n"
                f"- Seller Channel          : {v['seller_type']}\n\n"
                "VALUATION APPRAISAL RESULTS:\n"
                f"- Estimated Resale Value  : ₹{v['predicted_lakhs']:.2f} Lakhs ({format_inr(v['predicted_rupees'])})\n"
                f"- Retained Market Value   : {v['retained_pct']:.2f}%\n"
                f"- Depreciation Rate       : {v['depreciation_pct']:.2f}%\n"
                f"- Net Monetary Impact     : ₹{v['depreciation_amount_lakh']:.2f} Lakhs ({format_inr(v['depreciation_amount_rupees'])})\n\n"
                "Disclaimer: Statistical estimate based on HarCar AI machine-learning models.\n"
                "Actual market trade-in offers may vary based on physical vehicle inspection.\n"
                "================================================================================\n"
            )

            st.download_button(
                label="📥 Download Valuation Appraisal Summary (.txt)",
                data=summary_txt,
                file_name="harcar_ai_valuation_summary.txt",
                mime="text/plain",
                use_container_width=True
            )

    # ==========================================================================
    # SECTION 3: COMPARE CARS — MULTI-VEHICLE SCENARIO COMPARISON
    # ==========================================================================
    with tab_compare:
        st.markdown("<h2 style='font-size: 1.8rem; font-weight: 800; color: #FFFFFF; margin-bottom: 6px;'>Vehicle Valuation Comparison</h2>", unsafe_allow_html=True)
        st.markdown("<p style='color: #B8C0CC; font-size: 0.95rem; margin-bottom: 20px;'>Compare 2 or 3 vehicle scenarios side-by-side using real-time predictions from HarCar AI.</p>", unsafe_allow_html=True)

        count = st.radio("Number of Vehicles to Compare:", options=[2, 3], index=0, horizontal=True)

        sc_cols = st.columns(count, gap="medium")
        sc_inputs = []

        defaults = [
            {"name": "Vehicle 1", "price": 5.59, "age": 6, "kms": 27000, "owner": 0, "fuel": "Petrol", "trans": "Manual", "seller": "Dealer"},
            {"name": "Vehicle 2", "price": 7.50, "age": 4, "kms": 35000, "owner": 0, "fuel": "Diesel", "trans": "Manual", "seller": "Dealer"},
            {"name": "Vehicle 3", "price": 9.50, "age": 3, "kms": 20000, "owner": 0, "fuel": "Petrol", "trans": "Automatic", "seller": "Dealer"},
        ]

        for i in range(count):
            d = defaults[i]
            with sc_cols[i]:
                st.markdown(f"""
                <div style="background: rgba(18, 24, 32, 0.85); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 14px 18px 8px 18px; margin-bottom: 12px;">
                    <span style="font-size: 1.1rem; font-weight: 800; color: #FFFFFF;">{d['name']}</span>
                    <span style="float: right; font-size: 0.72rem; font-weight: 800; color: #FF8A34; letter-spacing: 1px;">PROFILE {i+1}</span>
                </div>
                """, unsafe_allow_html=True)

                p = st.number_input(f"Showroom Price (Lakhs) [{i+1}]", min_value=0.10, max_value=100.0, value=d['price'], step=0.25, key=f"comp_p_{i}")
                a = st.number_input(f"Age (Years) [{i+1}]", min_value=0, max_value=30, value=d['age'], step=1, key=f"comp_a_{i}")
                k = st.number_input(f"Mileage (km) [{i+1}]", min_value=0, max_value=1000000, value=d['kms'], step=5000, key=f"comp_k_{i}")
                f = st.selectbox(f"Fuel [{i+1}]", options=["Petrol", "Diesel", "CNG"], index=["Petrol", "Diesel", "CNG"].index(d['fuel']), key=f"comp_f_{i}")
                t = st.selectbox(f"Transmission [{i+1}]", options=["Manual", "Automatic"], index=["Manual", "Automatic"].index(d['trans']), key=f"comp_t_{i}")
                s = st.selectbox(f"Seller [{i+1}]", options=["Dealer", "Individual"], index=["Dealer", "Individual"].index(d['seller']), key=f"comp_s_{i}")
                o = st.selectbox(f"Owners [{i+1}]", options=[0, 1, 2, 3], index=d['owner'], key=f"comp_o_{i}")

                sc_inputs.append({
                    "name": f"Vehicle {i+1}",
                    "df": pd.DataFrame([{
                        'Present_Price': float(p),
                        'Kms_Driven': int(k),
                        'Owner': int(o),
                        'Car_Age': int(a),
                        'Fuel_Type': str(f),
                        'Seller_Type': str(s),
                        'Transmission': str(t)
                    }]),
                    "price": float(p)
                })

        results = []
        for car in sc_inputs:
            raw = pipeline.predict(car["df"])[0]
            pl = max(0.05, float(raw))
            val = calculate_valuation(pl, car["price"])
            results.append({
                "Car": car["name"],
                "Showroom_Price": car["price"],
                "Predicted_Price": pl,
                "Predicted_INR": pl * 100000.0,
                "Retained_Pct": val["retained_pct"],
                "Depreciation_Pct": val["depreciation_pct"],
                "Depreciation_Amount_Lakh": val["depreciation_amount_lakh"],
                "Depreciation_Amount_INR": val["depreciation_amount_rupees"],
            })

        df_res = pd.DataFrame(results)
        best_price = max(r["Predicted_Price"] for r in results)

        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 1.15rem; font-weight: 700; color: #FFFFFF; margin-bottom: 12px;'>Side-by-Side Valuation Matrix</div>", unsafe_allow_html=True)

        cmp_headers = ["Evaluation Metric"] + [r["Car"] for r in results]
        matrix_rows = [
            ("Original Showroom Price", [f"₹{r['Showroom_Price']:.2f}L ({format_inr(r['Showroom_Price']*100000)})" for r in results]),
            ("Estimated Resale Value", [
                f"<span style='color: #FF8A34; font-weight: 800;'>₹{r['Predicted_Price']:.2f}L</span> ({format_inr(r['Predicted_INR'])})" + 
                ("<span style='background: #FF6B00; color: #FFFFFF; font-size: 0.7rem; font-weight: 800; padding: 2px 6px; border-radius: 4px; margin-left: 6px;'>★ Top Value</span>" if r["Predicted_Price"] == best_price and len(results) > 1 else "")
                for r in results
            ]),
            ("Retained Market Value (%)", [f"<span style='color: #10B981; font-weight: 700;'>{r['Retained_Pct']:.1f}%</span>" for r in results]),
            ("Depreciation Rate (%)", [f"{r['Depreciation_Pct']:.1f}%" if r['Depreciation_Pct'] >= 0 else f"+{abs(r['Depreciation_Pct']):.1f}% (Apprec)" for r in results]),
            ("Net Depreciation Impact", [f"₹{r['Depreciation_Amount_Lakh']:.2f}L ({format_inr(r['Depreciation_Amount_INR'])})" for r in results]),
        ]

        matrix_html = ['<div class="dark-table-card"><table class="dark-data-table"><thead><tr>']
        for h in cmp_headers:
            matrix_html.append(f'<th>{h}</th>')
        matrix_html.append('</tr></thead><tbody>')

        for idx, (metric, vals) in enumerate(matrix_rows):
            alt_class = "row-alt" if idx % 2 == 1 else ""
            matrix_html.append(f'<tr class="{alt_class}"><td style="color: #B8C0CC; font-weight: 500;">{metric}</td>')
            for v_str in vals:
                matrix_html.append(f'<td style="color: #FFFFFF; font-weight: 600;">{v_str}</td>')
            matrix_html.append('</tr>')

        matrix_html.append('</tbody></table></div>')
        st.markdown("".join(matrix_html), unsafe_allow_html=True)

        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 1.15rem; font-weight: 700; color: #FFFFFF; margin-bottom: 12px;'>Estimated Resale Price Comparison (Lakhs INR)</div>", unsafe_allow_html=True)
        cmp_chart = alt.Chart(df_res).mark_bar(cornerRadius=6, size=45).encode(
            x=alt.X("Car:N", title=None, axis=alt.Axis(labelAngle=0, labelColor="#F5F7FA")),
            y=alt.Y("Predicted_Price:Q", title="Resale Estimate (Lakhs INR)", axis=alt.Axis(labelColor="#B8C0CC", gridColor="rgba(255,255,255,0.06)")),
            color=alt.Color("Car:N", scale=alt.Scale(range=["#FF6B00", "#10B981", "#38BDF8"]), legend=None),
            tooltip=[
                alt.Tooltip("Car:N"),
                alt.Tooltip("Predicted_Price:Q", title="Resale (Lakhs)", format=".2f"),
                alt.Tooltip("Retained_Pct:Q", title="Retained (%)", format=".1f")
            ]
        ).properties(height=240).configure_view(
            stroke=None, fill='transparent'
        ).configure(
            background='transparent'
        )
        st.altair_chart(cmp_chart, use_container_width=True)

    # Footer
    st.markdown("""
    <div class="app-clean-footer">
        HarCar AI • Intelligent Used-Car Price Prediction • <b>Know Your Car's Worth.</b>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
