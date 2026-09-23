import time

import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Aorta · Heart Risk",
    page_icon="🫀",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Theme (Aorta-inspired clinical desk)
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
    .stApp {
        background: #09090b;
        color: #f2f1ee;
    }
    h1, h2, h3, h4, p, label, span, div {
        color: #f2f1ee !important;
    }
    .stMarkdown, .stCaption {
        color: #9c9a94 !important;
    }
    [data-testid="stHeader"] {
        background: transparent;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 880px;
    }
    .aorta-kicker {
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 0.7rem;
        letter-spacing: 0.28em;
        text-transform: uppercase;
        color: #6e6c68 !important;
        margin-bottom: 0.35rem;
    }
    .aorta-title {
        font-size: 2.4rem;
        font-weight: 500;
        letter-spacing: -0.03em;
        line-height: 1.1;
        margin: 0 0 0.4rem 0;
        color: #f2f1ee !important;
    }
    .aorta-title em {
        color: #9c9a94 !important;
        font-style: italic;
        font-weight: 400;
    }
    .aorta-sub {
        color: #9c9a94 !important;
        font-size: 0.95rem;
        line-height: 1.55;
        max-width: 34rem;
        margin-bottom: 1.75rem;
    }
    .section-label {
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 0.7rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #6e6c68 !important;
        margin: 1.25rem 0 0.6rem 0;
    }
    .card {
        background: #131316;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 1.35rem 1.5rem 1.5rem;
        margin-bottom: 1rem;
    }
    .result-high {
        background: linear-gradient(160deg, #3a1816, #2a1210);
        border: 1px solid rgba(196, 92, 74, 0.45);
        border-radius: 16px;
        padding: 1.5rem 1.4rem;
        text-align: center;
        margin-top: 1rem;
    }
    .result-low {
        background: linear-gradient(160deg, #152018, #101812);
        border: 1px solid rgba(125, 154, 122, 0.45);
        border-radius: 16px;
        padding: 1.5rem 1.4rem;
        text-align: center;
        margin-top: 1rem;
    }
    .result-title {
        font-size: 1.35rem;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }
    .result-high .result-title { color: #e8a59a !important; }
    .result-low .result-title { color: #b5d0b2 !important; }
    .result-sub {
        font-size: 0.9rem;
        font-weight: 400;
        color: #9c9a94 !important;
    }
    .footer-note {
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255,255,255,0.08);
        font-size: 0.8rem;
        color: #6e6c68 !important;
        line-height: 1.5;
    }
    /* Controls */
    .stSlider label, .stSelectbox label, .stNumberInput label {
        color: #9c9a94 !important;
        font-size: 0.85rem !important;
    }
    div[data-baseweb="select"] > div {
        background-color: #1c1c20 !important;
        border-color: rgba(255,255,255,0.08) !important;
    }
    .stButton > button {
        width: 100%;
        background: #c9c4b8 !important;
        color: #09090b !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 0.7rem 0 !important;
        transition: transform 0.15s ease, opacity 0.15s ease;
    }
    .stButton > button:hover {
        opacity: 0.92;
        transform: translateY(-1px);
    }
    [data-testid="stMetricValue"] {
        color: #f2f1ee !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Model artifacts (must sit next to app.py)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("KNN_heart.pkl")
    scaler = joblib.load("scaler.pkl")
    expected_columns = joblib.load("columns.pkl")
    return model, scaler, expected_columns


try:
    model, scaler, expected_columns = load_artifacts()
except FileNotFoundError as e:
    st.error(
        "Missing model file. Put these next to `app.py`:\n"
        "`KNN_heart.pkl`, `scaler.pkl`, `columns.pkl`\n\n"
        f"Detail: {e}"
    )
    st.stop()
except Exception as e:
    # Fallback if the column file is named column.pkl (singular)
    try:
        model = joblib.load("KNN_heart.pkl")
        scaler = joblib.load("scaler.pkl")
        expected_columns = joblib.load("column.pkl")
    except Exception:
        st.error(f"Could not load model artifacts: {e}")
        st.stop()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<p class="aorta-kicker">Aorta · risk desk</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="aorta-title">Read the profile.<br><em>Not the diagnosis.</em></p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="aorta-sub">KNN clinical estimator using pain class, ST geometry, '
    "pressure, lipids, and heart-rate reserve. Educational only — not a medical device.</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------
PRESETS = {
    "Baseline": dict(
        age=40,
        sex="M",
        resting_bp=120,
        cholesterol=200,
        fasting_bs=0,
        chest_pain="ATA",
        resting_ecg="Normal",
        st_slope="Up",
        max_hr=150,
        oldpeak=1.0,
        exercise_angina="N",
    ),
    "Athlete": dict(
        age=28,
        sex="F",
        resting_bp=108,
        cholesterol=168,
        fasting_bs=0,
        chest_pain="ATA",
        resting_ecg="Normal",
        st_slope="Up",
        max_hr=186,
        oldpeak=0.2,
        exercise_angina="N",
    ),
    "Hypertensive": dict(
        age=64,
        sex="M",
        resting_bp=162,
        cholesterol=268,
        fasting_bs=1,
        chest_pain="ASY",
        resting_ecg="LVH",
        st_slope="Flat",
        max_hr=112,
        oldpeak=2.4,
        exercise_angina="Y",
    ),
}

preset_name = st.radio(
    "Profile preset",
    list(PRESETS.keys()),
    horizontal=True,
    label_visibility="collapsed",
)
P = PRESETS[preset_name]

# ---------------------------------------------------------------------------
# Intake
# ---------------------------------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<p class="section-label">Subject</p>', unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    age = st.slider("Age", 18, 100, int(P["age"]))
    sex = st.selectbox("Sex", ["M", "F"], index=0 if P["sex"] == "M" else 1)
with c2:
    resting_bp = st.number_input(
        "Resting blood pressure (mm Hg)", 80, 200, int(P["resting_bp"])
    )
    cholesterol = st.number_input(
        "Cholesterol (mg/dL)", 100, 600, int(P["cholesterol"])
    )

st.markdown('<p class="section-label">Cardiac</p>', unsafe_allow_html=True)

c3, c4 = st.columns(2)
with c3:
    chest_pain = st.selectbox(
        "Chest pain type",
        ["ATA", "NAP", "TA", "ASY"],
        index=["ATA", "NAP", "TA", "ASY"].index(P["chest_pain"]),
        help="ATA atypical · NAP non-anginal · TA typical · ASY asymptomatic",
    )
    resting_ecg = st.selectbox(
        "Resting ECG",
        ["Normal", "ST", "LVH"],
        index=["Normal", "ST", "LVH"].index(P["resting_ecg"]),
    )
    st_slope = st.selectbox(
        "ST slope",
        ["Up", "Flat", "Down"],
        index=["Up", "Flat", "Down"].index(P["st_slope"]),
    )
with c4:
    max_hr = st.slider("Max heart rate", 60, 220, int(P["max_hr"]))
    oldpeak = st.slider("Oldpeak (ST depression)", 0.0, 6.0, float(P["oldpeak"]), 0.1)
    exercise_angina = st.selectbox(
        "Exercise-induced angina",
        ["N", "Y"],
        index=0 if P["exercise_angina"] == "N" else 1,
    )

st.markdown('<p class="section-label">Metabolic</p>', unsafe_allow_html=True)
fasting_bs = st.selectbox(
    "Fasting blood sugar > 120 mg/dL",
    [0, 1],
    index=int(P["fasting_bs"]),
    format_func=lambda x: "No (≤ 120)" if x == 0 else "Yes (> 120)",
)
st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
run = st.button("Run analysis", type="primary")

if run:
    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        f"Sex_{sex}": 1,
        f"ChestPainType_{chest_pain}": 1,
        f"RestingECG_{resting_ecg}": 1,
        f"ExerciseAngina_{exercise_angina}": 1,
        f"ST_Slope_{st_slope}": 1,
    }

    input_df = pd.DataFrame([raw_input])
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_columns]

    scaled = scaler.transform(input_df)
    with st.spinner("Running clinical analysis..."):
        time.sleep(0.45)
        prediction = int(model.predict(scaled)[0])

    if prediction == 1:
        st.markdown(
            """
            <div class="result-high">
                <div class="result-title">High risk of heart disease</div>
                <div class="result-sub">Model flag is positive. Discuss with a cardiologist —
                this tool is not a diagnosis.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="result-low">
                <div class="result-title">Low risk of heart disease</div>
                <div class="result-sub">Model flag is negative on this profile. Keep preventive habits;
                this is still not a medical clearance.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    """
    <div class="footer-note">
        Educational use only. Not a substitute for examination, labs, imaging, or emergency care.
        If you have chest pain, pressure, or sudden shortness of breath, seek urgent medical help.
    </div>
    """,
    unsafe_allow_html=True,
)
