from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageOps
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
# pyrefly: ignore [missing-import]
import streamlit as st

try:
    import pytesseract
except ImportError:  # pragma: no cover - optional dependency
    pytesseract = None


BASE_DIR = Path(__file__).resolve().parents[1]
TELEMETRY_FEATURES = [
    "avg_acc_mag",
    "max_acc_mag",
    "std_acc_mag",
    "avg_gyro_mag",
    "max_gyro_mag",
    "avg_jerk_mag",
    "harsh_accel_events",
    "sharp_turn_events",
    "sudden_motion_events",
    "samples",
    "duration_ticks",
]


@st.cache_data
def load_base_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    driver_df = pd.read_csv(BASE_DIR / "data" / "driver_profile_dashboard.csv")
    trip_df = pd.read_csv(BASE_DIR / "data" / "telematics_trip_summary.csv")
    feedback_df = pd.read_csv(BASE_DIR / "data" / "feedback_proxy.csv")
    manifest_df = pd.read_csv(BASE_DIR / "data" / "license_manifest.csv")
    return driver_df, trip_df, feedback_df, manifest_df


@st.cache_resource
def train_models() -> tuple[RandomForestRegressor, RandomForestClassifier, Pipeline]:
    _, trip_df, feedback_df, _ = load_base_data()

    rating_model = RandomForestRegressor(n_estimators=300, random_state=42)
    rating_model.fit(trip_df[TELEMETRY_FEATURES], trip_df["driver_rating_proxy"])

    violation_model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
    )
    violation_model.fit(trip_df[TELEMETRY_FEATURES], trip_df["violation_flag"])

    sentiment_model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english")),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    sentiment_model.fit(feedback_df["feedback_text"], feedback_df["sentiment_label"])

    return rating_model, violation_model, sentiment_model


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #f3f7f4;
            --surface: #ffffff;
            --surface-strong: #ecf3ee;
            --ink: #10231b;
            --muted: #53685d;
            --line: #cbd9d0;
            --green: #157347;
            --green-soft: #d9efe4;
            --amber: #a15c00;
            --amber-soft: #fff0cf;
            --red: #b42318;
            --red-soft: #ffe1dd;
            --blue: #0b5cad;
            --blue-soft: #dcecff;
        }
        .stApp {
            background:
                linear-gradient(180deg, rgba(255,255,255,0.68), rgba(243,247,244,0.92)),
                radial-gradient(circle at top left, rgba(21,115,71,0.13), transparent 34rem),
                var(--bg);
            color: var(--ink);
        }
        .stApp,
        .stApp p,
        .stApp li,
        .stApp label,
        .stApp span,
        .stApp div {
            color: var(--ink);
        }
        .block-container {
            max-width: 1280px;
            padding-top: 0.8rem;
            padding-bottom: 1rem;
        }
        [data-testid="stHeader"] {
            display: none;
        }
        #MainMenu,
        footer {
            visibility: hidden;
        }
        h1, h2, h3 {
            color: var(--ink);
            letter-spacing: 0;
        }
        h1 {
            font-size: 1.78rem !important;
            margin-bottom: 0.1rem !important;
        }
        h2, h3 {
            font-size: 1.05rem !important;
        }
        [data-testid="stSidebar"] {
            background: #10231b;
            border-right: 1px solid rgba(255,255,255,0.08);
        }
        [data-testid="stSidebar"] * {
            color: #f5fff8 !important;
        }
        [data-testid="stSidebar"] [role="radiogroup"] label {
            border: 1px solid rgba(255,255,255,0.18);
            border-radius: 8px;
            padding: 0.55rem 0.65rem;
            margin-bottom: 0.35rem;
            background: rgba(255,255,255,0.06);
        }
        [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
            background: #d9efe4;
            border-color: #d9efe4;
        }
        [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) * {
            color: #10231b !important;
        }
        [data-testid="stMetric"] {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.75rem 0.8rem;
            box-shadow: 0 10px 24px rgba(16, 35, 27, 0.07);
        }
        [data-testid="stMetricLabel"] {
            color: var(--muted) !important;
        }
        [data-testid="stMetricLabel"] p {
            color: var(--muted) !important;
            font-weight: 600;
            font-size: 0.82rem;
        }
        [data-testid="stMetricValue"] {
            color: var(--ink) !important;
        }
        [data-testid="stMetricValue"] div {
            color: var(--ink) !important;
            font-size: 1.65rem !important;
        }
        [data-testid="stCaptionContainer"] {
            color: var(--muted) !important;
        }
        [data-testid="stCaptionContainer"] p {
            color: var(--muted) !important;
        }
        [data-testid="stDataFrame"] {
            background: var(--surface);
            border-radius: 8px;
            padding: 0.15rem;
            border: 1px solid var(--line);
            box-shadow: 0 10px 24px rgba(16, 35, 27, 0.07);
        }
        [data-testid="stFileUploader"] section,
        [data-testid="stTextArea"] textarea,
        [data-testid="stTextInput"] input,
        [data-baseweb="select"] > div,
        [data-baseweb="base-input"] > div {
            background: var(--surface) !important;
            color: var(--ink) !important;
            border-color: var(--line) !important;
        }
        [data-testid="stFileUploader"] label,
        [data-testid="stTextArea"] label,
        [data-testid="stSelectbox"] label,
        [data-testid="stFileUploaderDropzoneInstructions"] span,
        [data-testid="stFileUploaderDropzone"] span {
            color: var(--ink) !important;
        }
        [data-testid="stTextArea"] textarea::placeholder,
        [data-testid="stTextInput"] input::placeholder {
            color: var(--muted) !important;
            opacity: 1 !important;
        }
        [data-testid="stButton"] button,
        [data-testid="baseButton-secondary"],
        [data-testid="stBaseButton-secondary"],
        button[kind="secondary"] {
            background: var(--green) !important;
            color: #ffffff !important;
            border: 1px solid var(--green) !important;
            border-radius: 8px !important;
            font-weight: 700 !important;
        }
        [data-testid="stButton"] button *,
        [data-testid="stFileUploader"] button *,
        [data-testid="stBaseButton-secondary"] * {
            color: #ffffff !important;
        }
        [data-testid="stAlert"] {
            color: var(--ink) !important;
        }
        .panel {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.85rem 1rem;
            box-shadow: 0 10px 24px rgba(16, 35, 27, 0.07);
            margin-bottom: 0.75rem;
        }
        .panel-title {
            color: var(--ink);
            font-size: 0.94rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }
        .panel-copy {
            color: var(--muted);
            line-height: 1.45;
            margin-bottom: 0.75rem;
        }
        .feature-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.35rem 0.55rem;
        }
        .feature-chip {
            background: var(--surface-strong);
            border: 1px solid var(--line);
            border-radius: 8px;
            color: var(--ink);
            font-size: 0.82rem;
            padding: 0.38rem 0.5rem;
        }
        .dashboard-header {
            display: flex;
            align-items: end;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 0.7rem;
        }
        .dashboard-kicker {
            color: var(--green);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.2rem;
        }
        .dashboard-copy {
            color: var(--muted);
            font-size: 0.9rem;
            margin: 0;
        }
        .status-pill {
            border-radius: 999px;
            border: 1px solid var(--line);
            background: var(--surface);
            color: var(--ink);
            font-size: 0.8rem;
            font-weight: 800;
            padding: 0.4rem 0.7rem;
            white-space: nowrap;
        }
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.75rem;
        }
        .kpi-card {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.78rem 0.85rem;
            box-shadow: 0 10px 24px rgba(16, 35, 27, 0.07);
        }
        .kpi-label {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
        }
        .kpi-value {
            color: var(--ink);
            font-size: 1.55rem;
            font-weight: 900;
        }
        .overview-grid {
            display: grid;
            grid-template-columns: 1.12fr 0.88fr;
            gap: 0.75rem;
            margin-top: 0.75rem;
        }
        .mini-card {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.75rem;
            box-shadow: 0 10px 24px rgba(16, 35, 27, 0.07);
        }
        .mini-card h3 {
            font-size: 0.95rem !important;
            margin: 0 0 0.55rem 0 !important;
        }
        .risk-row {
            display: grid;
            grid-template-columns: 4.5rem 1fr 2.2rem;
            align-items: center;
            gap: 0.6rem;
            margin: 0.38rem 0;
            font-size: 0.84rem;
            font-weight: 700;
        }
        .bar-track {
            height: 0.55rem;
            background: #e7eee9;
            border-radius: 999px;
            overflow: hidden;
        }
        .bar-fill {
            height: 100%;
            border-radius: 999px;
        }
        .bar-high { background: var(--red); }
        .bar-medium { background: var(--amber); }
        .bar-low { background: var(--green); }
        .driver-row {
            display: grid;
            grid-template-columns: 3rem 1fr auto;
            align-items: center;
            gap: 0.55rem;
            padding: 0.42rem 0;
            border-bottom: 1px solid #e5eee8;
            font-size: 0.84rem;
        }
        .driver-row:last-child {
            border-bottom: 0;
        }
        .driver-id {
            font-weight: 800;
        }
        .score {
            font-weight: 800;
            color: var(--red);
        }
        .chip {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 999px;
            font-size: 0.73rem;
            font-weight: 800;
            padding: 0.18rem 0.5rem;
            min-width: 4rem;
        }
        .chip-high {
            background: var(--red-soft);
            color: var(--red);
        }
        .chip-medium {
            background: var(--amber-soft);
            color: var(--amber);
        }
        .chip-low {
            background: var(--green-soft);
            color: var(--green);
        }
        .event-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.5rem;
        }
        .event-tile {
            border-radius: 8px;
            padding: 0.6rem;
            border: 1px solid var(--line);
            background: var(--surface-strong);
        }
        .event-label {
            color: var(--muted);
            font-size: 0.72rem;
            font-weight: 800;
            text-transform: uppercase;
        }
        .event-value {
            color: var(--ink);
            font-size: 1.15rem;
            font-weight: 900;
        }
        @media (max-width: 900px) {
            .dashboard-header,
            .overview-grid {
                display: block;
            }
            .kpi-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .mini-card {
                margin-bottom: 0.75rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def validate_telemetry_columns(df: pd.DataFrame) -> tuple[bool, list[str]]:
    missing = [column for column in TELEMETRY_FEATURES if column not in df.columns]
    return not missing, missing


def prepare_telemetry_predictions(df: pd.DataFrame) -> pd.DataFrame:
    rating_model, violation_model, _ = train_models()
    scored = df.copy()
    scored["predicted_rating"] = rating_model.predict(scored[TELEMETRY_FEATURES]).round(2)
    scored["violation_probability"] = violation_model.predict_proba(scored[TELEMETRY_FEATURES])[:, 1].round(3)
    scored["predicted_violation_flag"] = violation_model.predict(scored[TELEMETRY_FEATURES]).astype(int)
    scored["risk_band"] = np.where(
        scored["violation_probability"] >= 0.70,
        "High",
        np.where(scored["violation_probability"] >= 0.40, "Medium", "Low"),
    )
    return scored


def classify_feedback(text: str) -> tuple[str, pd.DataFrame]:
    _, _, sentiment_model = train_models()
    label = sentiment_model.predict([text])[0]
    probabilities = sentiment_model.predict_proba([text])[0]
    prob_df = pd.DataFrame(
        {
            "sentiment": sentiment_model.classes_,
            "probability": probabilities,
        }
    ).sort_values("probability", ascending=False)
    return label, prob_df


def extract_ocr_text(image: Image.Image) -> tuple[str, bool]:
    if pytesseract is None:
        return "pytesseract is not installed in the active environment.", False
    text = pytesseract.image_to_string(image)
    cleaned = text.strip() or "No OCR text detected."
    return cleaned, True


def compute_forgery_score(image: Image.Image) -> tuple[float, list[str]]:
    gray = ImageOps.grayscale(image)
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_array = np.asarray(edges, dtype=float)
    gray_array = np.asarray(gray, dtype=float)

    edge_density = float((edge_array > 40).mean())
    intensity_std = float(gray_array.std() / 128.0)

    score = min(1.0, 0.65 * edge_density + 0.35 * min(intensity_std, 1.0))
    flags: list[str] = []
    if edge_density >= 0.22:
        flags.append("Dense edge pattern detected")
    if intensity_std >= 0.45:
        flags.append("High texture or noise variation detected")
    if not flags:
        flags.append("No strong anomaly flag detected")
    return round(score, 3), flags


def risk_chip_class(risk_level: str) -> str:
    normalized = risk_level.lower()
    if "high" in normalized:
        return "chip-high"
    if "medium" in normalized:
        return "chip-medium"
    return "chip-low"


def render_overview(driver_df: pd.DataFrame, trip_df: pd.DataFrame) -> None:
    driver_count = int(driver_df["driver_id"].nunique())
    trip_count = int(trip_df["trip_id"].nunique())
    avg_rating = round(float(driver_df["avg_driver_rating"].mean()), 2)
    flagged_trips = int(trip_df["violation_flag"].sum())
    high_risk_count = int((driver_df["risk_level"].str.lower() == "high").sum())
    avg_violation = round(float(driver_df["avg_violation_score"].mean()), 1)

    st.html(
        f"""
        <div class="dashboard-header">
            <div>
                <div class="dashboard-kicker">Smart Driver Monitoring</div>
                <h1>Operations Overview</h1>
                <p class="dashboard-copy">
                    {driver_count} drivers, {trip_count} trips, {flagged_trips} flagged trips, and {high_risk_count} high-risk drivers.
                </p>
            </div>
            <div class="status-pill">Average violation score: {avg_violation}</div>
        </div>
        """
    )

    top_risk = (
        driver_df.sort_values("avg_violation_score", ascending=False)
        .head(5)[["driver_id", "avg_violation_score", "risk_level"]]
        .copy()
    )
    driver_rows = "\n".join(
        f"""
        <div class="driver-row">
            <div class="driver-id">{row.driver_id}</div>
            <div><span class="chip {risk_chip_class(row.risk_level)}">{row.risk_level}</span></div>
            <div class="score">{row.avg_violation_score:.1f}</div>
        </div>
        """
        for row in top_risk.itertuples(index=False)
    )

    risk_counts = driver_df["risk_level"].value_counts()
    max_risk_count = max(int(risk_counts.max()), 1)
    risk_order = [("High", "bar-high"), ("Medium", "bar-medium"), ("Low", "bar-low")]
    risk_rows = "\n".join(
        f"""
        <div class="risk-row">
            <div>{label}</div>
            <div class="bar-track"><div class="bar-fill {bar_class}" style="width: {(int(risk_counts.get(label, 0)) / max_risk_count) * 100:.0f}%"></div></div>
            <div>{int(risk_counts.get(label, 0))}</div>
        </div>
        """
        for label, bar_class in risk_order
    )

    total_harsh = int(driver_df["total_harsh_accel_events"].sum())
    total_turns = int(driver_df["total_sharp_turn_events"].sum())
    total_motion = int(driver_df["total_sudden_motion_events"].sum())
    class_totals = {
        "Aggressive": int(driver_df["aggressive_trips"].sum()),
        "Normal": int(driver_df["normal_trips"].sum()),
        "Slow": int(driver_df["slow_trips"].sum()),
    }
    max_class_count = max(class_totals.values())
    class_rows = "\n".join(
        f"""
        <div class="risk-row">
            <div>{label}</div>
            <div class="bar-track"><div class="bar-fill {bar_class}" style="width: {(count / max_class_count) * 100:.0f}%"></div></div>
            <div>{count}</div>
        </div>
        """
        for (label, count), bar_class in zip(
            class_totals.items(),
            ["bar-high", "bar-low", "bar-medium"],
        )
    )

    st.html(
        f"""
        <div class="kpi-grid">
            <div class="kpi-card"><div class="kpi-label">Drivers</div><div class="kpi-value">{driver_count}</div></div>
            <div class="kpi-card"><div class="kpi-label">Trips</div><div class="kpi-value">{trip_count}</div></div>
            <div class="kpi-card"><div class="kpi-label">Avg Rating</div><div class="kpi-value">{avg_rating}</div></div>
            <div class="kpi-card"><div class="kpi-label">Flagged Trips</div><div class="kpi-value">{flagged_trips}</div></div>
        </div>
        <div class="overview-grid">
            <div class="mini-card">
                <h3>Highest Violation Drivers</h3>
                {driver_rows}
            </div>
            <div class="mini-card">
                <h3>Risk Level Mix</h3>
                {risk_rows}
            </div>
            <div class="mini-card">
                <h3>Event Totals</h3>
                <div class="event-grid">
                    <div class="event-tile">
                        <div class="event-label">Harsh Accel</div>
                        <div class="event-value">{total_harsh}</div>
                    </div>
                    <div class="event-tile">
                        <div class="event-label">Sharp Turns</div>
                        <div class="event-value">{total_turns}</div>
                    </div>
                    <div class="event-tile">
                        <div class="event-label">Sudden Motion</div>
                        <div class="event-value">{total_motion}</div>
                    </div>
                </div>
            </div>
            <div class="mini-card">
                <h3>Driver Class Mix</h3>
                {class_rows}
            </div>
        </div>
        """
    )


def render_telemetry_tab() -> None:
    st.subheader("Telemetry Upload and Scoring")
    st.markdown(
        """
        <div class="panel">
            <div class="panel-title">Upload a Trip-Level Telemetry File</div>
            <div class="panel-copy">
                Use a CSV that already contains the engineered trip features used by the rating and
                violation models. After upload, the app will score each row with a predicted driver
                rating, violation probability, and a trip risk band.
            </div>
            <div class="feature-grid">
                <div class="feature-chip">avg_acc_mag</div>
                <div class="feature-chip">max_acc_mag</div>
                <div class="feature-chip">std_acc_mag</div>
                <div class="feature-chip">avg_gyro_mag</div>
                <div class="feature-chip">max_gyro_mag</div>
                <div class="feature-chip">avg_jerk_mag</div>
                <div class="feature-chip">harsh_accel_events</div>
                <div class="feature-chip">sharp_turn_events</div>
                <div class="feature-chip">sudden_motion_events</div>
                <div class="feature-chip">samples</div>
                <div class="feature-chip">duration_ticks</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("Upload telemetry CSV", type=["csv"], key="telemetry_csv")
    if uploaded_file is None:
        return

    telemetry_df = pd.read_csv(uploaded_file)
    is_valid, missing = validate_telemetry_columns(telemetry_df)
    if not is_valid:
        st.error(f"Missing required columns: {', '.join(missing)}")
        return

    scored_df = prepare_telemetry_predictions(telemetry_df)
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows Scored", len(scored_df))
    col2.metric("Average Predicted Rating", round(scored_df["predicted_rating"].mean(), 2))
    col3.metric("Trips Flagged", int(scored_df["predicted_violation_flag"].sum()))

    st.subheader("Scored Telemetry Output")
    st.dataframe(scored_df, width="stretch")


def render_feedback_tab(feedback_df: pd.DataFrame) -> None:
    st.subheader("Feedback Sentiment Classifier")
    sample_text = st.selectbox(
        "Load a sample feedback comment",
        [""] + feedback_df["feedback_text"].head(20).tolist(),
    )
    user_text = st.text_area(
        "Enter passenger feedback",
        value=sample_text,
        placeholder="Example: Driver was rude and the braking felt too harsh.",
        height=140,
    )

    if not user_text.strip():
        return

    label, prob_df = classify_feedback(user_text)
    st.metric("Predicted Sentiment", label.title())
    st.subheader("Class Probabilities")
    st.dataframe(prob_df, width="stretch")


def render_forgery_tab(manifest_df: pd.DataFrame) -> None:
    st.subheader("License Forgery Check and OCR")
    st.markdown(
        """
        <div class="panel">
        Upload a license image to run OCR and a simple forgery heuristic. The score is intended as a screening signal,
        not a final decision.
        </div>
        """,
        unsafe_allow_html=True,
    )

    sample_id = st.selectbox("View a sample manifest record", [""] + manifest_df["record_id"].tolist())
    if sample_id:
        row = manifest_df.loc[manifest_df["record_id"] == sample_id].iloc[0]
        sample_path = BASE_DIR / row["image_path"]
        st.image(str(sample_path), caption=f"{row['record_id']} - {row['label']} ({row['issue_type']})", width=420)

    uploaded_image = st.file_uploader("Upload a license image", type=["png", "jpg", "jpeg"], key="license_image")
    if uploaded_image is None:
        return

    image = Image.open(BytesIO(uploaded_image.getvalue())).convert("RGB")
    forgery_score, flags = compute_forgery_score(image)
    ocr_text, ocr_available = extract_ocr_text(image)
    predicted_label = "forged" if forgery_score >= 0.50 else "genuine"

    left, right = st.columns([1, 1])
    with left:
        st.image(image, caption="Uploaded license image", width=420)
    with right:
        st.metric("Predicted Label", predicted_label.title())
        st.metric("Forgery Score", forgery_score)
        st.write("Flags:")
        for flag in flags:
            st.write(f"- {flag}")
        if not ocr_available:
            st.warning("OCR is unavailable in the current environment.")

    st.subheader("OCR Output")
    st.text_area("Extracted text", value=ocr_text, height=220)


def main() -> None:
    st.set_page_config(page_title="Driver Monitoring Demo", layout="wide")
    inject_styles()

    driver_df, trip_df, feedback_df, manifest_df = load_base_data()
    st.sidebar.title("Driver Monitor")
    page = st.sidebar.radio(
        "Navigation",
        ["Overview", "Telemetry Scoring", "Feedback NLP", "License Check"],
    )

    if page == "Overview":
        render_overview(driver_df, trip_df)
    elif page == "Telemetry Scoring":
        render_telemetry_tab()
    elif page == "Feedback NLP":
        render_feedback_tab(feedback_df)
    else:
        render_forgery_tab(manifest_df)


if __name__ == "__main__":
    main()
