from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path("/Users/ayet_dub/Documents/Codex/2026-05-11/files-mentioned-by-the-user-test")
SOURCE_FILES = {
    "train": Path("/Users/ayet_dub/Library/CloudStorage/OneDrive-Personal/Lamina_Documentation/archive/train_motion_data.csv"),
    "test": Path("/Users/ayet_dub/Library/CloudStorage/OneDrive-Personal/Lamina_Documentation/archive/test_motion_data.csv"),
}

DATA_DIR = BASE_DIR / "data"
NOTEBOOK_DIR = BASE_DIR / "notebooks"
DEMO_DIR = BASE_DIR / "demos"
SRC_DIR = BASE_DIR / "src"
LICENSE_DIR = DATA_DIR / "licenses"


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    LICENSE_DIR.mkdir(parents=True, exist_ok=True)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = ["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ", "Timestamp"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=numeric_cols + ["Class"]).copy()
    df["source_split"] = df["source_split"].astype(str)
    df["acc_mag"] = np.sqrt(df["AccX"] ** 2 + df["AccY"] ** 2 + df["AccZ"] ** 2)
    df["gyro_mag"] = np.sqrt(df["GyroX"] ** 2 + df["GyroY"] ** 2 + df["GyroZ"] ** 2)

    df["prev_acc_mag"] = df.groupby("source_split")["acc_mag"].shift(1)
    df["prev_gyro_mag"] = df.groupby("source_split")["gyro_mag"].shift(1)
    df["delta_t"] = df.groupby("source_split")["Timestamp"].diff().fillna(0).clip(lower=0)
    df["jerk_mag"] = (df["acc_mag"] - df["prev_acc_mag"]).abs().fillna(0)
    df["gyro_delta"] = (df["gyro_mag"] - df["prev_gyro_mag"]).abs().fillna(0)

    return df


def load_motion_frames() -> pd.DataFrame:
    frames = []
    for split_name, path in SOURCE_FILES.items():
        frame = pd.read_csv(path)
        frame["source_split"] = split_name
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined = compute_features(combined)
    return combined


def assign_trip_and_driver_ids(df: pd.DataFrame, trip_size: int = 60, driver_count: int = 18) -> pd.DataFrame:
    df = df.sort_values(["source_split", "Timestamp"]).reset_index(drop=True).copy()
    df["row_in_split"] = df.groupby("source_split").cumcount()
    df["trip_seq"] = df.groupby("source_split")["row_in_split"].transform(lambda s: s // trip_size)
    split_prefix = df["source_split"].map({"train": "TRN", "test": "TST"}).fillna("SRC")
    df["trip_id"] = split_prefix + "-TRIP-" + df["trip_seq"].add(1).astype(str).str.zfill(3)

    class_offsets = {"AGGRESSIVE": 0, "NORMAL": 6, "SLOW": 12}
    trip_lookup = (
        df.groupby("trip_id", as_index=False)
        .agg(
            source_split=("source_split", "first"),
            trip_seq=("trip_seq", "first"),
            dominant_class=("Class", lambda s: s.mode().iat[0]),
        )
        .copy()
    )

    trip_lookup["driver_num"] = (
        trip_lookup["trip_seq"]
        + trip_lookup["dominant_class"].map(class_offsets).fillna(0).astype(int)
    ) % driver_count + 1
    trip_lookup["driver_id"] = trip_lookup["driver_num"].map(lambda x: f"D{x:03d}")

    df = df.merge(trip_lookup[["trip_id", "driver_id", "dominant_class"]], on="trip_id", how="left")
    return df


def build_trip_summary(df: pd.DataFrame) -> pd.DataFrame:
    acc_p90 = float(df["acc_mag"].quantile(0.90))
    gyro_p90 = float(df["gyro_mag"].quantile(0.90))
    jerk_p90 = float(df["jerk_mag"].quantile(0.90))

    enriched = df.copy()
    enriched["harsh_accel_flag"] = (enriched["acc_mag"] >= acc_p90).astype(int)
    enriched["sharp_turn_flag"] = (enriched["gyro_mag"] >= gyro_p90).astype(int)
    enriched["sudden_motion_flag"] = (enriched["jerk_mag"] >= jerk_p90).astype(int)
    enriched["is_aggressive"] = (enriched["Class"] == "AGGRESSIVE").astype(int)
    enriched["is_normal"] = (enriched["Class"] == "NORMAL").astype(int)
    enriched["is_slow"] = (enriched["Class"] == "SLOW").astype(int)

    trip_summary = (
        enriched.groupby(["driver_id", "trip_id"], as_index=False)
        .agg(
            source_split=("source_split", "first"),
            trip_label=("dominant_class", "first"),
            start_timestamp=("Timestamp", "min"),
            end_timestamp=("Timestamp", "max"),
            samples=("Timestamp", "size"),
            avg_acc_mag=("acc_mag", "mean"),
            max_acc_mag=("acc_mag", "max"),
            std_acc_mag=("acc_mag", "std"),
            avg_gyro_mag=("gyro_mag", "mean"),
            max_gyro_mag=("gyro_mag", "max"),
            avg_jerk_mag=("jerk_mag", "mean"),
            harsh_accel_events=("harsh_accel_flag", "sum"),
            sharp_turn_events=("sharp_turn_flag", "sum"),
            sudden_motion_events=("sudden_motion_flag", "sum"),
            aggressive_rows=("is_aggressive", "sum"),
            normal_rows=("is_normal", "sum"),
            slow_rows=("is_slow", "sum"),
        )
        .fillna(0)
    )

    trip_summary["duration_ticks"] = trip_summary["end_timestamp"] - trip_summary["start_timestamp"]
    trip_summary["event_total"] = (
        trip_summary["harsh_accel_events"]
        + trip_summary["sharp_turn_events"]
        + trip_summary["sudden_motion_events"]
    )

    class_weight = {"AGGRESSIVE": 1.0, "NORMAL": 0.45, "SLOW": 0.2}
    trip_summary["base_risk"] = trip_summary["trip_label"].map(class_weight).fillna(0.3)
    trip_summary["violation_score"] = (
        trip_summary["base_risk"] * 45
        + trip_summary["harsh_accel_events"] * 0.70
        + trip_summary["sharp_turn_events"] * 0.85
        + trip_summary["sudden_motion_events"] * 0.55
        + trip_summary["max_acc_mag"] * 2.0
        + trip_summary["max_gyro_mag"] * 4.0
    ).round(2)

    trip_summary["driver_rating_proxy"] = (
        5.0
        - (trip_summary["violation_score"] / trip_summary["violation_score"].max()) * 3.7
        + np.where(trip_summary["trip_label"] == "SLOW", 0.2, 0.0)
        + np.where(trip_summary["trip_label"] == "NORMAL", 0.1, -0.1)
    ).clip(1.0, 5.0).round(2)

    trip_summary["feedback_flag"] = np.where(
        trip_summary["driver_rating_proxy"] <= 2.5,
        "Needs coaching",
        np.where(trip_summary["driver_rating_proxy"] <= 3.5, "Monitor", "On track"),
    )
    trip_summary["violation_flag"] = (trip_summary["violation_score"] >= trip_summary["violation_score"].median()).astype(int)
    return trip_summary


def build_driver_profile(trip_summary: pd.DataFrame) -> pd.DataFrame:
    profile = (
        trip_summary.groupby("driver_id", as_index=False)
        .agg(
            trips_completed=("trip_id", "nunique"),
            train_trips=("source_split", lambda s: int((s == "train").sum())),
            test_trips=("source_split", lambda s: int((s == "test").sum())),
            aggressive_trips=("trip_label", lambda s: int((s == "AGGRESSIVE").sum())),
            normal_trips=("trip_label", lambda s: int((s == "NORMAL").sum())),
            slow_trips=("trip_label", lambda s: int((s == "SLOW").sum())),
            avg_driver_rating=("driver_rating_proxy", "mean"),
            avg_violation_score=("violation_score", "mean"),
            max_violation_score=("violation_score", "max"),
            total_harsh_accel_events=("harsh_accel_events", "sum"),
            total_sharp_turn_events=("sharp_turn_events", "sum"),
            total_sudden_motion_events=("sudden_motion_events", "sum"),
            avg_acc_mag=("avg_acc_mag", "mean"),
            avg_gyro_mag=("avg_gyro_mag", "mean"),
        )
        .round(2)
    )

    profile["total_events"] = (
        profile["total_harsh_accel_events"]
        + profile["total_sharp_turn_events"]
        + profile["total_sudden_motion_events"]
    )
    profile["aggressive_trip_pct"] = (profile["aggressive_trips"] / profile["trips_completed"] * 100).round(2)
    profile["risk_level"] = np.select(
        [
            profile["avg_violation_score"] >= profile["avg_violation_score"].quantile(0.75),
            profile["avg_violation_score"] >= profile["avg_violation_score"].quantile(0.40),
        ],
        ["High", "Medium"],
        default="Low",
    )
    profile["coaching_priority"] = np.select(
        [
            profile["avg_driver_rating"] <= 2.5,
            profile["avg_driver_rating"] <= 3.5,
        ],
        ["Immediate", "Planned"],
        default="Routine",
    )
    return profile.sort_values(["avg_violation_score", "driver_id"], ascending=[False, True]).reset_index(drop=True)


def build_driver_ratings_table(trip_summary: pd.DataFrame) -> pd.DataFrame:
    ratings = trip_summary[
        [
            "driver_id",
            "trip_id",
            "source_split",
            "trip_label",
            "end_timestamp",
            "driver_rating_proxy",
            "event_total",
            "violation_score",
            "feedback_flag",
        ]
    ].copy()
    ratings["rating"] = ratings["driver_rating_proxy"].round().clip(1, 5).astype(int)
    ratings["complaints_count"] = np.select(
        [
            ratings["violation_score"] >= ratings["violation_score"].quantile(0.75),
            ratings["violation_score"] >= ratings["violation_score"].quantile(0.45),
        ],
        [2, 1],
        default=0,
    )
    ratings = ratings.rename(columns={"end_timestamp": "timestamp"})
    return ratings


def build_feedback_table(ratings_df: pd.DataFrame) -> pd.DataFrame:
    positive_openers = [
        "Smooth trip overall",
        "Driver was professional",
        "Ride felt safe",
        "Trip was comfortable",
    ]
    neutral_openers = [
        "Trip was acceptable",
        "Driver was okay",
        "Ride was average",
        "Some parts were fine",
    ]
    negative_openers = [
        "Trip felt unsafe",
        "Driver was too aggressive",
        "Ride was uncomfortable",
        "Braking felt too harsh",
    ]
    positive_closers = [
        "I would ride again.",
        "Very steady handling.",
        "No major issues noticed.",
        "The trip felt calm and controlled.",
    ]
    neutral_closers = [
        "There is room for improvement.",
        "A bit more consistency would help.",
        "The ride was not bad but not great.",
        "Some sharper turns were noticeable.",
    ]
    negative_closers = [
        "Please coach this driver.",
        "The driving needs attention.",
        "Frequent sudden motions were worrying.",
        "I noticed risky maneuvering during the trip.",
    ]

    rows = []
    for idx, row in ratings_df.reset_index(drop=True).iterrows():
        if row["rating"] >= 4:
            sentiment_label = "positive"
            opener = positive_openers[idx % len(positive_openers)]
            closer = positive_closers[idx % len(positive_closers)]
            topic = "safe_driving"
        elif row["rating"] == 3:
            sentiment_label = "neutral"
            opener = neutral_openers[idx % len(neutral_openers)]
            closer = neutral_closers[idx % len(neutral_closers)]
            topic = "consistency"
        else:
            sentiment_label = "negative"
            opener = negative_openers[idx % len(negative_openers)]
            closer = negative_closers[idx % len(negative_closers)]
            topic = "aggressive_behavior"

        feedback_text = (
            f"{opener}. "
            f"Trip label was {row['trip_label'].lower()} with violation score {row['violation_score']:.2f}. "
            f"{closer}"
        )
        rows.append(
            {
                "trip_id": row["trip_id"],
                "driver_id": row["driver_id"],
                "timestamp": row["timestamp"],
                "rating": row["rating"],
                "sentiment_label": sentiment_label,
                "complaint_topic": topic,
                "feedback_text": feedback_text,
            }
        )

    return pd.DataFrame(rows)


def build_license_manifest() -> pd.DataFrame:
    records = []
    issue_types = [
        ("genuine", "clean_template"),
        ("forged", "text_tamper"),
        ("forged", "photo_swap"),
        ("forged", "noise_overlay"),
    ]
    for idx in range(1, 13):
        label, issue_type = issue_types[(idx - 1) % len(issue_types)]
        records.append(
            {
                "record_id": f"LIC-{idx:03d}",
                "image_path": f"data/licenses/{label}_{idx:03d}.png",
                "label": label,
                "issue_type": issue_type,
                "expected_name": f"Driver {idx:03d}",
                "license_number": f"LAM{idx:05d}",
            }
        )
    return pd.DataFrame(records)


def write_csvs(
    sensor_df: pd.DataFrame,
    trip_summary: pd.DataFrame,
    driver_profile: pd.DataFrame,
    ratings_df: pd.DataFrame,
    feedback_df: pd.DataFrame,
    license_manifest: pd.DataFrame,
) -> None:
    sensor_df.to_csv(DATA_DIR / "motion_sensor_enriched.csv", index=False)
    trip_summary.to_csv(DATA_DIR / "telematics_trip_summary.csv", index=False)
    driver_profile.to_csv(DATA_DIR / "driver_profile_dashboard.csv", index=False)
    ratings_df.to_csv(DATA_DIR / "driver_ratings_proxy.csv", index=False)
    feedback_df.to_csv(DATA_DIR / "feedback_proxy.csv", index=False)
    license_manifest.to_csv(DATA_DIR / "license_manifest.csv", index=False)


def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip("\n").splitlines()],
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip("\n").splitlines()],
    }


def write_notebook(path: Path, cells: list[dict]) -> None:
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.x",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")


def notebook_data_prep() -> None:
    cells = [
        md_cell(
            """
            # Week 4 - Data Preparation for Smart Driver Monitoring

            This notebook prepares the driving-behavior motion dataset for Week 4 of the internship.

            It focuses on:
            - loading the train and test motion files
            - engineering telematics-style motion features
            - creating pseudo `trip_id` and `driver_id` fields for dashboarding
            - exporting cleaned outputs for later EDA, hypothesis testing, and violation analysis

            ## Important assumption

            The original dataset does **not** contain real `driver_id`, `trip_id`, ratings, or passenger feedback fields.
            To make the Week 4 dashboard possible, this notebook groups sequential sensor rows into fixed-size trips and assigns repeat pseudo drivers for analysis only.
            """
        ),
        code_cell(
            """
            !pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-learn
            """
        ),
        code_cell(
            f"""
            from pathlib import Path
            import numpy as np
            import pandas as pd

            BASE_DIR = Path(r"{BASE_DIR}")
            train_path = Path(r"{SOURCE_FILES['train']}")
            test_path = Path(r"{SOURCE_FILES['test']}")
            """
        ),
        code_cell(
            """
            def load_frame(path, split_name):
                df = pd.read_csv(path).copy()
                df["source_split"] = split_name
                return df

            train_df = load_frame(train_path, "train")
            test_df = load_frame(test_path, "test")
            raw_df = pd.concat([train_df, test_df], ignore_index=True)

            raw_df.head()
            """
        ),
        code_cell(
            """
            numeric_cols = ["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ", "Timestamp"]
            df = raw_df.copy()

            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.dropna(subset=numeric_cols + ["Class"]).copy()
            df["acc_mag"] = np.sqrt(df["AccX"]**2 + df["AccY"]**2 + df["AccZ"]**2)
            df["gyro_mag"] = np.sqrt(df["GyroX"]**2 + df["GyroY"]**2 + df["GyroZ"]**2)
            df["prev_acc_mag"] = df.groupby("source_split")["acc_mag"].shift(1)
            df["jerk_mag"] = (df["acc_mag"] - df["prev_acc_mag"]).abs().fillna(0)

            df.describe().T
            """
        ),
        code_cell(
            """
            trip_size = 60
            driver_count = 18
            class_offsets = {"AGGRESSIVE": 0, "NORMAL": 6, "SLOW": 12}

            df = df.sort_values(["source_split", "Timestamp"]).reset_index(drop=True)
            df["row_in_split"] = df.groupby("source_split").cumcount()
            df["trip_seq"] = df.groupby("source_split")["row_in_split"].transform(lambda s: s // trip_size)
            df["trip_id"] = df["source_split"].str.upper().str[:1] + "-TRIP-" + df["trip_seq"].add(1).astype(str).str.zfill(3)

            trip_lookup = (
                df.groupby("trip_id", as_index=False)
                .agg(
                    source_split=("source_split", "first"),
                    trip_seq=("trip_seq", "first"),
                    dominant_class=("Class", lambda s: s.mode().iat[0]),
                )
            )

            trip_lookup["driver_num"] = (
                trip_lookup["trip_seq"] + trip_lookup["dominant_class"].map(class_offsets).fillna(0).astype(int)
            ) % driver_count + 1
            trip_lookup["driver_id"] = trip_lookup["driver_num"].map(lambda x: f"D{x:03d}")

            df = df.merge(trip_lookup[["trip_id", "driver_id", "dominant_class"]], on="trip_id", how="left")
            df[["driver_id", "trip_id", "dominant_class"]].head()
            """
        ),
        code_cell(
            f"""
            output_path = BASE_DIR / "data" / "motion_sensor_enriched.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            print("Saved:", output_path)
            """
        ),
    ]
    write_notebook(NOTEBOOK_DIR / "01_data_prep.ipynb", cells)


def notebook_eda() -> None:
    cells = [
        md_cell(
            """
            # Week 4 - EDA and Feature Engineering

            This notebook explores the telematics-style motion features derived from the driving-behavior dataset.

            Suggested outputs:
            - class distribution plots
            - acceleration and gyroscope boxplots
            - feature correlation heatmap
            - trip-level severity patterns for dashboard design
            """
        ),
        code_cell(
            """
            !pip install pandas numpy matplotlib seaborn scipy statsmodels
            """
        ),
        code_cell(
            f"""
            from pathlib import Path
            import pandas as pd
            import seaborn as sns
            import matplotlib.pyplot as plt

            BASE_DIR = Path(r"{BASE_DIR}")
            df = pd.read_csv(BASE_DIR / "data" / "motion_sensor_enriched.csv")
            trip_df = pd.read_csv(BASE_DIR / "data" / "telematics_trip_summary.csv")
            driver_df = pd.read_csv(BASE_DIR / "data" / "driver_profile_dashboard.csv")
            """
        ),
        code_cell(
            """
            print(df.shape)
            print(trip_df.shape)
            print(driver_df.shape)
            df.head()
            """
        ),
        code_cell(
            """
            df["Class"].value_counts().plot(kind="bar", color=["#d73027", "#4575b4", "#91bfdb"])
            plt.title("Driving Behavior Class Distribution")
            plt.xlabel("Class")
            plt.ylabel("Sensor Rows")
            plt.show()
            """
        ),
        code_cell(
            """
            sns.boxplot(data=df, x="Class", y="acc_mag", order=["SLOW", "NORMAL", "AGGRESSIVE"])
            plt.title("Acceleration Magnitude by Driving Class")
            plt.show()
            """
        ),
        code_cell(
            """
            sns.boxplot(data=df, x="Class", y="gyro_mag", order=["SLOW", "NORMAL", "AGGRESSIVE"])
            plt.title("Gyroscope Magnitude by Driving Class")
            plt.show()
            """
        ),
        code_cell(
            """
            corr_cols = ["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ", "acc_mag", "gyro_mag", "jerk_mag"]
            plt.figure(figsize=(10, 6))
            sns.heatmap(df[corr_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
            plt.title("Feature Correlation Heatmap")
            plt.show()
            """
        ),
        md_cell(
            """
            ## Dashboard hint

            The `driver_profile_dashboard.csv` file is already shaped for Tableau, Power BI, or Metabase.
            Recommended views:
            - KPI cards: average rating proxy, total events, average violation score
            - bar chart: top high-risk drivers
            - stacked bar: aggressive vs normal vs slow trips per driver
            - scatter plot: `avg_driver_rating` vs `avg_violation_score`
            """
        ),
    ]
    write_notebook(NOTEBOOK_DIR / "02_eda_feature_engineering.ipynb", cells)


def notebook_driver_ratings() -> None:
    cells = [
        md_cell(
            """
            # Week 4 - Driver Ratings Regression

            This notebook follows the Week 4 guide's driver-rating module using the motion dataset.

            ## Modeling note

            The original dataset does not provide real passenger star ratings, so `driver_rating_proxy`
            is used as the regression target. It is derived from trip severity and event counts to support
            the Week 4 analytics workflow.
            """
        ),
        code_cell(
            """
            !pip install pandas numpy matplotlib seaborn scikit-learn
            """
        ),
        code_cell(
            f"""
            from pathlib import Path
            import pandas as pd
            import seaborn as sns
            import matplotlib.pyplot as plt
            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import LinearRegression
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            from sklearn.inspection import permutation_importance

            BASE_DIR = Path(r"{BASE_DIR}")
            df = pd.read_csv(BASE_DIR / "data" / "telematics_trip_summary.csv")
            """
        ),
        code_cell(
            """
            feature_cols = [
                "avg_acc_mag",
                "max_acc_mag",
                "std_acc_mag",
                "avg_gyro_mag",
                "max_gyro_mag",
                "avg_jerk_mag",
                "harsh_accel_events",
                "sharp_turn_events",
                "sudden_motion_events",
                "event_total",
                "samples",
            ]

            X = df[feature_cols]
            y = df["driver_rating_proxy"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            """
        ),
        code_cell(
            """
            linear_model = LinearRegression()
            linear_model.fit(X_train, y_train)
            linear_pred = linear_model.predict(X_test)

            rf_model = RandomForestRegressor(n_estimators=300, random_state=42)
            rf_model.fit(X_train, y_train)
            rf_pred = rf_model.predict(X_test)

            def rmse(y_true, y_pred):
                return mean_squared_error(y_true, y_pred) ** 0.5

            print("Linear Regression")
            print("MAE:", mean_absolute_error(y_test, linear_pred))
            print("RMSE:", rmse(y_test, linear_pred))
            print("R2:", r2_score(y_test, linear_pred))
            print()
            print("Random Forest")
            print("MAE:", mean_absolute_error(y_test, rf_pred))
            print("RMSE:", rmse(y_test, rf_pred))
            print("R2:", r2_score(y_test, rf_pred))
            """
        ),
        code_cell(
            """
            importance = permutation_importance(
                rf_model,
                X_test,
                y_test,
                n_repeats=20,
                random_state=42,
            )

            importance_df = (
                pd.Series(importance.importances_mean, index=feature_cols)
                .sort_values(ascending=False)
            )

            importance_df.plot(kind="bar", figsize=(10, 5), color="#2a9d8f")
            plt.title("Permutation Importance for Driver Rating Prediction")
            plt.ylabel("Importance")
            plt.show()
            """
        ),
        md_cell(
            """
            ## Interpretation

            Use this section to explain:
            - whether telematics features are enough to estimate driver quality
            - which features most influence the predicted rating
            - how the regression output can feed the driver profile dashboard
            """
        ),
    ]
    write_notebook(NOTEBOOK_DIR / "03_driver_ratings.ipynb", cells)


def notebook_feedback_nlp() -> None:
    cells = [
        md_cell(
            """
            # Week 4 - Feedback Analytics (NLP)

            This notebook mirrors the PDF's feedback analytics module using a synthetic feedback table
            derived from trip severity. The text is synthetic because the original motion dataset does not
            contain passenger comments.
            """
        ),
        code_cell(
            """
            !pip install pandas numpy matplotlib seaborn scikit-learn
            """
        ),
        code_cell(
            f"""
            from pathlib import Path
            import pandas as pd
            import seaborn as sns
            import matplotlib.pyplot as plt
            from sklearn.model_selection import train_test_split
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline
            from sklearn.metrics import classification_report, confusion_matrix

            BASE_DIR = Path(r"{BASE_DIR}")
            df = pd.read_csv(BASE_DIR / "data" / "feedback_proxy.csv")
            """
        ),
        code_cell(
            """
            df.head()
            """
        ),
        code_cell(
            """
            print(df["sentiment_label"].value_counts())
            sns.countplot(data=df, x="sentiment_label", order=["negative", "neutral", "positive"])
            plt.title("Feedback Sentiment Distribution")
            plt.show()
            """
        ),
        code_cell(
            """
            X_train, X_test, y_train, y_test = train_test_split(
                df["feedback_text"],
                df["sentiment_label"],
                test_size=0.25,
                random_state=42,
                stratify=df["sentiment_label"],
            )

            model = Pipeline(
                [
                    ("tfidf", TfidfVectorizer(stop_words="english")),
                    ("clf", LogisticRegression(max_iter=1000)),
                ]
            )
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            print(classification_report(y_test, pred))
            """
        ),
        code_cell(
            """
            cm = confusion_matrix(y_test, pred, labels=["negative", "neutral", "positive"])
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Oranges",
                xticklabels=["negative", "neutral", "positive"],
                yticklabels=["negative", "neutral", "positive"],
            )
            plt.title("Feedback Sentiment Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.show()
            """
        ),
        code_cell(
            """
            topic_counts = df["complaint_topic"].value_counts()
            topic_counts.plot(kind="bar", color="#f4a261")
            plt.title("Complaint Topics")
            plt.ylabel("Count")
            plt.show()
            """
        ),
        md_cell(
            """
            ## Interpretation

            In the final write-up, mention that the NLP table is synthetic and was added only because the
            Week 4 guide expects a feedback analysis module alongside the telematics work.
            """
        ),
    ]
    write_notebook(NOTEBOOK_DIR / "04_feedback_nlp.ipynb", cells)


def notebook_hypothesis_testing() -> None:
    cells = [
        md_cell(
            """
            # Week 4 - Hypothesis Testing Using SciPy and StatsModels

            This notebook adapts the hypothesis-testing hands-on exercise to the driving-behavior motion dataset instead of the `tips` sample dataset.

            ## Dataset context

            We use sensor-derived variables such as:
            - `acc_mag` for acceleration intensity
            - `gyro_mag` for turning or rotation intensity
            - `driver_rating_proxy` as a dashboard-friendly rating score derived from trip severity

            ## Questions tested

            1. Is the average acceleration magnitude significantly different from `1.40`?
            2. Do aggressive and slow trips have different average acceleration magnitudes?
            3. Is the average acceleration magnitude different from the average gyroscope magnitude within the same trips?
            4. Do trip-level ratings differ across `SLOW`, `NORMAL`, and `AGGRESSIVE` groups?
            5. Are `risk_level` and `coaching_priority` independent?
            6. What is the 95% confidence interval for the average rating proxy?
            """
        ),
        code_cell(
            """
            !pip install pandas numpy matplotlib seaborn scipy statsmodels
            """
        ),
        code_cell(
            f"""
            from pathlib import Path
            import numpy as np
            import pandas as pd
            import seaborn as sns
            import matplotlib.pyplot as plt
            from scipy import stats
            from scipy.stats import chi2_contingency
            import statsmodels.stats.api as sms

            BASE_DIR = Path(r"{BASE_DIR}")
            sensor_df = pd.read_csv(BASE_DIR / "data" / "motion_sensor_enriched.csv")
            trip_df = pd.read_csv(BASE_DIR / "data" / "telematics_trip_summary.csv")
            driver_df = pd.read_csv(BASE_DIR / "data" / "driver_profile_dashboard.csv")
            """
        ),
        code_cell(
            """
            print(sensor_df.head())
            print(sensor_df[["acc_mag", "gyro_mag", "jerk_mag"]].describe())
            print(trip_df[["driver_rating_proxy", "violation_score"]].describe())
            """
        ),
        code_cell(
            """
            sns.histplot(sensor_df["acc_mag"], kde=True)
            plt.title("Distribution of Acceleration Magnitude")
            plt.show()
            """
        ),
        md_cell(
            """
            ## Part 1: One-sample t-test

            **H0:** The mean acceleration magnitude is equal to `1.40`  
            **H1:** The mean acceleration magnitude is not equal to `1.40`
            """
        ),
        code_cell(
            """
            t_stat, p_val = stats.ttest_1samp(sensor_df["acc_mag"], 1.40)
            print("One-sample t-test")
            print("T-statistic:", t_stat)
            print("P-value:", p_val)
            """
        ),
        md_cell(
            """
            ## Part 2: Two-sample independent t-test

            **H0:** Aggressive and slow trips have the same mean acceleration magnitude  
            **H1:** They have different mean acceleration magnitudes
            """
        ),
        code_cell(
            """
            aggressive = trip_df.loc[trip_df["trip_label"] == "AGGRESSIVE", "avg_acc_mag"]
            slow = trip_df.loc[trip_df["trip_label"] == "SLOW", "avg_acc_mag"]

            t_stat, p_val = stats.ttest_ind(aggressive, slow, equal_var=False)
            print("Independent t-test")
            print("T-statistic:", t_stat)
            print("P-value:", p_val)
            """
        ),
        md_cell(
            """
            ## Part 3: Paired t-test

            We compare `avg_acc_mag` and `avg_gyro_mag` inside the same trips.

            **H0:** The mean paired difference is zero  
            **H1:** The mean paired difference is not zero
            """
        ),
        code_cell(
            """
            t_stat, p_val = stats.ttest_rel(trip_df["avg_acc_mag"], trip_df["avg_gyro_mag"])
            print("Paired t-test")
            print("T-statistic:", t_stat)
            print("P-value:", p_val)
            """
        ),
        md_cell(
            """
            ## Part 4: One-way ANOVA

            **H0:** Average rating proxy is the same across trip classes  
            **H1:** At least one class has a different mean rating proxy
            """
        ),
        code_cell(
            """
            grouped = [group["driver_rating_proxy"].values for _, group in trip_df.groupby("trip_label")]
            f_stat, p_val = stats.f_oneway(*grouped)
            print("ANOVA")
            print("F-statistic:", f_stat)
            print("P-value:", p_val)
            """
        ),
        md_cell(
            """
            ## Part 5: Chi-square test of independence

            **H0:** `risk_level` and `coaching_priority` are independent  
            **H1:** They are associated
            """
        ),
        code_cell(
            """
            contingency = pd.crosstab(driver_df["risk_level"], driver_df["coaching_priority"])
            chi2, p, dof, expected = chi2_contingency(contingency)
            print("Chi-square statistic:", chi2)
            print("P-value:", p)
            print("Degrees of freedom:", dof)
            print("Expected frequencies:\\n", expected)
            """
        ),
        md_cell(
            """
            ## Part 6: StatsModels confidence interval
            """
        ),
        code_cell(
            """
            ci = sms.DescrStatsW(trip_df["driver_rating_proxy"]).tconfint_mean()
            print("95% confidence interval for the mean driver rating proxy:", ci)
            """
        ),
        md_cell(
            """
            ## Short written summary

            After running the notebook, replace this section with:
            - which hypotheses were rejected
            - the meaning of the p-values
            - what the tests imply for driver monitoring decisions
            """
        ),
    ]
    write_notebook(NOTEBOOK_DIR / "04_hypothesis_testing.ipynb", cells)


def notebook_violations() -> None:
    cells = [
        md_cell(
            """
            # Week 4 - Violations Detection

            This notebook translates the motion dataset into a telematics-style violation problem.

            Because the source file does not provide explicit event labels such as `overspeed_flag` or `hard_brake_flag`,
            we derive proxy violation labels from high-intensity motion events and dominant class behavior.

            ## Target

            `violation_flag = 1` when a trip's violation score is at or above the dataset median.
            """
        ),
        code_cell(
            """
            !pip install pandas numpy matplotlib seaborn scikit-learn
            """
        ),
        code_cell(
            f"""
            from pathlib import Path
            import pandas as pd
            import seaborn as sns
            import matplotlib.pyplot as plt
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import classification_report, confusion_matrix

            BASE_DIR = Path(r"{BASE_DIR}")
            trip_df = pd.read_csv(BASE_DIR / "data" / "telematics_trip_summary.csv")
            """
        ),
        code_cell(
            """
            feature_cols = [
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

            X = trip_df[feature_cols]
            y = trip_df["violation_flag"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42, stratify=y
            )
            """
        ),
        code_cell(
            """
            clf = RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight="balanced",
            )
            clf.fit(X_train, y_train)

            pred = clf.predict(X_test)
            print(classification_report(y_test, pred))
            """
        ),
        code_cell(
            """
            cm = confusion_matrix(y_test, pred)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title("Violation Detection Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.show()
            """
        ),
        code_cell(
            """
            importances = (
                pd.Series(clf.feature_importances_, index=feature_cols)
                .sort_values(ascending=False)
            )

            importances.plot(kind="bar", figsize=(10, 5), color="#1f77b4")
            plt.title("Feature Importance for Violation Detection")
            plt.ylabel("Importance")
            plt.show()
            """
        ),
        md_cell(
            """
            ## Interpretation prompt

            In your final submission, explain:
            - which motion features were most useful
            - whether the classifier caught risky trips reliably
            - how this model could feed a driver-monitoring dashboard
            """
        ),
    ]
    write_notebook(NOTEBOOK_DIR / "05_violations_detection.ipynb", cells)


def notebook_forgery_detection() -> None:
    cells = [
        md_cell(
            """
            # Week 4 - Forgery Detection for Documents and Licenses

            This notebook follows the Week 4 forgery-detection guide with a synthetic setup.

            The original driving-behavior dataset does not include license images, so this notebook starts
            from a manifest and generates synthetic examples before showing the OCR and visual-check pipeline.
            """
        ),
        code_cell(
            """
            !pip install pandas numpy pillow opencv-python pytesseract matplotlib
            """
        ),
        code_cell(
            f"""
            from pathlib import Path
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt

            BASE_DIR = Path(r"{BASE_DIR}")
            manifest = pd.read_csv(BASE_DIR / "data" / "license_manifest.csv")
            manifest.head()
            """
        ),
        code_cell(
            """
            # Optional synthetic image generation step.
            # Run this if you want sample license cards to exist inside data/licenses/.
            from PIL import Image, ImageDraw

            out_dir = BASE_DIR / "data" / "licenses"
            out_dir.mkdir(parents=True, exist_ok=True)

            for _, row in manifest.iterrows():
                img = Image.new("RGB", (640, 400), color=(245, 247, 250))
                draw = ImageDraw.Draw(img)
                draw.rectangle((20, 20, 620, 380), outline=(50, 70, 90), width=4)
                draw.text((40, 50), "LAMINA DRIVER LICENSE", fill=(20, 20, 20))
                draw.text((40, 120), f"Name: {row['expected_name']}", fill=(20, 20, 20))
                draw.text((40, 170), f"License #: {row['license_number']}", fill=(20, 20, 20))
                draw.text((40, 220), f"Issue Type: {row['issue_type']}", fill=(20, 20, 20))

                if row["label"] == "forged":
                    draw.rectangle((420, 90, 590, 210), outline=(180, 20, 20), width=6)
                    draw.text((430, 145), "EDIT", fill=(180, 20, 20))

                img.save(BASE_DIR / row["image_path"])
            """
        ),
        code_cell(
            """
            # OCR and forensic workflow outline
            # 1. Load the image
            # 2. Extract text with pytesseract
            # 3. Compare extracted fields with expected formats
            # 4. Score suspicious artifacts such as abrupt edits or mismatched regions

            manifest[["record_id", "image_path", "label", "issue_type"]]
            """
        ),
        md_cell(
            """
            ## Suggested model path

            1. OCR checks with `pytesseract`
            2. Visual anomaly detection with OpenCV
            3. Binary classifier for `genuine` vs `forged`
            4. Final ensemble rule combining OCR mismatch and image-level anomaly score

            This matches the structure described in the Week 4 PDF, while acknowledging that the source
            motion dataset did not ship with real document images.
            """
        ),
    ]
    write_notebook(NOTEBOOK_DIR / "06_forgery_detection.ipynb", cells)


def write_streamlit_app() -> None:
    app_text = f'''from pathlib import Path

import pandas as pd
import streamlit as st


BASE_DIR = Path(r"{BASE_DIR}")
driver_df = pd.read_csv(BASE_DIR / "data" / "driver_profile_dashboard.csv")
trip_df = pd.read_csv(BASE_DIR / "data" / "telematics_trip_summary.csv")

st.set_page_config(page_title="Driver Profile Dashboard", layout="wide")
st.title("Week 4 Driver Profile Dashboard")
st.caption("Built from the driving behavior motion dataset using pseudo trips and pseudo drivers.")

col1, col2, col3 = st.columns(3)
col1.metric("Drivers", int(driver_df["driver_id"].nunique()))
col2.metric("Trips", int(trip_df["trip_id"].nunique()))
col3.metric("Average Rating Proxy", round(driver_df["avg_driver_rating"].mean(), 2))

st.subheader("Driver Risk Overview")
st.dataframe(driver_df, use_container_width=True)

st.subheader("Top Drivers by Violation Score")
st.bar_chart(driver_df.set_index("driver_id")["avg_violation_score"])

st.subheader("Trips by Driver")
trip_mix = driver_df.set_index("driver_id")[["aggressive_trips", "normal_trips", "slow_trips"]]
st.bar_chart(trip_mix)

st.subheader("Ratings vs Violation Score")
scatter_df = driver_df[["driver_id", "avg_driver_rating", "avg_violation_score"]].copy()
st.dataframe(scatter_df, use_container_width=True)
'''
    (DEMO_DIR / "streamlit_app.py").write_text(app_text, encoding="utf-8")


def write_forgery_helper() -> None:
    helper_text = '''from pathlib import Path

import pandas as pd


def load_license_manifest(base_dir: str | Path) -> pd.DataFrame:
    base_path = Path(base_dir)
    return pd.read_csv(base_path / "data" / "license_manifest.csv")


def simple_forgery_score(issue_type: str) -> float:
    weights = {
        "clean_template": 0.05,
        "text_tamper": 0.80,
        "photo_swap": 0.90,
        "noise_overlay": 0.65,
    }
    return weights.get(issue_type, 0.50)
'''
    (SRC_DIR / "forgery_check.py").write_text(helper_text, encoding="utf-8")


def write_readme(sensor_df: pd.DataFrame, driver_profile: pd.DataFrame, trip_summary: pd.DataFrame) -> None:
    readme = f"""# Week 4 Deliverables

This folder contains a starter Week 4 submission built from the driving-behavior motion files:

- `/Users/ayet_dub/Library/CloudStorage/OneDrive-Personal/Lamina_Documentation/archive/train_motion_data.csv`
- `/Users/ayet_dub/Library/CloudStorage/OneDrive-Personal/Lamina_Documentation/archive/test_motion_data.csv`

## What was created

- `data/motion_sensor_enriched.csv`
- `data/telematics_trip_summary.csv`
- `data/driver_profile_dashboard.csv`
- `data/driver_ratings_proxy.csv`
- `data/feedback_proxy.csv`
- `data/license_manifest.csv`
- `notebooks/01_data_prep.ipynb`
- `notebooks/02_eda_feature_engineering.ipynb`
- `notebooks/03_driver_ratings.ipynb`
- `notebooks/04_feedback_nlp.ipynb`
- `notebooks/04_hypothesis_testing.ipynb`
- `notebooks/05_violations_detection.ipynb`
- `notebooks/06_forgery_detection.ipynb`
- `demos/streamlit_app.py`
- `src/forgery_check.py`

## Key assumption

The source dataset does not include real `driver_id`, `trip_id`, ratings, feedback, or forgery labels.
To support the Week 4 dashboard and analytics flow, sequential sensor rows were grouped into fixed-size pseudo trips and assigned repeat pseudo drivers.
The `driver_rating_proxy` column is a dashboard-friendly heuristic based on the trip-level violation score.
The feedback and license datasets are synthetic support tables added so the submission aligns with the Week 4 guide in the PDF.

## Dataset snapshot

- Sensor rows: {len(sensor_df)}
- Trips created: {len(trip_summary)}
- Drivers created: {driver_profile['driver_id'].nunique()}

## Dashboard suggestion

Use `data/driver_profile_dashboard.csv` as the primary dashboard dataset in Tableau, Power BI, or Metabase.
Recommended visuals:

1. KPI cards for average rating proxy, total events, and average violation score.
2. Bar chart of highest-risk drivers by `avg_violation_score`.
3. Stacked bars of `aggressive_trips`, `normal_trips`, and `slow_trips`.
4. Scatter plot of `avg_driver_rating` vs `avg_violation_score`.
5. Table of `risk_level`, `coaching_priority`, and event totals.

## Week 4 module mapping

- Module 2: `notebooks/02_eda_feature_engineering.ipynb`
- Module 3: `notebooks/03_driver_ratings.ipynb`
- Module 4: `notebooks/04_feedback_nlp.ipynb`
- Supplemental stats exercise: `notebooks/04_hypothesis_testing.ipynb`
- Module 5: `notebooks/05_violations_detection.ipynb`
- Module 6: `notebooks/06_forgery_detection.ipynb`
- Module 7 demo: `demos/streamlit_app.py`

## Environment note

The notebooks include `pip install` cells because this local environment does not currently include `scipy`, `statsmodels`, or `scikit-learn`.
Run those cells in Jupyter or Colab before executing the analysis cells.
"""
    (BASE_DIR / "README_week4.md").write_text(readme, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    sensor_df = load_motion_frames()
    sensor_df = assign_trip_and_driver_ids(sensor_df)
    trip_summary = build_trip_summary(sensor_df)
    driver_profile = build_driver_profile(trip_summary)
    ratings_df = build_driver_ratings_table(trip_summary)
    feedback_df = build_feedback_table(ratings_df)
    license_manifest = build_license_manifest()
    write_csvs(sensor_df, trip_summary, driver_profile, ratings_df, feedback_df, license_manifest)
    notebook_data_prep()
    notebook_eda()
    notebook_driver_ratings()
    notebook_feedback_nlp()
    notebook_hypothesis_testing()
    notebook_violations()
    notebook_forgery_detection()
    write_streamlit_app()
    write_forgery_helper()
    write_readme(sensor_df, driver_profile, trip_summary)
    print("Created Week 4 deliverables in", BASE_DIR)


if __name__ == "__main__":
    main()
