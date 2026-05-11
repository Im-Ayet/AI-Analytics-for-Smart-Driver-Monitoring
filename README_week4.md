# Week 4 Deliverables

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

- Sensor rows: 6728
- Trips created: 113
- Drivers created: 18

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
