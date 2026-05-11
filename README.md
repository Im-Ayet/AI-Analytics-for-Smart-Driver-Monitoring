# AI and Analytics for Smart Driver Monitoring

This repository contains a Week 4 to 5 project for smart driver monitoring using Python notebooks, synthetic support datasets, and a small Streamlit demo.

The project covers:
- telemetry data preparation
- EDA and feature engineering
- driver rating prediction
- feedback NLP
- hypothesis testing
- violations detection
- document and license forgery detection
- a lightweight dashboard demo

## Repository Structure

```text
data/
  feedback_proxy.csv
  driver_profile_dashboard.csv
  driver_ratings_proxy.csv
  license_manifest.csv
  module2_driver_features.csv
  module2_trip_features.csv
  motion_sensor_enriched.csv
  telematics_trip_summary.csv
  telemetry_samples.csv
  licenses/
  raw/
demos/
  streamlit_app.py
notebooks/
  01_data_prep.ipynb
  02_eda_feature_engineering.ipynb
  03_driver_ratings.ipynb
  04_feedback_nlp.ipynb
  04_hypothesis_testing.ipynb
  05_violations_detection.ipynb
  06_forgery_detection.ipynb
src/
  forgery_check.py
generate_week4_deliverables.py
MULTI_DATASET_PLAN.md
README_week4.md
```

## Requirements

- Python 3.9+
- Jupyter Notebook or JupyterLab
- Streamlit
- Tesseract OCR installed at the OS level if you want OCR support

Suggested packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy statsmodels pillow opencv-python pytesseract streamlit
```

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv driver-ai
source driver-ai/bin/activate
pip install --upgrade pip
pip install pandas numpy matplotlib seaborn scikit-learn scipy statsmodels pillow opencv-python pytesseract streamlit
```

## How to Run the Notebooks

Start Jupyter:

```bash
jupyter lab
```

Recommended notebook order:

1. `notebooks/01_data_prep.ipynb`
2. `notebooks/02_eda_feature_engineering.ipynb`
3. `notebooks/03_driver_ratings.ipynb`
4. `notebooks/04_feedback_nlp.ipynb`
5. `notebooks/04_hypothesis_testing.ipynb`
6. `notebooks/05_violations_detection.ipynb`
7. `notebooks/06_forgery_detection.ipynb`

## How to Run the Demo

From the repository root:

```bash
streamlit run demos/streamlit_app.py
```

By default the app reads:
- `data/driver_profile_dashboard.csv`
- `data/telematics_trip_summary.csv`

## Key Outputs

- `data/telemetry_samples.csv`: simple synthetic telemetry table for Module 1 and Module 2
- `data/module2_trip_features.csv`: engineered trip-level features
- `data/module2_driver_features.csv`: engineered driver-level features
- `data/driver_profile_dashboard.csv`: dashboard-ready summary table
- `data/licenses/`: generated genuine and forged sample license images

## Notes

- Several datasets in this project are synthetic support datasets created to satisfy the training workflow.
- The notebooks already include written interpretation sections for regression, NLP, hypothesis testing, violations detection, and forgery detection.
- `MULTI_DATASET_PLAN.md` describes how the project can be extended with additional external datasets later.

## Deliverables Included

- Jupyter notebooks for each module
- processed CSV outputs
- synthetic document image samples
- Streamlit demo
- project documentation and run instructions
