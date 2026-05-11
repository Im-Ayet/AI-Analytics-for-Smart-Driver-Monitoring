# Week 4 Multi-Dataset Plan

This project will use different datasets for different Week 4 modules instead of expecting one dataset to cover everything.

## Selected Sources

### 1. Telematics and driver behavior

- Primary: `Driver behavior and route anomaly Dataset (DBRA24)`
- Support: `Driving Behavior`
- Optional support: `Aggressive Driving Behavior IoT Data`

Use these for:
- trip-level telematics
- driver behavior patterns
- route anomaly features
- driver profile dashboard metrics

Expected outputs:
- `data/raw/dbra24/`
- `data/raw/driving_behavior/`
- `data/processed/telematics_master.csv`
- `data/processed/trip_features.csv`
- `data/processed/driver_profile_dashboard.csv`

### 2. Violations

- Primary: `Traffic Violations` from Data.gov

Use this for:
- real violation categories
- frequency of violations
- enrichment of driver risk and compliance reporting

Expected outputs:
- `data/raw/traffic_violations/`
- `data/processed/violations_master.csv`

### 3. Video and computer vision

- Primary: `Highway Traffic Videos Dataset`
- Primary: `Vehicle Tracking Dataset`

Use these for:
- vehicle detection and tracking
- traffic scene analytics
- future CV expansion for violations or route-risk monitoring

Expected outputs:
- `data/raw/highway_traffic_videos/`
- `data/raw/vehicle_tracking/`

### 4. Ratings and feedback

These are still synthetic unless a separate customer-review dataset is added.

Current project approach:
- generate `driver_ratings_proxy.csv`
- generate `feedback_proxy.csv`

Reason:
- the selected driving and violation datasets do not provide passenger text feedback or a direct 1-to-5 driver rating table

### Forgery detection

Current project approach:
- synthetic `license_manifest.csv`
- synthetic `data/licenses/` image set to be generated locally

Reason:
- the selected driving and tracking datasets do not provide license or document images

## Recommended Module Mapping

### Data collection and synthetic data

- Download DBRA24
- Download Driving Behavior
- Download Traffic Violations
- Download Highway Traffic Videos
- Download Vehicle Tracking
- Generate synthetic ratings, feedback, and license images locally

### EDA and feature engineering

- Use DBRA24 as the main trip dataset
- Use Driving Behavior as motion-sensor support

### Driver ratings

- Use engineered telematics features
- Predict synthetic rating proxy or delivery-style rating if a new ratings dataset is later added

### Feedback analytics

- Use synthetic feedback table for now

### Violations detection

- Use Driving Behavior and DBRA24 features
- Optionally enrich labels or categories from Traffic Violations

### Forgery detection

- Use synthetic license images until a real document dataset is added

### Integration and dashboard

- Use the processed dashboard table as the Tableau, Power BI, or Metabase source

## Current Status

- Complete: starter notebooks, dashboard table, synthetic ratings table, synthetic feedback table, forgery scaffold
- Pending: raw dataset downloads into `data/raw/`
- Pending: normalized merge scripts for DBRA24 and traffic violations
- Pending: local synthetic license image generation
