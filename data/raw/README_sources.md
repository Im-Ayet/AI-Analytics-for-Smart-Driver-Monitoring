# Raw Data Staging

Put downloaded source datasets here before running any future merge or normalization scripts.

Suggested layout:

- `data/raw/dbra24/`
- `data/raw/driving_behavior/`
- `data/raw/traffic_violations/`
- `data/raw/highway_traffic_videos/`
- `data/raw/vehicle_tracking/`

Notes:

- `dbra24/`: main telematics and route anomaly dataset
- `driving_behavior/`: accelerometer and gyroscope behavior labels
- `traffic_violations/`: enforcement and violation records from Data.gov
- `highway_traffic_videos/`: surveillance-style traffic video files
- `vehicle_tracking/`: labeled object-detection and tracking files

This project currently works without these downloads because some support tables are synthetic.
Once the raw datasets are available, the dashboard and notebooks can be upgraded from synthetic proxies to richer real-source joins.
