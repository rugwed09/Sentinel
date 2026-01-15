# Sentinel - ML Model Performance Debugging System

An automated ML observability platform that detects, diagnoses, and explains model performance degradation in production.

## What This Does

Sentinel monitors your ML models in production and automatically:
- Detects when model performance is degrading
- Finds which data segments (slices) are failing
- Identifies root causes (data drift, feature issues, etc.)
- Provides actionable insights to fix problems

## Key Features

- **Drift Detection**: Statistical tests to catch data distribution changes
- **Slice Discovery**: Automatically finds underperforming segments
- **Feature Attribution**: Tracks how feature importance changes over time
- **Root Cause Analysis**: Diagnoses why models fail
- **Real-time Monitoring**: Dashboard with alerts

## Project Structure
```
sentinel/
├── src/              # Core system code
├── tests/            # Unit and integration tests
├── docs/             # Design docs and guides
├── notebooks/        # Exploratory analysis
├── examples/         # Usage examples
└── docker/           # Docker configurations
```

## Setup Instructions

Coming soon...

## Tech Stack

- Python 3.11+
- PostgreSQL
- Redis
- Docker
- FastAPI
- Streamlit

## Author

Built by rugwed09