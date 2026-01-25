# Sentinel - ML Model Performance Debugging System

An automated ML observability platform that detects, diagnoses, and explains model performance degradation in production.

![Sentinel Dashboard](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.11+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## What This Does

Sentinel monitors your ML models in production and automatically:
- ✅ Detects when model performance is degrading
- ✅ Identifies data drift using statistical tests
- ✅ Provides interactive dashboard for visualization
- ✅ Exposes REST API for programmatic access
- 🚧 Finds which data segments (slices) are failing (coming soon)
- 🚧 Diagnoses root causes (coming soon)

## Features

### ✅ Drift Detection Engine
Statistical tests to catch data distribution changes:
- **KS Test** (Kolmogorov-Smirnov) for continuous features
- **PSI** (Population Stability Index) - industry standard
- **Chi-Square test** for categorical features

### ✅ REST API
FastAPI-based endpoint for drift detection:
- `/detect-drift` - Analyze reference vs production data
- Automatic OpenAPI documentation at `/docs`
- JSON request/response format

### ✅ Interactive Dashboard
Streamlit-based web interface:
- Upload CSV files (reference & production data)
- Configurable thresholds (significance level, PSI)
- Visual charts (bar charts, pie charts)
- Detailed results table
- Real-time drift detection

### 🚧 Coming Soon
- Slice Discovery: Find underperforming segments
- Feature Attribution: Track feature importance changes
- Root Cause Analysis: Diagnose model failures
- Automated Alerts: Slack/email notifications

## Project Structure
```
sentinel/
├── src/
│   ├── api/              # FastAPI REST API
│   ├── drift_detection/  # Core drift detection engine
│   └── dashboard.py      # Streamlit dashboard
├── data/raw/             # Sample datasets
├── scripts/              # Data generation & testing
├── tests/                # Unit tests (coming soon)
└── docker/               # Docker configuration
```

## Setup Instructions

### Prerequisites
- Python 3.11+
- Docker Desktop (optional)

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/rugwed09/Sentinel.git
cd Sentinel
```

**2. Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Generate sample data**
```bash
python scripts/download_data.py
```

## Quick Start

### Option 1: Interactive Dashboard
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
streamlit run src/dashboard.py
```

Visit: http://localhost:8501

**Features:**
- Upload your CSV files
- Adjust detection thresholds
- View visualizations
- Export results

### Option 2: REST API
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
uvicorn src.api.main:app --reload
```

Visit API docs: http://127.0.0.1:8000/docs

**Test with curl:**
```bash
curl -X POST "http://127.0.0.1:8000/detect-drift" \
  -H "Content-Type: application/json" \
  -d '{
    "reference_data": [{"age": 25, "income": 50000}],
    "production_data": [{"age": 45, "income": 90000}]
  }'
```

### Option 3: Python Library
```python
from src.drift_detection.detector import DriftDetector
import pandas as pd

# Load data
reference_data = pd.read_csv("data/raw/reference_data.csv")
production_data = pd.read_csv("data/raw/production_data.csv")

# Create detector
detector = DriftDetector(reference_data, production_data)

# Run detection
results = detector.detect_drift()

# Check results
if results['drift_detected']:
    print(f"Drift detected in: {results['features_with_drift']}")
    for feature in results['features_with_drift']:
        details = results['feature_details'][feature]
        print(f"{feature}: PSI = {details.get('psi', {}).get('psi_value', 'N/A')}")
```

## Example Output

**Dashboard View:**
- Visual drift detection with charts
- Feature-by-feature breakdown
- Configurable thresholds

**API Response:**
```json
{
  "drift_detected": true,
  "features_with_drift": ["age", "income"],
  "summary": "⚠️ Drift detected in 2 feature(s): age, income",
  "feature_details": {
    "age": {
      "type": "continuous",
      "psi": {"psi_value": 0.32, "drift_detected": true},
      "ks_test": {"p_value": 0.001, "drift_detected": true}
    }
  }
}
```

## Tech Stack

- **Python 3.14** - Core language
- **FastAPI** - REST API framework
- **Streamlit** - Interactive dashboard
- **Pandas** - Data manipulation
- **SciPy** - Statistical tests
- **Plotly** - Interactive visualizations
- **Docker** - Containerization

## Use Cases

**Banking & FinTech:**
- Monitor credit scoring models for demographic shifts
- Detect fraud pattern changes

**E-commerce:**
- Track recommendation model performance
- Identify seasonal drift in user behavior

**Healthcare:**
- Monitor diagnostic model accuracy over time
- Detect population shifts in patient data

**General ML Ops:**
- Automated model monitoring
- Early warning system for model degradation
- Data quality checks in production

## Current Status

| Feature | Status |
|---------|--------|
| Drift Detection (KS, PSI, Chi-Square) | ✅ Complete |
| REST API Endpoint | ✅ Complete |
| Interactive Dashboard | ✅ Complete |
| Sample Dataset Generation | ✅ Complete |
| Unit Tests | 🚧 In Progress |
| Slice Discovery | 🚧 Planned |
| Root Cause Analysis | 🚧 Planned |
| Cloud Deployment | 🚧 Planned |

## Contributing

This is a portfolio project demonstrating production ML monitoring capabilities. Feedback and suggestions are welcome!

## License

MIT License

## Author

**Rugwed** - [@rugwed09](https://github.com/rugwed09)

## Acknowledgments

- Statistical methods based on industry best practices (banking, fintech)
- Dataset synthetically generated for demonstration

---

**⭐ Star this repo if you find it useful!**
