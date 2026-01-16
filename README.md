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
  - KS Test for continuous features
  - PSI (Population Stability Index) 
  - Chi-Square test for categorical features
- **Slice Discovery**: Automatically finds underperforming segments (coming soon)
- **Feature Attribution**: Tracks how feature importance changes over time (coming soon)
- **Root Cause Analysis**: Diagnoses why models fail (coming soon)
- **Real-time Monitoring**: Dashboard with alerts (coming soon)

## Project Structure
```
sentinel/
├── src/              # Core system code
│   ├── api/          # FastAPI application
│   └── drift_detection/  # Drift detection engine
├── tests/            # Unit and integration tests
├── data/             # Sample datasets
├── scripts/          # Utility scripts
├── docs/             # Documentation
└── docker/           # Docker configurations
```

## Setup Instructions

### Prerequisites

- Python 3.11+
- Docker Desktop (optional, for running full stack)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/rugwed09/Sentinel.git
cd Sentinel
```

2. **Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Generate sample data**
```bash
python scripts/download_data.py
```

### Quick Start

**Test the drift detector:**
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python scripts/test_drift_detector.py
```

**Run the API:**
```bash
uvicorn src.api.main:app --reload
```

Then visit: http://127.0.0.1:8000/docs

### Using Docker (Optional)
```bash
docker-compose up
```

Access API at: http://localhost:8000

## Usage Example
```python
from src.drift_detection.detector import DriftDetector
import pandas as pd

# Load your data
reference_data = pd.read_csv("data/raw/reference_data.csv")
production_data = pd.read_csv("data/raw/production_data.csv")

# Create detector
detector = DriftDetector(reference_data, production_data)

# Run drift detection
results = detector.detect_drift()

# Check results
if results['drift_detected']:
    print(f"Drift detected in: {results['features_with_drift']}")
```

## Tech Stack

- **Python 3.14**
- **FastAPI** - REST API framework
- **Pandas** - Data manipulation
- **SciPy** - Statistical tests
- **PostgreSQL** - Metadata storage (coming soon)
- **Redis** - Caching (coming soon)
- **Docker** - Containerization
- **Streamlit** - Dashboard (coming soon)

## Current Status

✅ Drift detection engine (KS test, PSI, Chi-Square)  
✅ Basic API structure  
✅ Sample dataset generation  
🚧 API endpoint for drift detection  
🚧 Dashboard visualization  
🚧 Slice discovery  
🚧 Root cause analysis  
🚧 Real-time monitoring  

## Roadmap

- [ ] Add drift detection API endpoint
- [ ] Build monitoring dashboard
- [ ] Implement slice discovery algorithm
- [ ] Add feature attribution tracking
- [ ] Create root cause analysis system
- [ ] Add unit tests
- [ ] Write comprehensive documentation
- [ ] Deploy to cloud

## Contributing

This is a portfolio project. Feedback and suggestions welcome!

## License

MIT

## Author

Built by [@rugwed09](https://github.com/rugwed09)

## Acknowledgments

- Dataset generated synthetically for demonstration purposes
