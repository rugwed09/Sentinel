"""
Test the drift detector with real data
"""
import pandas as pd
from src.drift_detection.detector import DriftDetector

# Load data
print("Loading data...")
reference_df = pd.read_csv("data/raw/reference_data.csv")
production_df = pd.read_csv("data/raw/production_data.csv")

print(f"Reference data: {reference_df.shape}")
print(f"Production data: {production_df.shape}")

# Create detector
print("\nInitializing drift detector...")
detector = DriftDetector(reference_df, production_df)

print(f"Continuous features: {detector.continuous_features}")
print(f"Categorical features: {detector.categorical_features}")

# Run detection
print("\nRunning drift detection...")
results = detector.detect_drift()

# Print results
print("\n" + "="*50)
print("DRIFT DETECTION RESULTS")
print("="*50)
print(f"\nOverall drift detected: {results['drift_detected']}")
print(f"Features with drift: {results['features_with_drift']}")

print("\nDetailed results:")
for feature, details in results['feature_details'].items():
    print(f"\n{feature} ({details['type']}):")
    if details['type'] == 'continuous':
        print(f"  KS Test p-value: {details['ks_test']['p_value']:.4f}")
        print(f"  PSI: {details['psi']['psi_value']:.4f}")
    else:
        print(f"  Chi-Square p-value: {details['chi_square']['p_value']:.4f}")
    print(f"  Drift: {details['drift_detected']}")