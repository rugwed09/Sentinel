"""
Generate synthetic credit default dataset for drift detection
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Create data directory
DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

print("Generating synthetic credit default dataset...")

# Number of samples
n_samples = 30000

# Generate features
data = {
    'LIMIT_BAL': np.random.randint(10000, 500000, n_samples),  # Credit limit
    'AGE': np.random.randint(21, 70, n_samples),  # Age
    'PAY_0': np.random.randint(-2, 9, n_samples),  # Payment status
    'PAY_2': np.random.randint(-2, 9, n_samples),
    'PAY_3': np.random.randint(-2, 9, n_samples),
    'BILL_AMT1': np.random.randint(-10000, 400000, n_samples),  # Bill amount
    'BILL_AMT2': np.random.randint(-10000, 400000, n_samples),
    'PAY_AMT1': np.random.randint(0, 100000, n_samples),  # Payment amount
    'PAY_AMT2': np.random.randint(0, 100000, n_samples),
    'default': np.random.binomial(1, 0.22, n_samples)  # 22% default rate
}

df = pd.DataFrame(data)

print(f"✅ Generated dataset with {n_samples} samples")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst few rows:")
print(df.head())

# Save full dataset
df.to_csv(DATA_DIR / "credit_default_full.csv", index=False)
print(f"\n✅ Saved full dataset to {DATA_DIR / 'credit_default_full.csv'}")

# Split into reference and production sets
reference_df = df.sample(frac=0.7, random_state=42)
production_df = df.drop(reference_df.index)

# Save splits
reference_df.to_csv(DATA_DIR / "reference_data.csv", index=False)
production_df.to_csv(DATA_DIR / "production_data.csv", index=False)

print(f"\n✅ Reference set shape: {reference_df.shape}")
print(f"✅ Production set shape: {production_df.shape}")
print("\n🎉 Data preparation complete!")