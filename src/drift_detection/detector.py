"""
Drift Detection Engine - Statistical tests for detecting data distribution changes
"""
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')


class DriftDetector:
    """Detects drift between reference and production data."""
    
    def __init__(
        self,
        reference_data: pd.DataFrame,
        production_data: pd.DataFrame,
        categorical_features: Optional[List[str]] = None,
        significance_level: float = 0.05,
        psi_threshold: float = 0.25
    ):
        self.reference_data = reference_data
        self.production_data = production_data
        self.significance_level = significance_level
        self.psi_threshold = psi_threshold
        
        if categorical_features is None:
            self.categorical_features = self._detect_categorical_features()
        else:
            self.categorical_features = categorical_features
        
        all_features = set(reference_data.columns)
        self.continuous_features = list(all_features - set(self.categorical_features))
    
    def _detect_categorical_features(self) -> List[str]:
        """Auto-detect categorical features."""
        categorical = []
        for col in self.reference_data.columns:
            if self.reference_data[col].dtype in ['object', 'category']:
                categorical.append(col)
            elif self.reference_data[col].nunique() < 10:
                categorical.append(col)
        return categorical
    
    def ks_test(self, feature: str) -> Dict:
        """Kolmogorov-Smirnov test for continuous features."""
        ref_data = self.reference_data[feature].dropna()
        prod_data = self.production_data[feature].dropna()
        
        statistic, p_value = ks_2samp(ref_data, prod_data)
        
        return {
            'test': 'KS',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'drift_detected': p_value < self.significance_level
        }
    
    def calculate_psi(self, feature: str, bins: int = 10) -> Dict:
        """Population Stability Index."""
        ref_data = self.reference_data[feature].dropna()
        prod_data = self.production_data[feature].dropna()
        
        # Create bins based on reference data
        breakpoints = np.percentile(ref_data, np.linspace(0, 100, bins + 1))
        breakpoints = np.unique(breakpoints)
        
        # Count samples in each bin
        ref_counts = np.histogram(ref_data, bins=breakpoints)[0]
        prod_counts = np.histogram(prod_data, bins=breakpoints)[0]
        
        # Calculate percentages
        ref_percents = ref_counts / len(ref_data)
        prod_percents = prod_counts / len(prod_data)
        
        # Avoid division by zero
        ref_percents = np.where(ref_percents == 0, 0.0001, ref_percents)
        prod_percents = np.where(prod_percents == 0, 0.0001, prod_percents)
        
        # PSI formula
        psi_values = (prod_percents - ref_percents) * np.log(prod_percents / ref_percents)
        psi = np.sum(psi_values)
        
        return {
            'test': 'PSI',
            'psi_value': float(psi),
            'drift_detected': psi >= self.psi_threshold
        }
    
    def chi_square_test(self, feature: str) -> Dict:
        """Chi-square test for categorical features."""
        ref_data = self.reference_data[feature].dropna()
        prod_data = self.production_data[feature].dropna()
        
        # Get all unique categories
        all_categories = set(ref_data.unique()) | set(prod_data.unique())
        
        # Count occurrences
        ref_counts = ref_data.value_counts()
        prod_counts = prod_data.value_counts()
        
        # Create contingency table
        contingency = []
        for cat in all_categories:
            contingency.append([
                ref_counts.get(cat, 0),
                prod_counts.get(cat, 0)
            ])
        
        contingency = np.array(contingency).T
        
        # Chi-square test
        chi2_stat, p_value, _, _ = chi2_contingency(contingency)
        
        return {
            'test': 'Chi-Square',
            'statistic': float(chi2_stat),
            'p_value': float(p_value),
            'drift_detected': p_value < self.significance_level
        }
    
    def detect_drift(self) -> Dict:
        """Run drift detection on all features."""
        results = {
            'drift_detected': False,
            'features_with_drift': [],
            'feature_details': {}
        }
        
        # Test continuous features with both KS and PSI
        for feature in self.continuous_features:
            ks_result = self.ks_test(feature)
            psi_result = self.calculate_psi(feature)
            
            drift = ks_result['drift_detected'] or psi_result['drift_detected']
            
            results['feature_details'][feature] = {
                'type': 'continuous',
                'ks_test': ks_result,
                'psi': psi_result,
                'drift_detected': drift
            }
            
            if drift:
                results['features_with_drift'].append(feature)
                results['drift_detected'] = True
        
        # Test categorical features with chi-square
        for feature in self.categorical_features:
            chi_result = self.chi_square_test(feature)
            
            results['feature_details'][feature] = {
                'type': 'categorical',
                'chi_square': chi_result,
                'drift_detected': chi_result['drift_detected']
            }
            
            if chi_result['drift_detected']:
                results['features_with_drift'].append(feature)
                results['drift_detected'] = True
        
        return results