"""
Streamlit Dashboard for Drift Detection
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from src.drift_detection.detector import DriftDetector

st.set_page_config(page_title="Sentinel - Drift Detection", layout="wide")

st.title("🔍 Sentinel - ML Drift Detection Dashboard")
st.markdown("Upload your reference and production datasets to detect drift")

# Sidebar for configuration
st.sidebar.header("Configuration")
significance_level = st.sidebar.slider("Significance Level", 0.01, 0.10, 0.05, 0.01)
psi_threshold = st.sidebar.slider("PSI Threshold", 0.1, 0.5, 0.25, 0.05)

# File uploaders
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Reference Data (Baseline)")
    reference_file = st.file_uploader("Upload Reference CSV", type=['csv'], key='ref')
    
with col2:
    st.subheader("📊 Production Data (Current)")
    production_file = st.file_uploader("Upload Production CSV", type=['csv'], key='prod')

if reference_file and production_file:
    # Load data
    reference_df = pd.read_csv(reference_file)
    production_df = pd.read_csv(production_file)
    
    # Show data previews
    with st.expander("Preview Reference Data"):
        st.dataframe(reference_df.head())
        st.write(f"Shape: {reference_df.shape}")
    
    with st.expander("Preview Production Data"):
        st.dataframe(production_df.head())
        st.write(f"Shape: {production_df.shape}")
    
    # Run detection button
    if st.button("🚀 Run Drift Detection", type="primary"):
        with st.spinner("Analyzing data..."):
            # Create detector
            detector = DriftDetector(
                reference_data=reference_df,
                production_data=production_df,
                significance_level=significance_level,
                psi_threshold=psi_threshold
            )
            
            # Run detection
            results = detector.detect_drift()
            
            # Display results
            st.markdown("---")
            
            # Summary
            if results['drift_detected']:
                st.error(f"⚠️ DRIFT DETECTED in {len(results['features_with_drift'])} feature(s)")
                st.warning(f"Features with drift: {', '.join(results['features_with_drift'])}")
            else:
                st.success("✅ No significant drift detected")
            
            # Detailed results
            st.subheader("Detailed Results by Feature")
            
            # Prepare data for visualization
            drift_data = []
            for feature, details in results['feature_details'].items():
                if details['type'] == 'continuous':
                    drift_data.append({
                        'Feature': feature,
                        'Type': 'Continuous',
                        'PSI': details['psi']['psi_value'],
                        'KS p-value': details['ks_test']['p_value'],
                        'Drift': '✅ Yes' if details['drift_detected'] else '❌ No'
                    })
                else:
                    drift_data.append({
                        'Feature': feature,
                        'Type': 'Categorical',
                        'PSI': None,
                        'Chi-Square p-value': details['chi_square']['p_value'],
                        'Drift': '✅ Yes' if details['drift_detected'] else '❌ No'
                    })
            
            # Display table
            drift_df = pd.DataFrame(drift_data)
            st.dataframe(drift_df, use_container_width=True)
            
            # Visualizations
            st.subheader("Visualizations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # PSI values chart
                continuous_features = [d for d in drift_data if d['Type'] == 'Continuous']
                if continuous_features:
                    psi_df = pd.DataFrame(continuous_features)
                    fig = px.bar(
                        psi_df,
                        x='Feature',
                        y='PSI',
                        color='Drift',
                        title='PSI Values by Feature',
                        color_discrete_map={'✅ Yes': 'red', '❌ No': 'green'}
                    )
                    fig.add_hline(y=psi_threshold, line_dash="dash", line_color="orange",
                                  annotation_text=f"PSI Threshold ({psi_threshold})")
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Drift summary pie chart
                drift_counts = drift_df['Drift'].value_counts()
                fig = px.pie(
                    values=drift_counts.values,
                    names=drift_counts.index,
                    title='Drift Detection Summary',
                    color=drift_counts.index,
                    color_discrete_map={'✅ Yes': 'red', '❌ No': 'green'}
                )
                st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👆 Upload both reference and production CSV files to get started")