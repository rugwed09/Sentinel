"""
Sentinel API - Main FastAPI application
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
from src.drift_detection.detector import DriftDetector

app = FastAPI(
    title="Sentinel ML Monitoring",
    description="Automated ML model performance debugging and root cause analysis",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class DriftDetectionRequest(BaseModel):
    """Request body for drift detection"""
    reference_data: List[Dict[str, Any]]
    production_data: List[Dict[str, Any]]
    categorical_features: List[str] = None
    significance_level: float = 0.05
    psi_threshold: float = 0.25


class DriftDetectionResponse(BaseModel):
    """Response for drift detection"""
    drift_detected: bool
    features_with_drift: List[str]
    feature_details: Dict[str, Any]
    summary: str


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Sentinel ML Monitoring",
        "version": "0.1.0",
        "endpoints": {
            "health": "/health",
            "drift_detection": "/detect-drift",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "api": "up",
        "drift_detector": "ready",
        "database": "not_configured",
        "redis": "not_configured"
    }


@app.post("/detect-drift", response_model=DriftDetectionResponse)
async def detect_drift(request: DriftDetectionRequest):
    """
    Detect drift between reference and production data.
    
    Args:
        request: DriftDetectionRequest with reference and production data
        
    Returns:
        DriftDetectionResponse with drift detection results
    """
    try:
        # Convert JSON to DataFrames
        reference_df = pd.DataFrame(request.reference_data)
        production_df = pd.DataFrame(request.production_data)
        
        # Validate data
        if reference_df.empty or production_df.empty:
            raise HTTPException(
                status_code=400,
                detail="Reference or production data is empty"
            )
        
        # Create detector
        detector = DriftDetector(
            reference_data=reference_df,
            production_data=production_df,
            categorical_features=request.categorical_features,
            significance_level=request.significance_level,
            psi_threshold=request.psi_threshold
        )
        
        # Run detection
        results = detector.detect_drift()
        
        # Generate summary
        if results['drift_detected']:
            num_drifted = len(results['features_with_drift'])
            summary = f"⚠️ Drift detected in {num_drifted} feature(s): {', '.join(results['features_with_drift'])}"
        else:
            summary = "✅ No significant drift detected"
        
        return DriftDetectionResponse(
            drift_detected=results['drift_detected'],
            features_with_drift=results['features_with_drift'],
            feature_details=results['feature_details'],
            summary=summary
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during drift detection: {str(e)}"
        )