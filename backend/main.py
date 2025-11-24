"""
FastAPI backend for AI-powered fault localization.
Provides REST API endpoints for prediction and history retrieval.
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from sqlalchemy.orm import Session
import uuid
from datetime import datetime
from pathlib import Path

from database import engine, get_db, Base
from models import Prediction, RankedLine
from feature_extraction import FeatureExtractor
# Use simplified model (Gemini only, no PyTorch)
try:
    from ai_model import FaultLocalizationModel
except ImportError:
    from ai_model_simple import FaultLocalizationModel
    print("⚠️ Using simplified model (Gemini only)")

# Create database tables
Base.metadata.create_all(bind=engine)

# Initialize FastAPI app
app = FastAPI(
    title="AI Fault Localization API",
    description="REST API for AI-powered fault localization in software repositories",
    version="1.0.0"
)

# Configure CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AI model (singleton)
fault_model = None

def get_model(db: Session = Depends(get_db)) -> FaultLocalizationModel:
    """
    Dependency to get or create the AI model instance.
    Model is initialized once and reused across requests.
    """
    global fault_model
    if fault_model is None:
        fault_model = FaultLocalizationModel(db=db, device="cpu")
    return fault_model


# ==================== Request/Response Models ====================

class PredictionRequest(BaseModel):
    """Request model for fault localization prediction."""
    repository_path: str = Field(..., description="Absolute path to the repository")
    failing_test: str = Field(..., description="Name of the failing test")
    error_message: Optional[str] = Field(None, description="Error message from test failure")
    stack_trace: Optional[str] = Field(None, description="Stack trace from test failure")
    max_candidates: Optional[int] = Field(100, description="Maximum number of candidates to analyze", ge=1, le=500)

    class Config:
        json_schema_extra = {
            "example": {
                "repository_path": "/path/to/repo",
                "failing_test": "test_user_authentication",
                "error_message": "AssertionError: Expected True but got False",
                "stack_trace": 'File "auth.py", line 42, in validate_user',
                "max_candidates": 50
            }
        }


class RankedLineResponse(BaseModel):
    """Response model for a single ranked line."""
    file: str
    line_number: int
    code: str
    probability: float
    rank: int
    features: Optional[Dict] = None


class PredictionResponse(BaseModel):
    """Response model for prediction results."""
    prediction_id: str
    ranked_lines: List[RankedLineResponse]
    timestamp: str
    total_candidates: int
    repository_path: str
    failing_test: str


class PredictionSummary(BaseModel):
    """Summary model for prediction history."""
    id: str
    repository_path: str
    failing_test: str
    timestamp: str
    total_candidates: int


class HistoryResponse(BaseModel):
    """Response model for prediction history."""
    predictions: List[PredictionSummary]
    total: int


# ==================== API Endpoints ====================

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "AI Fault Localization API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "POST /predict - Analyze code and predict fault locations",
            "history": "GET /history - Retrieve prediction history",
            "detail": "GET /history/{prediction_id} - Get detailed prediction results"
        },
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.post("/predict", response_model=PredictionResponse)
async def predict_faults(
    request: PredictionRequest,
    db: Session = Depends(get_db),
    model: FaultLocalizationModel = Depends(get_model)
):
    """
    Analyze repository and predict fault locations.
    
    Process:
    1. Validate repository path
    2. Extract candidate lines from code
    3. Compute CodeBERT embeddings (with caching)
    4. Run AI model to score each line
    5. Store results in database
    6. Return ranked suspicious lines
    
    Args:
        request: Prediction request with repository info
        db: Database session
        model: AI model instance
        
    Returns:
        Ranked list of suspicious lines with probabilities
    """
    # Validate repository path
    repo_path = Path(request.repository_path)
    if not repo_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Repository path does not exist: {request.repository_path}"
        )
    
    if not repo_path.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"Repository path is not a directory: {request.repository_path}"
        )
    
    try:
        # Extract candidate lines
        extractor = FeatureExtractor(str(repo_path))
        candidates = extractor.extract_candidate_lines(
            failing_test=request.failing_test,
            error_message=request.error_message,
            stack_trace=request.stack_trace,
            max_candidates=request.max_candidates
        )
        
        if not candidates:
            raise HTTPException(
                status_code=404,
                detail="No candidate lines found. Ensure repository contains Python files."
            )
        
        # Predict fault probabilities
        ranked_candidates = model.predict(candidates)
        
        # Create prediction record
        prediction_id = str(uuid.uuid4())
        prediction = Prediction(
            id=prediction_id,
            repository_path=request.repository_path,
            failing_test=request.failing_test,
            error_message=request.error_message,
            stack_trace=request.stack_trace,
            total_candidates=len(ranked_candidates)
        )
        db.add(prediction)
        
        # Store ranked lines
        for candidate in ranked_candidates:
            ranked_line = RankedLine(
                prediction_id=prediction_id,
                file_path=candidate['file_path'],
                line_number=candidate['line_number'],
                code=candidate['code'],
                probability=candidate['probability'],
                rank=candidate['rank'],
                features=candidate.get('features', {})
            )
            db.add(ranked_line)
        
        db.commit()
        
        # Prepare response
        response = PredictionResponse(
            prediction_id=prediction_id,
            ranked_lines=[
                RankedLineResponse(
                    file=c['file_path'],
                    line_number=c['line_number'],
                    code=c['code'],
                    probability=c['probability'],
                    rank=c['rank'],
                    features=c.get('features')
                )
                for c in ranked_candidates
            ],
            timestamp=prediction.timestamp.isoformat(),
            total_candidates=len(ranked_candidates),
            repository_path=request.repository_path,
            failing_test=request.failing_test
        )
        
        return response
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during prediction: {str(e)}"
        )


@app.get("/history", response_model=HistoryResponse)
async def get_prediction_history(
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """
    Retrieve prediction history.
    
    Args:
        limit: Maximum number of results to return
        offset: Number of results to skip (for pagination)
        db: Database session
        
    Returns:
        List of past predictions with summaries
    """
    # Query predictions ordered by timestamp (newest first)
    predictions = db.query(Prediction).order_by(
        Prediction.timestamp.desc()
    ).offset(offset).limit(limit).all()
    
    total = db.query(Prediction).count()
    
    return HistoryResponse(
        predictions=[
            PredictionSummary(
                id=p.id,
                repository_path=p.repository_path,
                failing_test=p.failing_test,
                timestamp=p.timestamp.isoformat(),
                total_candidates=p.total_candidates
            )
            for p in predictions
        ],
        total=total
    )


@app.get("/history/{prediction_id}", response_model=PredictionResponse)
async def get_prediction_detail(
    prediction_id: str,
    db: Session = Depends(get_db)
):
    """
    Get detailed results for a specific prediction.
    
    Args:
        prediction_id: UUID of the prediction
        db: Database session
        
    Returns:
        Detailed prediction results with ranked lines
    """
    # Query prediction
    prediction = db.query(Prediction).filter(
        Prediction.id == prediction_id
    ).first()
    
    if not prediction:
        raise HTTPException(
            status_code=404,
            detail=f"Prediction not found: {prediction_id}"
        )
    
    # Query ranked lines
    ranked_lines = db.query(RankedLine).filter(
        RankedLine.prediction_id == prediction_id
    ).order_by(RankedLine.rank).all()
    
    return PredictionResponse(
        prediction_id=prediction.id,
        ranked_lines=[
            RankedLineResponse(
                file=line.file_path,
                line_number=line.line_number,
                code=line.code,
                probability=line.probability,
                rank=line.rank,
                features=line.features
            )
            for line in ranked_lines
        ],
        timestamp=prediction.timestamp.isoformat(),
        total_candidates=prediction.total_candidates,
        repository_path=prediction.repository_path,
        failing_test=prediction.failing_test
    )


@app.delete("/history/{prediction_id}")
async def delete_prediction(
    prediction_id: str,
    db: Session = Depends(get_db)
):
    """
    Delete a prediction and its ranked lines.
    
    Args:
        prediction_id: UUID of the prediction
        db: Database session
        
    Returns:
        Success message
    """
    prediction = db.query(Prediction).filter(
        Prediction.id == prediction_id
    ).first()
    
    if not prediction:
        raise HTTPException(
            status_code=404,
            detail=f"Prediction not found: {prediction_id}"
        )
    
    db.delete(prediction)
    db.commit()
    
    return {"message": "Prediction deleted successfully", "prediction_id": prediction_id}


@app.get("/stats")
async def get_statistics(db: Session = Depends(get_db)):
    """
    Get statistics about predictions and cache.
    
    Args:
        db: Database session
        
    Returns:
        Statistics dictionary
    """
    from models import EmbeddingCache
    
    total_predictions = db.query(Prediction).count()
    total_ranked_lines = db.query(RankedLine).count()
    total_cached_embeddings = db.query(EmbeddingCache).count()
    
    return {
        "total_predictions": total_predictions,
        "total_ranked_lines": total_ranked_lines,
        "total_cached_embeddings": total_cached_embeddings,
        "cache_hit_rate": "N/A"  # Would need request tracking for this
    }


# ==================== Startup/Shutdown Events ====================

@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup."""
    print("🚀 Starting AI Fault Localization API...")
    print("📊 Database initialized")
    print("🤖 CodeBERT AI model ready")
    print("🔍 Advanced semantic analysis enabled")
    print("✅ Server is running on http://localhost:8000")
    print("📖 API docs: http://localhost:8000/docs")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown."""
    print("👋 Shutting down AI Fault Localization API...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
