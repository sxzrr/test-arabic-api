from fastapi import APIRouter, HTTPException
from app.models.prediction import PredictionRequest, PredictionResponse
from app.services.model import predict_text

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
async def predict(data: PredictionRequest):
    try:
        result = predict_text(data.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
