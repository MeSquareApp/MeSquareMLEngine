from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request

from config.settings import settings
from src.components.model_artifact import ArtifactError, QuantileModelBundle
from src.pipeline.prediction_pipeline import QuantilePredictor
from src.schemas.prediction import PredictRequest, PredictResponse
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _load_predictor() -> QuantilePredictor | None:
    try:
        bundle = QuantileModelBundle.load_latest(settings.model_dir)
    except ArtifactError as exc:
        logger.warning("No quantile model loaded: %s", exc)
        return None
    logger.info("Loaded quantile model version=%s from %s", bundle.version, settings.model_dir)
    return QuantilePredictor(bundle)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.predictor = _load_predictor()
    yield


app = FastAPI(title="MeSquareMLEngine", lifespan=lifespan)


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Hello World"}


@app.get("/health")
async def health(request: Request) -> dict[str, str]:
    predictor: QuantilePredictor | None = getattr(request.app.state, "predictor", None)
    return {
        "status": "healthy",
        "model_version": predictor.version if predictor is not None else "none",
    }


@app.post("/v1/predict", response_model=PredictResponse)
async def predict(body: PredictRequest, request: Request) -> PredictResponse:
    predictor: QuantilePredictor | None = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Quantile model is not loaded")

    records = [patient.model_dump() for patient in body.patients]
    results = predictor.predict(records)
    return PredictResponse(model_version=predictor.version, results=results)
