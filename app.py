from contextlib import asynccontextmanager
from typing import AsyncIterator

import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from config.settings import settings
from src.components.model_artifact import ArtifactError, QuantileModelBundle
from src.components.personalization.biomarker_config import BIOMARKERS
from src.components.prevent import init_prevent
from src.pipeline.prediction_pipeline import QuantilePredictor
from src.pipeline.twin_pipeline import TwinResult, predict_twin
from src.schemas.prediction import PredictRequest, PredictResponse
from src.schemas.twin import TwinRequest, TwinResponse
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
    # Embedded R must start on the main thread, before requests run in worker threads.
    if not init_prevent():
        logger.warning("AHA PREVENT R package not available; twin responses will omit PREVENT")
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


@app.post("/v1/twin/predict", response_model=TwinResponse)
def twin_predict(body: TwinRequest, request: Request) -> TwinResponse:
    predictor: QuantilePredictor | None = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Quantile model is not loaded")

    wearable_days = pd.DataFrame([day.model_dump() for day in body.wearable_days])
    twin = predict_twin(
        predictor,
        body.profile.model_dump(),
        labs=[lab.model_dump() for lab in body.labs],
        wearable_days=wearable_days,
        as_of=body.as_of,
        horizon_days=body.horizon_days,
    )
    return to_twin_response(twin, body.include_trajectory)


def to_twin_response(twin: TwinResult, include_trajectory: bool) -> TwinResponse:
    biomarkers = {}
    for name, result in twin.biomarkers.items():
        trajectory = result.trajectory
        points = trajectory[["date", "q10", "q50", "q90", "basis", "confidence", "forecast"]]
        records = points.to_dict(orient="records")
        current = next(r for r in records if r["date"] == twin.as_of)
        prior = twin.prior_quantiles[name]
        biomarkers[name] = {
            "unit": BIOMARKERS[name].unit,
            "prior": {"q10": prior[0.10], "q50": prior[0.50], "q90": prior[0.90]},
            "current": current,
            "forecast": records[-1],
            "change_vs_prior": current["q50"] - prior[0.50],
            "wearable_effect": float(trajectory["effect"].iloc[-1]),
            "innovations": [vars(i) for i in result.innovations],
            "trajectory": records if include_trajectory else None,
        }
    wearable = twin.wearable
    return TwinResponse(
        as_of=twin.as_of,
        biomarkers=biomarkers,
        wearable={
            "first_date": wearable.first_date,
            "baseline_end": wearable.baseline_end,
            "baseline_ready": wearable.baseline_ready,
            "baseline": wearable.baseline,
            "latest_delta": wearable.latest_delta,
        },
        prevent=twin.prevent,
        prevent_note=twin.prevent_note,
    )
