"""FastAPI service: population quantile predictions and the cardio digital twin."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Request

from config.settings import settings
from src.components.model_artifact import ArtifactError, QuantileModelBundle
from src.components.personalization.biomarker_config import BIOMARKERS
from src.components.prevent import prevent_available
from src.pipeline.prediction_pipeline import QuantilePredictor
from src.pipeline.twin_pipeline import TwinInputs, TwinResult, predict_twin
from src.schemas.prediction import PredictRequest, PredictResponse
from src.schemas.twin import TwinRequest, TwinResponse
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _load_predictor() -> QuantilePredictor | None:
    """Load the latest quantile bundle, or None when there is none."""
    try:
        bundle = QuantileModelBundle.load_latest(settings.model_dir)
    except ArtifactError as exc:
        logger.warning("No quantile model loaded: %s", exc)
        return None
    logger.info(
        "Loaded quantile model version=%s from %s", bundle.version, settings.model_dir
    )
    return QuantilePredictor(bundle)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the quantile model and start embedded R once at startup.

    Args:
        app: The FastAPI application.

    Yields:
        Control back to FastAPI while the app runs.
    """
    app.state.predictor = _load_predictor()
    # Embedded R must start on the main thread, before requests run in worker threads.
    if not prevent_available():
        logger.error("Twin responses will omit PREVENT until R and AHAprevent load")
    yield


app = FastAPI(title="MeSquareMLEngine", lifespan=lifespan)


@app.get("/")
async def root() -> dict[str, str]:
    """Return a greeting."""
    return {"message": "Hello World"}


@app.get("/health")
async def health(request: Request) -> dict[str, str]:
    """Report service health and the loaded model version.

    Args:
        request: Gives access to the loaded predictor.

    Returns:
        Status and model version.
    """
    predictor: QuantilePredictor | None = getattr(request.app.state, "predictor", None)
    return {
        "status": "healthy",
        "model_version": predictor.version if predictor is not None else "none",
    }


@app.post("/v1/predict", response_model=PredictResponse)
async def predict(body: PredictRequest, request: Request) -> PredictResponse:
    """Population p10/p50/p90 per biomarker for a batch of patients.

    Args:
        body: Patients to score.
        request: Gives access to the loaded predictor.

    Returns:
        Quantile predictions per patient.

    Raises:
        HTTPException: 503 when no quantile model is loaded.
    """
    predictor: QuantilePredictor | None = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Quantile model is not loaded")

    records = [patient.model_dump() for patient in body.patients]
    results = predictor.predict(records)
    return PredictResponse(model_version=predictor.version, results=results)


@app.post("/v1/twin/predict", response_model=TwinResponse)
def twin_predict(body: TwinRequest, request: Request) -> TwinResponse:
    """Personalized biomarker trajectories and PREVENT risk for one user.

    Args:
        body: Profile, labs, end-of-day wearables, as_of, and horizon.
        request: Gives access to the loaded quantile predictor.

    Returns:
        The twin response.

    Raises:
        HTTPException: 503 when no quantile model is loaded.
    """
    predictor: QuantilePredictor | None = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Quantile model is not loaded")

    inputs = TwinInputs(
        profile=body.profile.model_dump(),
        labs=[lab.model_dump() for lab in body.labs],
        wearable_days=[day.model_dump() for day in body.wearable_days],
        as_of=body.as_of,
        horizon_days=body.horizon_days,
    )
    return to_twin_response(predict_twin(predictor, inputs), body.include_trajectory)


TRAJECTORY_COLUMNS = ("date", "q10", "q50", "q90", "basis", "confidence", "forecast")


def to_twin_response(twin: TwinResult, include_trajectory: bool) -> TwinResponse:
    """Convert a twin result into the API response.

    Args:
        twin: Output of predict_twin().
        include_trajectory: Whether to include every daily point.

    Returns:
        The validated response model.
    """
    biomarkers = {}
    for name, result in twin.biomarkers.items():
        points = result.trajectory.select(TRAJECTORY_COLUMNS).to_dicts()
        current = next(point for point in points if point["date"] == twin.as_of)
        prior = twin.prior_quantiles[name]
        biomarkers[name] = {
            "unit": BIOMARKERS[name].unit,
            "prior": {"q10": prior[0.10], "q50": prior[0.50], "q90": prior[0.90]},
            "current": current,
            "forecast": points[-1],
            "change_vs_prior": current["q50"] - prior[0.50],
            "wearable_effect": result.trajectory["effect"][-1],
            "innovations": [asdict(innovation) for innovation in result.innovations],
            "trajectory": points if include_trajectory else None,
        }
    wearable = twin.wearable
    return TwinResponse.model_validate(
        {
            "as_of": twin.as_of,
            "biomarkers": biomarkers,
            "wearable": {
                "first_date": wearable.first_date,
                "baseline_end": wearable.baseline_end,
                "baseline_ready": wearable.baseline_ready,
                "baseline": wearable.baseline,
                "latest_delta": wearable.latest_delta,
                "valid_days_last_30": wearable.valid_days_last_30,
                "last_valid_date": wearable.last_valid_date,
                "status": wearable.status,
            },
            "prevent": twin.prevent,
            "prevent_note": twin.prevent_note,
            "prevent_out_of_range": twin.prevent_out_of_range,
        }
    )
