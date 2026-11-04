"""FastAPI service: population predictions and the cardio digital twin state."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request

from config.settings import settings
from src.components.model_artifact import ArtifactError, QuantileModelBundle
from src.components.personalization.activation import ActivationError, load_registry
from src.components.personalization.residual_model import load_residual_models
from src.components.prevent import package_version, prevent_available
from src.pipeline.prediction_pipeline import QuantilePredictor
from src.pipeline.twin_pipeline import TwinEngine, TwinRequest, TwinState, twin_state
from src.repository.shadow_store import ShadowBatch, ShadowStore
from src.schemas.prediction import (
    PatientPrediction,
    PredictRequest,
    PredictResponse,
)
from src.schemas.twin import TwinStateRequest, TwinStateResponse
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _load_engine() -> TwinEngine | None:
    """Load the prior, activation registry, residual models and PREVENT version."""
    try:
        bundle = QuantileModelBundle.load_latest(settings.model_dir)
        registry = load_registry()
    except (ArtifactError, ActivationError) as exc:
        logger.error("Twin engine not loaded: %s", exc)
        return None
    logger.info(
        "Loaded quantile model version=%s from %s", bundle.version, settings.model_dir
    )
    return TwinEngine(
        prior=QuantilePredictor(bundle),
        registry=registry,
        residual_models=load_residual_models(settings.residual_model_dir),
        prevent_version=package_version(),
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load models and start embedded R once at startup.

    Args:
        app: The FastAPI application.

    Yields:
        Control back to FastAPI while the app runs.
    """
    # Embedded R must start on the main thread, before requests run in worker threads.
    if not prevent_available():
        logger.error("Risk responses will be not_generated until R and AHAprevent load")
    app.state.engine = _load_engine()
    app.state.shadow_store = ShadowStore.from_settings(settings)
    yield


app = FastAPI(title="MeSquareMLEngine", lifespan=lifespan)


def _engine(request: Request) -> TwinEngine:
    """The loaded engine, or 503."""
    engine: TwinEngine | None = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Quantile model is not loaded")
    return engine


@app.get("/")
async def root() -> dict[str, str]:
    """Return a greeting."""
    return {"message": "Hello World"}


@app.get("/health")
async def health(request: Request) -> dict[str, str]:
    """Report service health and the loaded model versions.

    Args:
        request: Gives access to the loaded engine.

    Returns:
        Status, prior version and PREVENT package version.
    """
    engine: TwinEngine | None = getattr(request.app.state, "engine", None)
    return {
        "status": "healthy",
        "model_version": engine.prior.version if engine is not None else "none",
        "prevent": engine.prevent_version if engine is not None else "none",
    }


@app.post("/v1/predict", response_model=PredictResponse)
def predict(body: PredictRequest, request: Request) -> PredictResponse:
    """Population bands and an estimated PREVENT risk for each patient.

    Args:
        body: Patients to score.
        request: Gives access to the loaded engine.

    Returns:
        Bands (basis "population") and risk (risk_basis "estimated") per patient.

    Raises:
        HTTPException: 503 when no model is loaded.
    """
    engine = _engine(request)
    states = [
        twin_state(engine, TwinRequest(profile=patient.model_dump()))
        for patient in body.patients
    ]
    return PredictResponse(
        model_versions=states[0].model_versions,
        results=[
            PatientPrediction.model_validate(
                {
                    "predictions": [asdict(item) for item in state.predictions],
                    "risk": _risk_payload(state),
                    "input_hash": state.input_hash,
                }
            )
            for state in states
        ],
    )


@app.post("/v1/twin/state", response_model=TwinStateResponse)
def twin_state_endpoint(
    body: TwinStateRequest, request: Request, background: BackgroundTasks
) -> TwinStateResponse:
    """Personalized state for one user: bands, risk, and traceability.

    With a user_id, wearable-adjusted shadow bands are stored for later evaluation.
    They are never in the response until activated.

    Args:
        body: Profile, every record, as_of and user id.
        request: Gives access to the loaded engine and shadow store.
        background: Runs the shadow write after the response.

    Returns:
        The twin state.

    Raises:
        HTTPException: 503 when no model is loaded.
    """
    engine = _engine(request)
    state = twin_state(
        engine,
        TwinRequest(
            profile=body.profile.model_dump(),
            records=[record.model_dump() for record in body.records],
            as_of=body.as_of,
            user_id=body.user_id,
        ),
    )
    store: ShadowStore | None = getattr(request.app.state, "shadow_store", None)
    if body.user_id and state.shadow and store is not None:
        background.add_task(
            store.write,
            ShadowBatch(
                user_id=body.user_id,
                as_of=state.as_of,
                prior_version=state.model_versions["prior"],
                input_hash=state.input_hash,
                predictions=state.shadow,
            ),
        )
    return TwinStateResponse.model_validate(
        {
            "as_of": state.as_of,
            "predictions": [asdict(item) for item in state.predictions],
            "risk": _risk_payload(state),
            "rejected_records": [
                {
                    "id": item.record_id,
                    "biomarker_abbr": item.biomarker_abbr,
                    "reason": item.reason,
                }
                for item in state.rejected
            ],
            "wearable_windows": {
                key: asdict(window) for key, window in state.wearable_windows.items()
            },
            "lab_prompt": asdict(state.lab_prompt),
            "model_versions": state.model_versions,
            "input_hash": state.input_hash,
        }
    )


def _risk_payload(state: TwinState) -> dict[str, object]:
    """Risk block as plain data for the response models."""
    payload = asdict(state.risk)
    payload["inputs_used"] = {
        name: {"value": item.value, "source": item.source}
        for name, item in state.risk.inputs_used.items()
    }
    return payload
