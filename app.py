from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from src.components.personalization.biomarker_config import BIOMARKERS
from src.components.prior_model import ARTIFACT_DIR, ARTIFACT_FILE, PriorModel
from src.pipeline.twin_pipeline import TwinResult, predict_twin
from src.schemas.twin import TwinRequest, TwinResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Tests may set app.state.prior before startup.
    if getattr(app.state, "prior", None) is None:
        path = ARTIFACT_DIR / ARTIFACT_FILE
        app.state.prior = PriorModel.load() if path.exists() else None
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Hello World"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/twin/predict")
def twin_predict(body: TwinRequest, request: Request) -> TwinResponse:
    prior = getattr(request.app.state, "prior", None)
    if prior is None:
        raise HTTPException(503, "Prior models are not trained. Run train_prior.py.")
    wearable_days = pd.DataFrame([day.model_dump() for day in body.wearable_days])
    twin = predict_twin(
        prior,
        body.profile.model_dump(),
        labs=[lab.model_dump() for lab in body.labs],
        wearable_days=wearable_days,
        as_of=body.as_of,
        horizon_days=body.horizon_days,
    )
    return to_response(twin, body.include_trajectory)


def to_response(twin: TwinResult, include_trajectory: bool) -> TwinResponse:
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
