"""Digital twin state: personalized bands per analyte plus a PREVENT risk response.

Stages, per analyte (see the activation registry for what may be served):

0. Population: the NHANES quantile prior for the profile ("population").
1. First lab: a band centred on the latest measured lab ("lab").
2. Repeat labs: the latest lab anchors; the previous one feeds a 95% RCV change flag.
3. Wearables: a residual adjustment ("population_wearable" / "lab_wearable"),
   computed in shadow mode and served only once activated.

Stateless: the backend sends the profile and every record each time.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Literal, Protocol

import numpy as np
import orjson

from src.components.analytes import MODEL_ANALYTES
from src.components.personalization.activation import ActivationRegistry
from src.components.personalization.biomarker_config import BIOMARKERS
from src.components.personalization.lab_anchor import (
    ChangeFlag,
    TreatmentChanges,
    anchoring_labs,
    change_flag,
    lab_band,
)
from src.components.personalization.residual_model import (
    MIN_DAYS_AFTER_LAB,
    ResidualMode,
    ResidualModel,
)
from src.components.personalization.wearable_windows import (
    FEATURE_NAMES,
    WindowSummary,
    feature_vector,
    feature_windows,
)
from src.components.prevent import AGE_RANGE, prevent_available
from src.components.record_ingestion import (
    Measurement,
    RawRecord,
    Rejection,
    ingest_records,
)
from src.components.risk import (
    InputEstimate,
    RiskAssessment,
    RiskProfile,
    assess_risk,
)

Profile = Mapping[str, object]
Basis = Literal["population", "lab", "population_wearable", "lab_wearable"]
Band = tuple[float, float, float]

LAB_PROMPT_DAYS = 90
SCALE_WEIGHT_MAX_AGE_DAYS = 30
CM_PER_M = 100.0
REQUIRED_RISK_PROFILE = ("height_cm", "weight_kg")


class QuantilePrior(Protocol):
    """Population quantile model for one profile."""

    @property
    def version(self) -> str:
        """Model version."""
        ...

    def predict_quantiles(
        self, record: Mapping[str, str | float | int | None]
    ) -> dict[str, dict[float, float]]:
        """Quantiles keyed by target, then tau."""
        ...


@dataclass(frozen=True)
class TwinEngine:
    """Everything loaded once at startup.

    Attributes:
        prior: Population quantile model.
        registry: Served stages per analyte.
        residual_models: Wearable residual models keyed (analyte, mode).
        prevent_version: Version of the official AHA PREVENT package.
    """

    prior: QuantilePrior
    registry: ActivationRegistry
    residual_models: dict[tuple[str, str], ResidualModel] = field(default_factory=dict)
    prevent_version: str = "unavailable"


@dataclass(frozen=True)
class TwinRequest:
    """One user's state request.

    Attributes:
        profile: Profile with NHANES codes, height_cm, and treatment changes.
        records: Lab and wearable records in the input shape.
        as_of: Day of the state. Defaults to the latest record day, else today.
        user_id: Backend user id, used only to key shadow predictions.
    """

    profile: Profile
    records: Sequence[RawRecord] = field(default_factory=tuple)
    as_of: date | None = None
    user_id: str | None = None


@dataclass(frozen=True)
class AnalytePrediction:
    """The served band of one analyte.

    Attributes:
        analyte: Canonical analyte name.
        basis: What the band is based on.
        basis_label: The same, in words the app can show.
        p10: 10th percentile.
        p50: Median.
        p90: 90th percentile.
        unit: Canonical unit.
        as_of: Day the band is for.
        source_lab_id: Record id of the anchoring lab, when lab-based.
        lab_measured_at: Day of the anchoring lab, when lab-based.
        change: Latest vs previous lab against the 95% RCV, with 2+ labs.
    """

    analyte: str
    basis: Basis
    basis_label: str
    p10: float
    p50: float
    p90: float
    unit: str
    as_of: date
    source_lab_id: str | None = None
    lab_measured_at: date | None = None
    change: ChangeFlag | None = None


@dataclass(frozen=True)
class ShadowPrediction:
    """A wearable-adjusted band computed but not served (shadow mode).

    Attributes:
        analyte: Canonical analyte name.
        mode: "level" or "change".
        served: The band actually served.
        shadow: The wearable-adjusted band.
        model_version: Residual model version.
    """

    analyte: str
    mode: ResidualMode
    served: Band
    shadow: Band
    model_version: str


@dataclass(frozen=True)
class LabPrompt:
    """When the app should ask for the next lab.

    Attributes:
        due_on: Day the next lab is due.
        due: Whether it is due on or before as_of.
    """

    due_on: date
    due: bool


@dataclass
class TwinState:
    """Full twin response for one user.

    Attributes:
        as_of: Day of the state.
        predictions: One served band per analyte.
        risk: PREVENT risk block.
        rejected: Records that were not used, and why.
        wearable_windows: Wearable window summaries.
        shadow: Wearable-adjusted bands that are not served.
        lab_prompt: When to ask for the next lab.
        model_versions: Versions of every model and registry used.
        input_hash: SHA-256 of the inputs and model versions.
    """

    as_of: date
    predictions: list[AnalytePrediction]
    risk: RiskAssessment
    rejected: list[Rejection]
    wearable_windows: dict[str, WindowSummary]
    shadow: list[ShadowPrediction]
    lab_prompt: LabPrompt
    model_versions: dict[str, str]
    input_hash: str


def twin_state(engine: TwinEngine, request: TwinRequest) -> TwinState:
    """Build the personalized state for one user.

    Args:
        engine: Models and registry loaded at startup.
        request: Profile, records, as_of and user id.

    Returns:
        Served bands, risk, rejections, wearable windows, shadow bands, lab prompt,
        model versions and input hash.
    """
    profile = request.profile
    age, sex = _number(profile["age"]), int(_number(profile["sex"]))
    ingested = ingest_records(request.records, age, sex)
    as_of = request.as_of or _default_as_of(ingested.labs + ingested.wearables)
    treatments = TreatmentChanges(
        statin_started_at=_optional_date(profile.get("statin_started_at")),
        bp_meds_started_at=_optional_date(profile.get("bp_meds_started_at")),
        pregnant=bool(profile.get("pregnant")),
    )
    quantiles = engine.prior.predict_quantiles(_prior_features(profile))
    windows = feature_windows(ingested.wearables, as_of)
    predictions: list[AnalytePrediction] = []
    shadow: list[ShadowPrediction] = []
    anchors: dict[str, Measurement] = {}
    for analyte in MODEL_ANALYTES:
        labs = anchoring_labs(analyte, ingested.labs, as_of, treatments)
        if labs:
            anchors[analyte] = labs[-1]
        prediction, shadow_band = _analyte_prediction(
            engine,
            (analyte, as_of),
            (quantiles[analyte], labs),
            (windows, ingested.wearables),
        )
        predictions.append(prediction)
        if shadow_band is not None:
            shadow.append(shadow_band)
    model_versions = _model_versions(engine)
    return TwinState(
        as_of=as_of,
        predictions=predictions,
        risk=_risk(profile, predictions, (anchors, ingested.wearables, as_of)),
        rejected=ingested.rejected,
        wearable_windows=windows,
        shadow=shadow,
        lab_prompt=_lab_prompt(anchors, as_of),
        model_versions=model_versions,
        input_hash=input_hash(request, as_of, model_versions),
    )


def _analyte_prediction(
    engine: TwinEngine,
    target: tuple[str, date],
    evidence: tuple[dict[float, float], list[Measurement]],
    wearable: tuple[dict[str, WindowSummary], list[Measurement]],
) -> tuple[AnalytePrediction, ShadowPrediction | None]:
    """Served band of one analyte and, when a residual model applies, its shadow."""
    analyte, as_of = target
    population, labs = evidence
    spec = BIOMARKERS[analyte]
    latest = labs[-1] if labs and engine.registry.lab_active(analyte) else None
    if latest is not None:
        served: Band = lab_band(latest, as_of)
        basis: Basis = "lab"
        label = f"Based on your {latest.day:%b} {latest.day.day} lab"
    else:
        served = (population[0.10], population[0.50], population[0.90])
        basis = "population"
        label = "Typical for your profile"

    shadow = _shadow(engine, (analyte, served), latest, wearable)
    if shadow is not None and engine.registry.wearable_active(analyte):
        served = shadow.shadow
        basis = "lab_wearable" if latest is not None else "population_wearable"
        label += ", adjusted using your wearable trends"

    low, high = spec.clamp
    p10, p50, p90 = (float(np.clip(value, low, high)) for value in served)
    prediction = AnalytePrediction(
        analyte=analyte,
        basis=basis,
        basis_label=label,
        p10=p10,
        p50=p50,
        p90=p90,
        unit=spec.unit,
        as_of=as_of,
        source_lab_id=latest.record_id if latest is not None else None,
        lab_measured_at=latest.day if latest is not None else None,
        change=change_flag(labs[-2], labs[-1]) if len(labs) >= 2 else None,
    )
    return prediction, shadow


def _shadow(
    engine: TwinEngine,
    served: tuple[str, Band],
    latest: Measurement | None,
    wearable: tuple[dict[str, WindowSummary], list[Measurement]],
) -> ShadowPrediction | None:
    """Wearable-adjusted band from a residual model, if one applies."""
    analyte, band = served
    windows, wearables = wearable
    if latest is not None:
        mode: ResidualMode = "change"
        days_after = len({item.day for item in wearables if item.day > latest.day})
        eligible = days_after >= MIN_DAYS_AFTER_LAB
    else:
        mode = "level"
        eligible = any(w.sufficient for w in windows.values() if w.days == 30)
    model = engine.residual_models.get((analyte, mode))
    if model is None or not eligible:
        return None
    features = feature_vector(windows)
    row = np.array([[features[name] for name in FEATURE_NAMES]])
    adjusted = model.predict_band(row, np.array([band[1]]))[0]
    return ShadowPrediction(
        analyte=analyte,
        mode=mode,
        served=band,
        shadow=(float(adjusted[0]), float(adjusted[1]), float(adjusted[2])),
        model_version=model.version,
    )


def _risk(
    profile: Profile,
    predictions: list[AnalytePrediction],
    measured: tuple[dict[str, Measurement], list[Measurement], date],
) -> RiskAssessment:
    """PREVENT on measured labs where present, personalized p50s otherwise."""
    anchors, wearables, as_of = measured
    age = _number(profile["age"])
    if bool(profile.get("pregnant")):
        return RiskAssessment(
            "not_generated", reason="PREVENT is not validated in pregnancy"
        )
    if not AGE_RANGE[0] <= age <= AGE_RANGE[1]:
        return RiskAssessment(
            "not_generated",
            reason=f"PREVENT is validated for ages {AGE_RANGE[0]}-{AGE_RANGE[1]}",
        )
    missing = [name for name in REQUIRED_RISK_PROFILE if profile.get(name) is None]
    if missing:
        return RiskAssessment("not_generated", missing_inputs=missing)
    if not prevent_available():
        return RiskAssessment(
            "not_generated", reason="AHA PREVENT R package is not available"
        )

    inputs: dict[str, InputEstimate] = {}
    for prediction in predictions:
        lab = anchors.get(prediction.analyte)
        inputs[prediction.analyte] = (
            InputEstimate(lab.value, "measured")
            if lab is not None
            else InputEstimate(
                prediction.p50, "predicted", (prediction.p10, prediction.p90)
            )
        )
    inputs["bmi"] = _bmi(profile, wearables, as_of)
    diagnosed = (
        profile.get("diabetes_dx") == "yes"
        or profile.get("insulin") == 1
        or profile.get("diabetes_pills") == 1
    )
    inputs["diabetes"] = InputEstimate(1.0 if diagnosed else 0.0, "reported")
    return assess_risk(
        RiskProfile(
            age=age,
            sex=int(_number(profile["sex"])),
            smoking=1.0 if profile.get("smoking") == "current" else 0.0,
            bp_treated=_number(profile.get("bp_meds") or 0),
            statin=_number(profile.get("chol_meds") or 0),
        ),
        inputs,
    )


def _bmi(profile: Profile, wearables: list[Measurement], as_of: date) -> InputEstimate:
    """BMI from a recent smart-scale weight (measured) or the profile (reported)."""
    height_m = _number(profile["height_cm"]) / CM_PER_M
    recent = [
        item.value
        for item in wearables
        if item.analyte == "weight"
        and 0 <= (as_of - item.day).days < SCALE_WEIGHT_MAX_AGE_DAYS
    ]
    if recent:
        return InputEstimate(float(np.mean(recent)) / height_m**2, "measured")
    return InputEstimate(_number(profile["weight_kg"]) / height_m**2, "reported")


def _lab_prompt(anchors: dict[str, Measurement], as_of: date) -> LabPrompt:
    """Ask for a lab LAB_PROMPT_DAYS after the latest one, or now if none."""
    if not anchors:
        return LabPrompt(due_on=as_of, due=True)
    latest = max(lab.day for lab in anchors.values())
    due_on = latest + timedelta(days=LAB_PROMPT_DAYS)
    return LabPrompt(due_on=due_on, due=due_on <= as_of)


def _model_versions(engine: TwinEngine) -> dict[str, str]:
    """Versions of the prior, PREVENT, the registry, and any residual models."""
    versions = {
        "prior": engine.prior.version,
        "prevent": engine.prevent_version,
        "activation": engine.registry.version,
    }
    for (analyte, mode), model in sorted(engine.residual_models.items()):
        versions[f"residual_{analyte}_{mode}"] = model.version
    return versions


def input_hash(
    request: TwinRequest, as_of: date, model_versions: dict[str, str]
) -> str:
    """SHA-256 over the canonical inputs and model versions.

    Record order does not matter. The same inputs and models always give the same
    hash, so a changed hash means new data or a new model.

    Args:
        request: The request.
        as_of: Resolved as_of day.
        model_versions: Versions used.

    Returns:
        "sha256:<hex digest>".
    """
    canonical = orjson.OPT_SORT_KEYS | orjson.OPT_NON_STR_KEYS
    records = sorted(
        orjson.dumps(record, option=canonical, default=str)
        for record in request.records
    )
    payload = orjson.dumps(
        {
            "profile": dict(request.profile),
            "records": [orjson.loads(record) for record in records],
            "as_of": as_of,
            "model_versions": model_versions,
        },
        option=canonical,
        default=str,
    )
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _prior_features(profile: Profile) -> dict[str, str | float | int | None]:
    """Profile fields the prior model reads."""
    return {
        key: value
        for key, value in profile.items()
        if value is None or isinstance(value, str | float | int)
    }


def _default_as_of(measurements: list[Measurement]) -> date:
    """Latest measurement day, or today (UTC) without any."""
    if measurements:
        return max(item.day for item in measurements)
    return datetime.now(UTC).date()


def _optional_date(value: object) -> date | None:
    if value is None or isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _number(value: object) -> float:
    """Accept an int, float, or numeric string.

    Raises:
        ValueError: If value is not numeric.
    """
    if isinstance(value, int | float | str) and not isinstance(value, bool):
        return float(value)
    raise ValueError(f"Not a number: {value!r}")
