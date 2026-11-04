"""Which personalization stages may be served, per analyte ("never worse").

A stage is served only after it beat the stage below it on evaluation. The lab stage
is judged by simulate_lab_stage(); the wearable stage only by gate_residual_model()
on real consented pairs. Until then wearable adjustments run in shadow mode. The
registry is a committed JSON file with the evidence behind every decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import orjson

from src.utils.exception import MeSquareError

REGISTRY_PATH = Path(__file__).resolve().parents[3] / "config" / "activation.json"
# Wearable residuals are activated in this order; eGFR stays lab-only.
WEARABLE_ACTIVATION_ORDER = ("hba1c", "hdl", "non_hdl")
LAB_ONLY = frozenset({"egfr"})


class ActivationError(MeSquareError):
    """Raised when the registry is missing, malformed, or breaks the rules."""


@dataclass(frozen=True)
class StageDecision:
    """Whether one stage is served for one analyte, and why.

    Attributes:
        active: Served when True; shadow or off when False.
        evidence: Metrics or the reason behind the decision.
    """

    active: bool
    evidence: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ActivationRegistry:
    """Served stages per analyte.

    Attributes:
        version: Registry version, reported in model_versions.
        lab: Lab-anchored stage decision per analyte.
        wearable: Wearable residual stage decision per analyte.
    """

    version: str
    lab: dict[str, StageDecision]
    wearable: dict[str, StageDecision]

    def lab_active(self, analyte: str) -> bool:
        """Whether the lab-anchored band may be served for an analyte."""
        decision = self.lab.get(analyte)
        return decision is not None and decision.active

    def wearable_active(self, analyte: str) -> bool:
        """Whether wearable residual adjustments may be served for an analyte."""
        decision = self.wearable.get(analyte)
        return decision is not None and decision.active


def validate(registry: ActivationRegistry) -> None:
    """Enforce the activation rules.

    Args:
        registry: Registry to check.

    Raises:
        ActivationError: If a lab-only analyte has wearables on, or wearables are on
            for an analyte before every analyte ahead of it in the order.
    """
    for analyte in LAB_ONLY:
        if registry.wearable_active(analyte):
            raise ActivationError(f"{analyte} is lab-only; wearables cannot activate")
    for position, analyte in enumerate(WEARABLE_ACTIVATION_ORDER):
        ahead = WEARABLE_ACTIVATION_ORDER[:position]
        if registry.wearable_active(analyte) and not all(
            registry.wearable_active(earlier) for earlier in ahead
        ):
            raise ActivationError(
                f"{analyte} cannot activate before {', '.join(ahead)}"
            )
    allowed = set(WEARABLE_ACTIVATION_ORDER)
    for analyte in registry.wearable:
        if registry.wearable_active(analyte) and analyte not in allowed:
            raise ActivationError(f"{analyte} is not in the wearable activation order")


def load_registry(path: Path = REGISTRY_PATH) -> ActivationRegistry:
    """Read and validate the activation registry.

    Args:
        path: JSON file to read.

    Returns:
        The registry.

    Raises:
        ActivationError: If the file is missing, malformed, or breaks the rules.
    """
    try:
        payload = orjson.loads(path.read_bytes())
        registry = ActivationRegistry(
            version=str(payload["version"]),
            lab={
                name: StageDecision(
                    bool(item["active"]), dict(item.get("evidence", {}))
                )
                for name, item in payload["lab"].items()
            },
            wearable={
                name: StageDecision(
                    bool(item["active"]), dict(item.get("evidence", {}))
                )
                for name, item in payload["wearable"].items()
            },
        )
    except (OSError, orjson.JSONDecodeError, KeyError, TypeError) as error:
        raise ActivationError(
            f"Cannot read activation registry {path}: {error}"
        ) from error
    validate(registry)
    return registry


def save_registry(registry: ActivationRegistry, path: Path = REGISTRY_PATH) -> None:
    """Validate and write the activation registry.

    Args:
        registry: Registry to write.
        path: Destination file.

    Raises:
        ActivationError: If the registry breaks the rules.
    """
    validate(registry)
    payload = {
        "version": registry.version,
        "lab": {
            name: {"active": item.active, "evidence": item.evidence}
            for name, item in registry.lab.items()
        },
        "wearable": {
            name: {"active": item.active, "evidence": item.evidence}
            for name, item in registry.wearable.items()
        },
    }
    path.write_bytes(
        orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE)
    )
