"""Simulate repeat labs, judge the lab stage, and write config/activation.json."""

from src.pipeline.evaluation_pipeline import evaluate_lab_stage


def main() -> None:
    """Evaluate every biomarker and print which stages are served."""
    registry = evaluate_lab_stage()
    print(f"Wrote {registry.version}")
    for biomarker, decision in registry.lab.items():
        print(f"  {biomarker}: lab stage {'on' if decision.active else 'off'}")


if __name__ == "__main__":
    main()
