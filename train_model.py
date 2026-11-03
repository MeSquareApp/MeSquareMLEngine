"""Train the quantile models and save a new versioned bundle."""

from src.pipeline.training_pipeline import train_quantile_models


def main() -> None:
    """Train, save, and print the version and labeled row counts."""
    bundle = train_quantile_models()
    print(f"Trained quantile models version={bundle.version}")
    for target, n_rows in bundle.training_row_counts.items():
        print(f"  {target}: {n_rows} labeled rows")


if __name__ == "__main__":
    main()
