from src.pipeline.training_pipeline import train_quantile_models


def main() -> None:
    bundle = train_quantile_models()
    print(f"Trained quantile models version={bundle.version}")
    for target, n in bundle.training_row_counts.items():
        print(f"  {target}: {n} labeled rows")


if __name__ == "__main__":
    main()
