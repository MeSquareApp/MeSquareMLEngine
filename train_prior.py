from src.pipeline.train_prior_pipeline import train_prior


def main() -> None:
    prior = train_prior()
    summary = prior.metrics.groupby("target")[["test_pinball", "baseline_pinball"]].mean()
    summary["skill"] = 1.0 - summary["test_pinball"] / summary["baseline_pinball"]
    print(summary.round(4))


if __name__ == "__main__":
    main()
