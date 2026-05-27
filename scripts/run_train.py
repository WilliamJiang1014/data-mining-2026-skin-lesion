from __future__ import annotations

import argparse

from skin_lesion_risk.pipelines.train import build_model_from_experiment, list_experiment_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or inspect configured models.")
    parser.add_argument("--config", default="configs/experiments/baselines.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args()

    if args.list_models:
        for name in list_experiment_models(args.config):
            print(name)
        return

    if not args.model:
        raise SystemExit("--model is required unless --list-models is used")

    model = build_model_from_experiment(args.config, args.model)
    print(f"Created model: {model.model_name} ({model.model_type})")
    print("Training data loading is intentionally left to the manifest pipeline.")


if __name__ == "__main__":
    main()

