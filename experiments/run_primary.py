"""Run one primary synthetic experiment from a versioned YAML configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

from preference_shift.config import PrimaryConfig
from preference_shift.experiment import run_primary_experiment, save_primary_run


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/configs/primary.yaml"),
        help="Path to the experiment configuration.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/primary"),
        help="Directory in which to save the self-contained run output.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace known output files when the result directory is non-empty.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    config = PrimaryConfig.from_yaml(arguments.config)
    run = run_primary_experiment(config)
    output = save_primary_run(
        run,
        config,
        arguments.output,
        overwrite=arguments.overwrite,
    )
    print(f"Saved primary experiment to {output}")


if __name__ == "__main__":
    main()
