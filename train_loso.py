import argparse
from datetime import datetime
from pathlib import Path

from experiments.run_experiments import ensure_dirs, expand_config, run_jobs


def main():
    parser = argparse.ArgumentParser(description="Run speaker-independent IEMOCAP LOSO experiments.")
    parser.add_argument("--config", default="experiments/configs/core_models.yaml")
    parser.add_argument("--only-ids", nargs="*", default=["A3"], help="Experiment ids to run. Default: A3 Lite-GLSER.")
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--folds", nargs="*", type=int, default=None, help="Optional fold indices. Default: all 5 folds.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-experiments", type=int, default=None)
    args = parser.parse_args()

    ensure_dirs()
    jobs = expand_config(
        Path(args.config),
        seeds_override=args.seeds,
        epochs_override=args.epochs,
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
        max_experiments=args.max_experiments,
        max_batches=args.max_batches,
        num_workers=args.num_workers,
        only_ids=set(args.only_ids or []),
        folds_override=args.folds,
    )
    run_jobs(jobs)


if __name__ == "__main__":
    main()
