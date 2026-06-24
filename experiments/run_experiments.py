import argparse
import json
import os
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = EXPERIMENT_DIR / "output"


BASE_TRAIN_CONFIG = {
    "data": {
        "protocol": "loso",
        "feature_paths": [
            "dataset/IEMOCAP/session_data/ses_1.parquet",
            "dataset/IEMOCAP/session_data/ses_2.parquet",
            "dataset/IEMOCAP/session_data/ses_3.parquet",
            "dataset/IEMOCAP/session_data/ses_4.parquet",
            "dataset/IEMOCAP/session_data/ses_5.parquet",
        ],
        "metadata_paths": [],
        "fold_index": 0,
        "val_strategy": "cyclic_next",
        "norm_mode": "fixed_db",
    },
    "loader": {"batch_size": 32, "num_workers": 0},
    "model": {
        "input_dim": 80,
        "hidden_dim": 256,
        "num_layers": 3,
        "d_state": 32,
        "p_order": 3,
        "output_dim": 3,
        "dropout": 0.4,
        "sequence_core": "da_ls4",
        "noise_frontend": "identity",
        "remove_sequence_core": False,
        "remove_attention": False,
        "disable_ls4_dynamic": False,
    },
    "data_aug": {"disable_feature_aug": False},
    "label_refiner": {"sigma": 0.05, "apply_smoothing": True},
    "loss": {
        "loss_type": "improved",
        "ccc_weight": 0.5,
        "corr_weight": 0.25,
        "mse_weight": 0.15,
        "smooth_weight": 0.10,
        "task_importance": [1.5, 1.0, 1.0],
    },
    "optimizer": {"lr": 5e-5, "weight_decay": 0.05},
    "scheduler": {"factor": 0.5, "patience": 15},
    "training": {"epochs": 300, "early_stop": 50},
    "runtime": {"data_parallel": False, "amp": False},
}


def ensure_dirs():
    for name in ("checkpoints", "logs", "results", "efficiency", "noise", "summaries", "generated_configs"):
        (OUTPUT_ROOT / name).mkdir(parents=True, exist_ok=True)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_update(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def normalize_overrides(raw):
    raw = dict(raw)
    raw.pop("name", None)
    model_keys = {
        "sequence_core", "noise_frontend", "hidden_dim", "d_state", "p_order",
        "remove_sequence_core", "remove_attention", "disable_ls4_dynamic",
        "original_liquid_kernel",
    }
    normalized = {"model": {}, "data_aug": {}, "label_refiner": {}, "loss": {}}
    for key, value in raw.items():
        if key == "num_ls4_layers":
            normalized["model"]["num_layers"] = value
        elif key in model_keys:
            normalized["model"][key] = value
        elif key == "disable_feature_aug":
            normalized["data_aug"]["disable_feature_aug"] = value
        elif key == "disable_label_refine":
            normalized["label_refiner"]["apply_smoothing"] = not bool(value)
        elif key in {"loss_type", "task_importance"}:
            normalized["loss"][key] = value
        elif key in {"training", "optimizer", "scheduler", "loader", "data", "model", "loss", "label_refiner", "data_aug", "runtime"}:
            normalized[key] = value
        else:
            normalized["model"][key] = value
    return {k: v for k, v in normalized.items() if v}


def expand_config(
    config_path,
    seeds_override=None,
    epochs_override=None,
    timestamp=None,
    max_experiments=None,
    max_batches=None,
    num_workers=None,
    only_ids=None,
    folds_override=None,
):
    config_path = Path(config_path)
    cfg = load_yaml(config_path)
    group = cfg.get("group", config_path.stem)
    seeds = seeds_override or cfg.get("seeds", [42])
    save_checkpoints = bool(cfg.get("save_checkpoints", group != "ablations"))
    defaults = normalize_overrides(cfg.get("defaults", {}))
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

    jobs = []
    for exp_id, exp_cfg in cfg.get("experiments", {}).items():
        if only_ids and exp_id not in only_ids:
            continue
        exp_name = exp_cfg.get("name", exp_id.lower())
        for seed in seeds:
            train_cfg = deepcopy(BASE_TRAIN_CONFIG)
            deep_update(train_cfg, deepcopy(defaults))
            deep_update(train_cfg, normalize_overrides(exp_cfg))
            protocol = train_cfg.get("data", {}).get("protocol", "loso")
            fold_indices = [None]
            if protocol == "loso":
                fold_indices = list(folds_override) if folds_override is not None else list(range(5))

            for fold_index in fold_indices:
                fold_suffix = "" if fold_index is None else f"_fold{fold_index}"
                full_name = f"{group}_{exp_id}_{exp_name}_seed{seed}{fold_suffix}_{timestamp}"
                fold_cfg = deepcopy(train_cfg)
                if fold_index is not None:
                    fold_cfg["data"]["fold_index"] = int(fold_index)
                fold_cfg.update({
                    "experiment": exp_name,
                    "group": group,
                    "id": exp_id,
                    "seed": int(seed),
                    "save_checkpoint": save_checkpoints or bool(exp_cfg.get("save_checkpoint", False)),
                    "output": {
                        "root": str(OUTPUT_ROOT),
                        "checkpoint": str(OUTPUT_ROOT / "checkpoints" / f"{full_name}_best.pth"),
                        "log": str(OUTPUT_ROOT / "logs" / f"{full_name}.log"),
                        "result": str(OUTPUT_ROOT / "results" / f"{full_name}.json"),
                    },
                })
                if epochs_override is not None:
                    fold_cfg["training"]["epochs"] = epochs_override
                    fold_cfg["training"]["early_stop"] = max(epochs_override, 1)
                if max_batches is not None:
                    fold_cfg["training"]["max_train_batches"] = max_batches
                    fold_cfg["training"]["max_eval_batches"] = max_batches
                if num_workers is not None:
                    fold_cfg["loader"]["num_workers"] = num_workers
                generated = OUTPUT_ROOT / "generated_configs" / f"{full_name}.yaml"
                jobs.append({"name": full_name, "config": fold_cfg, "generated": generated})
                if max_experiments and len(jobs) >= max_experiments:
                    return jobs
    return jobs


def write_yaml(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def write_failure_result(job, returncode, stdout, stderr):
    error_log = OUTPUT_ROOT / "logs" / f"{job['name']}.error.log"
    with open(error_log, "w", encoding="utf-8") as f:
        f.write("STDOUT\n")
        f.write(stdout or "")
        f.write("\nSTDERR\n")
        f.write(stderr or "")
    result = {
        "experiment": job["config"]["experiment"],
        "group": job["config"]["group"],
        "id": job["config"]["id"],
        "seed": job["config"]["seed"],
        "status": "failed",
        "returncode": returncode,
        "error_log": str(error_log),
        "checkpoint": "",
        "log": job["config"]["output"]["log"],
        "config": job["config"],
    }
    result_path = Path(job["config"]["output"]["result"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def tail_text(text, max_lines=30):
    lines = (text or "").splitlines()
    return "\n".join(lines[-max_lines:])


def run_jobs(jobs):
    worker = EXPERIMENT_DIR / "train_worker.py"
    for index, job in enumerate(jobs, start=1):
        write_yaml(job["generated"], job["config"])
        print(f"[{index}/{len(jobs)}] Running {job['name']}")
        cmd = [sys.executable, str(worker), "--generated-config", str(job["generated"])]
        try:
            completed = subprocess.run(cmd, cwd=str(ROOT_DIR), capture_output=True, text=True)
        except KeyboardInterrupt:
            print(f"\n  INTERRUPTED: {job['name']}")
            write_failure_result(job, -2, "", "KeyboardInterrupt")
            raise

        if completed.returncode != 0:
            print(f"  FAILED: {job['name']} (code {completed.returncode})")
            stderr_tail = tail_text(completed.stderr)
            stdout_tail = tail_text(completed.stdout)
            if stderr_tail:
                print("  STDERR tail:")
                print(stderr_tail)
            elif stdout_tail:
                print("  STDOUT tail:")
                print(stdout_tail)
            write_failure_result(job, completed.returncode, completed.stdout, completed.stderr)
        else:
            print(f"  OK: {job['name']}")


def run_followups(limit=None):
    scripts = [
        [sys.executable, str(EXPERIMENT_DIR / "run_efficiency.py"), "--results", str(OUTPUT_ROOT / "results")],
        [sys.executable, str(EXPERIMENT_DIR / "run_noise_eval.py"), "--results", str(OUTPUT_ROOT / "results")],
        [sys.executable, str(EXPERIMENT_DIR / "summarize_results.py")],
    ]
    if limit:
        scripts[0].extend(["--limit", str(limit)])
        scripts[1].extend(["--limit", str(limit)])
    for cmd in scripts:
        subprocess.run(cmd, cwd=str(ROOT_DIR), check=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--max-experiments", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--only-ids", nargs="*", default=None)
    parser.add_argument("--folds", nargs="*", type=int, default=None, help="Optional LOSO fold indices, default: 0 1 2 3 4.")
    parser.add_argument("--skip-followups", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_paths = []
    if args.all:
        config_paths = [
            EXPERIMENT_DIR / "configs" / "core_models.yaml",
            EXPERIMENT_DIR / "configs" / "lightweight.yaml",
            EXPERIMENT_DIR / "configs" / "ablations.yaml",
        ]
    elif args.config:
        config_paths = [Path(args.config)]
    else:
        raise SystemExit("Provide --config or --all")

    jobs = []
    remaining = args.max_experiments
    for path in config_paths:
        batch = expand_config(
            path,
            seeds_override=args.seeds,
            epochs_override=args.epochs,
            timestamp=timestamp,
            max_experiments=remaining,
            max_batches=args.max_batches,
            num_workers=args.num_workers,
            only_ids=set(args.only_ids or []),
            folds_override=args.folds,
        )
        jobs.extend(batch)
        if remaining is not None:
            remaining -= len(batch)
            if remaining <= 0:
                break

    run_jobs(jobs)
    if args.all and not args.skip_followups:
        run_followups(limit=args.max_experiments)


if __name__ == "__main__":
    main()
