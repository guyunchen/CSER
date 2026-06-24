import argparse
import csv
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = EXPERIMENT_DIR / "output"


def read_json_results(path):
    rows = []
    for item in sorted(Path(path).glob("*.json")):
        with open(item, "r", encoding="utf-8") as f:
            row = json.load(f)
        flat = {
            "result_file": str(item),
            "result_mtime": item.stat().st_mtime,
            "experiment": row.get("experiment", ""),
            "group": row.get("group", ""),
            "id": row.get("id", ""),
            "seed": row.get("seed", ""),
            "status": row.get("status", ""),
            "protocol": row.get("protocol", ""),
            "speaker_independent": row.get("speaker_independent", ""),
            "fold_index": row.get("split", {}).get("fold_index", ""),
            "train_sessions": "|".join(row.get("split", {}).get("train_sessions", [])),
            "val_session": row.get("split", {}).get("val_session", ""),
            "test_session": row.get("split", {}).get("test_session", ""),
            "best_epoch": row.get("best_epoch", ""),
            "mean_ccc": row.get("mean_ccc", ""),
            "v_ccc": row.get("v_ccc", ""),
            "a_ccc": row.get("a_ccc", ""),
            "d_ccc": row.get("d_ccc", ""),
            "mean_mae": row.get("mean_mae", ""),
            "v_mae": row.get("v_mae", ""),
            "a_mae": row.get("a_mae", ""),
            "d_mae": row.get("d_mae", ""),
            "val_mean_ccc": row.get("val_mean_ccc", ""),
            "val_v_ccc": row.get("val_v_ccc", ""),
            "val_a_ccc": row.get("val_a_ccc", ""),
            "val_d_ccc": row.get("val_d_ccc", ""),
            "checkpoint": row.get("checkpoint", ""),
            "log": row.get("log", ""),
        }
        rows.append(flat)
    return rows


def read_csv(path):
    if not Path(path).exists():
        return []
    with open(path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def key(row):
    return (
        str(row.get("experiment", "")),
        str(row.get("group", "")),
        str(row.get("id", "")),
        str(row.get("seed", "")),
        str(row.get("fold_index", "")),
    )


def merge_results(results, efficiency):
    eff_by_key = {key(row): row for row in efficiency}
    rows = []
    for row in results:
        merged = dict(row)
        eff = eff_by_key.get(key(row), {})
        for field in ("params", "trainable_params", "mflops", "latency_ms", "rtf", "checkpoint_mb", "gpu_memory_mb"):
            merged[field] = eff.get(field, "")
        rows.append(merged)
    return rows


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    fields = []
    for row in rows:
        for name in row:
            if name not in fields:
                fields.append(name)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _to_float(value):
    try:
        if value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def aggregate_loso(rows):
    loso_rows = [row for row in rows if row.get("protocol") == "loso" and row.get("status") == "success"]
    latest_by_fold = {}
    for row in loso_rows:
        dedupe_key = (
            row.get("group", ""),
            row.get("id", ""),
            row.get("experiment", ""),
            row.get("seed", ""),
            row.get("fold_index", ""),
        )
        current = latest_by_fold.get(dedupe_key)
        if current is None or _to_float(row.get("result_mtime")) > _to_float(current.get("result_mtime")):
            latest_by_fold[dedupe_key] = row

    groups = defaultdict(list)
    for row in latest_by_fold.values():
        key_fields = ("group", "id", "experiment", "seed")
        groups[tuple(row.get(field, "") for field in key_fields)].append(row)

    metric_names = [
        "mean_ccc", "v_ccc", "a_ccc", "d_ccc",
        "mean_mae", "v_mae", "a_mae", "d_mae",
        "val_mean_ccc", "val_v_ccc", "val_a_ccc", "val_d_ccc",
    ]
    out = []
    for (group, exp_id, experiment, seed), fold_rows in sorted(groups.items()):
        result = {
            "group": group,
            "id": exp_id,
            "experiment": experiment,
            "seed": seed,
            "protocol": "loso",
            "n_folds": len(fold_rows),
            "folds": "|".join(str(row.get("fold_index", "")) for row in sorted(fold_rows, key=lambda x: str(x.get("fold_index", "")))),
        }
        for metric in metric_names:
            values = [_to_float(row.get(metric)) for row in fold_rows]
            values = [value for value in values if value is not None]
            if not values:
                result[f"{metric}_mean"] = ""
                result[f"{metric}_std"] = ""
                continue
            result[f"{metric}_mean"] = statistics.mean(values)
            result[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        out.append(result)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(OUTPUT_ROOT / "results"))
    parser.add_argument("--efficiency", default=str(OUTPUT_ROOT / "efficiency" / "efficiency.csv"))
    parser.add_argument("--noise", default=str(OUTPUT_ROOT / "noise" / "noise_results.csv"))
    parser.add_argument("--output-dir", default=str(OUTPUT_ROOT / "summaries"))
    args = parser.parse_args()

    results = read_json_results(args.results)
    efficiency = read_csv(args.efficiency)
    noise = read_csv(args.noise)
    merged = merge_results(results, efficiency)
    loso = aggregate_loso(merged)

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "summary_all.csv", merged)
    write_csv(output_dir / "summary_core.csv", [row for row in merged if row.get("group") == "core_models"])
    write_csv(output_dir / "summary_ablation.csv", [row for row in merged if row.get("group") == "ablations"])
    write_csv(output_dir / "summary_loso.csv", loso)
    write_csv(output_dir / "summary_noise.csv", noise)
    print(f"Saved summaries to: {output_dir}")


if __name__ == "__main__":
    main()
