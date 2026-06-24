import argparse
import csv
import os
import re


EPOCH_RE = re.compile(
    r"Epoch\s+(?P<epoch>\d+) \| Loss: (?P<loss>-?\d+\.\d+) \| "
    r"Mean-CCC: (?P<mean>\d+\.\d+) \| V:(?P<v>\d+\.\d+) A:(?P<a>\d+\.\d+) D:(?P<d>\d+\.\d+)"
)


def parse_best_epoch(log_path):
    best = None
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            match = EPOCH_RE.search(line)
            if not match:
                continue
            row = {
                "best_epoch": int(match.group("epoch")),
                "best_loss": float(match.group("loss")),
                "best_mean_ccc": float(match.group("mean")),
                "best_ccc_v": float(match.group("v")),
                "best_ccc_a": float(match.group("a")),
                "best_ccc_d": float(match.group("d")),
            }
            if best is None or row["best_mean_ccc"] > best["best_mean_ccc"]:
                best = row
    if best is None:
        raise ValueError(f"No epoch rows found in log: {log_path}")
    return best


def read_csv_by_label(path):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8-sig") as f:
        return {row["label"]: row for row in csv.DictReader(f)}


def read_noise_summary(path):
    if not path:
        return {}
    summary = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["condition"] == "clean":
                continue
            key = f"{row['noise_type']}_{float(row['snr_db']):g}db"
            summary[f"{key}_ccc"] = float(row["ccc_total"])
            summary[f"{key}_drop"] = float(row["ccc_drop"])
            summary[f"{key}_retention"] = float(row.get("ccc_retention", 0.0))
    return summary


def parse_named_paths(values):
    pairs = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=PATH, got: {value}")
        name, path = value.split("=", 1)
        pairs.append((name, path))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", nargs="+", required=True, help="NAME=training_log_path entries.")
    parser.add_argument("--efficiency", default=None, help="CSV from eval_efficiency_ls4.py")
    parser.add_argument("--noise", nargs="*", default=[], help="NAME=noise_eval_csv entries.")
    parser.add_argument("--output", default="output/evidence/ls4_evidence_summary.csv")
    args = parser.parse_args()

    efficiency = read_csv_by_label(args.efficiency)
    noise_by_name = {name: read_noise_summary(path) for name, path in parse_named_paths(args.noise)}

    rows = []
    for name, log_path in parse_named_paths(args.logs):
        row = {"label": name, "log": log_path}
        row.update(parse_best_epoch(log_path))

        if name in efficiency:
            eff = efficiency[name]
            for key in ("params_m", "mflops", "latency_ms", "rtf", "noise_frontend"):
                row[key] = eff.get(key, "")
        if name in noise_by_name:
            row.update(noise_by_name[name])
        rows.append(row)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved evidence summary: {args.output}")
    for row in rows:
        print(
            f"{row['label']}: CCC={row['best_mean_ccc']:.4f}, "
            f"ParamsM={row.get('params_m', 'n/a')}, RTF={row.get('rtf', 'n/a')}"
        )


if __name__ == "__main__":
    main()
