import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_utils.metrics import calculate_metrics
from models.unified_ser import UnifiedSERModel
from training.data import build_iemocap_datasets, build_loaders
from training.engine import load_model_state


def _canonical_id(exp_id):
    aliases = {
        "A3_B1": "B1",
        "A3_B3": "B3",
        "A3_B6": "B6",
        "A3_B1_TOPK": "B1_TOPK",
    }
    return aliases.get(exp_id, exp_id)


def _checkpoint_exists(row):
    checkpoint = row.get("checkpoint", "")
    return bool(checkpoint) and os.path.exists(checkpoint)


def collect_rows(result_dirs, ids, seeds):
    ids = {_canonical_id(item) for item in ids}
    seeds = {int(item) for item in seeds}
    by_fold_model = {}

    for result_dir in result_dirs:
        for path in sorted(Path(result_dir).glob("*.json")):
            with open(path, "r", encoding="utf-8") as f:
                row = json.load(f)
            if row.get("status") != "success":
                continue
            canonical_id = _canonical_id(row.get("id"))
            if canonical_id not in ids or int(row.get("seed")) not in seeds:
                continue
            if not _checkpoint_exists(row):
                continue
            fold = int(row.get("split", {}).get("fold_index"))
            key = (canonical_id, int(row["seed"]), fold)
            prev = by_fold_model.get(key)
            if prev is None or path.stat().st_mtime > prev["_mtime"]:
                row["_mtime"] = path.stat().st_mtime
                row["_result_path"] = str(path)
                row["_canonical_id"] = canonical_id
                by_fold_model[key] = row

    by_fold = defaultdict(list)
    for (_, _, fold), row in by_fold_model.items():
        by_fold[fold].append(row)
    return by_fold


@torch.no_grad()
def predict_one(row, fold, device):
    cfg = json.loads(json.dumps(row["config"]))
    cfg["data"]["fold_index"] = int(fold)

    datasets, split = build_iemocap_datasets(cfg["data"])
    loaders = build_loaders(datasets, cfg.get("loader", {"batch_size": 32, "num_workers": 0}), device)
    loader = loaders["test"]

    model = UnifiedSERModel(cfg["model"]).to(device)
    checkpoint = torch.load(row["checkpoint"], map_location=device)
    load_model_state(model, checkpoint["model"])
    model.eval()

    preds, labels = [], []
    for feats, y, lengths in loader:
        feats = feats.to(device)
        lengths = lengths.to(device)
        out = torch.sigmoid(model(feats, lengths=lengths)["emotion"])
        preds.append(out.cpu())
        labels.append(y.cpu())
    return torch.cat(preds), torch.cat(labels), split


def run_ensemble(args):
    by_fold = collect_rows(args.results, args.ids, args.seeds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_rows = []

    for fold in sorted(by_fold):
        rows = sorted(by_fold[fold], key=lambda r: (r["_canonical_id"], int(r["seed"])))
        pred_list = []
        labels = None
        split = None
        for row in rows:
            pred, labels, split = predict_one(row, fold, device)
            pred_list.append(pred)
        if not pred_list:
            continue
        avg_pred = torch.stack(pred_list, dim=0).mean(dim=0)
        metrics = calculate_metrics(avg_pred, labels)
        out_rows.append(
            {
                "tag": args.tag,
                "fold": fold,
                "test_session": split["test_session"],
                "n_models": len(rows),
                "models": "|".join(f'{r["_canonical_id"]}_seed{r["seed"]}' for r in rows),
                "mean_ccc": metrics["ccc_total"],
                "v_ccc": metrics["ccc_v"],
                "a_ccc": metrics["ccc_a"],
                "d_ccc": metrics["ccc_d"],
                "mean_mae": metrics["mae_total"],
                "v_mae": metrics["mae_v"],
                "a_mae": metrics["mae_a"],
                "d_mae": metrics["mae_d"],
            }
        )

    df = pd.DataFrame(out_rows)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.tag}.csv"
    df.to_csv(out_path, index=False)
    print(df)
    print("\nMean:")
    print(df[["mean_ccc", "v_ccc", "a_ccc", "d_ccc", "mean_mae"]].mean())
    print("\nStd:")
    print(df[["mean_ccc", "v_ccc", "a_ccc", "d_ccc", "mean_mae"]].std())
    print(f"\nSaved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+", default=["experiments/output/results"])
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--tag", default="ensemble")
    parser.add_argument("--output-dir", default="experiments/output/summaries")
    args = parser.parse_args()
    run_ensemble(args)


if __name__ == "__main__":
    main()
