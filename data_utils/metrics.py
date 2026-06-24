import torch


def ccc_score(pred, target):
    pred = pred.view(-1)
    target = target.view(-1)
    pred_mean = torch.mean(pred)
    target_mean = torch.mean(target)
    pred_var = torch.var(pred, unbiased=False)
    target_var = torch.var(target, unbiased=False)
    cov = torch.mean((pred - pred_mean) * (target - target_mean))
    denom = pred_var + target_var + (pred_mean - target_mean) ** 2 + 1e-8
    return (2 * cov) / denom


def pcc_score(pred, target):
    pred, target = pred.view(-1), target.view(-1)
    if len(pred) < 2: return torch.tensor(0.0)
    combined = torch.stack([pred, target])
    pcc = torch.corrcoef(combined)[0, 1]
    return torch.nan_to_num(pcc, nan=0.0)


def calculate_metrics(preds, targets):
    preds = preds.detach().cpu()
    targets = targets.detach().cpu()
    num_dims = preds.shape[1]

    # 计算 Valence
    ccc_v = ccc_score(preds[:, 0], targets[:, 0]).item()
    mae_v = torch.mean(torch.abs(preds[:, 0] - targets[:, 0])).item()

    # 计算 Arousal
    ccc_a = ccc_score(preds[:, 1], targets[:, 1]).item()
    mae_a = torch.mean(torch.abs(preds[:, 1] - targets[:, 1])).item()

    # 计算 Dominance
    if num_dims >= 3:
        ccc_d = ccc_score(preds[:, 2], targets[:, 2]).item()
        mae_d = torch.mean(torch.abs(preds[:, 2] - targets[:, 2])).item()
    else:
        ccc_d = mae_d = 0.0

    # --- 核心修改：计算总平均 MAE ---
    mae_total = (mae_v + mae_a + mae_d) / num_dims
    ccc_total = (ccc_v + ccc_a + ccc_d) / num_dims

    return {
        "ccc_v": ccc_v, "ccc_a": ccc_a, "ccc_d": ccc_d,
        "mae_v": mae_v, "mae_a": mae_a, "mae_d": mae_d,
        "mae_total": mae_total,  # 总 MAE
        "ccc_total": ccc_total
    }