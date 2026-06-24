import torch


def ccc_score(pred, target):
    # 保持不变，逻辑正确
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
    # 保持不变
    pred, target = pred.view(-1), target.view(-1)
    if len(pred) < 2: return torch.tensor(0.0)
    # 使用 corrcoef 更加简洁
    combined = torch.stack([pred, target])
    pcc = torch.corrcoef(combined)[0, 1]
    return torch.nan_to_num(pcc, nan=0.0)


def calculate_metrics(preds, targets):
    """
    针对 MESS 数据集 (VA) 优化的评估函数
    """
    preds = preds.detach().cpu()
    targets = targets.detach().cpu()

    # 1. 计算 Valence (索引 0)
    ccc_v = ccc_score(preds[:, 0], targets[:, 0]).item()
    pcc_v = pcc_score(preds[:, 0], targets[:, 0]).item()
    mae_v = torch.mean(torch.abs(preds[:, 0] - targets[:, 0])).item()

    # 2. 计算 Arousal (索引 1)
    ccc_a = ccc_score(preds[:, 1], targets[:, 1]).item()
    pcc_a = pcc_score(preds[:, 1], targets[:, 1]).item()
    mae_a = torch.mean(torch.abs(preds[:, 1] - targets[:, 1])).item()

    # 3. 计算总平均值 (仅针对已有的 2 个维度)
    ccc_mean = (ccc_v + ccc_a) / 2
    pcc_mean = (pcc_v + pcc_a) / 2
    mae_mean = (mae_v + mae_a) / 2

    # 返回结果：移除了 Dominance，加入了 PCC
    return {
        "ccc_v": ccc_v, "pcc_v": pcc_v, "mae_v": mae_v,
        "ccc_a": ccc_a, "pcc_a": pcc_a, "mae_a": mae_a,
        "ccc_mean": ccc_mean,
        "pcc_mean": pcc_mean,
        "mae_mean": mae_mean
    }