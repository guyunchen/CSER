import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedMultiTaskLoss(nn.Module):
    """
    Improved Multi-Task Loss for Continuous SER

    Features:
    - CCC Loss
    - Pearson Correlation Loss
    - MSE Auxiliary Loss
    - Temporal Smoothness Loss
    - Learnable Uncertainty Weighting
    - Valence Task Boosting
    """

    def __init__(
            self,
            num_tasks=3,
            ccc_weight=0.5,
            corr_weight=0.25,
            mse_weight=0.15,
            smooth_weight=0.10,
            task_importance=(1.5, 1.0, 1.0)
    ):
        super(ImprovedMultiTaskLoss, self).__init__()

        self.num_tasks = num_tasks

        # loss weights
        self.ccc_weight = ccc_weight
        self.corr_weight = corr_weight
        self.mse_weight = mse_weight
        self.smooth_weight = smooth_weight

        # task importance (Valence boost)
        self.task_importance = task_importance

        # learnable uncertainty parameters
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    # =========================================================
    # CCC Loss
    # =========================================================
    def ccc_loss(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)

        pred_mean = torch.mean(pred)
        target_mean = torch.mean(target)

        pred_var = torch.var(pred, unbiased=False)
        target_var = torch.var(target, unbiased=False)

        covariance = torch.mean(
            (pred - pred_mean) * (target - target_mean)
        )

        ccc = (2.0 * covariance) / (
                pred_var +
                target_var +
                (pred_mean - target_mean) ** 2 +
                1e-8
        )

        ccc = torch.clamp(ccc, min=-1.0, max=1.0)

        return 1.0 - ccc, ccc

    # =========================================================
    # Pearson Correlation Loss
    # =========================================================
    def corr_loss(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)

        pred_centered = pred - torch.mean(pred)
        target_centered = target - torch.mean(target)

        corr = torch.sum(pred_centered * target_centered) / (
                torch.sqrt(torch.sum(pred_centered ** 2)) *
                torch.sqrt(torch.sum(target_centered ** 2)) +
                1e-8
        )

        corr = torch.clamp(corr, min=-1.0, max=1.0)

        return 1.0 - corr, corr

    # =========================================================
    # Temporal Smoothness Loss
    # =========================================================
    def smoothness_loss(self, pred):
        """
        pred shape:
            [B, T] or [B, T, C]

        If no temporal dimension exists,
        smoothness loss returns 0.
        """

        if pred.dim() < 2:
            return torch.tensor(
                0.0,
                device=pred.device,
                dtype=pred.dtype
            )

        if pred.shape[1] <= 1:
            return torch.tensor(
                0.0,
                device=pred.device,
                dtype=pred.dtype
            )

        return torch.mean(
            torch.abs(pred[:, 1:] - pred[:, :-1])
        )

    # =========================================================
    # Forward
    # =========================================================
    def forward(self, preds, targets):
        """
        preds shape:
            [B, num_tasks]

        or:
            [B, T, num_tasks]

        targets shape:
            same as preds
        """

        total_loss = 0.0
        metrics = {}

        task_names = {
            0: 'v',
            1: 'a',
            2: 'd'
        }

        # ============================================
        # Case 1: [B, num_tasks]
        # ============================================
        if preds.dim() == 2:

            for i in range(self.num_tasks):

                pred_i = preds[:, i]
                target_i = targets[:, i]

                # CCC
                cur_ccc_loss, cur_ccc = self.ccc_loss(
                    pred_i,
                    target_i
                )

                # Corr
                cur_corr_loss, cur_corr = self.corr_loss(
                    pred_i,
                    target_i
                )

                # MSE
                cur_mse_loss = F.mse_loss(
                    pred_i,
                    target_i
                )

                # Combined base loss
                base_loss = (
                        self.ccc_weight * cur_ccc_loss +
                        self.corr_weight * cur_corr_loss +
                        self.mse_weight * cur_mse_loss
                )

                # uncertainty weighting
                precision = torch.exp(-self.log_vars[i])

                weighted_loss = (
                        self.task_importance[i] *
                        precision *
                        base_loss +
                        self.log_vars[i]
                )

                total_loss += weighted_loss

                name = task_names.get(i, f"t{i}")

                metrics[f"{name}_ccc"] = cur_ccc.item()
                metrics[f"{name}_corr"] = cur_corr.item()
                metrics[f"{name}_mse"] = cur_mse_loss.item()
                metrics[f"{name}_weight"] = precision.item()

        # ============================================
        # Case 2: [B, T, num_tasks]
        # ============================================
        elif preds.dim() == 3:

            for i in range(self.num_tasks):

                pred_i = preds[:, :, i]
                target_i = targets[:, :, i]

                # flatten for CCC/Corr
                pred_flat = pred_i.reshape(-1)
                target_flat = target_i.reshape(-1)

                # CCC
                cur_ccc_loss, cur_ccc = self.ccc_loss(
                    pred_flat,
                    target_flat
                )

                # Corr
                cur_corr_loss, cur_corr = self.corr_loss(
                    pred_flat,
                    target_flat
                )

                # MSE
                cur_mse_loss = F.mse_loss(
                    pred_i,
                    target_i
                )

                # Smoothness
                cur_smooth_loss = self.smoothness_loss(
                    pred_i
                )

                # Combined Loss
                base_loss = (
                        self.ccc_weight * cur_ccc_loss +
                        self.corr_weight * cur_corr_loss +
                        self.mse_weight * cur_mse_loss +
                        self.smooth_weight * cur_smooth_loss
                )

                # uncertainty weighting
                precision = torch.exp(-self.log_vars[i])

                weighted_loss = (
                        self.task_importance[i] *
                        precision *
                        base_loss +
                        self.log_vars[i]
                )

                total_loss += weighted_loss

                name = task_names.get(i, f"t{i}")

                metrics[f"{name}_ccc"] = cur_ccc.item()
                metrics[f"{name}_corr"] = cur_corr.item()
                metrics[f"{name}_mse"] = cur_mse_loss.item()
                metrics[f"{name}_smooth"] = cur_smooth_loss.item()
                metrics[f"{name}_weight"] = precision.item()

        else:
            raise ValueError(
                f"Unsupported preds shape: {preds.shape}"
            )

        metrics["loss_total"] = total_loss.item()

        return total_loss, metrics


class LearnableMultiTaskLoss(ImprovedMultiTaskLoss):
    """Backward-compatible wrapper for older training scripts."""

    def __init__(self, num_tasks=2, mse_helper_weight=0.2, **kwargs):
        kwargs.setdefault("mse_weight", mse_helper_weight)
        super().__init__(num_tasks=num_tasks, **kwargs)
