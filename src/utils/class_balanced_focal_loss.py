import torch
import torch.nn as nn
import torch.nn.functional as F

class BalancedFocalLoss(nn.Module):
    def __init__(
        self,
        samples_per_class: list,
        gamma: float = 2.0,
        ignore_index: int = None,
        reduction: str = "mean",
        eps: float = 1e-6
    ):
        """
        samples_per_class: list or 1D array of length C with the number of pixels for each class.
        gamma: focusing parameter γ ≥ 0.
        ignore_index: class index to ignore (will zero out its weight and mask it in the loss).
        reduction: 'mean', 'sum', or 'none'.
        eps: small constant to avoid division by zero.
        """
        super().__init__()
        counts = torch.tensor(samples_per_class, dtype=torch.float)
        # Inverse‐frequency weights
        weights = 1.0 / (counts + eps)
        # Normalize so that sum(weights) = C
        weights = weights / weights.sum() * counts.numel()
        # Zero out the ignored class weight
        if ignore_index is not None and 0 <= ignore_index < len(weights):
            weights[ignore_index] = 0.0

        # Register as buffer so it moves with .to(device)
        self.register_buffer("class_weights", weights)
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: Tensor of shape [B, C, H, W]
        targets: LongTensor of shape [B, H, W] with values in {0,...,C-1}
        """
        B, C, H, W = logits.shape
        # Flatten predictions and targets to [N, C] and [N]
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)
        targets_flat = targets.view(-1)

        # Compute log‐probs and probs
        log_probs = F.log_softmax(logits_flat, dim=1)  # [N, C]
        probs = torch.exp(log_probs)                   # [N, C]

        # Gather the log‐prob and prob for the true class
        idx = torch.arange(logits_flat.size(0), device=logits.device)
        target_log_p = log_probs[idx, targets_flat]    # [N]
        target_p     = probs[idx, targets_flat]        # [N]

        # Focusing term
        focal_factor = (1.0 - target_p).pow(self.gamma)  # [N]

        # Per‐sample class weight
        sample_w = self.class_weights[targets_flat]      # [N]

        # Raw loss
        loss = - sample_w * focal_factor * target_log_p  # [N]

        # Mask out ignore_index
        if self.ignore_index is not None:
            valid = targets_flat != self.ignore_index
            loss = loss[valid]

        # Reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss
