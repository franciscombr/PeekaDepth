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
        counts = torch.as_tensor(samples_per_class, dtype=torch.float)
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

class ClassBalancedFocalDiceLoss(nn.Module):
    def __init__(
        self,
        samples_per_class: list,
        gamma: float = 2.0,
        ignore_index: int = None,
        eps: float = 1e-6
    ):
        """
        Implements:
          (1) Class-Balanced Focal Loss  L_CBF = -(1/N) sum_i sum_j w_j (1-p_ij)^γ y_ij log p_ij
          (2) w_j = 1 - f_j / sum_k f_k
          (3) Dice loss per-class
          (4) L = 0.5*Dice + 0.5*L_CBF

        samples_per_class: [f_1, …, f_m]
        """
        super().__init__()
        counts = torch.as_tensor(samples_per_class, dtype=torch.float)
        total = counts.sum()
        # Eq.(2):
        weights = 1.0 - counts / (total + eps)
        # zero out ignored class
        if ignore_index is not None and 0 <= ignore_index < len(weights):
            weights[ignore_index] = 0.0

        self.register_buffer("class_weights", weights)
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, H, W = logits.shape
        N = B * H * W

        # flatten
        logit_flat = logits.permute(0,2,3,1).reshape(-1, C)   # [N, C]
        tgt_flat   = targets.view(-1)                         # [N]

        # mask valid pixels
        if self.ignore_index is not None:
            valid_mask = tgt_flat != self.ignore_index
        else:
            valid_mask = torch.ones_like(tgt_flat, dtype=torch.bool)

        # --- Class‐Balanced Focal (Eq.1) ---
        log_p = F.log_softmax(logit_flat, dim=1)              # [N, C]
        p     = log_p.exp()                                   # [N, C]

        idx        = torch.arange(logit_flat.size(0), device=logits.device)
        true_logp  = log_p[idx, tgt_flat]                     # [N]
        true_p     = p[idx,    tgt_flat]                      # [N]

        focal_fac  = (1 - true_p).pow(self.gamma)              # [N]
        w          = self.class_weights.to(logits.device)     # [C]
        sample_w   = w[tgt_flat]                              # [N]

        cb_focal   = - sample_w * focal_fac * true_logp       # [N]
        cb_focal   = cb_focal[valid_mask].mean()              # scalar

        # --- Dice (Eq.3) ---
        probs      = logit_flat.softmax(dim=1)                # [N, C]
        with torch.no_grad():
            one_hot = F.one_hot(tgt_flat.clamp(min=0), C).float()  # [N, C]

        probs  = probs[valid_mask]
        one_hot= one_hot[valid_mask]

        inter  = (probs * one_hot).sum(dim=0)                  # [C]
        denom  = one_hot.sum(dim=0) + (probs*probs).sum(dim=0) + self.eps  # [C]
        dice_c = 1 - 2*inter/denom                             # [C]

        # drop ignore class
        if self.ignore_index is not None:
            mask_c = torch.ones(C, dtype=torch.bool, device=logits.device)
            mask_c[self.ignore_index] = False
            dice_c = dice_c[mask_c]

        dice_loss = dice_c.mean()                              # scalar

        # --- Combine (Eq.4) ---
        return 0.5 * dice_loss + 0.5 * cb_focal
