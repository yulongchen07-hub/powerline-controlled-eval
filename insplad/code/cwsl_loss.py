"""
Class-Wise Soft Label (CWSL) classification loss.

The parameterisation implemented here follows the analysis in Section 3.4
of the accompanying paper, and differs from the formulation commonly used
for this family of losses in three respects:

  1) The soft-label weights are normalised by the largest weight rather
     than by their sum:  w_hat_c = w_c / max_j w_j = (N_min/N_c)^gamma, in (0, 1].
     Sum normalisation drives every target confidence towards 1 - eps as
     the class count grows; at C = 18 the head-to-tail spread falls below
     one percentage point, which makes the scheme numerically
     indistinguishable from uniform label smoothing.

  2) The focal exponent coefficient is negative:
     gamma_c = gamma_base + beta * (1 - N_c/N_max), with beta < 0.
     Because (1 - p) < 1, raising the exponent lowers a sample's
     contribution, so a positive beta would down-weight the rare classes —
     the opposite of the intended effect.

  3) The label weights and the loss weights are separated: labels use max
     normalisation so that they remain valid probabilities, while the loss
     weights are normalised to unit mean to preserve the overall loss
     scale (the convention used by Class-Balanced Loss).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CWSLBCE(nn.Module):
    def __init__(self, counts, gamma=0.5, eps=0.1, gamma_base=2.0, beta=-1.0):
        super().__init__()
        c = torch.as_tensor(counts, dtype=torch.float32).clamp(min=1.0)
        w = (c.min() / c) ** gamma                              # max-normalised, in (0, 1]
        self.register_buffer("y_soft", 1.0 - eps * (1.0 - w))   # target confidence of the true class
        self.register_buffer("w_loss", w / w.mean())            # loss weights, unit mean
        self.register_buffer("gamma_c", gamma_base + beta * (1.0 - c / c.max()))

    def forward(self, pred, target):
        """pred: class logits before the sigmoid; target: soft alignment scores (B, A, C)."""
        ys = self.y_soft.to(pred.device).view(1, 1, -1)
        gc = self.gamma_c.to(pred.device).view(1, 1, -1)
        wl = self.w_loss.to(pred.device).view(1, 1, -1)
        t = target * ys                                # frequency-aware soft labels
        ce = F.binary_cross_entropy_with_logits(pred, t, reduction="none")
        p = torch.sigmoid(pred)
        pt = p * t + (1.0 - p) * (1.0 - t)             # agreement between prediction and target
        mod = (1.0 - pt).clamp_(min=1e-6).pow(gc)      # class-adaptive focal modulation
        return ce * mod * wl


def describe(counts, gamma=0.5, eps=0.1, gamma_base=2.0, beta=-1.0, names=None):
    """Return the per-class factors, target confidences and focusing exponents.

    Reproduces the numerical example quoted in Section 3.4 for any class
    distribution.
    """
    c = torch.as_tensor(counts, dtype=torch.float32).clamp(min=1.0)
    w = (c.min() / c) ** gamma
    ys = 1.0 - eps * (1.0 - w)
    gc = gamma_base + beta * (1.0 - c / c.max())
    wl = w / w.mean()
    rows = []
    for i in range(len(c)):
        rows.append({
            "class": names[i] if names else str(i),
            "N_c": int(c[i].item()),
            "w_hat": round(w[i].item(), 4),
            "y_soft": round(ys[i].item(), 4),
            "gamma_c": round(gc[i].item(), 3),
            "w_loss": round(wl[i].item(), 4),
        })
    return rows
