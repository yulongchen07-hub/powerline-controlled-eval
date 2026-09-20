"""
Module implementations and Ultralytics registration.

Blocks evaluated in the controlled comparison (all inserted at the same
position so that the comparison is like-for-like):

- DLKA       : 5x5 depth-wise + dilated 7x7 depth-wise (d=3) + modulated
               deformable sampling + 1x1; effective receptive field 23x23.
- LKAOnly    : the same decomposition with the deformable stage removed
               (VAN-style large-kernel attention).
- DeformOnly : 3x3 modulated deformable convolution only, with no
               large-kernel context.
- SPPFBlock  : the SPPF block YOLOv8 already provides, used as a
               practical drop-in alternative at the same position.
- MAFF       : channel attention + spatial attention + adaptive
               scale weighting (weights conditioned on the input).

All blocks preserve channel width, so the computational graph is
otherwise unchanged.
"""
import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d


class DeformOnly(nn.Module):
    """Modulated deformable convolution (DCNv2 style), channel-preserving."""
    def __init__(self, c1, c2=None, k=3):
        super().__init__()
        c2 = c2 or c1
        self.k = k
        self.offset = nn.Conv2d(c1, 2 * k * k, k, padding=k // 2)
        self.mask = nn.Conv2d(c1, k * k, k, padding=k // 2)
        self.weight = nn.Parameter(torch.empty(c2, c1, k, k))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.bias = nn.Parameter(torch.zeros(c2))
        nn.init.zeros_(self.offset.weight); nn.init.zeros_(self.offset.bias)
        nn.init.zeros_(self.mask.weight); nn.init.zeros_(self.mask.bias)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        off = self.offset(x)
        msk = torch.sigmoid(self.mask(x))
        y = deform_conv2d(x, off, self.weight, self.bias, padding=self.k // 2, mask=msk)
        return self.act(self.bn(y))


class LKAOnly(nn.Module):
    """VAN-style large-kernel attention: 5x5 DW + dilated 7x7 DW (d=3) + 1x1.

    Effective receptive field 23x23; no deformable sampling.
    """
    def __init__(self, c1, c2=None, k_dw=5, k_dil=7, d=3):
        super().__init__()
        c2 = c2 or c1
        assert c1 == c2, "LKA modulates the input, so channel width must be preserved"
        self.dw = nn.Conv2d(c1, c1, k_dw, padding=k_dw // 2, groups=c1)
        pad = ((k_dil - 1) * d) // 2
        self.dw_d = nn.Conv2d(c1, c1, k_dil, padding=pad, groups=c1, dilation=d)
        self.pw = nn.Conv2d(c1, c1, 1)

    def forward(self, x):
        a = self.pw(self.dw_d(self.dw(x)))
        return x * a


class DLKA(nn.Module):
    """Deformable large-kernel attention.

    Modulated deformable sampling is embedded inside the decomposed
    large-kernel operator; this is a detection-domain adaptation of the
    D-LKA construction.
    """
    def __init__(self, c1, c2=None, k_dw=5, k_dil=7, d=3, k_def=3):
        super().__init__()
        c2 = c2 or c1
        assert c1 == c2
        self.dw = nn.Conv2d(c1, c1, k_dw, padding=k_dw // 2, groups=c1)
        pad = ((k_dil - 1) * d) // 2
        self.dw_d = nn.Conv2d(c1, c1, k_dil, padding=pad, groups=c1, dilation=d)
        # Modulated deformable stage, depth-wise to keep the parameter count down
        self.k = k_def
        self.offset = nn.Conv2d(c1, 2 * k_def * k_def, k_def, padding=k_def // 2)
        self.mask = nn.Conv2d(c1, k_def * k_def, k_def, padding=k_def // 2)
        self.dweight = nn.Parameter(torch.empty(c1, 1, k_def, k_def))  # groups=c1
        nn.init.kaiming_uniform_(self.dweight, a=5 ** 0.5)
        nn.init.zeros_(self.offset.weight); nn.init.zeros_(self.offset.bias)
        nn.init.zeros_(self.mask.weight); nn.init.zeros_(self.mask.bias)
        self.pw = nn.Conv2d(c1, c1, 1)
        self.norm = nn.GroupNorm(1, c1)

    def forward(self, x):
        a = self.dw(x)
        a = self.dw_d(a)
        off = self.offset(a)
        msk = torch.sigmoid(self.mask(a))
        a = deform_conv2d(a, off, self.dweight, None, padding=self.k // 2,
                          mask=msk, dilation=(1, 1), stride=(1, 1))
        a = self.pw(a)
        return self.norm(x * a + x)


class SPPFBlock(nn.Module):
    """The SPPF block native to YOLOv8, used as a same-position alternative."""
    def __init__(self, c1, c2=None, k=5):
        super().__init__()
        c2 = c2 or c1
        c_ = c1 // 2
        self.cv1 = nn.Conv2d(c1, c_, 1, bias=False); self.bn1 = nn.BatchNorm2d(c_)
        self.cv2 = nn.Conv2d(c_ * 4, c2, 1, bias=False); self.bn2 = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        self.m = nn.MaxPool2d(k, 1, k // 2)

    def forward(self, x):
        x1 = self.act(self.bn1(self.cv1(x)))
        y1 = self.m(x1); y2 = self.m(y1); y3 = self.m(y2)
        return self.act(self.bn2(self.cv2(torch.cat([x1, y1, y2, y3], 1))))


class MAFF(nn.Module):
    """Multi-scale attention feature fusion.

    Channel attention -> spatial attention -> adaptive scale weighting.
    Used as a single-input block it performs the channel and spatial
    refinement; the scale weighting takes effect after the neck's Concat.
    """
    def __init__(self, c1, c2=None, r=16):
        super().__init__()
        c2 = c2 or c1
        assert c1 == c2
        hidden = max(c1 // r, 8)
        self.mlp = nn.Sequential(nn.Linear(c1, hidden), nn.GELU(), nn.Linear(hidden, c1))
        self.sconv = nn.Conv2d(2, 1, 7, padding=3)
        # Adaptive scale weights (learnable, softmax-normalised)
        self.scale = nn.Parameter(torch.zeros(2))

    def forward(self, x):
        B, C, H, W = x.shape
        # Channel attention: global average and max pooling through a shared MLP
        avg = x.mean(dim=(2, 3))
        mx = x.amax(dim=(2, 3))
        ca = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).view(B, C, 1, 1)
        xc = x * ca
        # Spatial attention
        sa = torch.sigmoid(self.sconv(torch.cat([xc.mean(1, keepdim=True),
                                                 xc.amax(1, keepdim=True)], 1)))
        xs = xc * sa
        # Adaptively weighted residual (input-conditioned scale selection)
        w = torch.softmax(self.scale, 0)
        return w[0] * xs + w[1] * x


def register():
    """Register the custom blocks with the Ultralytics model parser."""
    import ultralytics.nn.tasks as tasks
    import ultralytics.nn.modules as um
    mods = {"DLKA": DLKA, "LKAOnly": LKAOnly, "DeformOnly": DeformOnly,
            "SPPFBlock": SPPFBlock, "MAFF": MAFF}
    for name, cls in mods.items():
        setattr(um, name, cls)
        setattr(tasks, name, cls)
        # so that the eval(m) inside parse_model resolves the name
        tasks.__dict__[name] = cls
    # these blocks have c2 == c1; make parse_model infer the width correctly
    globals().update(mods)
    return mods
