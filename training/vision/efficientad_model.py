"""EfficientAD-S (Batzner et al., "EfficientAD: Accurate Visual Anomaly Detection at
Millisecond-Level Latencies", WACV 2024) as a self-contained, export-friendly PyTorch module.

Written here instead of importing anomalib so that:
  - the exact graph we deploy (ExecuTorch QNN int8 on the Hexagon NPU, ADR-012) is the graph
    we train in W3;
  - the build environment's pinned torch isn't disturbed by anomalib's dependencies.

Networks (paper, "S" variant):
  teacher   PDN-S -> 384 channels    (distilled from a pretrained backbone during training)
  student   PDN-S -> 768 channels    (first 384 mimic the teacher, last 384 mimic the autoencoder)
  autoenc.  conv encoder to 1x1, bilinear-upsampling decoder -> 384 channels
Inference: two squared-difference maps (student-teacher, student-autoencoder), each scaled by
quantiles from validation images, averaged, upsampled to the input size.
Ops used: conv2d, relu, avg_pool2d, bilinear interpolate, mean, sub/mul/pow - all common.
"""
import torch
import torch.nn.functional as F
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
OUT_CHANNELS = 384


class PDNSmall(nn.Module):
    """Patch description network, small variant, padding=False everywhere (the paper's/anomalib's
    default). Layer names match anomalib's pretrained teacher checkpoint (conv1..conv4), so it
    loads with strict=True. 256 px input -> 56x56 feature map. (An earlier version padded only the
    pooling layers, giving 58x58 and matching neither configuration - P-44.)"""

    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 128, 4)
        self.conv2 = nn.Conv2d(128, 256, 4)
        self.conv3 = nn.Conv2d(256, 256, 3)
        self.conv4 = nn.Conv2d(256, out_channels, 4)
        self.pool = nn.AvgPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        return self.conv4(x)


class AutoEncoder(nn.Module):
    def __init__(self, out_channels: int, image_size: int, out_size: tuple[int, int]) -> None:
        super().__init__()
        s = image_size
        self.enc = nn.ModuleList([
            nn.Conv2d(3, 32, 4, 2, 1), nn.Conv2d(32, 32, 4, 2, 1), nn.Conv2d(32, 64, 4, 2, 1),
            nn.Conv2d(64, 64, 4, 2, 1), nn.Conv2d(64, 64, 4, 2, 1),
        ])
        self.enc_last = nn.Conv2d(64, 64, s // 32)  # -> 1x1
        # decoder target sizes follow the paper's schedule for a 256 px input
        self.dec_sizes = [s // 64 - 1, s // 32, s // 16 - 1, s // 8, s // 4 - 1, s // 2 - 1]
        self.dec = nn.ModuleList([nn.Conv2d(64, 64, 4, 1, 2) for _ in self.dec_sizes])
        self.out_size = out_size
        self.dec_last1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.dec_last2 = nn.Conv2d(64, out_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.enc:
            x = F.relu(conv(x))
        x = self.enc_last(x)
        for size, conv in zip(self.dec_sizes, self.dec):
            x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
            x = F.relu(conv(x))  # (dropout during training only)
        x = F.interpolate(x, size=self.out_size, mode="bilinear", align_corners=False)
        x = F.relu(self.dec_last1(x))
        return self.dec_last2(x)


class EfficientADS(nn.Module):
    """Inference wrapper: image (1x3xHxW, ImageNet-normalised) -> anomaly map (1x1xHxW)."""

    def __init__(self, image_size: int = 256) -> None:
        super().__init__()
        self.image_size = image_size
        self.teacher = PDNSmall(OUT_CHANNELS)
        self.student = PDNSmall(2 * OUT_CHANNELS)
        with torch.no_grad():
            pdn_out = self.teacher(torch.zeros(1, 3, image_size, image_size)).shape[-2:]
        self.autoencoder = AutoEncoder(OUT_CHANNELS, image_size, tuple(pdn_out))
        # Set during training: teacher channel stats, and the per-map quantile normalisation folded
        # into scale/shift tensors of shape 1x1x1x1. (Indexing a 2-element quantile tensor broke the
        # QNN backend's scalar-removal pass, P-35; broadcasting 4-D tensors is safe for every backend.)
        self.register_buffer("teacher_mean", torch.zeros(1, OUT_CHANNELS, 1, 1))
        self.register_buffer("teacher_std", torch.ones(1, OUT_CHANNELS, 1, 1))
        self.register_buffer("st_scale", torch.full((1, 1, 1, 1), 0.1))
        self.register_buffer("st_shift", torch.zeros(1, 1, 1, 1))
        self.register_buffer("ae_scale", torch.full((1, 1, 1, 1), 0.1))
        self.register_buffer("ae_shift", torch.zeros(1, 1, 1, 1))

    @torch.no_grad()
    def set_quantiles(self, st_a: float, st_b: float, ae_a: float, ae_b: float) -> None:
        """EfficientAD map normalisation 0.1 * (m - a) / (b - a), written as m * scale + shift."""
        for name, a, b in (("st", st_a, st_b), ("ae", ae_a, ae_b)):
            getattr(self, f"{name}_scale").fill_(0.1 / (b - a))
            getattr(self, f"{name}_shift").fill_(-0.1 * a / (b - a))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = (self.teacher(x) - self.teacher_mean) / self.teacher_std
        s = self.student(x)
        ae = self.autoencoder(x)
        map_st = torch.mean((t - s[:, :OUT_CHANNELS]) ** 2, dim=1, keepdim=True)
        map_ae = torch.mean((ae - s[:, OUT_CHANNELS:]) ** 2, dim=1, keepdim=True)
        map_st = map_st * self.st_scale + self.st_shift
        map_ae = map_ae * self.ae_scale + self.ae_shift
        amap = 0.5 * map_st + 0.5 * map_ae
        # The unpadded PDN's 56x56 output covers the image minus a border; padding by 4 cells before
        # upsampling aligns each cell with the patch it sees (original EfficientAD code / anomalib
        # pad_maps=True). Without it the heatmap is shifted/stretched (P-50).
        amap = F.pad(amap, (4, 4, 4, 4))
        return F.interpolate(amap, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)


def preprocess(image_rgb_uint8: torch.Tensor, image_size: int = 256) -> torch.Tensor:
    """HxWx3 uint8 RGB tensor -> 1x3xSxS float, ImageNet-normalised."""
    x = image_rgb_uint8.permute(2, 0, 1).float().unsqueeze(0) / 255.0
    # antialias=True: matches training (PIL bilinear resize, which filters on downscaling) and the app
    # (OpenCV INTER_AREA). Plain bilinear aliases badly at 1024 -> 256 and changed scores (P-60).
    x = F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False, antialias=True)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (x - mean) / std
