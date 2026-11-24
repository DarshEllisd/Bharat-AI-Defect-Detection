"""Inspect anomalib's pretrained PDN-small teacher and load it into our EfficientADS teacher.

    python training/vision/check_teacher.py
Prints the checkpoint's keys/shapes, the mapping onto efficientad_model.PDNSmall, and a forward
check (output shape on a 256 px input).
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from efficientad_model import OUT_CHANNELS, PDNSmall  # noqa: E402

CKPT = Path.home() / "bharatai-build/models/teachers/efficientad_pretrained_weights/pretrained_teacher_small.pth"


def load_teacher_state(path: Path = CKPT) -> dict:
    """Return a state_dict for PDNSmall from the anomalib checkpoint (keys mapped by layer order)."""
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    convs = [k[: -len(".weight")] for k in sd if k.endswith(".weight") and sd[k].dim() == 4]
    ours = PDNSmall(OUT_CHANNELS).state_dict()
    targets = [k[: -len(".weight")] for k in ours if k.endswith(".weight")]
    if len(convs) != len(targets):
        raise ValueError(f"layer count mismatch: checkpoint {convs} vs ours {targets}")
    mapped = {}
    for src, dst in zip(convs, targets):
        for suffix in (".weight", ".bias"):
            if sd[src + suffix].shape != ours[dst + suffix].shape:
                raise ValueError(f"shape mismatch {src}{suffix} {tuple(sd[src + suffix].shape)} "
                                 f"vs {dst}{suffix} {tuple(ours[dst + suffix].shape)}")
            mapped[dst + suffix] = sd[src + suffix]
    return mapped


if __name__ == "__main__":
    sd = torch.load(CKPT, map_location="cpu", weights_only=True)
    print("checkpoint keys:")
    for k, v in sd.items():
        print(f"  {k:24s} {tuple(v.shape)}")
    teacher = PDNSmall(OUT_CHANNELS)
    missing, unexpected = teacher.load_state_dict(load_teacher_state(), strict=True), None
    with torch.no_grad():
        out = teacher(torch.rand(1, 3, 256, 256))
    print(f"loaded into PDNSmall (strict) -> output {tuple(out.shape)}; "
          f"mean {out.mean():.3f} std {out.std():.3f}")
