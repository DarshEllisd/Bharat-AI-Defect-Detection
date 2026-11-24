"""Give MVTec anomaly categories a validation split (ADR-019: val decides quantiles + thresholds).

    python training/data/make_anomaly_val.py wood carpet

anomaly/<cat>/train/good holds ALL MVTec training images (prepare_mvtec.py). This creates
  anomaly/<cat>/train90/good   ~90% (used for training)
  anomaly/<cat>/val/good       ~10% (map quantiles + threshold), disjoint from train90
as symlinks, with a fixed seed - the same practice as the EfficientAD paper.
"""
import os
import random
import sys
from pathlib import Path

ROOT = Path.home() / "bharatai-build/datasets/anomaly"
SEED, VAL_FRAC = 42, 0.10


def main() -> None:
    for cat in sys.argv[1:]:
        imgs = sorted((ROOT / cat / "train/good").iterdir())
        rng = random.Random(SEED)
        rng.shuffle(imgs)
        n_val = max(5, round(len(imgs) * VAL_FRAC))
        for split, items in (("val", imgs[:n_val]), ("train90", imgs[n_val:])):
            d = ROOT / cat / split / "good"
            d.mkdir(parents=True, exist_ok=True)
            for old in d.iterdir():
                old.unlink()
            for p in items:
                os.symlink(p.resolve(), d / p.name)
        print(f"{cat}: train90 {len(imgs) - n_val}, val {n_val}")


if __name__ == "__main__":
    main()
