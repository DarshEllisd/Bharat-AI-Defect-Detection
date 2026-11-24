"""W3-B - train an EfficientAD-S anomaly model (Batzner et al., WACV 2024, Algorithm 1) on good images.

    ~/bharatai-build/train-venv/bin/python training/vision/train_efficientad.py --name own_wood \\
        --train ~/bharatai-build/datasets/anomaly_own/wood/train/good \\
        --val   ~/bharatai-build/datasets/anomaly_own/wood/val/good

The model is our export-friendly module (training/vision/efficientad_model.py), so the weights saved
here are exactly what gets quantised for the NPU (ADR-012/018). Teacher: anomalib's pretrained
PDN-small (frozen). Per iteration (batch 1, 256 px):
  L_ST   hard-feature loss: mean of (T'(I) - S_st(I))^2 over values >= their 0.999 quantile
         (+ optional penalty: mean S_st(P)^2 on an ImageNet-like image P, if --penalty-dir is given)
  L_AE   mean (T'(I_aug) - A(I_aug))^2           I_aug = brightness/contrast/saturation jitter
  L_STAE mean (A(I_aug) - S_ae(I_aug))^2
where T' = (T - mean) / std with channel stats over the training images.
After training, map normalisation quantiles (q 0.9 and 0.995 of all pixels) are taken from the
VALIDATION good images only (ADR-019) and folded into the model via set_quantiles().
Outputs: ~/bharatai-build/models/efficientad/<name>/{model.pt, config.json, train_log.csv}
"""
import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2 as T

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_teacher import load_teacher_state  # noqa: E402
from efficientad_model import IMAGENET_MEAN, IMAGENET_STD, OUT_CHANNELS, EfficientADS  # noqa: E402

IMG = 256
OUT_ROOT = Path.home() / "bharatai-build/models/efficientad"
EXTS = {".png", ".jpg", ".jpeg"}


def list_images(d: Path) -> list[Path]:
    return sorted(p for p in Path(d).expanduser().glob("*") if p.suffix.lower() in EXTS)


def load_rgb(p: Path, size: int = IMG) -> torch.Tensor:
    """-> 3 x size x size float in [0,1]."""
    img = Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR)
    return torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255.0


def tiles_of(x: torch.Tensor, t: int) -> list[torch.Tensor]:
    """3 x (t*IMG) x (t*IMG) -> t*t tiles of 3 x IMG x IMG (row-major)."""
    return [x[:, r * IMG:(r + 1) * IMG, c * IMG:(c + 1) * IMG] for r in range(t) for c in range(t)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--iters", type=int, default=70000)        # paper setting
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--penalty-dir", type=Path, default=None)  # ImageNet-like images (optional)
    ap.add_argument("--tiles", type=int, default=1, help="T: train on T x T tiles of the image at T*256 px (P-57)")
    ap.add_argument("--scale-aug", type=float, default=1.0,
                    help="random zoom-in up to this factor (e.g. 1.8): tolerance to camera distance (P-51)")
    ap.add_argument("--penalty-pool", type=int, default=3000, help="penalty images kept in VRAM")
    ap.add_argument("--geom-aug", action="store_true", help="random flips/90deg rotations (textures)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-amp", action="store_true", help="train in fp32/TF32 instead of bf16 autocast")
    args = ap.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = dev.type == "cuda" and not args.no_amp
    torch.backends.cudnn.benchmark = True  # fixed 256 px input
    torch.set_num_threads(1)  # all heavy work is on the GPU; keep the CPU free (P-48)
    out_dir = OUT_ROOT / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Everything the loop touches lives on the GPU (P-48): decoded once here, never again per iteration.
    # --tiles T: the capture square is resized to T*256 and cut into T*T tiles of 256 px; the model
    # learns (and later scores) tiles, i.e. T x the resolution for fine defects (P-57)
    t = args.tiles
    train_imgs = [tl.to(dev) for p in list_images(args.train) for tl in tiles_of(load_rgb(p, IMG * t), t)]
    val_imgs = [tl.to(dev) for p in list_images(args.val) for tl in tiles_of(load_rgb(p, IMG * t), t)]
    if len(train_imgs) == 0 or len(val_imgs) == 0:
        sys.exit("need train and val images")
    pen_paths = list_images(args.penalty_dir)[: args.penalty_pool] if args.penalty_dir else []
    pen_pool = (torch.stack([(load_rgb(p) * 255).round().to(torch.uint8) for p in pen_paths]).to(dev)
                if pen_paths else None)  # uint8 in VRAM: 3000 images ~ 0.6 GB
    print(f"[{args.name}] train {len(train_imgs)} | val {len(val_imgs)} | penalty images {len(pen_paths)} | {dev}")

    norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ae_aug = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)  # paper: factor in [0.8, 1.2]

    model = EfficientADS(IMG).to(dev)
    model.teacher.load_state_dict(load_teacher_state(), strict=True)
    model.teacher.eval().requires_grad_(False)

    # teacher channel statistics over the training images
    with torch.no_grad():
        feats = torch.cat([model.teacher(norm(x).unsqueeze(0)) for x in train_imgs])
        model.teacher_mean.copy_(feats.mean(dim=(0, 2, 3), keepdim=True))
        model.teacher_std.copy_(feats.std(dim=(0, 2, 3), keepdim=True))
        del feats

    params = list(model.student.parameters()) + list(model.autoencoder.parameters())
    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=int(0.95 * args.iters), gamma=0.1)
    model.student.train(); model.autoencoder.train()

    def t_norm(x: torch.Tensor) -> torch.Tensor:
        return (model.teacher(x) - model.teacher_mean) / model.teacher_std

    @torch.no_grad()
    def discrepancy(imgs) -> tuple[float, float]:
        """Overfitting monitor (no labels needed): mean student-teacher and student-autoencoder
        discrepancy, logged for VALIDATION good images and a fixed subset of TRAINING images. If the
        val value rises (or the val/train gap widens) while training continues, the student is
        memorising the few training photos instead of learning 'normal'."""
        st = stae = 0.0
        for xv in imgs:
            xv = norm(xv).unsqueeze(0)
            sv = model.student(xv)
            st += float(((t_norm(xv) - sv[:, :OUT_CHANNELS]) ** 2).mean())
            stae += float(((model.autoencoder(xv) - sv[:, OUT_CHANNELS:]) ** 2).mean())
        return st / len(imgs), stae / len(imgs)

    # Checkpoint every 5k iterations so a paused/killed run continues instead of restarting.
    # (Python RNG state is not restored, so a resumed run is equivalent, not bit-identical.)
    ckpt_path = out_dir / "ckpt.pt"
    start, elapsed0 = 1, 0.0
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        if ck.get("iters") == args.iters:
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
            start, elapsed0 = ck["it"] + 1, ck["elapsed"]
            print(f"[{args.name}] resumed from checkpoint at iteration {ck['it']}")
    mode = "a" if start > 1 else "w"
    log = open(out_dir / "train_log.csv", mode, newline="")
    w = csv.writer(log)
    vlog = open(out_dir / "val_log.csv", mode, newline="")
    vw = csv.writer(vlog)
    if start == 1:
        w.writerow(["iter", "loss", "l_st", "l_ae", "l_stae", "sec"])
        vw.writerow(["iter", "train_st", "train_stae", "val_good_st", "val_good_stae"])
    train_probe = train_imgs[:: max(1, len(train_imgs) // 25)][:25]
    n_hard = max(1, int(round(0.001 * OUT_CHANNELS * model.autoencoder.out_size[0] * model.autoencoder.out_size[1])))
    t0, running = time.time() - elapsed0, torch.zeros(4, device=dev)
    prev_done = None
    for it in range(start, args.iters + 1):
        x = random.choice(train_imgs)
        if args.geom_aug:  # flips + 90 degree rotations, on the GPU
            if random.random() < 0.5:
                x = torch.flip(x, dims=(2,))
            if random.random() < 0.5:
                x = torch.flip(x, dims=(1,))
            x = torch.rot90(x, k=random.randint(0, 3), dims=(1, 2))
        if args.scale_aug > 1.0:  # random zoom-in: crop a 1/z window and resize back (GPU)
            z = random.uniform(1.0, args.scale_aug)
            c = int(round(IMG / z))
            y0, x0 = random.randint(0, IMG - c), random.randint(0, IMG - c)
            x = F.interpolate(x[None, :, y0:y0 + c, x0:x0 + c], size=(IMG, IMG), mode="bilinear",
                              align_corners=False)[0]
        x_aug = ae_aug(x)  # ColorJitter on the GPU tensor
        x, x_aug = norm(x).unsqueeze(0), norm(x_aug).unsqueeze(0)

        # bf16 autocast for the convolutions (tensor cores, ~2x faster than TF32 here - P-45);
        # all losses and the hard-feature quantile are computed in fp32.
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
            with torch.no_grad():
                t = t_norm(x).float()
                t_aug = t_norm(x_aug).float()
            s = model.student(x)[:, :OUT_CHANNELS].float()
            pen = None
            if pen_pool is not None:
                p = norm(pen_pool[random.randrange(len(pen_pool))].float() / 255.0).unsqueeze(0)
                pen = model.student(p)[:, :OUT_CHANNELS].float()
            a = model.autoencoder(x_aug).float()
            s_ae = model.student(x_aug)[:, OUT_CHANNELS:].float()
        d = (t - s) ** 2
        # hard-feature loss: mean of the values at/above the 0.999 quantile = mean of the top 0.1%
        # (top-k avoids quantile + boolean indexing, which force a CPU-GPU sync every iteration)
        l_st = torch.topk(d.flatten(), n_hard, sorted=False).values.mean()
        if pen is not None:
            l_st = l_st + (pen ** 2).mean()
        l_ae = ((t_aug - a) ** 2).mean()
        l_stae = ((a - s_ae) ** 2).mean()
        loss = l_st + l_ae + l_stae

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step()
        running += torch.stack([loss, l_st, l_ae, l_stae]).detach()
        # let the CPU sleep (blocking event) instead of spin-waiting, at most one iteration ahead
        done = torch.cuda.Event(blocking=True) if dev.type == "cuda" else None
        if done is not None:
            done.record()
            if prev_done is not None:
                prev_done.synchronize()
            prev_done = done
        if it % 500 == 0 or it == args.iters:
            avg = (running / (500 if it % 500 == 0 else it % 500)).cpu().numpy()
            w.writerow([it, *[f"{v:.5f}" for v in avg], f"{time.time() - t0:.0f}"]); log.flush()
            if it % 5000 == 0 or it == args.iters:
                tr_st, tr_stae = discrepancy(train_probe)
                v_st, v_stae = discrepancy(val_imgs)
                vw.writerow([it, *[f"{v:.5f}" for v in (tr_st, tr_stae, v_st, v_stae)]]); vlog.flush()
                print(f"[{args.name}] it {it:6d}  loss {avg[0]:.4f}  st {avg[1]:.4f}  ae {avg[2]:.4f}  "
                      f"stae {avg[3]:.4f}  | mean st train {tr_st:.4f} val {v_st:.4f}  stae train {tr_stae:.4f} val {v_stae:.4f}  {time.time() - t0:.0f}s")
            running[:] = 0
            if it % 5000 == 0 and it < args.iters:
                torch.save({"it": it, "iters": args.iters, "elapsed": time.time() - t0, "model": model.state_dict(),
                            "opt": opt.state_dict(), "sched": sched.state_dict()}, ckpt_path.with_suffix(".tmp"))
                ckpt_path.with_suffix(".tmp").replace(ckpt_path)  # atomic: a kill mid-save keeps the old one
    log.close(); vlog.close()

    # map-normalisation quantiles from VALIDATION good images only (ADR-019)
    model.eval()
    st_maps, ae_maps = [], []
    with torch.no_grad():
        for x in val_imgs:
            x = norm(x).unsqueeze(0)
            t = t_norm(x); s = model.student(x); a = model.autoencoder(x)
            st_maps.append(((t - s[:, :OUT_CHANNELS]) ** 2).mean(1).flatten().cpu())
            ae_maps.append(((a - s[:, OUT_CHANNELS:]) ** 2).mean(1).flatten().cpu())
    st_all, ae_all = torch.cat(st_maps), torch.cat(ae_maps)
    q = [float(torch.quantile(st_all, 0.9)), float(torch.quantile(st_all, 0.995)),
         float(torch.quantile(ae_all, 0.9)), float(torch.quantile(ae_all, 0.995))]
    model.set_quantiles(*q)

    torch.save(model.cpu().state_dict(), out_dir / "model.pt")
    ckpt_path.unlink(missing_ok=True)
    (out_dir / "config.json").write_text(json.dumps({
        "name": args.name, "train": str(args.train), "val": str(args.val), "iters": args.iters,
        "lr": args.lr, "penalty_dir": str(args.penalty_dir) if args.penalty_dir else None,
        "geom_aug": args.geom_aug, "scale_aug": args.scale_aug, "tiles": args.tiles, "seed": args.seed, "image_size": IMG, "bf16_autocast": amp,
        "n_train": len(train_imgs), "n_val": len(val_imgs),
        "quantiles_st_a_b_ae_a_b": q, "train_seconds": round(time.time() - t0),
        "teacher": "anomalib pretrained_teacher_small.pth (sha256 of zip c09aeaa2...684a)",
    }, indent=2))
    print(f"[{args.name}] saved {out_dir / 'model.pt'}  quantiles {[round(v, 4) for v in q]}")


if __name__ == "__main__":
    main()
