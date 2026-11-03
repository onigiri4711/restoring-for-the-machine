"""Part B: fine-tuning DnCNN with a YOLO feature loss (notebook 08).

* ``YoloFeatures``  a differentiable forward through YOLOv8n's layers up to the deepest one we need, returning the
                    chosen layers' maps. YOLO is frozen (eval mode, no parameter gradients); gradients flow to the input.
* ``feature_distance``  batch version of Part A's distances, differentiable.
* ``CocoCrops``     random 320 x 320 crops of COCO images for training.
* ``train_dncnn``   the training loop: L = MSE / MSE0 + lam * D / D0.
"""
import copy
import math
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_


class YoloFeatures(nn.Module):
    """Runs YOLO's layer list the way ultralytics' DetectionModel does (each layer reads the outputs named by its
    ``f`` attribute), stops after the deepest requested layer, and returns {layer: map}. No Detect head, no NMS."""

    def __init__(self, detection_model, layers):
        super().__init__()
        self.layers = sorted(int(l) for l in layers)
        self.seq = detection_model.model
        self.stop = max(self.layers)
        for p in self.seq.parameters():
            p.requires_grad_(False)
        self.seq.eval()

    def train(self, mode=True):          # YOLO stays in eval mode (fixed BatchNorm) whatever the caller does
        return super().train(False)

    def forward(self, x):
        y, out = [], {}
        for m in self.seq:
            if m.i > self.stop:
                break
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x)
            if m.i in self.layers:
                out[m.i] = x
        return out


def feature_distance(fa, fb, metric="cos", eps=1e-8):
    """Per-sample distance between two batches of maps (B, C, H, W) -> (B,). Same definitions as Part A."""
    if metric == "cos":
        return 1.0 - F_.cosine_similarity(fa, fb, dim=1, eps=eps).flatten(1).mean(1)
    if metric == "rel_l2":
        return (fa - fb).pow(2).flatten(1).sum(1) / (fb.pow(2).flatten(1).sum(1) + eps)
    raise ValueError(metric)


class FeatureLoss(nn.Module):
    """Mean over the chosen layers of the per-sample distance between YOLO(restored) and YOLO(clean)."""

    def __init__(self, yolo_features, metric="cos"):
        super().__init__()
        self.yf, self.metric = yolo_features, metric

    def forward(self, restored, clean):
        with torch.no_grad():
            fc = self.yf(clean)
        fr = self.yf(restored.clamp(0, 1))
        d = torch.stack([feature_distance(fr[l], fc[l], self.metric) for l in self.yf.layers], 0).mean(0)
        return d.mean()


class CocoCrops(torch.utils.data.Dataset):
    """Random crops (default 320 x 320) with horizontal flips. Images smaller than the crop are scaled up first."""

    def __init__(self, paths, crop=320):
        self.paths, self.crop = list(paths), crop

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = cv2.imread(self.paths[i], cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32) / 255.0
        h, w = img.shape[:2]
        if min(h, w) < self.crop:
            s = self.crop / min(h, w)
            img = cv2.resize(img, (max(self.crop, round(w * s)), max(self.crop, round(h * s))), interpolation=cv2.INTER_CUBIC)
            h, w = img.shape[:2]
        rng = np.random.default_rng(torch.randint(0, 2**31, (1,)).item())   # follows torch's seeded worker RNG
        y0, x0 = rng.integers(0, h - self.crop + 1), rng.integers(0, w - self.crop + 1)
        img = img[y0:y0 + self.crop, x0:x0 + self.crop]
        if rng.random() < 0.5:
            img = img[:, ::-1]
        return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))


def make_loader(dataset, batch_size, seed, workers=4):
    g = torch.Generator(); g.manual_seed(seed)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=workers,
                                       generator=g, persistent_workers=False, pin_memory=True)


def add_noise(x, gen, sigma_max=55.0):
    """Blind Gaussian noise, sigma ~ U(0, sigma_max) per sample (0-255 scale), clipped to [0, 1] like degrade()."""
    sig = torch.rand(x.shape[0], 1, 1, 1, device=x.device, generator=gen) * (sigma_max / 255.0)
    return (x + sig * torch.randn(x.shape, device=x.device, generator=gen)).clamp(0, 1)


@torch.no_grad()
def loss_scales(net, feat_loss, loader, device, n_batches=20, seed=1234):
    """MSE0 and D0: mean MSE and feature distance of the *starting* DnCNN on n fixed training batches."""
    net.eval()
    gen = torch.Generator(device=device); gen.manual_seed(seed)
    torch.manual_seed(seed)
    mses, ds = [], []
    for k, x in enumerate(loader):
        if k >= n_batches:
            break
        x = x.to(device, non_blocking=True)
        y = add_noise(x, gen)
        r = y - net(y)
        mses.append(F_.mse_loss(r, x).item())
        ds.append(feat_loss(r, x).item())
    return float(np.mean(mses)), float(np.mean(ds))


def train_dncnn(net0, feat_loss, loader, device, lam, mse0, d0, steps=2000, lr=1e-4, drop_at=1500, clip=1.0,
                seed=0, log_every=100, log=print):
    """Fine-tune a copy of net0 with L = MSE/MSE0 + lam * D/D0. Returns (net, history list of dicts)."""
    net = copy.deepcopy(net0).to(device)
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[drop_at], gamma=0.1)
    torch.manual_seed(seed)
    gen = torch.Generator(device=device); gen.manual_seed(seed)
    hist, step, t0 = [], 0, time.time()
    run = {"mse": 0.0, "d": 0.0, "n": 0}
    while step < steps:
        for x in loader:
            if step >= steps:
                break
            x = x.to(device, non_blocking=True)
            y = add_noise(x, gen)
            r = y - net(y)
            mse = F_.mse_loss(r, x)
            if lam > 0:
                d = feat_loss(r, x)
                loss = mse / mse0 + lam * d / d0
            else:
                with torch.no_grad():
                    d = feat_loss(r, x) if ((step + 1) % log_every == 0) else torch.zeros((), device=device)
                loss = mse / mse0
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step(); sched.step()
            assert math.isfinite(loss.item()), f"loss is {loss.item()} at step {step}"
            run["mse"] += mse.item(); run["d"] += d.item(); run["n"] += 1
            step += 1
            if step % log_every == 0:
                n = run["n"]
                hist.append(dict(step=step, mse=run["mse"] / n, psnr=10 * math.log10(1 / max(run["mse"] / n, 1e-12)),
                                 feat=run["d"] / n if lam > 0 else d.item(), lr=sched.get_last_lr()[0],
                                 seconds=round(time.time() - t0)))
                log(f"  lam {lam:<5} step {step:5d}  train PSNR {hist[-1]['psnr']:.2f}  feat {hist[-1]['feat']:.4f}  "
                    f"({hist[-1]['seconds']} s)")
                run = {"mse": 0.0, "d": 0.0, "n": 0}
    net.eval()
    return net, hist


def high_freq_share(err, frac=0.5):
    """Share of the error image's energy above `frac` of the Nyquist frequency (radially). err: (H, W, 3) numpy."""
    e = err.mean(-1) - err.mean()
    P = np.abs(np.fft.fftshift(np.fft.fft2(e))) ** 2
    h, w = e.shape
    yy, xx = np.mgrid[-(h // 2):h - h // 2, -(w // 2):w - w // 2]
    r = np.sqrt((yy / (h / 2)) ** 2 + (xx / (w / 2)) ** 2)
    return float(P[r > frac].sum() / max(P.sum(), 1e-20))
