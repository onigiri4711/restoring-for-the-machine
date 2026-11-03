"""Restorers from the Degrade -> Restore -> Detect project (notebooks 04, 05, 05b), copied without changes,
plus one new damage model (dark_only) for notebook 07.

All functions take and return float32 RGB images in [0, 1], shape (H, W, 3)."""
import json
import os

import cv2
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
BEST_CLASSICAL_PATH = os.path.join(HERE, "..", "prior", "01", "best_classical.json")


# --- DnCNN, exactly as defined in notebook 04 (the checkpoint only stores weights) -------------------
class DnCNN(nn.Module):
    def __init__(self, channels=3, depth=17, features=64):
        super().__init__()
        layers = [nn.Conv2d(channels, features, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
                       nn.BatchNorm2d(features), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(features, channels, kernel_size=3, padding=1)]
        self.body = nn.Sequential(*layers)

    def forward(self, y):
        return self.body(y)            # predicts the noise; the clean estimate is y - body(y)


def load_dncnn(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    net = DnCNN(ck["cfg"]["channels"], ck["cfg"]["depth"], ck["cfg"]["features"]).to(device)
    net.load_state_dict(ck["model"])
    net.eval()
    return net, ck


def make_dncnn_restore(net, device):
    @torch.no_grad()
    def dncnn_restore(img):
        """Full float32: under fp16 autocast some activations overflow to NaN on out-of-distribution inputs (JPEG)."""
        y = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).unsqueeze(0).to(device)
        out = (y - net(y)).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
        assert np.isfinite(out).all(), "DnCNN produced NaN/inf"
        return out
    return dncnn_restore


# --- Classical filters, exactly as in notebook 01 ------------------------------------------------------
def to_u8(img):  return (np.clip(img, 0, 1) * 255).round().astype(np.uint8)
def from_u8(u8): return u8.astype(np.float32) / 255.0
def f_median(img, k=3):     return from_u8(cv2.medianBlur(to_u8(img), k))
def f_gaussian(img, sigma=1.0):
    return np.clip(cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigmaX=sigma, borderType=cv2.BORDER_REFLECT), 0, 1)
def f_bilateral(img, d=7, sigma_color=0.1, sigma_space=5):
    return np.clip(cv2.bilateralFilter(img.astype(np.float32), d, sigma_color, sigma_space), 0, 1)
def f_nlmeans(img, h=10):   return from_u8(cv2.fastNlMeansDenoisingColored(to_u8(img), None, h, h, 7, 21))
FILTERS = {"median": f_median, "gaussian": f_gaussian, "bilateral": f_bilateral, "nlmeans": f_nlmeans}

BEST = json.load(open(BEST_CLASSICAL_PATH))   # notebook 01: tuned filter per (degradation, severity)


def classical_restore(img, name, severity):
    """The filter + parameters notebook 01 tuned for exactly this (degradation, severity)."""
    entry = BEST[name][str(severity)]
    if entry["filter"] == "none":
        return img
    return FILTERS[entry["filter"]](img, **entry["params"])


def clahe_L(img, clip=2.0, tiles=8):
    """Notebook 01, Experiment 3: CLAHE on Lab's L channel only (brightness without colour shift)."""
    lab = cv2.cvtColor(to_u8(img), cv2.COLOR_RGB2Lab)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(lab[..., 0])
    return from_u8(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB))


def auto_gamma(img, target=0.45):
    """Notebook 05b: raise every channel to the power g chosen so that mean luma becomes `target`. Blind."""
    luma = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    m = float(np.clip(luma.mean(), 1e-3, 0.999))
    g = np.log(target) / np.log(m)
    return np.clip(img, 0, 1) ** g


def chain(*steps):
    """Compose restorers left to right: chain(a, b)(img) = b(a(img))."""
    def run(img, name, s):
        for step in steps:
            img = np.clip(step(img, name, s), 0, 1).astype(np.float32)
        return img
    return run


# --- BM3D (notebook 05b), given the true sigma like the tuned filters -----------------------------------
def bm3d_restore(img, sigma):
    import bm3d
    return np.clip(bm3d.bm3d_rgb(img.astype(np.float64), sigma / 255.0), 0, 1).astype(np.float32)


def _bm3d_job(args):
    img, sigma = args
    return bm3d_restore(img, sigma)


def bm3d_many(imgs, sigma):
    """BM3D is CPU-only and slow (seconds per image), so spread it over all CPU cores."""
    from multiprocessing import get_context
    with get_context("fork").Pool(os.cpu_count()) as pool:
        return pool.map(_bm3d_job, [(im, sigma) for im in imgs], chunksize=2)


# --- New in notebook 07 -----------------------------------------------------------------------------------
def dark_only(img, level):
    """Step 1 of degradations.low_light (darken + gamma) with NO shot or read noise.
    Separates 'the image is dark' from 'the image is dark and noisy'."""
    gain = 1.0 - 0.8 * level
    gamma = 1.0 + 2.0 * level
    return np.clip(gain * np.power(img.astype(np.float32), gamma), 0, 1).astype(np.float32)


def build_restorers(dncnn_restore):
    """The restorer table used by notebook 07: name -> f(img, degradation, severity)."""
    R = {
        "none":      lambda img, name, s: img,
        "classical": classical_restore,
        "dncnn":     lambda img, name, s: dncnn_restore(img),
        "clahe_L":   lambda img, name, s: clahe_L(img),
        "gamma":     lambda img, name, s: auto_gamma(img),
    }
    R["gamma>dncnn"] = chain(R["gamma"], R["dncnn"])
    R["dncnn>gamma"] = chain(R["dncnn"], R["gamma"])
    return R
