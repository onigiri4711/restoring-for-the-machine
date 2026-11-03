"""Hooks into YOLOv8n's layers and distances between two sets of feature maps.

Conventions
-----------
* A feature map is a tensor of shape (1, C, H, W) for one image at one layer.
* The clean image and its damaged/restored version always have the same size, so YOLO letterboxes them
  identically and their feature maps line up position by position.
* Distances are computed only on the part of each map that covers the real image, not the grey letterbox
  padding (padding is identical in both images and would dilute every distance by a different amount per image).

Three distances per layer (all 0 for identical maps, larger = further from clean):

* ``cos``    1 - mean over spatial positions of the cosine similarity between the two C-dim channel vectors.
             Ignores the length of each vector, so a uniform brightness gain that only scales activations is
             (almost) invisible to it.
* ``cka``    1 - linear CKA (Kornblith et al., 2019), spatial positions as samples. Invariant to rotations and
             isotropic scaling of the channel space: asks "is the same spatial pattern represented?".
* ``rel_l2`` ||F_x - F_c||^2 / ||F_c||^2. Sensitive to everything, including pure scale changes.

Plus ``norm_ratio`` = ||F_x|| / ||F_c|| (is the damaged map louder or quieter?), which is not a distance.
"""
import math

import torch
import torch.nn as nn

SKIP_TYPES = {"Upsample", "Concat", "Detect"}   # no learned transform (Upsample, Concat) or not a feature map (Detect)
NECK_OUTPUTS = (15, 18, 21)                     # YOLOv8 C2f blocks whose outputs feed the Detect head: P3, P4, P5


def _find_sequential(detector):
    """The nn.Sequential of layers that the ultralytics predictor actually runs.

    After the first predict() call the predictor wraps the (fused) DetectionModel in an AutoBackend. Fusing
    happens in place, so this is normally the same object as detector.model.model, but we look it up through the
    predictor to be safe across ultralytics versions."""
    candidates = []
    pred = getattr(detector, "predictor", None)
    if pred is not None and getattr(pred, "model", None) is not None:
        candidates.append(pred.model)                      # AutoBackend
    candidates.append(detector.model)                      # DetectionModel
    for root in candidates:
        for m in root.modules():
            seq = getattr(m, "model", None)
            if isinstance(seq, nn.Sequential) and type(seq[-1]).__name__ == "Detect":
                return seq
    raise RuntimeError("could not find YOLO's layer list: has the ultralytics API changed?")


class FeatureTap:
    """Records the input tensor and the output of every hooked YOLO layer during one predict() call.

    Uses torch's *global* module hooks and keeps only calls to YOLO's top-level layers, which ultralytics marks
    with the attributes ``i`` (layer index) and ``f`` (where its input comes from). This works whichever copy of the
    model the predictor runs: some ultralytics versions run a copy, not ``detector.model``, so hooks attached to
    ``detector.model``'s layers never fire. Other networks (DnCNN) have no such attributes and are ignored.

    Usage:
        tap = FeatureTap(detector)
        detector.predict(img, ...)
        feats = tap.pop()                   # {"input": (1,3,H,W), 0: (1,16,H/2,W/2), ...}
    """

    def __init__(self, detector, layers=None):
        seq = _find_sequential(detector)
        self.layer_types = {i: type(m).__name__ for i, m in enumerate(seq)}
        if layers is None:
            layers = [i for i, t in self.layer_types.items() if t not in SKIP_TYPES]
        self.layers = list(layers)
        self._wanted = set(self.layers)
        self._store = {}
        from torch.nn.modules.module import register_module_forward_hook, register_module_forward_pre_hook
        self._handles = [register_module_forward_pre_hook(self._pre_hook), register_module_forward_hook(self._hook)]

    @staticmethod
    def _yolo_index(module):
        i = getattr(module, "i", None)
        return i if isinstance(i, int) and hasattr(module, "f") else None

    def _pre_hook(self, module, args):
        if self._yolo_index(module) == 0 and args and torch.is_tensor(args[0]):
            self._store["input"] = args[0].detach()

    def _hook(self, module, args, out):
        i = self._yolo_index(module)
        if i in self._wanted and torch.is_tensor(out):
            self._store[i] = out.detach()

    def pop(self):
        feats, self._store = self._store, {}
        missing = [i for i in ["input"] + self.layers if i not in feats]
        assert not missing, f"hooks did not fire for layers {missing}: has the ultralytics layer API changed?"
        return feats

    def remove(self):
        for h in self._handles:
            h.remove()


def letterbox_valid_box(h, w, H_in, W_in, imgsz=640):
    """Rows/cols of the letterboxed input tensor (H_in, W_in) that hold the real image, for an h x w original.

    Mirrors ultralytics' LetterBox (center=True): scale r = min(imgsz/h, imgsz/w), resize to round(w*r) x
    round(h*r), pad symmetrically to (H_in, W_in) with top = round(dh - 0.1), left = round(dw - 0.1)."""
    r = min(imgsz / h, imgsz / w)
    new_h, new_w = int(round(h * r)), int(round(w * r))
    dh, dw = (H_in - new_h) / 2, (W_in - new_w) / 2
    top, left = int(round(dh - 0.1)), int(round(dw - 0.1))
    return top, top + new_h, left, left + new_w


def crop_valid(f, box, H_in, W_in):
    """Crop feature map f (1, C, h, w) to the cells that overlap the real image (box in input-pixel coords).
    A cell is kept if it overlaps the image at all, so border cells (half image, half padding) stay in."""
    t, b, l, r = box
    sy, sx = H_in / f.shape[2], W_in / f.shape[3]
    y0, y1 = int(math.floor(t / sy)), int(math.ceil(b / sy))
    x0, x1 = int(math.floor(l / sx)), int(math.ceil(r / sx))
    return f[:, :, y0:y1, x0:x1]


def cosine_distance(a, b, eps=1e-8):
    """1 - mean over positions of cos(a[:, :, y, x], b[:, :, y, x])."""
    a, b = a.double(), b.double()
    num = (a * b).sum(1)
    den = a.norm(dim=1) * b.norm(dim=1)
    return float(1.0 - (num / (den + eps)).mean())


def linear_cka(a, b, eps=1e-12):
    """Linear CKA between two (1, C, H, W) maps, with the H*W positions as samples and channels as features.
    1 for identical maps (or maps equal up to rotation / isotropic scale of the channels), near 0 for unrelated."""
    X = a.double().flatten(2)[0].T          # (n, C_a)
    Y = b.double().flatten(2)[0].T          # (n, C_b)
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    xy = (X.T @ Y).pow(2).sum()
    xx = (X.T @ X).pow(2).sum().sqrt()
    yy = (Y.T @ Y).pow(2).sum().sqrt()
    return float(xy / (xx * yy + eps))


def feature_distances(fc, fx):
    """All per-layer numbers for one (clean, other) pair of maps of the same shape."""
    assert fc.shape == fx.shape, (fc.shape, fx.shape)
    nc = fc.double().pow(2).sum()
    diff = (fx.double() - fc.double()).pow(2).sum()
    return {
        "cos": cosine_distance(fc, fx),
        "cka": 1.0 - linear_cka(fc, fx),
        "rel_l2": float(diff / (nc + 1e-12)),
        "norm_ratio": float((fx.double().pow(2).sum() / (nc + 1e-12)).sqrt()),
    }


def compare_feature_sets(feats_clean, feats_other, h, w, imgsz=640):
    """Distances at every hooked layer (and at the input tensor) between two feature dicts from FeatureTap.pop().
    Returns {layer: {metric: value}} with layer = "input" or an int."""
    H_in, W_in = feats_clean["input"].shape[2:]
    assert feats_other["input"].shape == feats_clean["input"].shape, "the two images were letterboxed differently"
    box = letterbox_valid_box(h, w, H_in, W_in, imgsz)
    out = {}
    for k in feats_clean:
        fc = crop_valid(feats_clean[k], box, H_in, W_in)
        fx = crop_valid(feats_other[k], box, H_in, W_in)
        out[k] = feature_distances(fc, fx)
    return out


def describe_layers(tap, feats):
    """Table of hooked layers: index, type, channels, stride (input size / map size)."""
    H_in = feats["input"].shape[2]
    rows = []
    for k in ["input"] + tap.layers:
        f = feats[k]
        rows.append(dict(layer=k, type="input" if k == "input" else tap.layer_types[k], channels=f.shape[1],
                         stride=round(H_in / f.shape[2]), shape=tuple(f.shape[2:]),
                         neck_output=(k in NECK_OUTPUTS)))
    return rows
