"""Part B analysis on synthetic tables shaped like notebook 08's outputs (numpy / pandas only)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rftm import analysis8 as B  # noqa: E402

LAMS = [0, 0.03, 0.1, 0.3, 1, 3]


def synthetic(n=60, seed=0, star_gain=0.03):
    rng = np.random.default_rng(seed)
    models = ["noisy", "orig"] + [f"lam_{l:g}" for l in LAMS] + ["neck_0.3"]
    effect = {"noisy": -0.25, "orig": 0.0, "neck_0.3": star_gain}
    for l in LAMS:
        effect[f"lam_{l:g}"] = star_gain * (l / 0.3) * np.exp(1 - l / 0.3) if l > 0 else 0.0   # peaks at 0.3
    psnr_cost = {m: (lam_of_safe(m) * 0.8) for m in models}
    base = rng.uniform(0.3, 0.8, n)
    runs, pi = [], []
    for split, ids in (("val", range(1000, 1000 + n)), ("test", range(n))):
        for m in models:
            for det, k in (("yolov8n", 1.0), ("yolov8s", 0.8), ("frcnn_r50_v2", 0.6)):
                for s in (0, 15, 25, 50):
                    eff = effect[m] * k * (s / 50) - (0.0 if m != "noisy" else 0.0)
                    ap = np.clip(base - 0.3 * s / 50 + eff + rng.normal(0, 0.03, n), 0, 1)
                    ps = 30 - 0.2 * s - psnr_cost[m]
                    if split == "test":
                        pi += [dict(model=m, detector=det, sigma=s, image_id=i, ap=a, psnr=ps) for i, a in zip(ids, ap)]
                    runs.append(dict(split=split, model=m, detector=det, sigma=s, map=ap.mean() * 0.66, map50=0.5, psnr=ps))
    return pd.DataFrame(runs), pd.DataFrame(pi)


def lam_of_safe(m):
    try:
        return B.lam_of(m)
    except (ValueError, IndexError):
        return 0.0


def test_end_to_end():
    runs, pi = synthetic()
    star, t = B.choose_lambda_star(runs)
    assert star == "lam_0.3", t                       # lam 1 and 3 have a lower curve; all within 1 dB except 3
    sc = B.scorecard8(runs, pi, star)
    assert set(sc) == {"H7", "H8", "H9", "H10"}
    assert sc["H7"][0] and sc["H8"][0] and sc["H9"][0] and sc["H10"][0], sc
    cmp_ = B.comparisons(pi, ["orig", "lam_0.3"], ["yolov8n"], [50])
    assert len(cmp_) == 2 and cmp_.loc[cmp_.model == "lam_0.3", "clear"].item()


def test_no_effect_fails():
    runs, pi = synthetic(star_gain=0.0, seed=3)
    star, _ = B.choose_lambda_star(runs)
    sc = B.scorecard8(runs, pi, star)
    assert not sc["H7"][0]


def test_psnr_cost_excludes():
    runs, pi = synthetic()
    star, t = B.choose_lambda_star(runs, max_psnr_cost=0.01)   # every lam > 0 costs >= 0.024 dB here
    assert star is None and not t.eligible.any()


if __name__ == "__main__":
    test_end_to_end(); test_no_effect_fails(); test_psnr_cost_excludes()
    print("all Part B analysis tests passed")
