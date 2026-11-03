"""Runs the whole Part A analysis on synthetic tables shaped like notebook 07's outputs.
Needs only numpy / pandas / scipy:  python -m pytest tests/  (or python tests/test_analysis.py)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rftm import analysis as A  # noqa: E402

LAYERS = ["input"] + [str(i) for i in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 18, 19, 21)]
RUNS = ([("gaussian", s, r) for s in (15, 25, 50) for r in ("none", "classical", "dncnn")]
        + [("gaussian", s, "bm3d") for s in (25, 50)]
        + [("jpeg", s, r) for s in (30, 10) for r in ("none", "classical", "dncnn")]
        + [("low_light", s, r) for s in (0.6, 0.8)
           for r in ("none", "classical", "dncnn", "clahe_L", "gamma", "gamma>dncnn", "dncnn>gamma")]
        + [("dark_only", 0.8, r) for r in ("none", "gamma", "clahe_L", "dncnn")]
        + [("gaussian_seed1", 25, "none")])
STRENGTH = {"none": 1.0, "classical": 0.6, "dncnn": 0.25, "bm3d": 0.27, "clahe_L": 1.3, "gamma": 1.1,
            "gamma>dncnn": 0.3, "dncnn>gamma": 0.4}


def synthetic(n_img=40, seed=0):
    rng = np.random.default_rng(seed)
    ids = np.arange(1000, 1000 + n_img)
    hard = rng.uniform(0.5, 1.5, n_img)                 # per-image sensitivity, drives both distance and AP loss
    clean_ap = rng.uniform(0.3, 0.9, n_img)
    dist, pi, runs = [], [], []
    pi += [dict(degradation="clean", severity=0, restorer="none", image_id=i, ap=a, psnr=np.inf, mse=0.0)
           for i, a in zip(ids, clean_ap)]
    runs.append(dict(degradation="clean", severity=0, restorer="none", map=0.387, map50=0.54, psnr=np.inf, mse=0.0))
    for d, s, r in RUNS:
        sev = {"gaussian": s / 50, "gaussian_seed1": s / 50, "jpeg": (100 - s) / 90, "low_light": s, "dark_only": 0.3 * s}[d]
        k = sev * STRENGTH[r]
        for j, i in enumerate(ids):
            dmg = k * hard[j] * rng.uniform(0.8, 1.2)
            for li, l in enumerate(LAYERS):
                depth = (li + 1) / len(LAYERS)
                prof = (1.2 - depth) if d.startswith("gaussian") else 0.6 + 0.2 * depth
                v = dmg * prof * 0.1
                dist.append(dict(degradation=d, severity=s, restorer=r, image_id=i, layer=l,
                                 cos=v, cka=0.5 * v, rel_l2=2 * v, norm_ratio=1.0))
            ap = float(np.clip(clean_ap[j] - 0.5 * dmg + rng.normal(0, 0.05), 0, 1))
            psnr = 30 - 8 * sev * (1.0 if r == "none" else 0.6) + rng.normal(0, 1)
            pi.append(dict(degradation=d, severity=s, restorer=r, image_id=i, ap=ap, psnr=psnr, mse=10 ** (-psnr / 10)))
        sub = [p for p in pi if (p["degradation"], p["severity"], p["restorer"]) == (d, s, r)]
        runs.append(dict(degradation=d, severity=s, restorer=r, map=np.mean([p["ap"] for p in sub]) * 0.6, map50=0.5,
                         psnr=np.mean([p["psnr"] for p in sub]), mse=np.mean([p["mse"] for p in sub])))
    return pd.DataFrame(dist), pd.DataFrame(pi), pd.DataFrame(runs)


def test_scorecard_runs_end_to_end():
    dist, pi, runs = synthetic()
    sc = A.scorecard(dist, pi, runs, clean_map=0.387)
    expected = {"H1a", "H1b", "H2a", "H2b", "H2c", "H3a", "H3b", "H3c", "H3d", "H4a", "H4b", "H5", "H6"}
    assert expected <= set(sc)
    for k, (verdict, ev) in sc.items():
        assert verdict in (True, False), (k, verdict, ev)
    # In the synthetic data the feature distance drives AP loss, so H4a must come out supported
    assert sc["H4a"][0], sc["H4a"]
    # and the Gaussian depth profile was built to decrease with depth
    assert sc["H1a"][0], sc["H1a"]
    # gain correlations must be computed on real image pairs, not empty ones
    g = A.gain_rhos(A.ap_loss_table(dist, pi))
    assert (g.n == 40).all() and g.rho_feat.notna().all(), g


def test_recovery_and_ranking_shapes():
    dist, pi, runs = synthetic(n_img=20, seed=1)
    rec = A.recovery_table(dist, runs, 0.387)
    assert {"R_neck", "R_pixel", "map_recovery", "R_input"} <= set(rec.columns)
    assert (rec.restorer != "none").all()
    ra = A.ranking_agreement(dist, runs)
    assert len(ra) >= 7


def test_part_b_choice_and_ceiling():
    dist, pi, runs = synthetic(n_img=30, seed=2)
    pick, opts = A.choose_part_b_layers(dist, pi)
    assert pick["metric"] in ("cos", "rel_l2") and len(pick["layers"]) >= 1
    c = A.noise_ceiling(pi, ("gaussian", 25, "none"), ("gaussian_seed1", 25, "none"))
    assert -1 <= c <= 1


if __name__ == "__main__":
    test_scorecard_runs_end_to_end(); test_recovery_and_ranking_shapes(); test_part_b_choice_and_ceiling()
    print("all analysis tests passed")
