"""Part B analysis on plain tables (checked without a GPU in tests/test_analysis8.py).

* ``runs``  one row per (model, detector, sigma): map, map50, psnr. Split "val" or "test" in column ``split``.
* ``pi``    one row per (model, detector, sigma, image_id) on the test split: ap, psnr.

Model names: "noisy" (no restoration), "orig" (dncnn_last.pt), "lam_<value>" (fine-tuned, cos @ layer 21),
"neck_<value>" (fine-tuned with the neck-mean loss at lambda*). "lam_0" is the control.
"""
import numpy as np
import pandas as pd

CONTROL = "lam_0"


def lam_of(name):
    return float(name.split("_", 1)[1])


def lam_models(names):
    return sorted([n for n in set(names) if n.startswith("lam_")], key=lam_of)


def choose_lambda_star(runs, detector="yolov8n", sigma=50, max_psnr_cost=1.0):
    """PLAN.md 2.1: the lambda > 0 with the highest validation mAP at sigma 50 among those whose validation PSNR is at
    most 1 dB below the control's. Returns (model name, table)."""
    v = runs[(runs.split == "val") & (runs.detector == detector) & (runs.sigma == sigma)].set_index("model")
    cands = [m for m in lam_models(v.index) if lam_of(m) > 0]
    t = v.loc[[CONTROL] + cands, ["map", "psnr"]].copy()
    t["psnr_cost"] = t.loc[CONTROL, "psnr"] - t.psnr
    t["eligible"] = [(m != CONTROL) and (t.loc[m, "psnr_cost"] <= max_psnr_cost) for m in t.index]
    elig = t[t.eligible]
    star = elig["map"].idxmax() if len(elig) else None
    return star, t


def paired_bootstrap(pi, a, b, detector, sigma, B=2000, seed=12345):
    """Image-averaged AP of model a minus model b on the same images: mean and 95% CI (paired resampling)."""
    g = pi[(pi.detector == detector) & (pi.sigma == sigma)]
    va = g[g.model == a].set_index("image_id").ap
    vb = g[g.model == b].set_index("image_id").ap
    ids = va.index.intersection(vb.index).sort_values()
    d = (va.loc[ids] - vb.loc[ids]).to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    bs = d[idx].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(diff=float(d.mean()), ci_low=float(lo), ci_high=float(hi), n=len(d))


def test_table(runs):
    t = runs[runs.split == "test"]
    return t.pivot_table(index=["detector", "sigma"], columns="model", values="map")


def scorecard8(runs, pi, star):
    res = {}
    T = runs[runs.split == "test"].set_index(["model", "detector", "sigma"])
    get = lambda m, d, s, col="map": float(T.loc[(m, d, s), col]) if (m, d, s) in T.index else np.nan
    if star is None:
        return {k: (None, {"reason": "no eligible lambda on the validation split"}) for k in ("H7", "H8", "H9", "H10")}

    # H7: YOLOv8n gain at sigma 50, CI above 0, PSNR cost <= 1 dB
    gain_n = get(star, "yolov8n", 50) - get(CONTROL, "yolov8n", 50)
    bn = paired_bootstrap(pi, star, CONTROL, "yolov8n", 50)
    cost = get(CONTROL, "yolov8n", 50, "psnr") - get(star, "yolov8n", 50, "psnr")
    res["H7"] = (bool(gain_n >= 0.010 and bn["ci_low"] > 0 and cost <= 1.0),
                 dict(lambda_star=star, map_gain=gain_n, ci=[bn["ci_low"], bn["ci_high"]], psnr_cost_db=cost))

    # H8: Faster R-CNN keeps >= 50% of the YOLOv8n gain, CI above 0
    gain_f = get(star, "frcnn_r50_v2", 50) - get(CONTROL, "frcnn_r50_v2", 50)
    bf = paired_bootstrap(pi, star, CONTROL, "frcnn_r50_v2", 50)
    gs = get(star, "yolov8s", 50) - get(CONTROL, "yolov8s", 50)
    ok = bool(gain_n > 0 and gain_f >= 0.5 * gain_n and bf["ci_low"] > 0)
    res["H8"] = (ok, dict(frcnn_gain=gain_f, frcnn_ci=[bf["ci_low"], bf["ci_high"]], yolov8n_gain=gain_n,
                          share_kept=(gain_f / gain_n) if gain_n else np.nan, yolov8s_gain_reported=gs))

    # H9: interior optimum over lambda for YOLOv8n at sigma 50 on the test split
    lm = lam_models(T.index.get_level_values(0))
    curve = {m: get(m, "yolov8n", 50) for m in lm}
    best = max(curve, key=curve.get)
    res["H9"] = (bool(lam_of(best) < max(lam_of(m) for m in lm)), dict(curve=curve, best=best))

    # H10: clean images not hurt
    c_star, c_ctrl = get(star, "yolov8n", 0), get(CONTROL, "yolov8n", 0)
    res["H10"] = (bool(c_star >= c_ctrl - 0.005), dict(clean_map_star=c_star, clean_map_control=c_ctrl, diff=c_star - c_ctrl))
    return res


def comparisons(pi, models, detectors, sigmas, baseline=CONTROL):
    rows = []
    for d in detectors:
        for s in sigmas:
            for m in models:
                if m == baseline:
                    continue
                r = paired_bootstrap(pi, m, baseline, d, s)
                rows.append(dict(detector=d, sigma=s, model=m, vs=baseline, **r, clear=bool(r["ci_low"] > 0 or r["ci_high"] < 0)))
    return pd.DataFrame(rows)
