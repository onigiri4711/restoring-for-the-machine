"""Analysis for notebook 07, on plain pandas tables so it can be checked without a GPU.

Tables (all written by notebook 07):

* ``dist``  one row per (run, image, layer): degradation, severity, restorer, image_id, layer, cos, cka, rel_l2,
            norm_ratio. ``layer`` is a string: "input" or the YOLO module index ("0" ... "21").
* ``pi``    one row per (run, image): degradation, severity, restorer, image_id, ap, psnr, mse. Includes the clean run
            (degradation "clean", restorer "none").
* ``runs``  one row per run: degradation, severity, restorer, map, map50, psnr, mse.

A *condition* is (degradation, severity); a *run* is (degradation, severity, restorer).
Every threshold below is the one written in PLAN.md before any result existed.
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

NECK = ["15", "18", "21"]
EARLY = ["0", "1", "2", "3", "4"]
STEM = ["0", "1", "2"]
METRICS = ["cos", "cka", "rel_l2"]
KEY = ["degradation", "severity", "restorer"]
COND = ["degradation", "severity"]


def layer_order(dist):
    ls = [l for l in dist.layer.unique() if l != "input"]
    return ["input"] + sorted(ls, key=int)


def rho(x, y):
    """Spearman correlation, NaN if either side is constant."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan
    with warnings.catch_warnings():            # constant halves of tiny QUICK runs: NaN is the right answer
        warnings.simplefilter("ignore")
        return float(spearmanr(x, y, nan_policy="omit").correlation)


# --- per-run summaries -------------------------------------------------------------------------------------
def layer_means(dist, metric):
    """Mean distance over images for each run (rows) and layer (columns, in network order)."""
    t = dist.pivot_table(index=KEY, columns="layer", values=metric, aggfunc="mean")
    return t[[l for l in layer_order(dist) if l in t.columns]]


def per_image_group(dist, metric, layers):
    """Per-image mean of `metric` over a set of layers -> Series indexed by (degradation, severity, restorer, image_id)."""
    sub = dist[dist.layer.isin(layers)]
    return sub.groupby(KEY + ["image_id"])[metric].mean()


def neck_means(dist, metric="cos"):
    return per_image_group(dist, metric, NECK).groupby(level=KEY).mean()


def recovery_table(dist, runs, clean_map, metric="cos"):
    """For every restorer run: share of the damage it removes, measured three ways.

    * feature removal at layer l = 1 - mean d_l(restored) / mean d_l(damaged)   (1 = back to clean features)
    * pixel removal              = 1 - mean MSE(restored) / mean MSE(damaged)    (from native-resolution pixels)
    * mAP recovery               = (mAP_restored - mAP_damaged) / (clean mAP - mAP_damaged)
    """
    lm = layer_means(dist, metric)
    lm["neck"] = neck_means(dist, metric)
    lm["early"] = per_image_group(dist, metric, EARLY).groupby(level=KEY).mean()
    r = runs.set_index(KEY)
    out = []
    for (d, s, rest), row in lm.iterrows():
        if rest == "none" or (d, s, "none") not in lm.index:
            continue
        base = lm.loc[(d, s, "none")]
        rec = {"degradation": d, "severity": s, "restorer": rest}
        for col in lm.columns:
            rec[f"R_{col}"] = 1 - row[col] / base[col] if base[col] > 0 else np.nan
        mse_r, mse_0 = r.loc[(d, s, rest), "mse"], r.loc[(d, s, "none"), "mse"]
        rec["R_pixel"] = 1 - mse_r / mse_0
        m_r, m_0 = r.loc[(d, s, rest), "map"], r.loc[(d, s, "none"), "map"]
        rec["map_recovery"] = (m_r - m_0) / (clean_map - m_0) if clean_map - m_0 > 1e-9 else np.nan
        rec["map"], rec["psnr"] = m_r, r.loc[(d, s, rest), "psnr"]
        out.append(rec)
    return pd.DataFrame(out)


# --- per-image prediction (H4) -----------------------------------------------------------------------------
def ap_loss_table(dist, pi, metric="cos", layers=None):
    """One row per (run, image): AP loss vs clean, PSNR, and the feature distance (mean over `layers`)."""
    layers = layers or NECK
    clean = pi[pi.degradation == "clean"].set_index("image_id")["ap"]
    d = per_image_group(dist, metric, layers).rename("dist").reset_index()
    t = d.merge(pi, on=KEY + ["image_id"], how="inner")
    t["ap_loss"] = clean.loc[t.image_id].to_numpy() - t.ap.to_numpy()
    return t


def within_run_rhos(t):
    """Spearman with AP loss inside each run: feature distance vs minus PSNR (both: larger = worse)."""
    rows = []
    for k, g in t.groupby(KEY):
        rows.append(dict(zip(KEY, k), rho_feat=rho(g.dist, g.ap_loss), rho_psnr=rho(-g.psnr, g.ap_loss), n=len(g)))
    return pd.DataFrame(rows)


def gain_rhos(t):
    """Inside each condition, for each restorer vs 'none' on the same images: does the drop in feature distance
    (or the rise in PSNR) predict the per-image AP gain?"""
    rows = []
    for (d, s), g in t.groupby(COND):
        base = g[g.restorer == "none"].set_index("image_id")
        if base.empty:
            continue
        for rest, gr in g[g.restorer != "none"].groupby("restorer"):
            gr = gr.set_index("image_id")
            common = base.index.intersection(gr.index)
            gr, b = gr.loc[common], base.loc[common]
            ap_gain = gr.ap - b.ap
            rows.append(dict(degradation=d, severity=s, restorer=rest,
                             rho_feat=rho(b.dist - gr.dist, ap_gain), rho_psnr=rho(gr.psnr - b.psnr, ap_gain), n=len(gr)))
    return pd.DataFrame(rows)


def split_half_layer_choice(dist, pi, metric="cos", runs_filter=None):
    """Pick the single layer whose distance best predicts AP loss on the even-position images, then score that
    layer (and the neck mean) on the odd-position images. Position = rank of image_id in the sorted subset."""
    ids = np.sort(pi.image_id.unique())
    even, odd = set(ids[0::2]), set(ids[1::2])
    scores = {}
    for layer in layer_order(dist):
        t = ap_loss_table(dist, pi, metric, [layer])
        if runs_filter is not None:
            t = t[runs_filter(t)]
        r_even = within_run_rhos(t[t.image_id.isin(even)]).rho_feat.mean()
        r_odd = within_run_rhos(t[t.image_id.isin(odd)]).rho_feat.mean()
        scores[layer] = (r_even, r_odd)
    best = max(scores, key=lambda l: -np.inf if np.isnan(scores[l][0]) else scores[l][0])
    t = ap_loss_table(dist, pi, metric, NECK)
    if runs_filter is not None:
        t = t[runs_filter(t)]
    neck_odd = within_run_rhos(t[t.image_id.isin(odd)]).rho_feat.mean()
    table = pd.DataFrame(scores, index=["rho_even (select)", "rho_odd (test)"]).T
    return best, table, neck_odd


def noise_ceiling(pi, cond_a, cond_b):
    """Spearman between the per-image AP losses of two runs that differ only in the noise draw: how predictable
    per-image AP loss can be at all, from anything."""
    clean = pi[pi.degradation == "clean"].set_index("image_id")["ap"]
    get = lambda c: pi[(pi.degradation == c[0]) & (pi.severity == c[1]) & (pi.restorer == c[2])].set_index("image_id")["ap"]
    a, b = get(cond_a), get(cond_b)
    common = a.index.intersection(b.index)
    return rho(clean.loc[common] - a.loc[common], clean.loc[common] - b.loc[common])


# --- ranking agreement (H6) ----------------------------------------------------------------------------------
def ranking_agreement(dist, runs, metric="cos", conditions=None):
    """Inside each condition: is the restorer with the best neck distance / best PSNR also the best for mAP?"""
    nm = neck_means(dist, metric).rename("neck").reset_index()
    t = runs.merge(nm, on=KEY)
    rows = []
    for (d, s), g in t.groupby(COND):
        if conditions is not None and (d, s) not in conditions:
            continue
        if len(g) < 2:
            continue
        best_map = g.loc[g["map"].idxmax(), "restorer"]
        rows.append(dict(degradation=d, severity=s, n_restorers=len(g), best_map=best_map,
                         best_feat=g.loc[g.neck.idxmin(), "restorer"], best_psnr=g.loc[g.psnr.idxmax(), "restorer"],
                         rank_rho_feat=rho(-g.neck, g["map"]), rank_rho_psnr=rho(g.psnr, g["map"])))
    out = pd.DataFrame(rows)
    out["feat_ok"] = out.best_feat == out.best_map
    out["psnr_ok"] = out.best_psnr == out.best_map
    return out


# --- the scorecard -----------------------------------------------------------------------------------------------
def scorecard(dist, pi, runs, clean_map):
    """Every pre-registered hypothesis of Part A -> (verdict, the numbers behind it). See PLAN.md for the wording."""
    res = {}
    lm = layer_means(dist, "cos")
    neck = neck_means(dist, "cos")
    rec = recovery_table(dist, runs, clean_map, "cos").set_index(KEY)
    have = lambda *k: k in lm.index

    # H1a: Gaussian noise does the most damage in the stem (layers 0-2) and less by the neck
    ev = {}
    for s in (25, 50):
        if have("gaussian", s, "none"):
            row = lm.loc[("gaussian", s, "none")].drop("input")
            ev[f"sigma {s}"] = dict(argmax_layer=row.idxmax(), layer0=row["0"], neck=neck.loc[("gaussian", s, "none")])
    res["H1a"] = (all(v["argmax_layer"] in STEM and v["neck"] < v["layer0"] for v in ev.values()) if ev else None, ev)

    # H1b: JPEG q=10 damage survives depth better than Gaussian sigma=50 (higher neck / layer-0 ratio)
    if have("jpeg", 10, "none") and have("gaussian", 50, "none"):
        rj = neck.loc[("jpeg", 10, "none")] / lm.loc[("jpeg", 10, "none"), "0"]
        rg = neck.loc[("gaussian", 50, "none")] / lm.loc[("gaussian", 50, "none"), "0"]
        res["H1b"] = (bool(rj > rg), dict(jpeg10_neck_over_l0=rj, gauss50_neck_over_l0=rg))
    else:
        res["H1b"] = (None, {})

    # H2: DnCNN on Gaussian noise
    ev = {}
    for s in (15, 25, 50):
        if ("gaussian", s, "dncnn") in rec.index:
            r = rec.loc[("gaussian", s, "dncnn")]
            ev[f"sigma {s}"] = dict(R_neck=r.R_neck, R_pixel=r.R_pixel, map_recovery=r.map_recovery)
    res["H2a"] = (all(v["R_neck"] >= 0.60 for v in ev.values()) if ev else None, ev)
    res["H2b"] = (all(v["R_neck"] < v["R_pixel"] for v in ev.values()) if ev else None, ev)
    res["H2c"] = (all(abs(v["R_neck"] - v["map_recovery"]) <= 0.15 for v in ev.values()) if ev else None, ev)

    # H3: low light and CLAHE
    ev = {}
    for s in (0.6, 0.8):
        if ("low_light", s, "clahe_L") in rec.index:
            r = rec.loc[("low_light", s, "clahe_L")]
            ev[f"level {s}"] = dict(R_pixel=r.R_pixel, R_early=r.R_early, R_neck=r.R_neck)
    res["H3a"] = (all(v["R_pixel"] > 0 and v["R_early"] < 0 for v in ev.values()) if ev else None, ev)

    rr = runs.set_index(KEY)
    if ("dark_only", 0.8, "none") in rr.index:
        m_dark = rr.loc[("dark_only", 0.8, "none"), "map"]
        m_ll = rr.loc[("low_light", 0.8, "none"), "map"] if ("low_light", 0.8, "none") in rr.index else 0.121
        ev = dict(map_dark_only=m_dark, map_low_light=m_ll, threshold=clean_map - 0.5 * (clean_map - m_ll))
        res["H3b"] = (bool(m_dark >= ev["threshold"]), ev)
        ev = {r_: rr.loc[("dark_only", 0.8, r_), "map"] for r_ in ("none", "gamma", "clahe_L") if ("dark_only", 0.8, r_) in rr.index}
        res["H3c"] = (all(ev[r_] > ev["none"] for r_ in ("gamma", "clahe_L") if r_ in ev), ev)
    else:
        res["H3b"] = res["H3c"] = (None, {})

    ev = {}
    for s in (0.6, 0.8):
        sub = neck[[k for k in neck.index if k[0] == "low_light" and k[1] == s and k[2] != "none"]]
        if len(sub):
            ev[f"level {s}"] = dict(best=sub.idxmin()[2], neck=sub.droplevel([0, 1]).round(4).to_dict())
    res["H3d"] = (all(v["best"] == "gamma>dncnn" for v in ev.values()) if ev else None, ev)

    # H4: per-image prediction of AP loss and of AP gain
    t = ap_loss_table(dist, pi, "cos", NECK)
    wr = within_run_rhos(t[t.degradation != "clean"])
    wr = wr[wr.restorer.notna()]
    beats = (wr.rho_feat > wr.rho_psnr).mean()
    res["H4a"] = (bool(wr.rho_feat.mean() >= 0.30 and beats >= 0.75),
                  dict(mean_rho_feat=wr.rho_feat.mean(), mean_rho_psnr=wr.rho_psnr.mean(), share_runs_feat_beats_psnr=beats,
                       n_runs=len(wr)))
    gr = gain_rhos(t)
    gr = gr[gr.degradation != "dark_only"] if len(gr) else gr
    if len(gr):
        beats = (gr.rho_feat > gr.rho_psnr).mean()
        res["H4b"] = (bool(gr.rho_feat.mean() >= 0.30 and beats >= 0.75),
                      dict(mean_rho_feat=gr.rho_feat.mean(), mean_rho_psnr=gr.rho_psnr.mean(),
                           share_pairs_feat_beats_psnr=beats, n_pairs=len(gr)))
    else:
        res["H4b"] = (None, {})

    # H5: BM3D and DnCNN end up equally close to clean in feature space
    ev = {}
    for s in (25, 50):
        if have("gaussian", s, "bm3d") and have("gaussian", s, "dncnn"):
            nb, nd = neck.loc[("gaussian", s, "bm3d")], neck.loc[("gaussian", s, "dncnn")]
            ev[f"sigma {s}"] = dict(neck_bm3d=nb, neck_dncnn=nd, rel_gap=(nb - nd) / nd,
                                    psnr_gap=rr.loc[("gaussian", s, "dncnn"), "psnr"] - rr.loc[("gaussian", s, "bm3d"), "psnr"])
    res["H5"] = (all(abs(v["rel_gap"]) <= 0.15 for v in ev.values()) if ev else None, ev)

    # H6: ranking agreement across restorers inside each damaged condition
    main = {("gaussian", 15), ("gaussian", 25), ("gaussian", 50), ("jpeg", 30), ("jpeg", 10),
            ("low_light", 0.6), ("low_light", 0.8)}
    ra = ranking_agreement(dist, runs, "cos", main)
    if len(ra):
        need = int(np.ceil(6 / 7 * len(ra)))
        res["H6"] = (bool(ra.feat_ok.sum() >= need and ra.feat_ok.sum() > ra.psnr_ok.sum()),
                     dict(feat_matches=int(ra.feat_ok.sum()), psnr_matches=int(ra.psnr_ok.sum()), conditions=len(ra),
                          needed=need, mean_rank_rho_feat=ra.rank_rho_feat.mean(), mean_rank_rho_psnr=ra.rank_rho_psnr.mean()))
    else:
        res["H6"] = (None, {})
    return res


def choose_part_b_layers(dist, pi):
    """The pre-registered rule for Part B's loss: among {neck mean, early mean, best single layer} x {cos, rel_l2},
    take the option with the highest held-out (odd-half) Spearman with AP loss on the Gaussian-noise runs."""
    gauss = lambda t: t.degradation == "gaussian"
    ids = np.sort(pi.image_id.unique())
    odd = set(ids[1::2])
    options = []
    for metric in ("cos", "rel_l2"):
        best, table, neck_odd = split_half_layer_choice(dist, pi, metric, gauss)
        options.append(dict(metric=metric, layers=NECK, rho_odd=neck_odd))
        t = ap_loss_table(dist, pi, metric, EARLY)
        t = t[gauss(t) & t.image_id.isin(odd)]
        options.append(dict(metric=metric, layers=EARLY, rho_odd=within_run_rhos(t).rho_feat.mean()))
        options.append(dict(metric=metric, layers=[best], rho_odd=table.loc[best, "rho_odd (test)"]))
    opts = pd.DataFrame(options)
    pick = opts.loc[opts.rho_odd.idxmax()].to_dict()
    return pick, opts
