"""Figures for notebook 07. Every function takes the tables described in rftm/analysis.py and returns a Figure."""
import matplotlib.pyplot as plt
import numpy as np

from . import analysis as A

DEG_COLORS = {"gaussian": "#d62728", "jpeg": "#1f77b4", "low_light": "#2ca02c", "dark_only": "#8c564b",
              "gaussian_seed1": "#ff9896"}
REST_COLORS = {"none": "#888888", "classical": "#1f77b4", "dncnn": "#d62728", "bm3d": "#9467bd", "clahe_L": "#2ca02c",
               "gamma": "#bcbd22", "gamma>dncnn": "#ff7f0e", "dncnn>gamma": "#e377c2"}
METRIC_LABEL = {"cos": "cosine distance", "cka": "1 − linear CKA", "rel_l2": "relative L2  ‖F−F₀‖²/‖F₀‖²"}


def _shade_neck(ax, order):
    for l in A.NECK:
        if l in order:
            i = order.index(l)
            ax.axvspan(i - 0.4, i + 0.4, color="k", alpha=0.07, lw=0)


def depth_profiles(dist, conditions):
    """Distance vs layer for unrestored damage, one line per condition, one panel per metric. Shaded = neck outputs."""
    order = A.layer_order(dist)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for ax, metric in zip(axes, A.METRICS):
        lm = A.layer_means(dist, metric)
        for (d, s) in conditions:
            if (d, s, "none") not in lm.index:
                continue
            ls = "-" if d != "dark_only" else "--"
            ax.plot(range(len(order)), lm.loc[(d, s, "none"), order].values, ls, marker="o", ms=3,
                    color=DEG_COLORS.get(d, "k"), alpha=0.35 + 0.65 * _strength(d, s), label=f"{d} {s}")
        _shade_neck(ax, order)
        ax.set_xticks(range(len(order))); ax.set_xticklabels(order, fontsize=7)
        ax.set_xlabel("YOLOv8n layer (input → backbone → neck)"); ax.set_title(METRIC_LABEL[metric])
        if metric == "rel_l2":
            ax.set_yscale("log")
    axes[0].set_ylabel("damaged vs clean (mean over images)")
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Where damage shows up inside YOLOv8n (no restoration). Shaded: P3/P4/P5 neck outputs.")
    fig.tight_layout()
    return fig


def _strength(d, s):
    """0..1 shade for a severity level, mild -> light."""
    table = {"gaussian": {15: 0.3, 25: 0.6, 50: 1.0}, "jpeg": {30: 0.5, 10: 1.0}, "low_light": {0.6: 0.6, 0.8: 1.0},
             "dark_only": {0.8: 1.0}, "gaussian_seed1": {25: 0.6}}
    return table.get(d, {}).get(s, 1.0)


def removal_by_layer(dist, runs, clean_map, degradation, levels, restorers, metric="cos"):
    """Share of the feature damage each restorer removes, layer by layer, with its pixel-MSE removal (dotted) and
    mAP recovery (dashed) for comparison. One panel per level."""
    rec = A.recovery_table(dist, runs, clean_map, metric).set_index(A.KEY)
    order = A.layer_order(dist)
    fig, axes = plt.subplots(1, len(levels), figsize=(5.2 * len(levels), 4), sharey=True, squeeze=False)
    for ax, s in zip(axes[0], levels):
        for r in restorers:
            if (degradation, s, r) not in rec.index:
                continue
            row = rec.loc[(degradation, s, r)]
            c = REST_COLORS.get(r, "k")
            ax.plot(range(len(order)), [row[f"R_{l}"] for l in order], "o-", ms=3, color=c, label=r)
            ax.axhline(row.R_pixel, color=c, ls=":", lw=1)
            ax.axhline(row.map_recovery, color=c, ls="--", lw=1)
        ax.axhline(0, color="k", lw=0.8)
        _shade_neck(ax, order)
        ax.set_xticks(range(len(order))); ax.set_xticklabels(order, fontsize=7)
        ax.set_title(f"{degradation} {s}"); ax.set_xlabel("layer")
    axes[0][0].set_ylabel(f"feature damage removed ({metric})\n1 = clean features, <0 = worse than damaged")
    axes[0][0].legend(fontsize=8)
    fig.suptitle("Solid: removal at each layer · dotted: pixel-MSE removal · dashed: mAP recovery")
    fig.tight_layout()
    return fig


def low_light_panel(dist, runs, levels=(0.6, 0.8)):
    """Low light: mAP, PSNR and neck distance of every restorer, side by side, at each level."""
    nm = A.neck_means(dist, "cos")
    rr = runs.set_index(A.KEY)
    rests = ["none", "classical", "dncnn", "clahe_L", "gamma", "gamma>dncnn", "dncnn>gamma"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    w = 0.8 / len(levels)
    for j, s in enumerate(levels):
        have = [r for r in rests if ("low_light", s, r) in rr.index]
        x = np.arange(len(have)) + j * w
        vals = [[rr.loc[("low_light", s, r), "map"] for r in have],
                [rr.loc[("low_light", s, r), "psnr"] for r in have],
                [nm.get(("low_light", s, r), np.nan) for r in have]]
        for ax, v in zip(axes, vals):
            ax.bar(x, v, w, label=f"ℓ = {s}", color=[REST_COLORS[r] for r in have], alpha=0.5 + 0.5 * j / max(1, len(levels) - 1),
                   edgecolor="k", lw=0.5)
            ax.set_xticks(np.arange(len(have)) + w * (len(levels) - 1) / 2)
            ax.set_xticklabels(have, rotation=30, ha="right", fontsize=8)
    for ax, t in zip(axes, ["mAP@[.5:.95] (higher = better)", "PSNR, dB (higher = 'better')", "neck cosine distance (lower = closer to clean)"]):
        ax.set_title(t)
    axes[0].legend(fontsize=8)
    fig.suptitle("Low light: which number agrees with mAP? (light bars ℓ = 0.6, dark bars ℓ = 0.8)")
    fig.tight_layout()
    return fig


def norm_profiles(dist, conditions_restorers):
    """Exploratory: ‖F‖/‖F_clean‖ by layer. Above 1 = louder activations than clean, below 1 = quieter."""
    order = A.layer_order(dist)
    lm = A.layer_means(dist, "norm_ratio")
    fig, ax = plt.subplots(figsize=(8, 4))
    for (d, s, r) in conditions_restorers:
        if (d, s, r) in lm.index:
            ax.plot(range(len(order)), lm.loc[(d, s, r), order].values, marker="o", ms=3, color=REST_COLORS.get(r, "k"),
                    ls="-" if d == "low_light" else "--", label=f"{d} {s} {r}")
    ax.axhline(1, color="k", lw=0.8); _shade_neck(ax, order)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, fontsize=7)
    ax.set_ylabel("‖F‖ / ‖F_clean‖"); ax.set_xlabel("layer"); ax.legend(fontsize=7)
    ax.set_title("Activation energy relative to clean (solid: low light, dashed: dark only)")
    fig.tight_layout()
    return fig


def per_image_rho_by_layer(dist, pi, ceiling=None):
    """Mean within-run Spearman(distance at layer l, AP loss) for each metric, against PSNR's (flat line)."""
    order = A.layer_order(dist)
    fig, ax = plt.subplots(figsize=(9, 4))
    psnr_line = None
    for metric, c in zip(A.METRICS, ["#d62728", "#1f77b4", "#2ca02c"]):
        vals = []
        for l in order:
            t = A.ap_loss_table(dist, pi, metric, [l])
            wr = A.within_run_rhos(t)
            vals.append(wr.rho_feat.mean())
            psnr_line = wr.rho_psnr.mean()
        ax.plot(range(len(order)), vals, "o-", ms=3, color=c, label=METRIC_LABEL[metric])
    ax.axhline(psnr_line, color="k", ls="--", label=f"−PSNR ({psnr_line:.2f})")
    ax.axhline(0.30, color="grey", ls=":", label="H4a threshold 0.30")
    if ceiling is not None and np.isfinite(ceiling):
        ax.axhline(ceiling, color="purple", ls="-.", label=f"noise ceiling ({ceiling:.2f})")
    _shade_neck(ax, order)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, fontsize=7)
    ax.set_xlabel("layer"); ax.set_ylabel("mean Spearman with per-image AP loss")
    ax.set_title("Which layer's distance predicts which images lose AP? (mean over all damaged runs)")
    ax.legend(fontsize=7, ncol=2); fig.tight_layout()
    return fig


def per_image_scatter(dist, pi, run):
    """One run: per-image AP loss against neck distance and against PSNR."""
    t = A.ap_loss_table(dist, pi, "cos", A.NECK)
    g = t[(t.degradation == run[0]) & (t.severity == run[1]) & (t.restorer == run[2])]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    axes[0].scatter(g.dist, g.ap_loss, s=10, alpha=0.6, color="#d62728")
    axes[0].set_xlabel("neck cosine distance"); axes[0].set_title(f"ρ = {A.rho(g.dist, g.ap_loss):.2f}")
    axes[1].scatter(g.psnr, g.ap_loss, s=10, alpha=0.6, color="#555555")
    axes[1].set_xlabel("PSNR (dB)"); axes[1].set_title(f"ρ(−PSNR, loss) = {A.rho(-g.psnr, g.ap_loss):.2f}")
    axes[0].set_ylabel("AP loss (clean AP − damaged AP)")
    fig.suptitle(f"Per image, {run[0]} {run[1]} {run[2]}"); fig.tight_layout()
    return fig
