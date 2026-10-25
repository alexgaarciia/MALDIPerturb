import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

INCLUDE_MSUMG = False
EXCLUDE_EPS = [0.01, 0.02, 0.05]
SHOW_BANDS = True

RES = Path("/export/usuarios_ml4ds/agnavarr/MALDIPerturb/experiments/results")
CSV = Path("experiments/results/20261025_111431/mlp_lowdata_perspecies_grid.csv")
df = pd.read_csv(CSV)

suffix = "_with_msumg" if INCLUDE_MSUMG else ""
wanted = ["DRIAMS", "MARISMA"] + (["MS-UMG"] if INCLUDE_MSUMG else [])
centers = [c for c in wanted if c in df["center"].unique()]
df = df[df["center"].isin(centers)].copy()
ks = sorted(df["k"].unique())

def name(r):
    if r["attack"] == "fgsm": return "FGSM"
    if r["attack"] == "pgd": return f"PGD-{int(r['pgd_steps'])}"
    return r["attack"]
df["method"] = df.apply(name, axis=1)

STYLE = {"FGSM": (plt.cm.Oranges, "o"), "PGD-10": (plt.cm.Blues, "s"), "PGD-40": (plt.cm.Greens, "^")}
methods = [m for m in STYLE if m in df["method"].unique()]
epss = [e for e in sorted(df[df["method"].isin(methods)]["eps"].unique()) if not any(np.isclose(e, x) for x in EXCLUDE_EPS)]
shade = np.linspace(0.45, 0.95, len(epss))

pert = df[df["method"].isin(methods) & df["eps"].isin(epss)]


fig, axes = plt.subplots(1, len(centers), figsize=(5.2 * len(centers), 4.8), squeeze=False)
for j, c in enumerate(centers):
    ax = axes[0][j]

    r = df[(df["center"] == c) & (df["attack"] == "none")].groupby("k")["ba_test"].agg(["mean", "std"]).fillna(0)
    if SHOW_BANDS:
        ax.fill_between(r.index, r["mean"] - r["std"], r["mean"] + r["std"], color="black", alpha=0.10)
    ax.plot(r.index, r["mean"], color="black", ls="--", lw=2, marker="D", ms=4, zorder=5)

    all_means = [r["mean"].values]
    for m in methods:
        cmap, mk = STYLE[m]
        for e, sh in zip(epss, shade):
            col = cmap(sh)
            sub = pert[(pert["center"] == c) & (pert["method"] == m) & (pert["eps"] == e)]
            if sub.empty:
                continue
            g = sub.groupby("k")["ba_test"].agg(["mean", "std"]).fillna(0)
            all_means.append(g["mean"].values)
            if SHOW_BANDS:
                ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"], color=col, alpha=0.07)
            ax.plot(g.index, g["mean"], color=col, marker=mk, ms=4.5, lw=1.7)

    hi = min(1.0, max(np.max(a) for a in all_means) + 0.01)
    ax.set_ylim(0.94, hi)
    ax.set_title(c, fontsize=12, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels(ks)
    ax.minorticks_off()
    ax.set_xlabel("Training spectra per species (k)", fontsize=10)
    ax.grid(ls=":", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    if j == 0:
        ax.set_ylabel("Balanced accuracy\n(held-out test, same center)", fontsize=10)

n_rows = len(epss)
blank = Line2D([], [], color="none", label=" ")
handles = [Line2D([0], [0], color="black", ls="--", lw=2, marker="D", ms=4, label="Standard training")] + [blank] * (n_rows - 1)
for m in methods:
    cmap, mk = STYLE[m]
    for e, sh in zip(epss, shade):
        handles.append(Line2D([0], [0], color=cmap(sh), marker=mk, lw=1.7, ms=4.5, label=f"{m}, ε={e:g}"))
fig.legend(handles=handles, loc="lower center", ncol=len(methods) + 1, fontsize=9, frameon=False)
fig.suptitle("Training with perturbations in low-data settings (mean ± SD over seeds)", fontsize=12, fontweight="bold")
plt.tight_layout(rect=[0, 0.17, 1, 0.94])
plt.savefig(RES / f"low_data_regime_results{suffix}.png", dpi=200, bbox_inches="tight")
plt.close()
