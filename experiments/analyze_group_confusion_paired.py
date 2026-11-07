# ============================================================
# IMPORTS
# ============================================================
import re
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
CM_DIR = Path("/export/usuarios_ml4ds/agnavarr/MALDIPerturb/experiments/results/20261106_185905/confusion_matrices")
RES = Path("/export/usuarios_ml4ds/agnavarr/MALDIPerturb/experiments/results")
EXCLUDE_EPS = [0.01, 0.02, 0.05, 0.2]

TAG_RE = re.compile(
    r"^(?P<center>.+?)_seed(?P<seed>\d+)_k(?P<k>\d+)_"
    r"(?P<attack>none|fgsm|pgd)(?P<steps>\d+)_eps(?P<eps>[0-9.]+)\.txt$"
)

GROUPS = {
    "Enterobacteriaceae": [
        "Escherichia_Coli", "Klebsiella_Pneumoniae", "Klebsiella_Oxytoca",
        "Enterobacter_cloacae_complex", "Citrobacter_Koseri", "Citrobacter_Freundii",
        "Serratia_Marcescens",
    ],
    "Morganellaceae": [
        "Proteus_Mirabilis", "Proteus_Vulgaris", "Morganella_Morganii",
    ],
    "CoNS": [
        "Staphylococcus_Epidermidis", "Staphylococcus_Haemolyticus",
        "Staphylococcus_Capitis", "Staphylococcus_Lugdunensis", "Staphylococcus_Hominis",
    ]
}


# ============================================================
# HELPER: within-group confusion rate from one confusion matrix
# ============================================================
def group_confusion_rate(cm, species, group_species):
    idx = [species.index(s) for s in group_species if s in species]
    sub = cm.iloc[idx, idx].values
    total_true = cm.iloc[idx, :].values.sum()
    off_diag = sub.sum() - np.trace(sub)
    return off_diag / total_true if total_true > 0 else np.nan


# ============================================================
# LOAD EACH MATRIX PER SEED
# ============================================================
rows = []
species = None

for f in CM_DIR.glob("*.txt"):
    m = TAG_RE.match(f.name)
    if not m:
        print(f"filename does not match pattern: {f.name}")
        continue

    center = m["center"]
    if center == "MS-UMG":
        continue

    seed, k, steps = int(m["seed"]), int(m["k"]), int(m["steps"])
    attack, eps = m["attack"], float(m["eps"])

    cm = pd.read_csv(f, sep="\t", index_col=0)
    if species is None:
        species = cm.index.tolist()

    for gname, gspecies in GROUPS.items():
        rate = group_confusion_rate(cm, species, gspecies)
        rows.append({"center": center, "seed": seed, "k": k, "attack": attack,
                     "steps": steps, "eps": eps, "group": gname, "confusion_rate": rate})

per_seed = pd.DataFrame(rows)

MAX_K = 50
per_seed = per_seed[per_seed["k"] <= MAX_K].copy()


# ============================================================
# Paire comparison: for each (center, k, group, AT config),
# subtract seed-by-seed against 'none', then aggregate the deltas
# ============================================================
def paired_comparison(df, center, k, group, attack, steps, eps):
    none_rates = df[(df.center == center) & (df.k == k) & (df.group == group) & (df.attack == "none")] \
                   .set_index("seed")["confusion_rate"]
    at_rates = df[(df.center == center) & (df.k == k) & (df.group == group) &
                  (df.attack == attack) & (df.steps == steps) & (df.eps == eps)] \
                 .set_index("seed")["confusion_rate"]
    common = none_rates.index.intersection(at_rates.index)
    if len(common) < 2:
        return None

    deltas = none_rates.loc[common] - at_rates.loc[common]
    pct_deltas = 100 * deltas / none_rates.loc[common]  

    return {
        "center": center, "k": k, "group": group, "attack": attack, "steps": steps, "eps": eps,
        "n_seeds": len(common),
        "mean_rate_none": none_rates.loc[common].mean(),
        "mean_delta": deltas.mean(),
        "std_delta": deltas.std(ddof=1),
        "mean_reduction_pct": pct_deltas.mean(),
        "std_reduction_pct": pct_deltas.std(ddof=1),
    }

at_configs = per_seed[(per_seed.attack != "none") & (~per_seed.eps.isin(EXCLUDE_EPS))][["attack", "steps", "eps"]].drop_duplicates()
centers_ks_groups = per_seed[["center", "k", "group"]].drop_duplicates()

results = []
for _, (center, k, group) in centers_ks_groups.iterrows():
    for _, (attack, steps, eps) in at_configs.iterrows():
        r = paired_comparison(per_seed, center, k, group, attack, steps, eps)
        if r is not None:
            results.append(r)

paired_df = pd.DataFrame(results)
paired_df.to_csv("/export/usuarios_ml4ds/agnavarr/MALDIPerturb/experiments/results/confusion_paired_comparison.csv", index=False)
print(paired_df.sort_values(["group", "center", "k", "mean_reduction_pct"], ascending=[True, True, True, False]))
