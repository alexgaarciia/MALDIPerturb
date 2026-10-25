# ============================================================
# IMPORTS
# ============================================================
import numpy as np
import pandas as pd
from pathlib import Path

import torch

from datetime import datetime

from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config.loader import load_config
from src.data.datasets import *
from src.data.preprocessing import *
from src.evaluation.eval import *
from src.evaluation.adversarial_attacks import train_adversarial

from src.models.mlp import MLPClassifier_Extended


# ============================================================
# CONFIG
# ============================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

EXPERIMENT_DIR = Path(f"experiments/results/{timestamp}")
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = EXPERIMENT_DIR / "mlp_lowdata_perspecies_grid.csv"
SPECIES_PATH = EXPERIMENT_DIR / "species_used.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

ADV_LAMBDA = 0.5
EPS_LIST = [0.01, 0.02, 0.05, 0.1, 0.2]
ATTACKS = [("fgsm", 1), ("pgd", 10)]
SEEDS = [42, 123, 456, 789, 1011]
K_LIST = [5, 10, 25, 50, 75, 100]
K_MAX = max(K_LIST)
MIN_SPECIES = 200
TEST_FRAC = 0.3


# ============================================================
# DATA LOADING
# ============================================================
cfg = load_config()
raw = {
    "DRIAMS":  load_driams(cfg["data"]["DRIAMS_FULL"]),
    "MARISMA": load_marisma(cfg["data"]["MARISMa_FULL"]),
    "MS-UMG":  load_msumg(cfg["data"]["MSUMG_FULL"]),
}

counts = {c: pd.Series(d["label"]).value_counts() for c, d in raw.items()}
species = sorted(set.intersection(*[set(s[s >= MIN_SPECIES].index) for s in counts.values()]))
n_cls = len(species)
species_counts = pd.DataFrame({c: counts[c].reindex(species) for c in raw})

le = LabelEncoder().fit(species)
CENTERS = {}
for c, d in raw.items():
    m = np.isin(d["label"], species)
    CENTERS[c] = (row_minmax_normalize(d["data"][m]), le.transform(d["label"][m]))

print(f"\n{n_cls} species with >={MIN_SPECIES} spectra in every center:")
print(species_counts.to_string())
print("\nSpectra per center after filtering:")
for c, (X_c, y_c) in CENTERS.items():
    print(f"  {c}: {len(y_c)}")
print("\nSpecies used:")
for s in species:
    print(f"  {s}")

with open(SPECIES_PATH, "w") as f:
    f.write(f"Species used ({n_cls}), criterion: >= {MIN_SPECIES} spectra in each of: {', '.join(raw.keys())}\n\n")
    for s in species:
        f.write(f"{s}\n")
    f.write("\nSpectra per species and center (before the train/test split):\n")
    f.write(species_counts.to_string())
    f.write("\n\nSpectra per center after filtering:\n")
    for c, (X_c, y_c) in CENTERS.items():
        f.write(f"{c}: {len(y_c)}\n")
print(f"\nSpecies list saved: {SPECIES_PATH}")


# ============================================================
# GRID
# ============================================================
configs = [("none", 0, 0.0)] + [(a, s, e) for (a, s) in ATTACKS for e in EPS_LIST]
rows = []
skipped = []

for center, (X_c, y_c) in CENTERS.items():
    for seed in SEEDS:
        X_pool, X_te, y_pool, y_te = train_test_split(X_c, y_c, test_size=TEST_FRAC, stratify=y_c, random_state=seed)
        rng = np.random.default_rng(seed)
        order = {s: rng.permutation(np.where(y_pool == s)[0]) for s in range(n_cls)}
        min_pool = min(len(v) for v in order.values())
        need = K_MAX + max(2, K_MAX // 4)
        print(f"\n##### {center} | seed={seed} | pool={len(X_pool)} | test={len(X_te)} | min per species in pool={min_pool} #####")

        if min_pool < need:
            print(f"  [warning] smallest pool per species ({min_pool}) < {need}, skipping this center/seed")
            skipped.append(f"{center} seed={seed}: smallest pool per species {min_pool} < {need}")
            continue

        for k in K_LIST:
            v = max(2, k // 4)
            tr_idx = np.concatenate([order[s][:k] for s in range(n_cls)])
            va_idx = np.concatenate([order[s][K_MAX:K_MAX + v] for s in range(n_cls)])
            X_tr, y_tr = X_pool[tr_idx], y_pool[tr_idx]
            X_v, y_v = X_pool[va_idx], y_pool[va_idx]

            for attack, steps, eps in configs:
                torch.manual_seed(seed)
                np.random.seed(seed)
                print(f"\n===== {center} | seed={seed} | k={k} | attack={attack} | steps={steps} | eps={eps} =====")

                mlp = MLPClassifier_Extended(input_dim=X_tr.shape[1], n_species=n_cls, epochs=200, lr=1e-3, patience=10)
                trl = make_loader(X_tr, y_tr, shuffle=True)
                val = make_loader(X_v, y_v)

                if attack == "none":
                    mlp.trainloop(trl, val, device)
                else:
                    train_adversarial(mlp, trl, val, device, eps=eps, adv_lambda=ADV_LAMBDA, attack=attack, pgd_steps=steps, epochs=200)

                mlp.to(device)
                pred = predict(mlp, X_te, device)
                ba = balanced_accuracy_score(y_te, pred)
                print(f"BA test={ba:.4f}")
                rows.append({"center": center, "seed": seed, "k": k, "n_train": k * n_cls, "attack": attack, "pgd_steps": steps, "eps": eps, "ba_test": ba})
            pd.DataFrame(rows).to_csv(OUT_PATH, index=False)

with open(SPECIES_PATH, "a") as f:
    f.write("\nSkipped center/seed runs:\n")
    if skipped:
        for s in skipped:
            f.write(f"{s}\n")
    else:
        f.write("none\n")

print(f"\nResults saved: {OUT_PATH}")
print(f"Species list saved: {SPECIES_PATH}")
print("\n===== Experiment completed successfully =====")
