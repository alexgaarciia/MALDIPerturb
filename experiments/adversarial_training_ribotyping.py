# ============================================================
# IMPORTS
# ============================================================
import numpy as np
import pandas as pd
from pathlib import Path
from pyprojroot import here

import torch

from datetime import datetime

from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

from src.config.loader import load_config
from src.data.datasets import *
from src.evaluation.eval import *
from src.evaluation.adversarial_attacks import train_adversarial

from src.models.mlp import RiboClassifier_Extended


# ============================================================
# CONFIG
# ============================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

EXPERIMENT_DIR = Path(f"experiments/results/{timestamp}")
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = EXPERIMENT_DIR / "mlp_ribotyping_cross_domain.csv"

ADV_LAMBDA = 0.5
EPS_LIST = [0.01, 0.02, 0.05, 0.1, 0.2]
ATTACKS = [("fgsm", 1), ("pgd", 10)]
SEEDS = [42, 123, 456, 789, 1011]


# ============================================================
# DATA LOADING
# ============================================================
PROJECT_ROOT = here()
cfg = load_config()
clostris = load_clostris(str(PROJECT_ROOT / cfg["data"]["CDIFF_PKL"]))
data, label, meta = clostris["data"], clostris["label"], clostris["meta"]


# ============================================================
# VALID TRAINING CELLS
# ============================================================
repro = meta[meta["subset"] == "Reproducibilidad"]
val_meta = meta[meta["subset"] == "Validacion"]

cell_counts = repro.groupby(["centre", "media", "extraction", "week"])["class"].value_counts().unstack(fill_value=0)

cells = cell_counts.index.to_frame(index=False)
cells["n_train_total"] = cell_counts.sum(axis=1).values
cells["n_min_per_class"] = cell_counts.min(axis=1).values

print(f"{len(cells)} training cells")
print(cells.to_string())


# ============================================================
# GRID
# ============================================================
configs = [("none", 0, 0.0)] + [(a, s, e) for (a, s) in ATTACKS for e in EPS_LIST]
rows = []

for _, cell in cells.iterrows():
    train_mask = ((repro["centre"] == cell["centre"]) & (repro["media"] == cell["media"]) & (repro["extraction"] == cell["extraction"]) & (repro["week"] == cell["week"]))
    test_mask = ~train_mask

    X_tr, y_tr = data[repro.index[train_mask]], label[repro.index[train_mask]]
    test_meta = repro[test_mask]

    for seed in SEEDS:
        X_pool, X_val, y_pool, y_val = train_test_split(X_tr, y_tr, test_size=0.1, stratify=y_tr, random_state=seed)

        for attack, steps, eps in configs:
            torch.manual_seed(seed)
            np.random.seed(seed)

            print(f"\n===== centre={cell['centre']} | media={cell['media']} | extraction={cell['extraction']} | week={cell['week']} | seed={seed} | attack={attack} | eps={eps} =====")

            mlp = RiboClassifier_Extended(input_dim=X_tr.shape[1], n_species=6, epochs=200, lr=1e-3, patience=10)
            trl = make_loader(X_pool, y_pool, shuffle=True)
            val = make_loader(X_val, y_val)

            if attack == "none":
                mlp.trainloop(trl, val, device)
            else:
                train_adversarial(mlp, trl, val, device, eps=eps, adv_lambda=ADV_LAMBDA, attack=attack, pgd_steps=steps, epochs=200)

            mlp.to(device)

            # Eval on rest of Reproducibilidad
            for (c, m, e, w), idx in test_meta.groupby(["centre", "media", "extraction", "week"]).groups.items():
                X_ev, y_ev = data[idx], label[idx]
                pred = predict(mlp, X_ev, device)
                ba = balanced_accuracy_score(y_ev, pred)
                print(f"  [repro]  eval centre={c} media={m} extraction={e} week={w} | n={len(idx)} | BA={ba:.4f}")
                rows.append({
                    "train_centre": cell["centre"], "train_media": cell["media"],
                    "train_extraction": cell["extraction"], "train_week": cell["week"],
                    "n_train": len(X_tr), "seed": seed, "attack": attack, "pgd_steps": steps, "eps": eps,
                    "eval_subset": "reproducibilidad",
                    "eval_centre": c, "eval_media": m, "eval_extraction": e, "eval_week": w,
                    "n_eval": len(idx), "ba": ba
                })

            # Eval on Validation
            for (c, m, e), idx in val_meta.groupby(["centre", "media", "extraction"]).groups.items():
                X_ev, y_ev = data[idx], label[idx]
                pred = predict(mlp, X_ev, device)
                ba = balanced_accuracy_score(y_ev, pred)
                print(f"  [val]    eval centre={c} media={m} extraction={e} | n={len(idx)} | BA={ba:.4f}")
                rows.append({
                    "train_centre": cell["centre"], "train_media": cell["media"],
                    "train_extraction": cell["extraction"], "train_week": cell["week"],
                    "n_train": len(X_tr), "seed": seed, "attack": attack, "pgd_steps": steps, "eps": eps,
                    "eval_subset": "validacion",
                    "eval_centre": c, "eval_media": m, "eval_extraction": e, "eval_week": None,
                    "n_eval": len(idx), "ba": ba 
                })

        pd.DataFrame(rows).to_csv(OUT_PATH, index=False)

print(f"\nResults saved: {OUT_PATH}")
print("\n===== Experiment completed successfully =====")
