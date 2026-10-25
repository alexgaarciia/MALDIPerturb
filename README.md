# MALDIPerturb

**How robust are deep classifiers of MALDI-TOF mass spectra to adversarial perturbations, and can adversarial training help when labelled data is scarce?**

MALDIPerturb is a small PyTorch research codebase for bacterial species identification from MALDI-TOF spectra. It provides:

- loaders for several public/private spectral collections (DRIAMS, MARISMa, MS-UMG),
- an MLP species classifier,
- L∞ adversarial attacks (**FGSM**, **PGD**) and **adversarial training**,
- a ready-to-run experiment: *adversarial training in low-data regimes* (k spectra per species), across centers, seeds and perturbation budgets, plus a plotting script.

---

## Repository structure

```
MALDIPerturb/
├── configs/                  # (git-ignored) local config with dataset paths
│   └── config.yaml
├── experiments/
│   ├── adversarial_training_low_data_regimes_id.py            # runs the grid, writes CSV
│   ├── visualize_adversarial_training_low_data_regimes.py     # CSV -> figure
│   └── results/<timestamp>/  # (git-ignored) outputs of each run
├── src/
│   ├── config/loader.py      # load_config(): YAML -> dict
│   ├── data/
│   │   ├── io.py             # load_pkl()
│   │   ├── datasets.py       # load_driams / load_marisma / load_msumg
│   │   └── preprocessing.py  # row_minmax_normalize()
│   ├── models/mlp.py         # MLPClassifier, MLPClassifier_Extended (training loop + early stopping)
│   └── evaluation/
│       ├── adversarial_attacks.py  # fgsm_attack, pgd_attack, eval_under_attack,
│       │                           # sweep_attacks, train_adversarial
│       └── eval.py
├── pyproject.toml            # package metadata (installable as `maldiperturb`)
└── requirements.txt          # pinned dependencies
```

---

## Quick start

### 1. Clone and create an environment

```bash
git clone <this-repo-url> MALDIPerturb
cd MALDIPerturb

python -m venv .venv      
source .venv/bin/activate
```

### 2. Install

```bash
pip install -r requirements.txt
pip install -e .
```

### 3. Provide the data

The datasets are **not** included in this repository. Each one is expected as a pickle (`.pkl`) containing a dict with the keys:

| key     | content                                                          |
|---------|------------------------------------------------------------------|
| `data`  | `np.ndarray` `(n_spectra, n_bins)` binned spectra                |
| `label` | `np.ndarray` `(n_spectra,)` species names                        |
| `meta`  | iterable of per-spectrum dicts (e.g. `hospital` for DRIAMS, `agar` for MS-UMG) |

Create `configs/config.yaml` (it is git-ignored on purpose, since paths are machine-specific):

```yaml
data:
  DRIAMS_FULL:   /path/to/driams.pkl
  MARISMa_FULL:  /path/to/marisma.pkl
  MSUMG_FULL:    /path/to/msumg.pkl
```

The low-data experiment only reads `DRIAMS_FULL`, `MARISMa_FULL` and `MSUMG_FULL`. Always run scripts **from the repository root**, since the config is loaded from the relative path `configs/config.yaml`.

### 4. Run the experiment

```bash
python -m experiments.adversarial_training_low_data_regimes_id
```
(or `python experiments/adversarial_training_low_data_regimes_id.py` after `pip install -e .`)

### 5. Plot the results

Edit the `CSV` (and `RES`) path at the top of the visualization script to point to your run, then:

```bash
python experiments/visualize_adversarial_training_low_data_regimes.py
```

The figure `low_data_regime_results.png` is saved in `experiments/results/`.

---

## What the experiment does

1. **Harmonise species.** Keeps only species with ≥ `MIN_SPECIES` (200) spectra in *every* center, so all centers share one label space.
2. **Normalise.** Each spectrum is min-max scaled to [0, 1] (`row_minmax_normalize`), the domain on which attacks are clipped.
3. **Split.** For each center and seed: stratified 70/30 train-pool / test split.
4. **Low-data sampling.** Train on `k` spectra per species (`k ∈ {5, 10, 25, 50, 75, 100}`), with a validation set of `max(2, k/4)` per species.
5. **Train a model for each configuration**: standard training (`none`) or adversarial training with FGSM or PGD-10 at ε ∈ {0.01, 0.02, 0.05, 0.1, 0.2}, mixing clean and adversarial loss with `ADV_LAMBDA = 0.5`.
6. **Evaluate** balanced accuracy on the held-out clean test split, and append to the results CSV after each `k`.


### Outputs (`experiments/results/<YYYYMMDD_HHMMSS>/`)

- `mlp_lowdata_perspecies_grid.csv` with columns `center, seed, k, n_train, attack, pgd_steps, eps, ba_test`
- `species_used.txt`: the species list, per-center counts, and any skipped center/seed runs

---

## Using the building blocks

```python
import torch
from src.data.preprocessing import row_minmax_normalize
from src.models.mlp import MLPClassifier_Extended
from src.evaluation.adversarial_attacks import (
    pgd_attack, eval_under_attack, train_adversarial,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLPClassifier_Extended(input_dim=X.shape[1], n_species=n_cls,
                               epochs=200, lr=1e-3, patience=10)

# adversarial training (clean + adversarial loss)
train_adversarial(model, train_loader, val_loader, device,
                  eps=0.05, adv_lambda=0.5, attack="pgd", pgd_steps=10, epochs=200)

# robust accuracy under PGD
ba = eval_under_attack(model, X_test, y_test,
                       lambda x, y: pgd_attack(model, x, y, eps=0.05, alpha=0.0125, steps=10),
                       device)
```

To add a **new dataset**, write a `load_*` function in `src/data/datasets.py` following the existing ones and add its path to `config.yaml`. To add a **new model**, create it in `src/models/` exposing the same `trainloop(trainloader, validloader, device)` interface.
