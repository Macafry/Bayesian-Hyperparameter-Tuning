# Bayesian Hyperparameter Optimization on the HIGGS Dataset

This project explores Bayesian optimization for hyperparameter tuning using [Optuna](https://optuna.org/). It benchmarks the performance of **XGBoost**, **LightGBM**, and **CatBoost** on the **HIGGS dataset** across three tuning stages. The goal is to demonstrate the value of Optuna in finding performant model configurations in an efficient and structured way.

---

## Project Highlights

- **Data**: Large-scale binary classification using the HIGGS dataset (28 continuous features)
- **Models**: XGBoost, LightGBM, CatBoost
- **Tuning Library**: Optuna with TPE sampler and pruning (where supported)
- **Pipeline**:
  1. Coarse search to identify key hyperparameters
  2. Narrow search around best values
  3. Final evaluation on full dataset
- **Frameworks Used**: Polars, Optuna, Matplotlib, Seaborn, Scikit-learn, PyPlot, HTML/Jinja2 for interactive plotting

---

## File Structure

```
.
├── higgs_data/                     # Dataset in Parquet + CD files
├── models/                         # Saved baseline and tuned models
├── studies/                        # Saved Optuna study objects (.pkl)
├── training/                       # Temporary training data splits
├── docs/                           # Rendered analysis html & files
├── lazy_model_evaluation.py        # Helper to evaluate trained models
├── manual_splits.py                # Custom file-based train/test splits
├── download_data.py                # Script to download and convert HIGGS dataset
├── bayesian_hyperparameters.qmd    # File containing the analysis
└── README.qmd                      # This file
```

---

## Quickstart

1. Clone the repo and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   I used the gpu version of LightGBM. To install it, please follow [this tutorial](https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html).

2. Run the notebook or render it via Quarto:

   ```bash
   quarto render bayesian_hyperparameters.qmd
   ```

3. If running from a clean slate, set `FORCE_RERUN = True` in the top cell to regenerate all files.

---

## Methodology

Each model is tuned in **three stages**:

| Stage | Data Size | Purpose                         | Notes                                     |
|-------|-----------|----------------------------------|-------------------------------------------|
| 1     | 10%       | Coarse parameter space scan      | Fast, many trials, MedianPruner           |
| 2     | 20%       | Focused tuning on top params     | Tighter ranges, SuccessiveHalvingPruner   |
| 3     | 100%      | Final training and evaluation    | Compares baseline vs tuned configuration  |

Cross-validation is skipped in favor of fixed train/valid/test splits for computational efficiency.

---

## Results Summary

- **CatBoost** showed the largest test AUC improvement: 0.81 → 0.84
- **XGBoost** displayed overfitting signs despite tuning
- **LightGBM** had modest gains
- Test AUCs and train-test differences are visualized via layered bar plots

See the final interactive table and plots in the last section of the notebook.

---

## Why Optuna?

- Flexible search space definitions
- Seamless pruning integration
- GPU compatibility with all three model types
- Visual diagnostics: parameter importances, slice plots, optimization history
