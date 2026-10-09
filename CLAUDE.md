# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`optsamp` is an early-stage research prototype for **sampling an external data pool so it mirrors an internal credit portfolio's multivariate distribution**. The idea: a bank has a small internal dataset and buys data from a provider, so it needs to choose which external records best resemble its own book.

There is no package config yet (no `pyproject.toml`, no tests, no lint setup). The sampling methods live in their own modules (`src/gower_sampling.py`, `src/stratified_sampling.py`). There are two drivers:
- `src/optsamp.py` is the **main pipeline**. It uses OpenML 43454 "Credit-Risk-Dataset" (32,581 loans, target `loan_status`) and writes a comparison memo to `output/`.
- `src/sampling.py` is the original German Credit experiment script.

## Running

```bash
python src/import_data.py                    # OpenML 43454 → data/credit_risk.csv (--dataset german|all, --force)
python src/optsamp.py                        # pipeline → output/sampling_memo.html + figures + metrics csvs (~2 min)
python src/sampling.py                       # older German Credit experiment
```

Dependencies (not pinned anywhere yet): `numpy`, `pandas`, `matplotlib`, `scipy`, `scikit-learn` (used for `fetch_openml` and the discriminator AUC), and `certifi` (optional; supplies SSL certificates for macOS python.org builds). If you add packaging, follow the parent-folder convention (`uv` + `pyproject.toml`).

Known caveats when running locally:
- `sampling.py` writes its outputs (`metrics.csv`, `fig1`–`fig6` PNGs) to `OUT_DIR`, which is the repo's `data/` folder (gitignored and created if missing). The path is resolved from the script's location, so the working directory doesn't matter.
- `sampling.py` still downloads German Credit from a GitHub raw URL on every run instead of reading `data/german.csv`.

## Code structure

**Sampling methods** (side-effect-free modules that print progress only):
- `src/gower_sampling.py`
  - `gower_matrix`: a vectorised Gower distance (numeric `|x−y|/range`, categorical mismatch 0/1, then averaged). A feature missing on either side is left out of that pair's average. Pass `ranges` (from `numeric_ranges`) when calling it on chunks so every chunk is scaled the same way.
  - `gower_sample` (the proposed method): each internal row acts as a centroid and picks its `n_per_centroid` nearest external rows. The union is deduplicated. Set `chunk_size` for large pools: the full `n_int × n_ext` matrix is then never held in memory, and `D` is returned as `None`. `class` and `default` are dropped before the distance is computed, so the target is never used for matching.
- `src/stratified_sampling.py`
  - `stratified_sample` (baseline): matches the joint distribution of `strat_cols`. Strata missing from the external pool are skipped, and small strata are sampled with replacement. It takes a `random_state` argument.

**Pipeline** (`src/optsamp.py`): load → `split` (random `INTERNAL_FRAC` internal portfolio vs external pool) → `run_once` (Gower, stratified, and a same-size **random reference** sample, scored by KS/TVD, default-rate error and a **discriminator AUC**, the CV AUC of a classifier separating internal rows from sample rows, where 0.5 is ideal) → repeat over `N_RUNS` seeds → `build_memo` (Markdown) → `memo_html.markdown_to_html` (a self-contained HTML page with base64-embedded figures; it only supports the Markdown subset `build_memo` emits, so check the rendering if you add new syntax). The memo's conclusions are generated from the numbers, so its wording changes if the results change. `loan_status` is renamed to `default` and is never used for selection.

**Experiment** (`src/sampling.py`): it runs at import time (there is no `main()`) and imports the two modules by bare name, so run it as `python src/sampling.py`, which puts `src/` on the path. Its numbered sections are:

1. **Data**: German Credit (1,000 rows, 20 features). `default = (class == 2)`. Numeric and categorical columns are inferred from dtypes.
2. **Simulated scenario**: a random `INTERNAL_FRAC` (15%) split becomes the "internal" portfolio and the rest becomes the "external" pool.
3. **Run both methods**: the Gower sample's size is used as the stratified `target_size`, so both samples are the same size.
4. **Metrics**: each feature is compared with the internal set using KS for numeric features and TVD for categorical ones (lower is better), plus default-rate replication.
5. **Plots**: six figures with a fixed palette (`C_INT`/`C_GOW`/`C_STR`/`C_EXT`).
6. **Summary** printed to stdout.

Hyperparameters (`SEED`, `INTERNAL_FRAC`, `N_PER_CENTROID`, `STRAT_COLS`) are module-level constants at the top of the file.

Note: `NUM_COLS` has 7 columns but the numeric histogram grid is 2×3, so `zip` silently drops the last numeric feature from fig1.
