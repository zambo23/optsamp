# optsamp

**Choose records from an external data pool so the sample mirrors the multivariate distribution of an internal credit portfolio.**

A bank has a small internal loan book and wants to buy data from a provider. It shouldn't buy records at random. It wants the ones that look like its own customers. `optsamp` is a research prototype that compares ways of making that selection and scores how closely each sample matches the internal portfolio.

> Status: early-stage research prototype. There is no packaging, no tests, and the API may change.

## Methods

| Method | Idea |
|---|---|
| **Gower nearest-neighbour** (proposed) | Every internal loan acts as a centroid and picks its `N_PER_CENTROID` nearest external records by [Gower distance](https://en.wikipedia.org/wiki/Gower%27s_distance), which mixes numeric range-scaled differences with categorical mismatches. The union of picks is deduplicated. The distance matrix is built in chunks, so large pools fit in memory. |
| **Stratified** (baseline) | Matches the joint distribution of a few stratification columns (`loan_grade`, `person_home_ownership`, `loan_intent`), then draws at random within each stratum. |
| **Random** (reference) | A simple random sample of the same size. It shows what "doing nothing clever" achieves. |

The target (`loan_status` → `default`) is never used for selection. It is only used afterwards to check whether the default rate carries over.

## Evaluation

The pipeline runs on the [OpenML 43454 Credit-Risk-Dataset](https://www.openml.org/d/43454) (32,581 loans). A random 15% becomes the "internal" portfolio and the remaining 85% becomes the "external" pool. Each method's sample is then compared with the internal set on:

- **Per-feature fit**: KS statistic for numeric features and total variation distance for categorical ones (lower is better).
- **Default-rate error**: the absolute gap between the sample's default rate and the internal default rate.
- **Discriminator AUC**: the cross-validated AUC of a gradient-boosted classifier trained to tell internal rows from sample rows. **0.5 means the two can't be told apart**, which makes this the main multivariate test.

All of this is repeated over 10 random splits to check stability.

### Current results (mean over 10 splits, ~10.7k-row samples)

| Method | Discriminator AUC ↓ | Mean KS ↓ | Mean TVD ↓ | Default-rate error ↓ |
|---|---|---|---|---|
| Gower | 0.538 | 0.020 | 0.004 | 1.1 pp |
| Stratified | 0.499 | 0.015 | 0.002 | 0.6 pp |
| Random | 0.501 | 0.014 | 0.011 | 0.5 pp |

In this setup, the Gower sample is slightly *easier* to tell apart from the internal portfolio than a random sample is. The likely reason is that the simulated internal book is a random draw from the same population as the pool, so a random sample is already close to ideal, and nearest-neighbour matching adds selection bias rather than removing it. A fair test of the method needs an internal portfolio whose distribution differs from the external pool (for example, a biased or segment-specific split). That is the next step.

The full write-up, with figures, is generated as [`output/sampling_memo.html`](output/sampling_memo.html). Its conclusions are computed from the numbers, so they update whenever the results change.

## Quick start

Requires Python 3.10+.

```bash
git clone git@github.com:zambo23/optsamp.git
cd optsamp
pip install numpy pandas matplotlib scipy scikit-learn certifi

python src/import_data.py   # download OpenML 43454 → data/credit_risk.csv
python src/optsamp.py       # run the pipeline (~2 min) → output/
```

`import_data.py` also accepts `--dataset german|all` (German Credit) and `--force` (re-download). If the data file is missing, `optsamp.py` downloads it automatically. `certifi` is optional. It supplies SSL certificates for python.org builds on macOS.

### Outputs (`output/`)

| File | Contents |
|---|---|
| `sampling_memo.html` | Self-contained comparison memo with embedded figures |
| `metrics_summary.csv` | Per-method summary for the main run (seed 42) |
| `metrics_features.csv` | Per-feature KS/TVD for the main run |
| `metrics_robustness.csv` | Summary metrics for every seed |
| `fig_*.png` | Feature scores, default rates, robustness across seeds |

## Using the samplers directly

```python
from gower_sampling import gower_sample
from stratified_sampling import stratified_sample

sample, _ = gower_sample(df_int, df_ext, num_cols, cat_cols,
                         n_per_centroid=3, drop_cols=("default",),
                         chunk_size=500)

baseline = stratified_sample(df_int, df_ext,
                             ["loan_grade", "person_home_ownership", "loan_intent"],
                             target_size=len(sample), drop_cols=("default",),
                             random_state=0)
```

## Repository layout

```
src/
  optsamp.py              main pipeline: split → sample → score → memo
  gower_sampling.py       Gower distance + nearest-neighbour sampler
  stratified_sampling.py  stratified baseline
  import_data.py          dataset download (OpenML / German Credit)
  memo_html.py            minimal Markdown → self-contained HTML renderer
output/                   generated memo, metrics and figures
data/                     downloaded datasets (gitignored)
```

## Configuration

Hyperparameters are module-level constants at the top of `src/optsamp.py`:

| Constant | Default | Meaning |
|---|---|---|
| `SEED` | 42 | seed for the main run |
| `INTERNAL_FRAC` | 0.15 | share of the data used as the internal portfolio |
| `N_PER_CENTROID` | 3 | Gower neighbours picked per internal loan |
| `CHUNK_SIZE` | 500 | centroids per distance-matrix chunk |
| `STRAT_COLS` | grade, home ownership, intent | stratification columns |
| `N_RUNS` | 10 | splits for the robustness check |

## License

[MIT](LICENSE)
