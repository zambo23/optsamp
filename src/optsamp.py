"""
Gower vs Stratified sampling — Credit Risk Dataset (OpenML 43454)
─────────────────────────────────────────────────────────────────
End-to-end pipeline:
  1. import the data          (data/credit_risk.csv, fetched if missing)
  2. simulate the scenario     internal portfolio (INTERNAL_FRAC) vs
                               external provider pool (the rest)
  3. apply both sampling methods
  4. score each sample against the internal portfolio
  5. repeat steps 2–4 over N_RUNS seeds to check the result is stable
  6. write a comparison memo  →  output/sampling_memo.html
                               (+ metrics csv and figures in output/)

A simple random sample of the same size is scored alongside as a
reference point: it shows what "doing nothing clever" achieves.

Usage
─────
  python src/optsamp.py
"""

import datetime as dt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from gower_sampling import gower_sample
from import_data import CREDIT_RISK_FILE, OPENML_ID, fetch_credit_risk
from memo_html import markdown_to_html
from stratified_sampling import stratified_sample

# ── palette ───────────────────────────────────────────────────────────────────
C_INT  = "#378ADD"   # blue   internal
C_GOW  = "#1D9E75"   # teal   Gower
C_STR  = "#D85A30"   # coral  stratified
C_RND  = "#888780"   # gray   random reference

# ── hyper-parameters ──────────────────────────────────────────────────────────
SEED            = 42
INTERNAL_FRAC   = 0.15          # fraction of full dataset used as "internal"
N_PER_CENTROID  = 3             # Gower: neighbours per centroid
CHUNK_SIZE      = 500           # Gower: centroids per distance-matrix chunk
STRAT_COLS      = ["loan_grade", "person_home_ownership", "loan_intent"]
N_RUNS          = 10            # seeds for the robustness check
TARGET          = "loan_status" # 1 = default

# ── outputs ───────────────────────────────────────────────────────────────────
OUT_DIR   = Path(__file__).resolve().parent.parent / "output"
MEMO_FILE = OUT_DIR / "sampling_memo.html"

METHODS = ["Gower", "Stratified", "Random"]


# ══════════════════════════════════════════════════════════════════════════════
# 1.  IMPORT DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_data() -> tuple:
    """Return (df, num_cols, cat_cols); `default` is the binary target."""
    if not CREDIT_RISK_FILE.exists():
        fetch_credit_risk()
    df = pd.read_csv(CREDIT_RISK_FILE)
    df = df.rename(columns={TARGET: "default"})

    num_cols = [c for c in df.select_dtypes(include="number").columns
                if c != "default"]
    cat_cols = [c for c in df.columns if c not in num_cols and c != "default"]
    return df, num_cols, cat_cols


# ══════════════════════════════════════════════════════════════════════════════
# 2.  SCENARIO: INTERNAL (bank portfolio) / EXTERNAL (provider pool)
# ══════════════════════════════════════════════════════════════════════════════

def split(df: pd.DataFrame, seed: int) -> tuple:
    rng     = np.random.default_rng(seed)
    idx_int = rng.choice(len(df), size=int(INTERNAL_FRAC * len(df)), replace=False)
    idx_ext = np.setdiff1d(np.arange(len(df)), idx_int)
    return (df.iloc[idx_int].reset_index(drop=True),
            df.iloc[idx_ext].reset_index(drop=True))


# ══════════════════════════════════════════════════════════════════════════════
# 3–4.  SAMPLE AND SCORE
# ══════════════════════════════════════════════════════════════════════════════

def ks(ref: pd.Series, smp: pd.Series) -> float:
    return stats.ks_2samp(ref.dropna().values, smp.dropna().values).statistic


def tvd(ref: pd.Series, smp: pd.Series) -> float:
    cats = set(ref.dropna().unique()) | set(smp.dropna().unique())
    p = ref.value_counts(normalize=True).reindex(cats, fill_value=0)
    q = smp.value_counts(normalize=True).reindex(cats, fill_value=0)
    return 0.5 * np.abs(p - q).sum()


def discriminator_auc(df_int: pd.DataFrame, df_smp: pd.DataFrame,
                      num_cols: list, cat_cols: list, seed: int) -> float:
    """
    Multivariate check: cross-validated AUC of a classifier trying to tell
    internal rows from sample rows. 0.5 = indistinguishable, 1 = fully
    separable. The target is excluded, so this is about the features only.
    """
    X = pd.concat([df_int, df_smp], ignore_index=True)[num_cols + cat_cols]
    X[cat_cols] = X[cat_cols].astype("category")
    y = np.r_[np.zeros(len(df_int)), np.ones(len(df_smp))]
    clf = HistGradientBoostingClassifier(categorical_features="from_dtype",
                                         random_state=seed)
    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return cross_val_score(clf, X, y, cv=cv, scoring="roc_auc").mean()


def run_once(df: pd.DataFrame, num_cols: list, cat_cols: list, seed: int) -> dict:
    df_int, df_ext = split(df, seed)

    df_gow, _ = gower_sample(df_int, df_ext, num_cols, cat_cols,
                             n_per_centroid=N_PER_CENTROID,
                             drop_cols=("default",), chunk_size=CHUNK_SIZE)
    df_str = stratified_sample(df_int, df_ext, STRAT_COLS,
                               target_size=len(df_gow),
                               drop_cols=("default",), random_state=seed)
    df_rnd = df_ext.sample(n=len(df_gow), random_state=seed)
    samples = {"Gower": df_gow, "Stratified": df_str, "Random": df_rnd}

    rows = []
    for col in num_cols:
        rows.append({"feature": col, "type": "numeric", "metric": "KS",
                     **{m: ks(df_int[col], s[col]) for m, s in samples.items()}})
    for col in cat_cols:
        rows.append({"feature": col, "type": "categorical", "metric": "TVD",
                     **{m: tvd(df_int[col], s[col]) for m, s in samples.items()}})
    features = pd.DataFrame(rows)
    features["winner"] = np.where(features["Gower"] < features["Stratified"],
                                  "Gower", "Stratified")

    summary = pd.DataFrame({
        m: {
            "size"          : len(s),
            "coverage"      : len(s) / len(df_ext),
            "default_rate"  : s["default"].mean(),
            "dr_abs_error"  : abs(s["default"].mean() - df_int["default"].mean()),
            "mean_ks"       : features.loc[features["type"] == "numeric", m].mean(),
            "mean_tvd"      : features.loc[features["type"] == "categorical", m].mean(),
            "mean_score"    : features[m].mean(),
            "disc_auc"      : discriminator_auc(df_int, s, num_cols, cat_cols, seed),
        }
        for m, s in samples.items()
    }).T

    return {"seed": seed, "df_int": df_int, "df_ext": df_ext,
            "samples": samples, "features": features, "summary": summary}


# ══════════════════════════════════════════════════════════════════════════════
# 5.  FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def plot_feature_scores(features: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.8))
    x, w = np.arange(len(features)), 0.27
    for k, (m, c) in enumerate(zip(METHODS, [C_GOW, C_STR, C_RND])):
        ax.bar(x + (k - 1) * w, features[m], w, color=c, alpha=0.85, label=m)

    n_num = (features["type"] == "numeric").sum()
    ax.axvline(n_num - 0.5, color="#888780", linewidth=0.8, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{f}\n({m})" for f, m in
                        zip(features["feature"], features["metric"])],
                       rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("KS / TVD vs internal  (lower = closer)")
    ax.set_title("Per-feature distance to the internal portfolio", fontweight="bold")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_default_rates(run: dict, path: Path) -> None:
    labels = ["Internal", *METHODS, "External pool"]
    values = [run["df_int"]["default"].mean(),
              *[run["samples"][m]["default"].mean() for m in METHODS],
              run["df_ext"]["default"].mean()]
    colors = [C_INT, C_GOW, C_STR, C_RND, "#B4B2A9"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, values, color=colors, alpha=0.85, width=0.55)
    ax.axhline(values[0], color=C_INT, linestyle="--", linewidth=1.3, alpha=0.7)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.004, f"{v:.1%}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.25)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("Default rate")
    ax.set_title("Default-rate replication", fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_robustness(robust: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (col, title) in zip(axes, [("mean_score", "Mean KS/TVD"),
                                       ("disc_auc", "Discriminator AUC"),
                                       ("dr_abs_error", "Default-rate abs. error")]):
        data = [robust.loc[robust["method"] == m, col] for m in METHODS]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5)
        ax.set_xticks(range(1, len(METHODS) + 1), METHODS)
        for patch, c in zip(bp["boxes"], [C_GOW, C_STR, C_RND]):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        ax.set_title(f"{title}\n(lower = better)", fontsize=10)
    fig.suptitle(f"Stability across {robust['seed'].nunique()} random splits",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 6.  MEMO
# ══════════════════════════════════════════════════════════════════════════════

def md_table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(df.columns) + " |"
    sep  = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.values]
    return "\n".join([head, sep, *body])


def better(a: float, b: float, rel_tol: float = 0.05) -> str:
    """'Gower' / 'Stratified' / 'tie' for two lower-is-better numbers."""
    if abs(a - b) <= rel_tol * max(abs(a), abs(b), 1e-12):
        return "tie"
    return "Gower" if a < b else "Stratified"


def build_memo(df: pd.DataFrame, num_cols: list, cat_cols: list,
               main: dict, robust: pd.DataFrame) -> str:
    s, f = main["summary"], main["features"]
    agg  = robust.groupby("method")[["mean_score", "disc_auc", "dr_abs_error"]]
    mu, sd = agg.mean(), agg.std()
    n_runs = robust["seed"].nunique()

    # per-seed head-to-head
    wide = robust.pivot(index="seed", columns="method")
    wins = {col: int((wide[col]["Gower"] < wide[col]["Stratified"]).sum())
            for col in ["mean_score", "disc_auc", "dr_abs_error"]}

    verdicts = {col: better(mu.loc["Gower", col], mu.loc["Stratified", col])
                for col in ["mean_score", "disc_auc", "dr_abs_error"]}
    decided  = [v for v in verdicts.values() if v != "tie"]
    if decided and all(v == decided[0] for v in decided) and len(decided) >= 2:
        overall = (f"**{decided[0]} sampling gives the closer match to the internal "
                   f"portfolio** on every criterion that separates the two methods.")
    else:
        overall = ("**Neither method dominates**: the criteria point in different "
                   "directions, so the choice depends on which property matters most.")

    # each method vs the random reference, criterion by criterion
    crit_names = {"mean_score": "marginal fit", "disc_auc": "joint fit",
                  "dr_abs_error": "default-rate error"}

    def vs_random(m: str) -> str:
        beats = [crit_names[c] for c in crit_names
                 if better(mu.loc[m, c], mu.loc["Random", c]) == "Gower"]
        loses = [crit_names[c] for c in crit_names
                 if better(mu.loc[m, c], mu.loc["Random", c]) == "Stratified"]
        if not beats and loses:
            return f"**{m} is worse than a random draw on {', '.join(loses)}** and better on nothing"
        if beats and not loses:
            return f"{m} beats a random draw on {', '.join(beats)}"
        return (f"{m} beats a random draw on {', '.join(beats) or 'nothing'}"
                f" but is worse on {', '.join(loses) or 'nothing'}")

    def verdict_txt(col: str) -> str:
        v = verdicts[col]
        return "the two methods are effectively tied" if v == "tie" else f"{v} is closer"

    # dedup effect: how many neighbour picks collapsed into the same record
    n_picks   = len(main["df_int"]) * N_PER_CENTROID
    collapsed = 1 - main["summary"].loc["Gower", "size"] / n_picks

    def pct(x): return f"{x:.1%}"
    def f3(x):  return f"{x:.3f}"
    def ms(m, c, fmt=f3): return f"{fmt(mu.loc[m, c])} ± {fmt(sd.loc[m, c])}"

    # ── tables ────────────────────────────────────────────────────────────────
    t_main = pd.DataFrame({
        "Sample"            : ["Internal portfolio", *METHODS],
        "Rows"              : [f"{len(main['df_int']):,}",
                               *[f"{int(s.loc[m, 'size']):,}" for m in METHODS]],
        "% of pool"         : ["–", *[pct(s.loc[m, "coverage"]) for m in METHODS]],
        "Default rate"      : [pct(main["df_int"]["default"].mean()),
                               *[pct(s.loc[m, "default_rate"]) for m in METHODS]],
        "Mean KS (numeric)" : ["–", *[f3(s.loc[m, "mean_ks"]) for m in METHODS]],
        "Mean TVD (categ.)" : ["–", *[f3(s.loc[m, "mean_tvd"]) for m in METHODS]],
        "Discriminator AUC" : ["–", *[f3(s.loc[m, "disc_auc"]) for m in METHODS]],
    })
    t_feat = f[["feature", "metric", *METHODS, "winner"]].copy()
    for m in METHODS:
        t_feat[m] = t_feat[m].map(f3)
    t_feat = t_feat.rename(columns={"winner": "Closer (G vs S)"})

    t_rob = pd.DataFrame({
        "Method"                  : METHODS,
        "Mean KS/TVD"             : [ms(m, "mean_score") for m in METHODS],
        "Discriminator AUC"       : [ms(m, "disc_auc") for m in METHODS],
        "Default-rate abs. error" : [ms(m, "dr_abs_error", lambda x: f"{x*100:.2f} pp")
                                     for m in METHODS],
    })

    # ── data-quality facts used in the limitations section ────────────────────
    desc   = df[num_cols].describe(percentiles=[0.99]).T
    skewed = desc[desc["max"] > 3 * desc["99%"]]
    skew_txt = ", ".join(f"`{c}` (99th pct {r['99%']:,.0f}, max {r['max']:,.0f})"
                         for c, r in skewed.iterrows()) or "none"
    na = df.isna().sum()
    na_txt = ", ".join(f"`{c}` ({n:,} rows)" for c, n in na[na > 0].items()) or "none"

    gw_f = (f["winner"] == "Gower").sum()

    return f"""# Memo — Gower-distance vs stratified sampling of an external credit pool

**Date:** {dt.date.today():%d %B %Y}
**Data:** OpenML {OPENML_ID} "Credit-Risk-Dataset" ({len(df):,} loans, default rate {pct(df['default'].mean())})
**Produced by:** `src/optsamp.py` (all figures below are regenerated on every run)

---

## 1. Bottom line

- {overall}
- Averaged over {n_runs} random internal/external splits:
  - **Marginal fit** (mean KS/TVD, lower is better): Gower {ms('Gower', 'mean_score')} vs Stratified {ms('Stratified', 'mean_score')}. Gower was closer in {wins['mean_score']}/{n_runs} splits.
  - **Joint (multivariate) fit** (discriminator AUC, 0.5 is ideal): Gower {ms('Gower', 'disc_auc')} vs Stratified {ms('Stratified', 'disc_auc')}. Gower was closer in {wins['disc_auc']}/{n_runs} splits.
  - **Default-rate replication** (absolute error vs internal): Gower {ms('Gower', 'dr_abs_error', lambda x: f'{x*100:.2f} pp')} vs Stratified {ms('Stratified', 'dr_abs_error', lambda x: f'{x*100:.2f} pp')}. Gower was closer in {wins['dr_abs_error']}/{n_runs} splits.
- Against a plain random draw of the same size (the do-nothing option): {vs_random('Gower')}; {vs_random('Stratified')}. Differences within 5% of each other count as ties.
- Caveat: the simulated internal portfolio is itself a random draw from the same population as the pool, so a random sample is already a strong benchmark (see §7). This set-up tests whether a method *preserves* the portfolio's distribution, not whether it can *correct* a pool that differs from it.

## 2. Question

A bank with a small internal loan book wants to buy data from an external provider. Which records should it take so that the purchased sample looks like its own portfolio? We compare two ways of choosing them.

## 3. Set-up

| Item | Value |
|---|---|
| Source data | {len(df):,} loans, {len(num_cols)} numeric and {len(cat_cols)} categorical features, target `{TARGET}` |
| Internal portfolio | random {pct(INTERNAL_FRAC)} of rows ({len(main['df_int']):,} loans) |
| External pool | the remaining {len(main['df_ext']):,} loans |
| Main run | seed {SEED} |
| Robustness | {n_runs} further splits (seeds 0–{n_runs - 1}) |

The target (`{TARGET}`) is **never** used to select records. The default-rate comparison is therefore an out-of-sample check.

## 4. Methods

**Gower-distance sampling (proposed).** Each internal loan acts as a centroid. The {N_PER_CENTROID} external loans nearest to it by Gower distance are taken, using all {len(num_cols) + len(cat_cols)} features. Numeric features are scaled by their range and categoricals score 0 for a match and 1 for a mismatch; a feature missing on either side is skipped for that pair. The union of all the selected loans, with duplicates removed, is the sample, so its size is an *output* of the method.

**Stratified sampling (baseline).** The external pool is sampled so that the joint distribution of {", ".join(f"`{c}`" for c in STRAT_COLS)} matches the internal portfolio. Its target size is set to the Gower sample's size. Strata that are too small are sampled with replacement and then de-duplicated, so the sample can end up a little smaller than the target.

**Random reference.** This is a simple random draw from the pool, the same size as the Gower sample.

**Metrics** (all compare a sample with the internal portfolio, and lower is better):
- *KS statistic* for each numeric feature and *total variation distance (TVD)* for each categorical feature. These measure **marginal** fit.
- *Discriminator AUC* is the 5-fold cross-validated AUC of a gradient-boosting classifier trained to tell internal loans from sampled ones. It measures **joint** fit: 0.5 means the two cannot be told apart.
- *Default-rate error* is the absolute difference between the sample's default rate and the internal default rate.

## 5. Results — main run (seed {SEED})

{md_table(t_main)}

Gower is closer than Stratified on **{gw_f} of {len(f)}** features:

{md_table(t_feat)}

![Per-feature scores](fig_feature_scores.png)

![Default rates](fig_default_rates.png)

### Stability across {n_runs} splits (mean ± sd)

{md_table(t_rob)}

![Robustness](fig_robustness.png)

## 6. Interpretation

- **What Gower optimises.** Gower sampling looks for close matches to each internal loan, so it should preserve the *joint* structure of the features, including interactions that stratification on {len(STRAT_COLS)} columns cannot see. The discriminator AUC is the metric that tests this. Here **{verdict_txt('disc_auc')}**.
- **Why Gower can distort the mix.** The union of {n_picks:,} neighbour picks (internal loans × {N_PER_CENTROID}) collapsed to {int(main['summary'].loc['Gower', 'size']):,} unique records, so {pct(collapsed)} of picks were duplicates. Duplicates arise where internal loans share nearest neighbours, which happens mostly in dense regions, so de-duplication under-weights typical profiles relative to unusual ones. This is the likely reason the Gower sample is easier to tell apart from the portfolio than a random draw is.
- **What stratification optimises.** Stratified sampling reproduces the {len(STRAT_COLS)} stratification columns almost exactly by construction (see their TVD in the feature table). Within each stratum it draws at random, so it inherits the random draw's fit on the other features.
- **Default rate.** Neither method sees the target. Any default-rate gap therefore comes from how well the features that drive default are reproduced. Here **{verdict_txt('dr_abs_error')}**.
- **Sample size.** With {N_PER_CENTROID} neighbours per centroid, the Gower sample covers {pct(s.loc['Gower', 'coverage'])} of the pool. Changing `N_PER_CENTROID` changes the size, and with it the trade-off between how closely the sample matches and how many records are bought.

## 7. Limitations

1. **The pool is already representative.** The internal portfolio is a random subset of the same population as the pool, so even a random draw matches it well. A realistic test would use an internal book that differs from the pool, for example a bank that skews towards certain grades or loan intents.
2. **Outliers compress Gower's numeric distances.** Range scaling means a few extreme values squeeze everyone else together: {skew_txt}. In those features nearly all pairs look alike, so the categorical features dominate the distance.
3. **Missing values.** These are {na_txt}. Gower skips them pair by pair; KS ignores them; the discriminator handles them natively.
4. **The two samples differ in size.** Stratified de-duplication can leave that sample a few rows smaller than the Gower sample, and KS/TVD are slightly noisier for smaller samples.
5. **No downstream test.** The fit metrics are proxies. The decision-relevant question is whether a PD model or scorecard trained on each sample performs well on the internal book.

## 8. Suggested next steps

- Repeat the comparison with a **deliberately biased internal portfolio**, so the pool no longer matches it by construction.
- **Keep Gower's duplicates as weights** (each external record weighted by how many internal loans picked it) instead of de-duplicating them. This restores the one-to-{N_PER_CENTROID} proportionality with the portfolio.
- Make Gower **robust to outliers**: cap features at their 1st and 99th percentiles, or scale by the IQR instead of the range.
- Add a **downstream check**: train a PD model on each sample and score it on the internal book (Gini, calibration).
- Tune `N_PER_CENTROID` to trade closeness of fit against the number of records bought.
"""


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("1. Importing data …")
    df, num_cols, cat_cols = load_data()
    print(f"   {len(df):,} rows · default rate {df['default'].mean():.1%}")
    print(f"   Numeric     : {num_cols}")
    print(f"   Categorical : {cat_cols}")

    print(f"\n2. Main run (seed {SEED}) …")
    main_run = run_once(df, num_cols, cat_cols, SEED)
    print(main_run["summary"].round(4).to_string())

    print(f"\n3. Robustness: {N_RUNS} splits …")
    rob_rows = []
    for seed in range(N_RUNS):
        print(f"   — seed {seed}")
        res = run_once(df, num_cols, cat_cols, seed)
        for m in METHODS:
            rob_rows.append({"seed": seed, "method": m, **res["summary"].loc[m].to_dict()})
    robust = pd.DataFrame(rob_rows)

    print("\n4. Writing outputs …")
    main_run["features"].to_csv(OUT_DIR / "metrics_features.csv", index=False)
    main_run["summary"].to_csv(OUT_DIR / "metrics_summary.csv")
    robust.to_csv(OUT_DIR / "metrics_robustness.csv", index=False)
    plot_feature_scores(main_run["features"], OUT_DIR / "fig_feature_scores.png")
    plot_default_rates(main_run, OUT_DIR / "fig_default_rates.png")
    plot_robustness(robust, OUT_DIR / "fig_robustness.png")
    memo_md = build_memo(df, num_cols, cat_cols, main_run, robust)
    MEMO_FILE.write_text(markdown_to_html(memo_md, "Gower vs stratified sampling memo", OUT_DIR),
                         encoding="utf-8")
    print(f"✓ Memo saved to {MEMO_FILE}")


if __name__ == "__main__":
    main()
