"""
Stratified sampling  (baseline)
───────────────────────────────
The external pool is sampled so that the joint distribution of the
stratification columns matches the internal dataset.
"""

import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# STRATIFIED SAMPLING  (baseline)
# ══════════════════════════════════════════════════════════════════════════════

def stratified_sample(df_internal: pd.DataFrame,
                      df_external: pd.DataFrame,
                      strat_cols: list,
                      target_size: int,
                      drop_cols: tuple = ("class", "default"),
                      random_state: int = 42) -> pd.DataFrame:
    """
    Sample target_size rows from df_external so that the joint distribution
    of strat_cols matches df_internal.

    Strata absent in df_external are skipped.
    Strata with insufficient rows are sampled with replacement.
    """
    dist  = df_internal.groupby(strat_cols).size() / len(df_internal)
    parts = []

    for key, prop in dist.items():
        n    = max(1, round(prop * target_size))
        keys = key if isinstance(key, tuple) else (key,)

        mask = pd.Series(True, index=df_external.index)
        for col, val in zip(strat_cols, keys):
            mask &= (df_external[col] == val)

        stratum = df_external[mask]
        if len(stratum) == 0:
            continue

        replace = len(stratum) < n
        parts.append(stratum.sample(n=n, replace=replace, random_state=random_state))

    result = pd.concat(parts).drop_duplicates().reset_index(drop=True)
    coverage = len(result) / len(df_external)
    print(f"   Stratified sample : {len(result)} obs  "
          f"({coverage:.1%} of external pool)")
    return result
