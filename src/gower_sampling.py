"""
Gower-distance-based sampling
─────────────────────────────
Each internal observation is used as a centroid; the n_per_centroid
nearest records in the external pool (by Gower distance) are selected
and the union is deduplicated.

Missing values follow Gower's original rule: a feature that is missing
on either side of a pair is left out of that pair's average.
"""

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# GOWER DISTANCE
# ══════════════════════════════════════════════════════════════════════════════

def numeric_ranges(frames: list, num_cols: list) -> pd.Series:
    """Per-feature range (max − min) over all frames; zero ranges become 1."""
    combined = pd.concat([f[num_cols] for f in frames], ignore_index=True)
    r = combined.max() - combined.min()
    return r.where(r > 0, 1.0)


def gower_matrix(src: pd.DataFrame,
                 tgt: pd.DataFrame,
                 num_cols: list,
                 cat_cols: list,
                 ranges: pd.Series | None = None) -> np.ndarray:
    """
    Compute the (n_src × n_tgt) Gower distance matrix.

    Partial distance per feature:
      Numeric     : |x − y| / range(feature)  ∈ [0, 1]
      Categorical : 0 if x == y else 1

    Overall distance = mean of the partial distances that are not missing.
    Pass `ranges` when calling on chunks of src so every chunk is scaled
    the same way; by default ranges come from src ∪ tgt.
    """
    if ranges is None:
        ranges = numeric_ranges([src, tgt], num_cols)

    D = np.zeros((len(src), len(tgt)))
    W = np.zeros((len(src), len(tgt)))     # number of usable features per pair

    for col in num_cols:
        s_v   = src[col].to_numpy(dtype=float)[:, None]   # (n_src, 1)
        t_v   = tgt[col].to_numpy(dtype=float)[None, :]   # (1, n_tgt)
        d     = np.abs(s_v - t_v) / ranges[col]
        valid = ~np.isnan(d)
        D    += np.where(valid, d, 0.0)
        W    += valid

    for col in cat_cols:
        cats  = pd.unique(pd.concat([src[col], tgt[col]]).dropna())
        s_v   = pd.Categorical(src[col], categories=cats).codes[:, None]
        t_v   = pd.Categorical(tgt[col], categories=cats).codes[None, :]
        valid = (s_v >= 0) & (t_v >= 0)                    # code −1 = missing
        D    += (s_v != t_v) & valid
        W    += valid

    return np.divide(D, W, out=np.full_like(D, np.nan), where=W > 0)


# ══════════════════════════════════════════════════════════════════════════════
# GOWER-BASED SAMPLING  (novel method)
# ══════════════════════════════════════════════════════════════════════════════

def gower_sample(df_internal: pd.DataFrame,
                 df_external: pd.DataFrame,
                 num_cols: list,
                 cat_cols: list,
                 n_per_centroid: int = 3,
                 drop_cols: tuple = ("class", "default"),
                 chunk_size: int | None = None) -> tuple:
    """
    For every observation in df_internal (centroid), select the
    n_per_centroid closest records in df_external by Gower distance.
    The union (deduplicated) forms the Gower sample.

    chunk_size : process this many centroids at a time so the full
                 (n_int × n_ext) matrix is never held in memory.
                 None = one pass, and the full matrix is returned.

    Returns
    -------
    df_sample : pd.DataFrame       – sampled rows from df_external
    D         : np.ndarray | None  – full distance matrix (n_int × n_ext),
                                     None when chunk_size is set
    """
    feat_int = df_internal.drop(columns=list(drop_cols), errors="ignore")
    feat_ext = df_external.drop(columns=list(drop_cols), errors="ignore")

    num_ = [c for c in num_cols if c in feat_int.columns]
    cat_ = [c for c in cat_cols if c in feat_int.columns]

    ranges = numeric_ranges([feat_int, feat_ext], num_)
    step   = chunk_size or len(feat_int)

    print("   Computing Gower distances …", end=" ", flush=True)
    selected = set()
    for start in range(0, len(feat_int), step):
        D  = gower_matrix(feat_int.iloc[start:start + step], feat_ext,
                          num_, cat_, ranges)
        nn = np.argsort(D, axis=1)[:, :n_per_centroid]
        selected.update(nn.ravel().tolist())
    print(f"done  ({len(feat_int)}×{len(feat_ext)})")

    df_sample = df_external.iloc[sorted(selected)].reset_index(drop=True)
    coverage  = len(df_sample) / len(df_external)
    print(f"   Gower sample : {len(df_sample)} obs  "
          f"({coverage:.1%} of external pool)")
    return df_sample, (None if chunk_size else D)
