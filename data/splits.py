from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch

# =========================================================
# Split strategy
# =========================================================
STRATEGY_MAP = {
    0: "random_subject",
    1: "stratify_dose_even",
    2: "leave_one_dose_out",
    3: "only_bw_range",
    4: "stratify_dose_even_no_placebo_test",
    5: "robust_deterministic",
    6: "stratify_bw_even",
    7: "leave_high_bw_out",
    8: "task3_scenario",
    9: "highest_bw_one_test",
    10: "stratify_dose_even_no_placebo_valtest", 
}


def normalize_split_strategy(x) -> str:
    s = str(x).strip().lower()
    if s.isdigit():
        code = int(s)
        if code in STRATEGY_MAP:
            return STRATEGY_MAP[code]
        raise ValueError(f"Unknown split strategy code: {code}")
    if s in STRATEGY_MAP.values():
        return s
    raise ValueError(f"Unknown split strategy: {x}")

def parse_float_list(val) -> List[float]:
    if val is None or val == "":
        return []
    if isinstance(val, (list, tuple, np.ndarray)):
        return [float(x) for x in val]
    return [float(x.strip()) for x in str(val).split(",") if str(x).strip()]

# =========================================================
# Utilities
# =========================================================
def _first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _mode_or_nan(s: pd.Series):
    s = s.dropna()
    if s.empty:
        return np.nan
    m = s.mode()
    return m.iloc[0] if len(m) else np.nan

def _detect_id_col(df: pd.DataFrame) -> str:
    return _first_existing(df, ["ID", "SUBJ", "SUBJECT", "USUBJID"]) or "ID"

def _subject_primary_dose(df: pd.DataFrame, id_col: str, dose_col: str) -> pd.Series:
    """
    Calculate representative dose:
    - If subject has any DOSE==0 → representative dose = 0 (placebo-safe)
    - Else if >0 doses exist → use most frequent >0 dose
    - Else return 0
    """
    d = df.loc[df.get("EVID", 0).eq(1), [id_col, dose_col]].dropna()
    if d.empty:
        d = df[[id_col, dose_col]].dropna()

    def per_id(s: pd.Series) -> float:
        vals = s.values.astype(float)
        if (vals == 0).any():
            return 0.0
        nz = vals[vals > 0]
        if nz.size:
            uniq, cnt = np.unique(nz, return_counts=True)
            return float(uniq[np.argmax(cnt)])
        return 0.0

    return d.groupby(id_col)[dose_col].apply(per_id).astype(float)

def _split_ids(ids: np.ndarray, test_size: float, rng: np.random.RandomState) -> tuple[np.ndarray, np.ndarray]:
    ids = np.array(ids)
    if ids.size == 0:
        return np.array([], dtype=ids.dtype), np.array([], dtype=ids.dtype)

    perm = rng.permutation(ids)
    if 0 < test_size < 1:
        n_test = max(0, int(round(len(ids) * test_size)))
    else:
        n_test = int(max(0, test_size))

    n_test = min(n_test, len(ids) - 1)
    n_test = max(n_test, 0)

    test_ids = np.sort(perm[:n_test])
    train_ids = np.sort(perm[n_test:])
    return train_ids, test_ids

# =========================================================
# Core splitting
# =========================================================
def split_dataset(
    df: pd.DataFrame,
    *,
    id_col: Optional[str] = None,
    strategy: str = "random_subject",
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 123,
    dose_col: Optional[str] = None,
    n_dose_bins: int = 4,
    leaveout_doses: Optional[Iterable[float]] = None,
    bw_col: Optional[str] = None,
    bw_range: Optional[Tuple[float, float]] = None,
    quantile_range: Optional[Tuple[float, float]] = None,
    n_bw_bins: int = 4,
) -> Dict[str, pd.DataFrame]:
    df = df.copy()
    id_col = id_col or _detect_id_col(df)
    if id_col not in df.columns:
        raise ValueError(f"ID column not found: '{id_col}'")


    df = df.copy()
    id_col = id_col or _detect_id_col(df)
    if id_col not in df.columns:
        raise ValueError(f"ID column not found: '{id_col}'")

    rng = np.random.RandomState(random_state)
    all_ids = np.array(sorted(df[id_col].unique()))
    strategy = normalize_split_strategy(strategy)

    if strategy == "random_subject":
        tr_ids, te_ids = _split_ids(all_ids, test_size, rng)
        # Maintain global ratio: adjust val ratio in train set to (val_size / (1 - test_size))
        denom = max(1e-9, (1.0 - (len(te_ids) / max(1, len(all_ids)))))
        adj_val = min(0.99, max(0.0, val_size / denom))
        tr_ids, va_ids = _split_ids(tr_ids, adj_val, rng)

    elif strategy == "stratify_dose_even":
        dose_col = dose_col or _first_existing(df, ["DOSE", "AMT"]) or "DOSE"
        if dose_col not in df.columns:
            raise ValueError("dose_col not found.")
        subj_dose = _subject_primary_dose(df, id_col=id_col, dose_col=dose_col)
        zero_mask = subj_dose.eq(0.0)
        bins = pd.Series(index=subj_dose.index, dtype=object)
        bins[zero_mask] = "dose=0"
        nonzero = subj_dose[~zero_mask]
        if nonzero.size:
            q = min(n_dose_bins, max(1, nonzero.nunique()))
            qbins = pd.qcut(nonzero, q=q, duplicates="drop")
            bins.loc[nonzero.index] = qbins.astype(str)

        tr_ids, va_ids, te_ids = [], [], []
        # Apply same adjustment per bin to maintain global ratio
        for bin_name, idx in bins.groupby(bins):
            ids_b = np.array(sorted(idx.index))
            
            # Handle small bins properly
            if len(ids_b) < 3:
                # Too few subjects in this bin, assign to train
                tr_ids.extend(ids_b)
                continue
            
            # Use consistent random state for all bins
            bin_rng = rng
            tr_b, te_b = _split_ids(ids_b, test_size, bin_rng)
            
            # Adjust global val ratio within bin as well
            denom_b = max(1e-9, (1.0 - (len(te_b) / max(1, len(ids_b)))))
            adj_val_b = min(0.99, max(0.0, val_size / denom_b))
            tr_b, va_b = _split_ids(tr_b, adj_val_b, bin_rng)
            tr_ids += tr_b.tolist(); va_ids += va_b.tolist(); te_ids += te_b.tolist()
        tr_ids, va_ids, te_ids = np.array(sorted(tr_ids)), np.array(sorted(va_ids)), np.array(sorted(te_ids))

    elif strategy == "leave_one_dose_out":
        dose_col = dose_col or _first_existing(df, ["DOSE", "AMT"]) or "DOSE"
        if dose_col not in df.columns:
            raise ValueError("dose_col not found.")
        subj_dose = _subject_primary_dose(df, id_col=id_col, dose_col=dose_col)
        excl = set(parse_float_list(leaveout_doses))
        if not excl:
            raise ValueError("leave_one_dose_out requires non-empty leaveout_doses.")

        te_ids = np.array(sorted([i for i, v in subj_dose.items()
                                  if any(np.isclose(v, e, atol=1e-6) for e in excl)]))
        remain = np.array(sorted(np.setdiff1d(all_ids, te_ids)))

        # Global ratio adjustment: adjust val ratio in remaining considering test ratio
        test_ratio = len(te_ids) / max(1, len(all_ids))
        denom = max(1e-9, 1.0 - test_ratio)
        adj_val = min(0.99, max(0.0, val_size / denom))
        tr_ids, va_ids = _split_ids(remain, adj_val, rng)

    elif strategy == "stratify_dose_even_no_placebo_test":
        # Same as stratify_dose_even but exclude dose=0 from test set
        dose_col = dose_col or _first_existing(df, ["DOSE", "AMT"]) or "DOSE"
        if dose_col not in df.columns:
            raise ValueError("dose_col not found.")
        subj_dose = _subject_primary_dose(df, id_col=id_col, dose_col=dose_col)
        zero_mask = subj_dose.eq(0.0)
        bins = pd.Series(index=subj_dose.index, dtype=object)
        bins[zero_mask] = "dose=0"
        nonzero = subj_dose[~zero_mask]
        if nonzero.size:
            q = min(n_dose_bins, max(1, nonzero.nunique()))
            qbins = pd.qcut(nonzero, q=q, duplicates="drop")
            bins.loc[nonzero.index] = qbins.astype(str)

        tr_ids, va_ids, te_ids = [], [], []
        # Apply same adjustment per bin to maintain global ratio
        for bin_name, bin_ids in bins.groupby(bins):
            bin_ids = np.array(sorted(bin_ids.index))
            if len(bin_ids) < 2:
                # Too few subjects in this bin, assign to train
                tr_ids.extend(bin_ids)
                continue
            
            # For dose=0 bin, exclude from test set
            if bin_name == "dose=0":
                # Split dose=0 subjects between train and validation only
                n_test = 0  # No test subjects for placebo
                n_val = max(1, int(len(bin_ids) * val_size))
                n_tr = len(bin_ids) - n_val
                
                # Shuffle and assign
                rng.shuffle(bin_ids)
                tr_ids.extend(bin_ids[:n_tr])
                va_ids.extend(bin_ids[n_tr:n_tr + n_val])
            else:
                # For non-placebo doses, use normal split
                n_test = max(1, int(len(bin_ids) * test_size))
                n_val = max(1, int(len(bin_ids) * val_size))
                n_tr = len(bin_ids) - n_test - n_val
                
                # Shuffle and assign
                rng.shuffle(bin_ids)
                tr_ids.extend(bin_ids[:n_tr])
                va_ids.extend(bin_ids[n_tr:n_tr + n_val])
                te_ids.extend(bin_ids[n_tr + n_val:])

    elif strategy == "only_bw_range":
        bw_col = bw_col or _first_existing(df, ["BW", "WT", "WEIGHT", "BODYWEIGHT"]) or "BW"
        if bw_col not in df.columns:
            raise ValueError("bw_col not found.")
        subj_bw = df.groupby(id_col)[bw_col].median()
        if quantile_range:
            lo, hi = subj_bw.quantile(quantile_range[0]), subj_bw.quantile(quantile_range[1])
        elif bw_range:
            lo, hi = bw_range
        else:
            raise ValueError("only_bw_range requires bw_range or quantile_range.")
        keep = np.array(sorted(subj_bw[(subj_bw >= lo) & (subj_bw <= hi)].index))
        tr_ids, te_ids = _split_ids(keep, test_size, rng)

        # Global ratio adjustment
        denom = max(1e-9, (1.0 - (len(te_ids) / max(1, len(keep)))))
        adj_val = min(0.99, max(0.0, val_size / denom))
        tr_ids, va_ids = _split_ids(tr_ids, adj_val, rng)

    elif strategy == "stratify_bw_even":
        # BW-based stratified splitting (similar to stratify_dose_even but for BW)
        bw_col = bw_col or _first_existing(df, ["BW", "WT", "WEIGHT", "BODYWEIGHT"]) or "BW"
        if bw_col not in df.columns:
            raise ValueError("bw_col not found.")
        subj_bw = df.groupby(id_col)[bw_col].median()
        
        # Create BW bins (similar to dose bins)
        n_bw_bins = 4  # Default number of BW bins
        try:
            qbins = pd.qcut(subj_bw, q=n_bw_bins, duplicates="drop")
            bins = qbins.astype(str)
        except ValueError:
            # If qcut fails, use simple quantile-based bins
            bins = pd.Series(index=subj_bw.index, dtype=object)
            for i, (q_low, q_high) in enumerate(zip([0, 0.25, 0.5, 0.75], [0.25, 0.5, 0.75, 1.0])):
                mask = (subj_bw >= subj_bw.quantile(q_low)) & (subj_bw <= subj_bw.quantile(q_high))
                bins[mask] = f"bw_bin_{i+1}"

        tr_ids, va_ids, te_ids = [], [], []
        # Apply same adjustment per bin to maintain global ratio
        for bin_name, idx in bins.groupby(bins):
            ids_b = np.array(sorted(idx.index))
            
            # Handle small bins properly
            if len(ids_b) < 3:
                # Too few subjects in this bin, assign to train
                tr_ids.extend(ids_b)
                continue
            
            # Use consistent random state for all bins
            bin_rng = rng
            tr_b, te_b = _split_ids(ids_b, test_size, bin_rng)
            
            # Adjust global val ratio within bin as well
            denom_b = max(1e-9, (1.0 - (len(te_b) / max(1, len(ids_b)))))
            adj_val_b = min(0.99, max(0.0, val_size / denom_b))
            tr_b, va_b = _split_ids(tr_b, adj_val_b, bin_rng)
            tr_ids += tr_b.tolist(); va_ids += va_b.tolist(); te_ids += te_b.tolist()
        tr_ids, va_ids, te_ids = np.array(sorted(tr_ids)), np.array(sorted(va_ids)), np.array(sorted(te_ids))

    elif strategy == "leave_high_bw_out":
        # Leave high BW subjects out for testing (for Question 3 evaluation)
        bw_col = bw_col or _first_existing(df, ["BW", "WT", "WEIGHT", "BODYWEIGHT"]) or "BW"
        if bw_col not in df.columns:
            raise ValueError("bw_col not found.")
        subj_bw = df.groupby(id_col)[bw_col].median()
        
        # Define high BW threshold (e.g., 75th percentile or 80 kg)
        high_bw_threshold = subj_bw.quantile(0.75)  # 75th percentile
        if high_bw_threshold < 80:
            high_bw_threshold = 80  # Minimum threshold of 80 kg
        
        # Separate high BW and normal BW subjects
        high_bw_subjects = np.array(sorted(subj_bw[subj_bw >= high_bw_threshold].index))
        normal_bw_subjects = np.array(sorted(subj_bw[subj_bw < high_bw_threshold].index))
        
        print(f"   High BW threshold: {high_bw_threshold:.1f} kg")
        print(f"   High BW subjects: {len(high_bw_subjects)}")
        print(f"   Normal BW subjects: {len(normal_bw_subjects)}")
        
        # Use high BW subjects as test set
        te_ids = high_bw_subjects
        
        # Split normal BW subjects into train/val
        if len(normal_bw_subjects) < 2:
            raise ValueError("Not enough normal BW subjects for train/val split")
        
        # Adjust val ratio considering test ratio
        test_ratio = len(te_ids) / max(1, len(all_ids))
        denom = max(1e-9, 1.0 - test_ratio)
        adj_val = min(0.99, max(0.0, val_size / denom))
        
    elif strategy == "task3_scenario":
        bw_col = bw_col or _first_existing(df, ["BW", "WT", "WEIGHT", "BODYWEIGHT"]) or "BW"
        if bw_col not in df.columns:
            raise ValueError("bw_col not found.")
        subj_bw = df.groupby(id_col)[bw_col].median()
        
        # Define BW categories
        train_subjects = np.array(sorted(subj_bw[(subj_bw >= 60) & (subj_bw <= 90)].index))
        test_subjects = np.array(sorted(subj_bw[(subj_bw >= 80) & (subj_bw <= 120)].index))
        common_subjects = np.array(sorted(subj_bw[(subj_bw >= 80) & (subj_bw <= 90)].index))
        
        if len(common_subjects) > 0:
            np.random.seed(random_state)
            common_indices = np.random.permutation(len(common_subjects))
            mid_point = len(common_subjects) // 2
            common_train = common_subjects[common_indices[:mid_point]]
            common_test = common_subjects[common_indices[mid_point:]]
        else:
            common_train, common_test = np.array([]), np.array([])
        
        # Dose info for placebo filter
        subj_dose = _subject_primary_dose(df, id_col=id_col, dose_col=dose_col or "DOSE")
        placebo_ids = set(subj_dose[subj_dose == 0].index)

        # Ensure placebo never enters test
        te_ids = np.setdiff1d(np.concatenate([test_subjects, common_test]), list(placebo_ids))
        tr_ids = np.concatenate([train_subjects, common_train, list(placebo_ids)])
        
        adj_val = min(0.99, max(0.0, val_size))
        tr_ids, va_ids = _split_ids(tr_ids, adj_val, rng)

    elif strategy == "highest_bw_one_test":
        bw_col = bw_col or _first_existing(df, ["BW", "WT", "WEIGHT", "BODYWEIGHT"]) or "BW"
        if bw_col not in df.columns:
            raise ValueError("bw_col not found.")

        # 각 subject의 대표 BW 계산 (중앙값 사용)
        subj_bw = df.groupby(id_col)[bw_col].median()

        # 최고 BW subject 1명만 test set
        highest_id = subj_bw.idxmax()
        te_ids = np.array([highest_id])

        # 나머지 subject
        remain_ids = np.array(sorted(set(all_ids) - {highest_id}))

        # train/val split (val_size 비율)
        tr_ids, va_ids = _split_ids(remain_ids, val_size, rng)

    elif strategy == "stratify_dose_even_no_placebo_valtest":
        # Same as stratify_dose_even, but placebo subjects go only to train (no val/test)
        dose_col = dose_col or _first_existing(df, ["DOSE", "AMT"]) or "DOSE"
        if dose_col not in df.columns:
            raise ValueError("dose_col not found.")
        subj_dose = _subject_primary_dose(df, id_col=id_col, dose_col=dose_col)

        zero_mask = subj_dose.eq(0.0)
        bins = pd.Series(index=subj_dose.index, dtype=object)
        bins[zero_mask] = "dose=0"
        nonzero = subj_dose[~zero_mask]
        if nonzero.size:
            q = min(n_dose_bins, max(1, nonzero.nunique()))
            qbins = pd.qcut(nonzero, q=q, duplicates="drop")
            bins.loc[nonzero.index] = qbins.astype(str)

        tr_ids, va_ids, te_ids = [], [], []
        for bin_name, bin_ids in bins.groupby(bins):
            bin_ids = np.array(sorted(bin_ids.index))
            if len(bin_ids) < 2:
                tr_ids.extend(bin_ids)
                continue

            if bin_name == "dose=0":
                # Placebo subjects → train only
                tr_ids.extend(bin_ids)
            else:
                # Normal split for non-placebo bins
                n_test = max(1, int(len(bin_ids) * test_size))
                n_val = max(1, int(len(bin_ids) * val_size))
                n_tr = len(bin_ids) - n_test - n_val

                rng.shuffle(bin_ids)
                tr_ids.extend(bin_ids[:n_tr])
                va_ids.extend(bin_ids[n_tr:n_tr + n_val])
                te_ids.extend(bin_ids[n_tr + n_val:])

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    train_mask = df[id_col].isin(tr_ids)
    val_mask = df[id_col].isin(va_ids)
    test_mask = df[id_col].isin(te_ids)
    return {"train": df[train_mask].copy(), "val": df[val_mask].copy(), "test": df[test_mask].copy()}

# =========================================================
# Universe & split-ready DF
# =========================================================
def choose_universe_and_build_split_df(
    df_final: pd.DataFrame,
    df_dose: Optional[pd.DataFrame] = None,
    *,
    id_universe: str = 'union'
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Select PK/PD universe from df_final and construct representative DOSE for splitting.
    - Priority: AMT/DOSE mode from df_dose(EVID==1) → if not available, DOSE mode from df_final
    Return: (df_universe, pk_df, pd_df, dose_grp, df_for_split)
    """
    pk_df = df_final[df_final['DVID'] == 1].copy()
    pd_df = df_final[df_final['DVID'] == 2].copy()

    pk_ids = set(pk_df['ID'].unique())
    pd_ids = set(pd_df['ID'].unique())
    matched = pk_ids & pd_ids
    union = pk_ids | pd_ids

    used = matched if id_universe == 'intersection' else union
    if not used:
        raise ValueError("No subjects in chosen universe!")

    pk_df = pk_df[pk_df['ID'].isin(used)].copy()
    pd_df = pd_df[pd_df['ID'].isin(used)].copy()
    df_universe = df_final[df_final['ID'].isin(used)].copy()

    # Calculate representative DOSE: df_dose first
    if df_dose is not None and not df_dose.empty:
        id_col = _detect_id_col(df_dose)
        dose_name = _first_existing(df_dose, ["AMT", "DOSE"]) or "AMT"
        tmp = df_dose[df_dose[id_col].isin(used)].copy()
        rep_from_dose = _subject_primary_dose(tmp, id_col=id_col, dose_col=dose_name)
        rep_from_dose.name = "DOSE_SPLIT"
    else:
        rep_from_dose = pd.Series(dtype=float, name="DOSE_SPLIT")

    # Auxiliary: observation-based (mode)
    rep_from_obs = df_universe.groupby("ID")["DOSE"].agg(_mode_or_nan)
    rep_from_obs.name = "DOSE_SPLIT"

    # Merge: dosing-based first, if not available use observation-based
    dose_grp = rep_from_dose.reindex(list(used))
    dose_grp = dose_grp.fillna(rep_from_obs)
    dose_grp = dose_grp.fillna(0.0)

    df_universe = df_universe.merge(dose_grp.rename("DOSE_SPLIT"), left_on='ID', right_index=True, how='left')

    # DF for splitting: drop original DOSE and use DOSE_SPLIT as DOSE
    df_for_split = df_universe.copy()
    df_for_split.drop(columns=['DOSE'], inplace=True, errors='ignore')
    df_for_split.rename(columns={'DOSE_SPLIT': 'DOSE'}, inplace=True)
    return df_universe, pk_df, pd_df, dose_grp, df_for_split

def project_split(sub_df: pd.DataFrame, global_splits: Dict[str, pd.DataFrame], id_col: str = "ID") -> Dict[str, pd.DataFrame]:
    """Project a global (train/val/test) split onto PK/PD subset."""
    idsets = {k: set(global_splits[k][id_col].unique()) for k in ("train","val","test")}
    return {k: sub_df[sub_df[id_col].isin(idsets[k])].copy() for k in ("train","val","test")}

# =========================================================
# Reporting
# =========================================================
def _col_as_series(df: pd.DataFrame, name: str) -> pd.Series:
    """Return df[name] if exists else a NaN series with proper name."""
    if name in df.columns:
        s = df[name]; s.name = name
        return s
    return pd.Series(np.nan, index=df.index, name=name)

def report_dose_and_bw(df: pd.DataFrame, tag: str, topn: int = 5):
    print(f"\n=== [{tag}] brief summary ===")
    dose_col = _col_as_series(df, 'DOSE')
    uniq = sorted(pd.unique(dose_col.dropna()))
    head = uniq[:topn]
    more = f" (+{len(uniq)-len(head)} more)" if len(uniq) > len(head) else ""
    zero_ratio = float((dose_col.fillna(0) == 0).mean())

    print(f"Rows={len(df):,} | IDs={df['ID'].nunique():,}")
    print(f"DOSE levels (row, non-null): {head}{more}")
    print(f"DOSE==0 (row): {zero_ratio:.3%}")

    subj_dose = df.groupby("ID")[dose_col.name].apply(_mode_or_nan)

    bw_name = _first_existing(df, ["BW","WT","WEIGHT","BODYWEIGHT"])
    if bw_name is not None:
        subj_bw = df.groupby("ID")[bw_name].first()
    else:
        subj_bw = pd.Series(np.nan, index=subj_dose.index, name="BW")

    idx = subj_dose.index.intersection(subj_bw.index)
    subj_dose = subj_dose.loc[idx]
    subj_bw   = subj_bw.loc[idx]

    dose_perkg = (subj_dose / subj_bw).replace([np.inf, -np.inf], np.nan)

    vc = subj_dose.value_counts(dropna=False)
    vc_head = vc.head(topn).to_dict()
    more2 = f" (+{len(vc)-len(vc_head)} more)" if len(vc) > len(vc_head) else ""

    if bw_name is not None and subj_bw.notna().any():
        bw_mean, bw_std = float(np.nanmean(subj_bw)), float(np.nanstd(subj_bw))
        bw_min, bw_med, bw_max = map(float, (np.nanmin(subj_bw), np.nanmedian(subj_bw), np.nanmax(subj_bw)))
        print(f"BW by ID: mean={bw_mean:.2f}±{bw_std:.2f}, min|med|max={bw_min:.1f}|{bw_med:.1f}|{bw_max:.1f}")
    else:
        print("BW by ID: (missing)")

    if dose_perkg.notna().any():
        dpk_med = float(np.nanmedian(dose_perkg))
        q25 = float(np.nanpercentile(dose_perkg.dropna(), 25))
        q75 = float(np.nanpercentile(dose_perkg.dropna(), 75))
        print(f"Primary DOSE/BW by ID: median={dpk_med:.4f} [IQR {q25:.4f}–{q75:.4f}]")
    else:
        print("Primary DOSE/BW by ID: (insufficient data)")

    print(f"Primary DOSE by ID (top): {vc_head}{more2}")

# =========================================================
# High-level helper: prepare_for_split
# =========================================================
def prepare_for_split(
    df_final: pd.DataFrame,
    df_dose: pd.DataFrame,
    pk_df: pd.DataFrame,
    pd_df: pd.DataFrame,
    *,
    split_strategy,
    test_size: float,
    val_size: float,
    random_state: int,
    dose_bins: int = 4,
    leaveout_doses: Optional[Iterable[float]] = None,
    bw_range: Optional[Tuple[float, float]] = None,
    bw_quantiles: Optional[Tuple[float, float]] = None,
    id_universe: str = 'intersection',
    verbose: bool = True,
):
    """Prepare global PK/PD splits under a chosen universe and strategy."""
    df_univ, pk_df_u, pd_df_u, dose_grp_obs, df_for_split = choose_universe_and_build_split_df(
        df_final, df_dose=df_dose, id_universe=id_universe
    )

    strategy = normalize_split_strategy(split_strategy)
    leaveout = parse_float_list(leaveout_doses)

    split_kwargs = dict(
        strategy=strategy,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        dose_col='DOSE',
        bw_col='BW',
    )
    if strategy == "stratify_dose_even":
        split_kwargs["n_dose_bins"] = dose_bins
    elif strategy == "leave_one_dose_out":
        if not leaveout:
            raise ValueError("leave_one_dose_out requires leaveout_doses (e.g., [0,10,30]).")
        split_kwargs["leaveout_doses"] = leaveout
    elif strategy == "only_bw_range":
        if bw_range:
            split_kwargs["bw_range"] = bw_range
        elif bw_quantiles:
            split_kwargs["quantile_range"] = bw_quantiles
        else:
            raise ValueError("only_bw_range requires bw_range or bw_quantiles.")
    elif strategy == "robust_deterministic":
        # Use the new robust deterministic splitter
        splitter = RobustDeterministicSplitter(random_state=random_state)
        pk_splits, pd_splits, global_splits, _ = splitter.split_data_robustly(
            df_univ, df_dose, pk_df_u, pd_df_u, test_size, val_size
        )
        return pk_splits, pd_splits, global_splits, df_for_split
    elif strategy in ["stratify_bw_even", "leave_high_bw_out"]:
        # BW-based splitting strategies
        split_kwargs["bw_col"] = 'BW'
        if strategy == "stratify_bw_even":
            split_kwargs["n_bw_bins"] = 4

    global_splits = split_dataset(df_for_split, **split_kwargs)
    pk_splits = project_split(pk_df_u, global_splits)
    pd_splits = project_split(pd_df_u, global_splits)

    pk_ids_u = set(pk_df_u["ID"].unique())
    pd_ids_u = set(pd_df_u["ID"].unique())
    used_ids = pk_ids_u | pd_ids_u if id_universe == 'union' else (pk_ids_u & pd_ids_u)
    for k in ("train","val","test"):
        pk_ids_k = set(pk_splits[k]["ID"].unique())
        pd_ids_k = set(pd_splits[k]["ID"].unique())
        assert pk_ids_k.issubset(used_ids) and pd_ids_k.issubset(used_ids)
        if id_universe == 'intersection':
            assert pk_ids_k == pd_ids_k, f"{k} IDs not aligned: PK {len(pk_ids_k)} vs PD {len(pd_ids_k)}"

    if verbose:
        for split in ("train","val","test"):
            ids = global_splits[split]['ID'].unique()
            levels = sorted(dose_grp_obs.loc[ids].unique().tolist())
            print(f"[{split}] representative DOSE levels: {levels}")
        report_dose_and_bw(global_splits['train'], "Train (matched IDs)")
        report_dose_and_bw(global_splits['val'],   "Val (matched IDs)")
        report_dose_and_bw(global_splits['test'],  "Test (matched IDs)")

    return pk_splits, pd_splits, global_splits, df_for_split


# =========================================================
# Robust Deterministic Data Splitter
# =========================================================
class RobustDeterministicSplitter:
    """
    Robust Deterministic Data Splitter
    
    Solves the issues with stratify_dose_even:
    1. ✅ Consistent splits across runs (deterministic)
    2. ✅ Subject integrity (same subject in same split)
    3. ✅ Balanced dose distribution
    4. ✅ Handles small bins properly
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
        torch.manual_seed(random_state)
    
    def split_data_robustly(self, df_final, df_dose, pk_df, pd_df, 
                           test_size=0.1, val_size=0.1, min_subjects_per_bin=3):
        """
        Split data robustly with dose stratification and subject integrity
        
        Args:
            df_final: Final processed dataframe
            df_dose: Dose dataframe
            pk_df: PK dataframe
            pd_df: PD dataframe
            test_size: Test set size ratio
            val_size: Validation set size ratio
            min_subjects_per_bin: Minimum subjects per dose bin
            
        Returns:
            Tuple of (pk_splits, pd_splits, global_splits, None)
        """
        print("🔄 Robust Deterministic Data Splitting")
        print("=" * 50)
        
        # Get subject dose information
        dose_col = _first_existing(df_final, ["DOSE", "AMT"]) or "DOSE"
        subj_dose = _subject_primary_dose(df_final, id_col="ID", dose_col=dose_col)
        
        # Create dose bins deterministically
        dose_bins = self._create_dose_bins_deterministically(subj_dose, min_subjects_per_bin)
        
        # Split subjects within each bin deterministically
        train_subjects, val_subjects, test_subjects = self._split_subjects_by_bins(
            dose_bins, test_size, val_size
        )
        
        print(f"   Total subjects: {len(subj_dose)}")
        print(f"   Train subjects: {len(train_subjects)}")
        print(f"   Val subjects: {len(val_subjects)}")
        print(f"   Test subjects: {len(test_subjects)}")
        
        # Create splits based on subject IDs
        pk_splits = {
            'train': pk_df[pk_df['ID'].isin(train_subjects)].copy(),
            'val': pk_df[pk_df['ID'].isin(val_subjects)].copy(),
            'test': pk_df[pk_df['ID'].isin(test_subjects)].copy()
        }
        
        pd_splits = {
            'train': pd_df[pd_df['ID'].isin(train_subjects)].copy(),
            'val': pd_df[pd_df['ID'].isin(val_subjects)].copy(),
            'test': pd_df[pd_df['ID'].isin(test_subjects)].copy()
        }
        
        # Create global splits
        global_splits = {
            'train': df_final[df_final['ID'].isin(train_subjects)].copy(),
            'val': df_final[df_final['ID'].isin(val_subjects)].copy(),
            'test': df_final[df_final['ID'].isin(test_subjects)].copy()
        }
        
        print(f"   PK splits: Train={pk_splits['train'].shape}, Val={pk_splits['val'].shape}, Test={pk_splits['test'].shape}")
        print(f"   PD splits: Train={pd_splits['train'].shape}, Val={pd_splits['val'].shape}, Test={pd_splits['test'].shape}")
        
        return pk_splits, pd_splits, global_splits, None
    
    def _create_dose_bins_deterministically(self, subj_dose, min_subjects_per_bin):
        """Create dose bins deterministically"""
        # Separate placebo and non-placebo
        zero_mask = subj_dose.eq(0.0)
        placebo_subjects = subj_dose[zero_mask].index.tolist()
        nonzero_subjects = subj_dose[~zero_mask]
        
        bins = {}
        
        # Placebo bin
        if len(placebo_subjects) >= min_subjects_per_bin:
            bins["placebo"] = placebo_subjects
        else:
            # If too few placebo subjects, merge with lowest dose bin
            bins["low_dose"] = placebo_subjects
        
        # Non-placebo bins
        if len(nonzero_subjects) > 0:
            # Sort by dose and create bins
            sorted_subjects = nonzero_subjects.sort_values().index.tolist()
            
            # Create 2-3 bins for non-placebo doses
            n_bins = min(3, max(2, len(sorted_subjects) // min_subjects_per_bin))
            bin_size = len(sorted_subjects) // n_bins
            
            for i in range(n_bins):
                start_idx = i * bin_size
                end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(sorted_subjects)
                bin_subjects = sorted_subjects[start_idx:end_idx]
                
                if len(bin_subjects) >= min_subjects_per_bin:
                    bin_name = f"dose_bin_{i+1}"
                    bins[bin_name] = bin_subjects
                else:
                    # Merge with previous bin if too small
                    if bins:
                        last_bin = list(bins.keys())[-1]
                        bins[last_bin].extend(bin_subjects)
                    else:
                        bins["low_dose"] = bin_subjects
        
        return bins
    
    def _split_subjects_by_bins(self, dose_bins, test_size, val_size):
        """Split subjects within each bin deterministically"""
        train_subjects = []
        val_subjects = []
        test_subjects = []
        
        for bin_name, subjects in dose_bins.items():
            # Sort subjects deterministically
            sorted_subjects = sorted(subjects)
            
            # Calculate split sizes for this bin
            n_subjects = len(sorted_subjects)
            n_test = max(1, int(n_subjects * test_size))
            n_val = max(1, int(n_subjects * val_size))
            n_train = n_subjects - n_test - n_val
            
            # Ensure at least 1 subject in each split
            if n_train < 1:
                n_train = 1
                n_val = max(1, n_subjects - n_train - 1)
                n_test = n_subjects - n_train - n_val
            
            # Split deterministically
            train_subjects.extend(sorted_subjects[:n_train])
            val_subjects.extend(sorted_subjects[n_train:n_train + n_val])
            test_subjects.extend(sorted_subjects[n_train + n_val:])
            
            print(f"   {bin_name}: {n_subjects} subjects -> Train:{n_train}, Val:{n_val}, Test:{n_test}")
        
        return train_subjects, val_subjects, test_subjects

# Legacy class for backward compatibility
class DeterministicDataSplitter(RobustDeterministicSplitter):
    """Legacy class - use RobustDeterministicSplitter instead"""
    
    def split_data_deterministically(self, df_final, df_dose, pk_df, pd_df, 
                                   test_size=0.1, val_size=0.1):
        """Legacy method - redirects to robust splitting"""
        return self.split_data_robustly(df_final, df_dose, pk_df, pd_df, 
                                      test_size, val_size)
