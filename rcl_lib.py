"""rcl_lib - shared code for the RetailCreditLifecycle notebook (LendingClub only).

ONE dataset end to end: LendingClub accepted loans 2007-2018Q4 (public Kaggle dataset
wordsforthewise/lending-club).  PD, term structure, LGD, EAD, staging and ECL are all
built on this same book; nothing is borrowed from another dataset.

Everything a stage needs lives here so the notebook cells stay short and readable.
Fixed seeds, no randomness outside seeded paths, ASCII-safe prints.
"""
from __future__ import annotations

import json
import math
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (auc as sk_auc, average_precision_score, brier_score_loss,
                             log_loss, precision_recall_curve, roc_auc_score)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

SEED = 42
# The "2018Q4" file was pulled in March 2019: last_pymnt_d of Current loans is Mar-2019 and
# last_credit_pull_d is Mar-2019.  The snapshot month is therefore READ FROM THE DATA
# (max last_pymnt_d), not assumed to be Dec-2018 (the old project's 3-month age error).
SNAP_IDX = 2019 * 12 + 3
HORIZON_M = 12                             # PD horizon (months from issue)
LAST_PAY_MAX = 9                           # stopped paying by month 9 => 90+ DPD inside 12 months
MIN_AGE = 15                               # 12m window + ~3m to reach 90 DPD + charge-off posting lag
CO_LAG_M = 4                               # est. charge-off date = last payment + 4 months (120 DPD)

PD_FEATURES = [
    "loan_amnt", "term_months", "emp_length_years", "annual_inc", "dti",
    "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec", "revol_bal",
    "revol_util", "total_acc", "mort_acc", "fico_mid", "purpose", "addr_state",
    "home_ownership", "verification_status", "application_type",
]
CATEGORICAL = ["purpose", "addr_state", "home_ownership", "verification_status",
               "application_type"]
NUMERIC = [c for c in PD_FEATURES if c not in CATEGORICAL]
SCALING = {"base_score": 600, "pdo": 20, "base_odds": 20.0}

# columns that are known only AFTER origination (never PD features)
LEAKAGE_FIELDS = [
    "out_prncp", "total_rec_prncp", "total_rec_int", "total_pymnt", "recoveries",
    "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d",
    "last_credit_pull_d", "last_fico_range_high", "last_fico_range_low",
    "loan_status", "settlement_status", "debt_settlement_flag", "months_on_book",
    "months_to_last_pymnt",
    "grade", "sub_grade", "int_rate", "installment",   # LendingClub's own price/risk output
]

RAW_COLS = [
    "id", "loan_amnt", "funded_amnt", "term", "int_rate", "installment", "grade",
    "sub_grade", "emp_length", "home_ownership", "annual_inc", "verification_status",
    "issue_d", "loan_status", "purpose", "addr_state", "dti", "delinq_2yrs",
    "fico_range_low", "fico_range_high", "inq_last_6mths", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "mort_acc", "out_prncp", "total_rec_prncp",
    "total_rec_int", "total_pymnt", "recoveries", "collection_recovery_fee",
    "last_pymnt_d", "last_pymnt_amnt", "application_type", "last_fico_range_high",
    "last_fico_range_low", "last_credit_pull_d",
]
NUM_RAW = ["id", "loan_amnt", "funded_amnt", "installment", "annual_inc", "dti",
           "delinq_2yrs", "fico_range_low", "fico_range_high", "inq_last_6mths",
           "open_acc", "pub_rec", "revol_bal", "total_acc", "mort_acc", "out_prncp",
           "total_rec_prncp", "total_rec_int", "total_pymnt", "recoveries",
           "collection_recovery_fee", "last_pymnt_amnt", "last_fico_range_high",
           "last_fico_range_low"]
CAT_RAW = ["grade", "sub_grade", "home_ownership", "verification_status", "purpose",
           "addr_state", "loan_status", "application_type"]

# defaults seen by the lender: charged off / defaulted / 31-120 days late
EVENT_STATUS = ["Charged Off", "Default", "Late (31-120 days)",
                "Does not meet the credit policy. Status:Charged Off"]
CHARGED_OFF_STATUS = ["Charged Off", "Does not meet the credit policy. Status:Charged Off"]
ACTIVE_STATUS = ["Current", "Late (16-30 days)", "Late (31-120 days)", "In Grace Period",
                 "Default"]

_T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - _T0:7.0f}s] {msg}", flush=True)


# --------------------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------------------
def month_index(dt: pd.Series) -> pd.Series:
    return dt.dt.year * 12 + dt.dt.month


def _pct(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.strip().str.rstrip("%"), errors="coerce")


def _parse_chunk(c: pd.DataFrame) -> pd.DataFrame:
    for col in NUM_RAW:
        c[col] = pd.to_numeric(c[col], errors="coerce")
    c["issue_d"] = pd.to_datetime(c["issue_d"], format="%b-%Y", errors="coerce")
    c["last_pymnt_d"] = pd.to_datetime(c["last_pymnt_d"], format="%b-%Y", errors="coerce")
    c["last_credit_pull_d"] = pd.to_datetime(c["last_credit_pull_d"], format="%b-%Y",
                                             errors="coerce")
    c["int_rate"] = _pct(c["int_rate"])
    c["revol_util"] = _pct(c["revol_util"])
    c["term_months"] = pd.to_numeric(c["term"].astype(str).str.extract(r"(\d+)",
                                                                       expand=False),
                                     errors="coerce")
    emp = c["emp_length"].astype(str).str.strip()
    ey = pd.Series(np.nan, index=c.index, dtype="float64")
    ey[emp == "< 1 year"] = 0.0
    parsed = pd.to_numeric(emp.str.extract(r"^(\d+)\+? years?$", expand=False),
                           errors="coerce")
    ey[parsed.notna()] = parsed
    c["emp_length_years"] = ey
    c["fico_mid"] = (c["fico_range_low"] + c["fico_range_high"]) / 2.0
    c["last_fico_mid"] = (c["last_fico_range_low"] + c["last_fico_range_high"]) / 2.0
    for col in CAT_RAW:
        c[col] = c[col].astype("category")
    return c.drop(columns=["term", "emp_length"])


def load_accepted(path, sample_frac: float | None = None, chunk: int = 400_000,
                  seed: int = SEED):
    """Read the raw LendingClub accepted file chunk by chunk (memory-safe).

    Returns (frame, audit).  Footer/summary rows (no loan_amnt / issue_d) are dropped and
    counted.  `sample_frac` keeps a seeded random share of rows (FAST/smoke mode only).
    """
    rng = np.random.default_rng(seed)
    parts, audit = [], {"raw_rows": 0, "footer_rows_dropped": 0}
    reader = pd.read_csv(path, usecols=lambda c: c in set(RAW_COLS), dtype=str,
                         low_memory=False, chunksize=chunk)
    for raw in reader:
        audit["raw_rows"] += len(raw)
        parsed = _parse_chunk(raw)
        footer = parsed["loan_amnt"].isna() | parsed["issue_d"].isna() | parsed["id"].isna()
        audit["footer_rows_dropped"] += int(footer.sum())
        parsed = parsed[~footer]
        if sample_frac and sample_frac < 1.0:
            parsed = parsed[rng.random(len(parsed)) < sample_frac]
        parts.append(parsed)
    frame = pd.concat(parts, ignore_index=True)
    for col in CAT_RAW:   # categories must be consistent after concat
        frame[col] = frame[col].astype("category")
    frame["id"] = frame["id"].astype("int64")
    audit["rows_kept"] = int(len(frame))
    audit["sampled"] = bool(sample_frac and sample_frac < 1.0)
    return frame, audit


def snapshot_index(df: pd.DataFrame) -> int:
    """Snapshot month index = latest last-payment month in the file."""
    return int(month_index(df["last_pymnt_d"].dropna()).max())


def derive_basics(df: pd.DataFrame, snap_idx: int | None = None) -> pd.DataFrame:
    """Time fields measured against the data snapshot (Mar-2019, read from the file)."""
    df = df.copy()
    snap_idx = snap_idx or snapshot_index(df)
    df.attrs["snap_idx"] = snap_idx
    df["issue_month"] = df["issue_d"].dt.strftime("%Y-%m")
    df["mob"] = (snap_idx - month_index(df["issue_d"])).astype("int32")
    df["mtlp"] = month_index(df["last_pymnt_d"]) - month_index(df["issue_d"])  # months to last payment
    return df


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Duplicates are removed BEFORE any split so the same loan can never sit in two splits."""
    n0 = len(df)
    dup_id = int(df["id"].duplicated().sum())
    df = df.drop_duplicates(subset="id", keep="first")
    keys = ["issue_d", "loan_amnt", "funded_amnt", "term_months", "annual_inc", "dti",
            "fico_mid", "revol_bal", "revol_util", "open_acc", "total_acc", "addr_state",
            "purpose", "emp_length_years", "inq_last_6mths", "delinq_2yrs"]
    dup_app = int(df.duplicated(subset=keys, keep="first").sum())
    df = df.drop_duplicates(subset=keys, keep="first")
    return df.reset_index(drop=True), {"rows_in": n0, "duplicate_ids_removed": dup_id,
                                       "duplicate_applications_removed": dup_app,
                                       "rows_out": int(len(df))}


# --------------------------------------------------------------------------------------
# labels and splits
# --------------------------------------------------------------------------------------
def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """12-month PD label (frozen definition, see notebook cell 'Framing').

    bad_12m = 1 : loan ends charged off / defaulted / 31-120 days late AND its last payment
                 was made in month <= 9 (so it was 90+ days past due inside the first 12
                 months).  A loan that never paid anything (last_pymnt_d missing) is a
                 first-payment default and counts as bad.
    bad_12m = 0 : every other loan old enough to be observed (age >= 15 months),
                 INCLUDING loans still Current - they survived the window.
    undefined   : age < 15 months (window or charge-off posting lag not complete), or a
                 still-active loan that stopped paying inside the window (indeterminate).
    """
    df = df.copy()
    status = df["loan_status"].astype(str)
    event = status.isin(EVENT_STATUS)
    mtlp = df["mtlp"].astype(float)
    never_paid = event & mtlp.isna()
    mtlp_eff = mtlp.where(~never_paid, 0.0)
    stopped_in_window = mtlp_eff <= LAST_PAY_MAX
    active_nonevent = status.isin(["Current", "Late (16-30 days)", "In Grace Period"])
    indeterminate = active_nonevent & stopped_in_window & mtlp.notna()
    seasoned = df["mob"] >= MIN_AGE
    bad = (event & stopped_in_window).astype(float)
    pop = seasoned & ~indeterminate
    df["is_event"] = event
    df["never_paid"] = never_paid
    df["resolved_12m"] = pop
    df["bad_12m"] = np.where(pop, bad, np.nan)
    # legacy definition (for the comparison table only; NOT used for modelling)
    co = status.isin(["Charged Off", "Default"])
    fp = status.isin(["Fully Paid"])
    old_bad = co & (mtlp.fillna(np.inf) <= 12)
    old_good = fp & (df["mob"] >= 12)
    df["bad_12m_legacy"] = np.where(old_bad, 1.0, np.where(old_good, 0.0, np.nan))
    # default time (month in which the loan first reaches 90 DPD) for hazard work
    df["t_default"] = np.where(event, mtlp_eff + 3.0, np.nan)
    return df


def add_split(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    y = df["issue_d"].dt.year
    df["split"] = np.select(
        [df["issue_d"] <= pd.Timestamp("2015-12-31"), y == 2016, y == 2017, y >= 2018],
        ["train", "valid", "oot", "psi_only"], default="unassigned")
    return df


# --------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------
def gini(y, s) -> float:
    return float(2.0 * roc_auc_score(y, s) - 1.0)


def ks_stat(y, s) -> float:
    """KS evaluated at distinct score values (tie-safe)."""
    y = np.asarray(y, float)
    g = (pd.DataFrame({"y": y, "s": np.asarray(s, float)}).groupby("s", sort=True)["y"]
         .agg(["sum", "count"]))
    cb = g["sum"].cumsum() / y.sum()
    cg = (g["count"] - g["sum"]).cumsum() / (len(y) - y.sum())
    return float(np.max(np.abs(cb.to_numpy() - cg.to_numpy())))


def lift_at(y, s, frac=0.10) -> float:
    y = np.asarray(y)
    k = int(round(len(y) * frac))
    top = np.argsort(-np.asarray(s))[:k]
    return float(y[top].mean() / y.mean()) if y.mean() else float("nan")


def pr_auc(y, s) -> float:
    """Average precision (step-wise area under the precision-recall curve)."""
    return float(average_precision_score(y, s))


def ece_and_deciles(y, p, n_bins=10):
    f = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(p, float)})
    f["d"] = pd.qcut(f["p"].rank(method="first"), n_bins, labels=False)
    rows, ece = [], 0.0
    for d, g in f.groupby("d"):
        w = len(g) / len(f)
        e, a = float(g["p"].mean()), float(g["y"].mean())
        rows.append({"decile": int(d) + 1, "n": int(len(g)), "expected_pd": round(e, 5),
                     "actual_pd": round(a, 5), "difference": round(a - e, 5)})
        ece += w * abs(a - e)
    return float(ece), rows


def ece_point(y, p, n_bins=10) -> float:
    return ece_and_deciles(y, p, n_bins)[0]


def evaluate(y, p) -> dict:
    return {"n": int(len(y)), "bad_rate": round(float(np.mean(y)), 5),
            "auc": round(float(roc_auc_score(y, p)), 5), "gini": round(gini(y, p), 5),
            "ks": round(ks_stat(y, p), 5), "pr_auc": round(pr_auc(y, p), 5),
            "brier": round(float(brier_score_loss(y, p)), 5),
            "log_loss": round(float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))), 5),
            "lift_10": round(lift_at(y, p, 0.10), 4)}


def _midrank(v):
    order = np.argsort(v)
    z = v[order]
    n = len(v)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    out = np.empty(n)
    out[order] = t
    return out


def delong_test(y, a, b) -> dict:
    """Paired DeLong test (Sun & Xu 2014) for two correlated AUCs on the same rows."""
    y = np.asarray(y).astype(int)
    sc = np.vstack([np.asarray(a, float), np.asarray(b, float)])
    order = np.argsort(-y, kind="mergesort")
    m = int(y.sum())
    ss = sc[:, order]
    n = ss.shape[1] - m
    pos, neg = ss[:, :m], ss[:, m:]
    tx, ty, tz = np.empty((2, m)), np.empty((2, n)), np.empty((2, m + n))
    for r in range(2):
        tx[r], ty[r], tz[r] = _midrank(pos[r]), _midrank(neg[r]), _midrank(ss[r])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    d = aucs[0] - aucs[1]
    if var <= 0:
        z, p, se = 0.0, 1.0, 0.0
    else:
        se = math.sqrt(var)
        z = d / se
        p = 2.0 * float(stats.norm.sf(abs(z)))
    return {"auc_a": round(float(aucs[0]), 5), "auc_b": round(float(aucs[1]), 5),
            "delta_auc": round(float(d), 5), "delta_gini": round(float(2 * d), 5),
            "delta_gini_ci95_delong": [round(float(2 * (d - 1.96 * se)), 5),
                                       round(float(2 * (d + 1.96 * se)), 5)],
            "z": round(float(z), 4), "p_value": float(p)}


def bootstrap_gini_delta(y, a, b, n_iter=1000, seed=SEED) -> dict:
    """Paired row bootstrap of Gini(a) - Gini(b)."""
    rng = np.random.default_rng(seed)
    y, a, b = np.asarray(y), np.asarray(a), np.asarray(b)
    idx = np.arange(len(y))
    d = np.full(n_iter, np.nan)
    for i in range(n_iter):
        s = rng.choice(idx, size=len(idx), replace=True)
        if y[s].sum() in (0, len(s)):
            continue
        d[i] = gini(y[s], a[s]) - gini(y[s], b[s])
    k = ~np.isnan(d)
    return {"delta": round(float(np.nanmean(d)), 5),
            "ci95": [round(float(np.percentile(d[k], 2.5)), 5),
                     round(float(np.percentile(d[k], 97.5)), 5)],
            "p_two_sided": round(float(2 * min((d[k] <= 0).mean(), (d[k] >= 0).mean())), 5),
            "n_iter": int(k.sum())}


def psi(expected, actual, n_bins=10) -> float:
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(expected[~np.isnan(expected)], np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return float("nan")
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.clip(np.histogram(expected[~np.isnan(expected)], edges)[0] / np.sum(~np.isnan(expected)), 1e-6, None)
    a = np.clip(np.histogram(actual[~np.isnan(actual)], edges)[0] / np.sum(~np.isnan(actual)), 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_band(v: float) -> str:
    return "unknown" if not np.isfinite(v) else ("stable" if v < 0.10 else ("watch" if v <= 0.25 else "shift"))


def score_from_pd(p):
    f = SCALING["pdo"] / math.log(2.0)
    off = SCALING["base_score"] - f * math.log(SCALING["base_odds"])
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return off - f * np.log(p / (1 - p))


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return [jsonable(v) for v in o.tolist()]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o.date())
    return o


def write_json(obj, path: Path) -> None:
    Path(path).write_text(json.dumps(jsonable(obj), indent=2, sort_keys=False, default=str),
                          encoding="utf-8")


# --------------------------------------------------------------------------------------
# univariate WoE / IV and multicollinearity
# --------------------------------------------------------------------------------------
class WOETransformer(BaseEstimator, TransformerMixin):
    """Supervised WoE encoding, quantile bins for numerics, categories as-is.
    Fitted inside the pipeline (so inside every CV fold), never on validation/test."""

    def __init__(self, n_bins: int = 8):
        self.n_bins = n_bins

    def _keys(self, col, values):
        if col in self.edges_:
            e = self.edges_[col]
            v = pd.to_numeric(values, errors="coerce").to_numpy(float)
            k = np.searchsorted(e[1:-1], v, side="right").astype(int)
            k[np.isnan(v)] = -1
            return pd.Series(k, index=values.index)
        return values.astype(str)

    def fit(self, frame, y):
        y = np.asarray(y, float)
        self.columns_ = list(frame.columns)
        self.maps_, self.edges_ = {}, {}
        for col in self.columns_:
            v = frame[col]
            if pd.api.types.is_numeric_dtype(v):
                vals = v.dropna()
                e = np.unique(np.quantile(vals, np.linspace(0, 1, self.n_bins + 1)))
                if len(e) >= 3:
                    self.edges_[col] = e
            keys = self._keys(col, v)
            t = pd.DataFrame({"k": keys.to_numpy(), "y": y}).groupby("k")["y"].agg(["sum", "count"])
            good = t["count"] - t["sum"]
            pb = (t["sum"] + 0.5) / (t["sum"] + 0.5).sum()
            pg = (good + 0.5) / (good + 0.5).sum()
            self.maps_[col] = np.log(pb / pg).to_dict()
        return self

    def transform(self, frame):
        out = {}
        for col in self.columns_:
            m = self.maps_[col]
            fb = float(np.mean(list(m.values())))
            out[col] = self._keys(col, frame[col]).map(m).fillna(fb).to_numpy(float)
        return pd.DataFrame(out, index=frame.index)

    def bin_label(self, col, key):
        if col not in self.edges_:
            return str(key)
        if key == -1:
            return "missing"
        e = self.edges_[col]
        lo = "-inf" if key == 0 else f"{e[key]:.4g}"
        hi = "+inf" if key >= len(e) - 2 else f"{e[key + 1]:.4g}"
        return f"[{lo}, {hi})"


def iv_table(x: pd.DataFrame, y, n_bins=10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Information value per feature + the bin-level WoE table (training data only)."""
    w = WOETransformer(n_bins=n_bins).fit(x, y)
    y = np.asarray(y, float)
    rows, bins = [], []
    for col in w.columns_:
        keys = w._keys(col, x[col]).to_numpy()
        t = pd.DataFrame({"k": keys, "y": y}).groupby("k")["y"].agg(["sum", "count"])
        good = t["count"] - t["sum"]
        pb = (t["sum"] + 0.5) / (t["sum"] + 0.5).sum()
        pg = (good + 0.5) / (good + 0.5).sum()
        woe = np.log(pb / pg)
        iv = float(((pb - pg) * woe).sum())
        rows.append({"feature": col, "iv": iv,
                     "strength": ("suspect (>0.5)" if iv > 0.5 else "strong" if iv > 0.3
                                  else "medium" if iv > 0.1 else "weak" if iv >= 0.02
                                  else "useless (<0.02)")})
        if col in w.edges_ or x[col].nunique() <= 12:
            for k in t.index:
                bins.append({"feature": col, "bin": w.bin_label(col, k), "n": int(t.loc[k, "count"]),
                             "bad_rate": float(t.loc[k, "sum"] / t.loc[k, "count"]),
                             "woe": float(woe.loc[k])})
    return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True), pd.DataFrame(bins)


def vif_table(x_num: pd.DataFrame, max_rows=100_000, drop_above=5.0):
    """VIF on median-imputed numerics, then iterative drop of the worst feature above 5."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    d = x_num.sample(min(len(x_num), max_rows), random_state=SEED)
    d = d.fillna(d.median())
    d = (d - d.mean()) / d.std().replace(0, 1)
    cols = list(d.columns)
    first = None
    dropped = []
    while True:
        arr = d[cols].to_numpy()
        v = pd.Series([variance_inflation_factor(arr, i) for i in range(len(cols))], index=cols)
        if first is None:
            first = v.copy()
        if v.max() <= drop_above or len(cols) <= 2:
            break
        worst = v.idxmax()
        dropped.append({"feature": worst, "vif_at_drop": float(v.max())})
        cols.remove(worst)
    out = pd.DataFrame({"feature": first.index, "vif_initial": first.values})
    out["dropped_in_iteration"] = out["feature"].isin([r["feature"] for r in dropped])
    return out.sort_values("vif_initial", ascending=False).reset_index(drop=True), dropped


# --------------------------------------------------------------------------------------
# PD models
# --------------------------------------------------------------------------------------
class SimplePrep(BaseEstimator, TransformerMixin):
    """Median-impute numerics, integer-code categoricals (unseen -> -1), plain dict params."""

    def __init__(self, numeric=None, categorical=None, min_category_freq: int = 50):
        self.numeric = numeric
        self.categorical = categorical
        self.min_category_freq = min_category_freq

    def fit(self, frame, y=None):
        self.medians_ = {c: float(pd.to_numeric(frame[c], errors="coerce").median())
                         for c in self.numeric}
        self.category_codes_ = {}
        for col in self.categorical:
            counts = frame[col].astype(str).value_counts()
            levels = sorted(counts[counts >= self.min_category_freq].index)
            self.category_codes_[col] = {lv: i for i, lv in enumerate(levels)}
        return self

    def transform(self, frame):
        n, k = len(frame), len(self.numeric)
        out = np.zeros((n, k + len(self.categorical)), dtype=np.float32)
        for j, col in enumerate(self.numeric):
            out[:, j] = pd.to_numeric(frame[col], errors="coerce").astype(float).fillna(
                self.medians_[col]).to_numpy(np.float32)
        for j, col in enumerate(self.categorical):
            out[:, k + j] = frame[col].astype(str).map(self.category_codes_[col]).fillna(-1).to_numpy(np.float32)
        return pd.DataFrame(out, columns=list(self.numeric) + list(self.categorical), index=frame.index)


def one_hot_prep():
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler(with_mean=False))]), NUMERIC),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                                   min_frequency=0.01, sparse_output=True))]),
         CATEGORICAL)], sparse_threshold=1.0)


def tree_pipeline(est, min_category_freq=50):
    return Pipeline([("prep", SimplePrep(NUMERIC, CATEGORICAL, min_category_freq)), ("clf", est)])


def woe_pipeline(clf):
    return Pipeline([("woe", WOETransformer()), ("clf", clf)])


def build_candidates(fast: bool = False) -> list:
    import lightgbm
    import xgboost
    import catboost
    it = 60 if fast else 400
    nj = 4
    C = []
    C.append({"name": "LogisticRegression L2 (one-hot, tuned C)", "family": "classical linear",
              "pipeline": Pipeline([("prep", one_hot_prep()),
                                    ("clf", LogisticRegression(max_iter=3000, random_state=SEED))]),
              "space": {"clf__C": [0.01, 0.1, 1.0, 10.0]}, "kind": "grid", "n_iter": None})
    C.append({"name": "WOE logistic scorecard (tuned C x bins)", "family": "credit scorecard",
              "pipeline": woe_pipeline(LogisticRegression(max_iter=3000, random_state=SEED)),
              "space": {"woe__n_bins": [6, 10], "clf__C": [0.01, 0.1, 1.0, 10.0]},
              "kind": "grid", "n_iter": None})
    C.append({"name": "WOE L1 logistic (lasso, tuned C)", "family": "classical linear",
              "pipeline": woe_pipeline(LogisticRegression(penalty="l1", solver="liblinear",
                                                          max_iter=3000, random_state=SEED)),
              "space": {"clf__C": [0.01, 0.05, 0.2, 1.0]}, "kind": "grid", "n_iter": None})
    C.append({"name": "RandomForest (tuned)", "family": "tree ensemble",
              "pipeline": tree_pipeline(RandomForestClassifier(random_state=SEED, n_jobs=nj)),
              "space": {"clf__n_estimators": [150, 250], "clf__max_depth": [8, 12, 18, None],
                        "clf__min_samples_leaf": [20, 50, 150], "clf__max_features": ["sqrt", 0.3],
                        "clf__class_weight": [None, "balanced_subsample"]},
              "kind": "random", "n_iter": 3 if fast else 6})
    C.append({"name": "LightGBM (tuned)", "family": "tree ensemble",
              "pipeline": tree_pipeline(lightgbm.LGBMClassifier(random_state=SEED, n_jobs=nj,
                                                                verbosity=-1, n_estimators=it)),
              "space": {"clf__n_estimators": [200, 400], "clf__learning_rate": [0.03, 0.06, 0.1],
                        "clf__num_leaves": [15, 31, 63], "clf__min_child_samples": [50, 200, 500],
                        "clf__subsample": [0.7, 0.9], "clf__colsample_bytree": [0.7, 0.9],
                        "clf__reg_alpha": [0.0, 0.5], "clf__reg_lambda": [0.0, 1.0, 5.0]},
              "kind": "random", "n_iter": 3 if fast else 15})
    C.append({"name": "XGBoost (tuned)", "family": "tree ensemble",
              "pipeline": tree_pipeline(xgboost.XGBClassifier(random_state=SEED, n_jobs=nj,
                                                              tree_method="hist", eval_metric="logloss",
                                                              n_estimators=it)),
              "space": {"clf__n_estimators": [200, 400], "clf__learning_rate": [0.03, 0.06, 0.1],
                        "clf__max_depth": [3, 5, 7], "clf__min_child_weight": [1, 5, 20],
                        "clf__subsample": [0.7, 0.9], "clf__colsample_bytree": [0.7, 0.9],
                        "clf__reg_alpha": [0.0, 0.5], "clf__reg_lambda": [1.0, 5.0],
                        "clf__gamma": [0.0, 1.0]},
              "kind": "random", "n_iter": 3 if fast else 12})
    C.append({"name": "CatBoost (tuned)", "family": "tree ensemble",
              "pipeline": tree_pipeline(catboost.CatBoostClassifier(random_seed=SEED, verbose=0,
                                                                    allow_writing_files=False,
                                                                    thread_count=nj, iterations=it)),
              "space": {"clf__iterations": [200, 400], "clf__learning_rate": [0.03, 0.06, 0.1],
                        "clf__depth": [4, 6, 8], "clf__l2_leaf_reg": [1.0, 3.0, 10.0],
                        "clf__random_strength": [0.5, 1.0], "clf__bagging_temperature": [0.0, 0.5]},
              "kind": "random", "n_iter": 3 if fast else 10})
    return C


def search_and_refit(cand, x_search, y_search, cv, x_train, y_train):
    """One CV scheme for every tunable family (roc_auc); the best config is refit on full train."""
    t0 = time.time()
    if cand["kind"] == "grid":
        s = GridSearchCV(cand["pipeline"], cand["space"], cv=cv, scoring="roc_auc", refit=False, n_jobs=1)
        n_cand = int(np.prod([len(v) for v in cand["space"].values()]))
    else:
        s = RandomizedSearchCV(cand["pipeline"], cand["space"], n_iter=cand["n_iter"], cv=cv,
                               scoring="roc_auc", refit=False, n_jobs=1, random_state=SEED)
        n_cand = int(cand["n_iter"])
    s.fit(x_search, y_search)
    best = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in s.best_params_.items()}
    t_search = time.time() - t0
    model = clone(cand["pipeline"]).set_params(**best)
    t1 = time.time()
    model.fit(x_train, y_train)
    info = {"search": "GridSearchCV" if cand["kind"] == "grid" else "RandomizedSearchCV",
            "n_candidates": n_cand, "n_folds": cv.get_n_splits(), "best_cv_roc_auc": round(float(s.best_score_), 5),
            "best_cv_gini": round(2 * float(s.best_score_) - 1, 5), "best_params": best,
            "search_seconds": round(t_search, 1), "refit_seconds": round(time.time() - t1, 1)}
    log(f"  {cand['name']}: CV auc={s.best_score_:.4f} ({n_cand}x{cv.get_n_splits()} fits, {t_search:.0f}s) {best}")
    return model, info


# --------------------------------------------------------------------------------------
# calibration (fitted on VALIDATION only; own implementation = no hidden refits)
# --------------------------------------------------------------------------------------
class Calibrator:
    """Maps a raw model probability to a calibrated PD.  method in {raw, isotonic, platt}."""

    def __init__(self, method: str):
        self.method = method

    def fit(self, p_raw, y):
        p_raw = np.asarray(p_raw, float)
        if self.method == "isotonic":
            self.m_ = IsotonicRegression(out_of_bounds="clip", y_min=1e-5, y_max=1 - 1e-5).fit(p_raw, y)
        elif self.method == "platt":
            lg = np.log(np.clip(p_raw, 1e-6, 1 - 1e-6) / (1 - np.clip(p_raw, 1e-6, 1 - 1e-6)))
            self.m_ = LogisticRegression(C=1e6, max_iter=1000).fit(lg.reshape(-1, 1), y)
        return self

    def predict(self, p_raw):
        p_raw = np.asarray(p_raw, float)
        if self.method == "raw":
            return p_raw
        if self.method == "isotonic":
            return self.m_.predict(p_raw)
        lg = np.log(np.clip(p_raw, 1e-6, 1 - 1e-6) / (1 - np.clip(p_raw, 1e-6, 1 - 1e-6)))
        return self.m_.predict_proba(lg.reshape(-1, 1))[:, 1]


def cv_ece_of_calibrator(method, p_raw, y, n_splits=5, seed=SEED):
    """Cross-validated ECE of a calibrator fitted on part of VALID and scored on the rest."""
    p_raw, y = np.asarray(p_raw, float), np.asarray(y, float)
    out = np.zeros(len(y))
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(p_raw, y):
        out[te] = Calibrator(method).fit(p_raw[tr], y[tr]).predict(p_raw[te])
    return ece_point(y, out)


def binom_row(k: int, n: int, p_pred: float) -> dict:
    """Exact two-sided binomial test of observed defaults vs predicted PD (+ Wilson CI)."""
    if n == 0:
        return {"binom_p": None, "ci95_actual": [None, None]}
    p = float(stats.binomtest(int(k), int(n), float(np.clip(p_pred, 1e-9, 1 - 1e-9))).pvalue)
    ci = stats.binomtest(int(k), int(n)).proportion_ci(confidence_level=0.95, method="wilson")
    return {"binom_p": p, "ci95_actual": [float(ci.low), float(ci.high)]}


# --------------------------------------------------------------------------------------
# term structure (Kaplan-Meier default hazard + cumulative incidence with prepayment)
# --------------------------------------------------------------------------------------
def hazard_by_month(df: pd.DataFrame, max_m: int = 84, min_at_risk: int = 200) -> dict:
    """Exposure-based monthly default hazard by month on book.

    Event time = month the loan reaches 90 DPD (months_to_last_payment + 3).  Loans leave the
    risk set when they are paid off (competing risk) or when the file ends (administrative
    censoring at their age).  h_k = defaults in month k / loans still at risk at the start
    of month k.  `cif` = cumulative incidence of default allowing for prepayment;
    `km` = default-only cumulative (as if nobody prepaid).  Months where fewer than
    `min_at_risk` loans remain at risk get hazard 0 (tail of the observed curve).
    """
    ev = df["is_event"].to_numpy(bool)
    t_def = df["t_default"].to_numpy(float)
    status = df["loan_status"].astype(str).to_numpy()
    fully = np.isin(status, ["Fully Paid", "Does not meet the credit policy. Status:Fully Paid"])
    t_pre = df["mtlp"].to_numpy(float)
    mob = df["mob"].to_numpy(float)
    t_cens = np.where(fully, t_pre, mob)
    t_cens = np.where(np.isnan(t_cens), 0, t_cens)
    t_def = np.where(np.isnan(t_def), 0, t_def)

    def cnt(x, mask):
        x = np.clip(np.floor(x[mask]).astype(int), 0, max_m)
        return np.bincount(x, minlength=max_m + 1)[: max_m + 1]

    d = cnt(t_def, ev)
    pre = cnt(t_cens, (~ev) & fully)
    cen = cnt(t_cens, (~ev) & (~fully))
    n = len(df)
    exits_before = np.concatenate([[0], np.cumsum(d + pre + cen)[:-1]])
    at_risk = n - exits_before
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where(at_risk > 0, d / at_risk, 0.0)
        hp = np.where(at_risk > 0, pre / at_risk, 0.0)
    h[0] = 0.0
    hp[0] = 0.0
    thin = at_risk < min_at_risk          # a handful of survivors makes the ratio meaningless
    h[thin], hp[thin] = 0.0, 0.0
    surv_all = np.cumprod(1 - h - hp)
    s_prev = np.concatenate([[1.0], surv_all[:-1]])
    cif = np.cumsum(s_prev * h)
    km = 1 - np.cumprod(1 - h)
    months = np.arange(max_m + 1)
    return {"months": months, "at_risk": at_risk, "defaults": d, "prepaid": pre, "hazard": h,
            "prepay_hazard": hp, "cif": cif, "km": km}


# --------------------------------------------------------------------------------------
# LGD / EAD
# --------------------------------------------------------------------------------------
def amortised_balance(P, r_annual_pct, A, m):
    """Level-payment balance after m payments (clipped at 0)."""
    r = np.asarray(r_annual_pct, float) / 1200.0
    P, A, m = np.asarray(P, float), np.asarray(A, float), np.asarray(m, float)
    g = (1 + r) ** m
    with np.errstate(divide="ignore", invalid="ignore"):
        b = np.where(r > 0, P * g - A * (g - 1) / r, P - A * m)
    return np.clip(b, 0, None)


def lgd_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Charged-off loans with realised EAD, realised recovery and LGD (workout, undiscounted).

    EAD  = funded principal - principal repaid before charge-off (what was owed at default).
    rr   = (recoveries - collection fees) / EAD, clipped to [0,1];   LGD = 1 - rr.
    The charge-off date is not published: est. date = issue + months to last payment + 4m
    (LendingClub charges off at 120+ days past due).
    """
    co = df[df["loan_status"].astype(str).isin(CHARGED_OFF_STATUS)].copy()
    co["ead_obs"] = co["funded_amnt"] - co["total_rec_prncp"]
    co = co[co["ead_obs"] > 1.0].copy()
    mt = co["mtlp"].fillna(0).astype(float)
    co["mtlp_f"] = mt
    co["co_date"] = co["issue_d"] + pd.to_timedelta((mt + CO_LAG_M) * 30.4375, unit="D")
    co["co_year"] = co["co_date"].dt.year
    net_rec = co["recoveries"].fillna(0) - co["collection_recovery_fee"].fillna(0)
    co["rec_rate"] = (net_rec / co["ead_obs"]).clip(0, 1)
    co["lgd_obs"] = 1.0 - co["rec_rate"]
    co["any_rec"] = (net_rec > 0).astype(int)
    # recovery-lag-discounted LGD (assumption: cash arrives ~12 months after charge-off, EIR discount)
    disc = (1 + co["int_rate"] / 100.0) ** (-12 / 12.0)
    co["lgd_disc"] = 1.0 - (co["rec_rate"] * disc)
    return co


LGD_NUM = ["loan_amnt", "funded_amnt", "term_months", "int_rate", "installment", "grade_n",
           "emp_length_years", "annual_inc", "dti", "fico_mid", "revol_util", "open_acc",
           "total_acc", "mort_acc", "mtlp_f", "ead_obs", "ead_ratio"]
LGD_CAT = ["purpose", "home_ownership", "verification_status", "addr_state"]


def lgd_features(co: pd.DataFrame) -> pd.DataFrame:
    f = co.copy()
    f["grade_n"] = f["grade"].astype(str).map({g: i for i, g in enumerate("ABCDEFG", 1)}).astype(float)
    f["ead_ratio"] = f["ead_obs"] / f["funded_amnt"].clip(lower=1)
    return f[LGD_NUM + LGD_CAT]


def lgd_metrics(y, p, base) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    mae = float(np.mean(np.abs(y - p)))
    rmse_b = float(np.sqrt(np.mean((y - base) ** 2)))
    r2 = 1 - np.sum((y - p) ** 2) / np.sum((y - base) ** 2)   # skill vs the training-mean benchmark
    rho = float(stats.spearmanr(y, p).correlation) if np.std(p) > 0 else 0.0
    return {"n": int(len(y)), "mean_actual": round(float(y.mean()), 5), "mean_pred": round(float(p.mean()), 5),
            "rmse": round(rmse, 5), "mae": round(mae, 5), "rmse_mean_baseline": round(rmse_b, 5),
            "skill_vs_mean": round(float(r2), 5), "spearman": round(rho, 5)}


def bootstrap_rmse_gain(y, p, base, n_iter=1000, seed=SEED) -> dict:
    """Paired bootstrap of RMSE(mean baseline) - RMSE(model); > 0 means the model beats the mean."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y, float), np.asarray(p, float)
    se_m, se_b = (y - p) ** 2, (y - base) ** 2
    n = len(y)
    d = np.empty(n_iter)
    for i in range(n_iter):
        s = rng.integers(0, n, n)
        d[i] = math.sqrt(se_b[s].mean()) - math.sqrt(se_m[s].mean())
    return {"gain": round(float(math.sqrt(se_b.mean()) - math.sqrt(se_m.mean())), 6),
            "ci95": [round(float(np.percentile(d, 2.5)), 6), round(float(np.percentile(d, 97.5)), 6)],
            "p_model_not_better": round(float((d <= 0).mean()), 4)}


# --------------------------------------------------------------------------------------
# staging + ECL engine (contractual life, EIR discounting, per-loan hazards)
# --------------------------------------------------------------------------------------
def project_ecl(term, mob, bal, int_rate, installment, pd12, stage, lgd, haz36, haz60,
                pool_p12, horizon12=12, pd_mult=1.0, max_h=60, pre36=None, pre60=None):
    """Marginal-PD x LGD x EAD x DF summed month by month.

    Per-loan hazard at seasoning s:  h_i(s) = 1 - (1 - h_term(s)) ** m_i , with the
    relative-risk multiplier  m_i = ln(1 - PD12_i) / ln(1 - pool_P12(term)).
    Stage 1 sums the next 12 months; Stage 2 sums the remaining contractual life;
    Stage 3 is credit-impaired: ECL = LGD x balance (PD = 1).  EAD_k = scheduled balance
    after k payments; DF_k = (1 + EIR)^(-k/12).  If `pre36/pre60` (empirical monthly
    prepayment hazards by month on book) are given, survival also falls with prepayment, so
    the expected life shortens; without them the projection assumes nobody prepays
    (conservative).  `pool_p12` should be the pooled 12-month default probability on the same
    basis (cumulative incidence when prepayment is modelled).
    """
    n = len(term)
    term, mob = np.asarray(term, int), np.asarray(mob, int)
    remaining = np.clip(term - mob, 1, max_h)
    pd12 = np.clip(np.asarray(pd12, float) * pd_mult, 1e-5, 0.95)
    pool = np.where(term == 36, pool_p12[36], pool_p12[60])
    m = np.log1p(-pd12) / np.log1p(-pool)
    r = np.asarray(int_rate, float) / 1200.0
    eir = np.asarray(int_rate, float) / 100.0
    bal0 = np.asarray(bal, float)
    A = np.asarray(installment, float)
    surv = np.ones(n)
    b = bal0.copy()
    ecl = np.zeros(n)
    pd_h = np.zeros(n)            # PD over the stage horizon (12m for S1, lifetime for S2)
    horizon = np.where(stage == 1, np.minimum(horizon12, remaining), remaining)
    kmax = int(horizon.max()) if n else 0
    lgd_arr = np.broadcast_to(np.asarray(lgd, float), (n,))
    for k in range(1, kmax + 1):
        s = np.clip(mob + k, 0, len(haz36) - 1)
        hz = np.where(term == 36, haz36[s], haz60[s])
        h = 1.0 - (1.0 - np.clip(hz, 0, 0.999)) ** m
        hp = (np.where(term == 36, pre36[s], pre60[s]) if pre36 is not None else 0.0)
        b = np.maximum(b * (1 + r) - A, 0.0)
        active = (k <= horizon) & (stage != 3)
        marg = surv * h
        df_k = (1 + eir) ** (-k / 12.0)
        ecl += np.where(active, marg * lgd_arr * b * df_k, 0.0)
        pd_h += np.where(active, marg, 0.0)
        surv = surv * (1 - h) * (1 - hp)
    s3 = stage == 3
    ecl = np.where(s3, lgd_arr * bal0, ecl)
    pd_h = np.where(s3, 1.0, pd_h)
    return ecl, pd_h


def sicr_flags(pd_orig, pd_now, status, ratio=2.0, abs_pp=0.01, backstop=True):
    """SICR = relative PD deterioration (>= ratio x AND >= abs_pp) OR 30-DPD backstop."""
    rel = (pd_now >= ratio * pd_orig) & ((pd_now - pd_orig) >= abs_pp)
    back = np.isin(status, ["Late (31-120 days)"]) if backstop else np.zeros(len(pd_now), bool)
    return rel | back, rel, back
