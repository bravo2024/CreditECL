"""Unit tests for rcl_lib (run: pytest tests/)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rcl_lib as L  # noqa: E402


def _toy():
    snap = L.SNAP_IDX
    rows = []
    def add(status, mob, mtlp, term=36):
        issue = pd.Timestamp(year=(snap - mob - 1) // 12, month=(snap - mob - 1) % 12 + 1, day=1)
        last = None if mtlp is None else issue + pd.DateOffset(months=int(mtlp))
        rows.append({"id": len(rows), "loan_status": status, "issue_d": issue, "last_pymnt_d": last, "term_months": term})
    add("Charged Off", 40, 6)          # defaulted inside the window  -> bad
    add("Charged Off", 40, 20)         # defaulted after the window   -> good for 12m
    add("Charged Off", 40, None)       # never paid                   -> bad (first-payment default)
    add("Fully Paid", 40, 30)          # good
    add("Current", 30, 30)             # surviving Current loan       -> good (legacy label dropped it)
    add("Current", 8, 8)               # too young                    -> undefined
    add("Late (16-30 days)", 30, 6)    # active but stopped paying    -> indeterminate
    df = pd.DataFrame(rows)
    df["last_pymnt_d"] = pd.to_datetime(df["last_pymnt_d"])
    df = L.derive_basics(df, snap)
    return L.add_labels(df)


def test_label_rules():
    d = _toy()
    assert d.bad_12m.iloc[0] == 1 and d.bad_12m.iloc[1] == 0 and d.bad_12m.iloc[2] == 1
    assert d.bad_12m.iloc[3] == 0 and d.bad_12m.iloc[4] == 0
    assert np.isnan(d.bad_12m.iloc[5]) and np.isnan(d.bad_12m.iloc[6])
    assert np.isnan(d.bad_12m_legacy.iloc[4])      # legacy label dropped the Current loan


def test_no_leakage_in_features():
    assert not (set(L.PD_FEATURES) & set(L.LEAKAGE_FIELDS))


def test_delong_matches_known_case():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 4000)
    a = y + rng.normal(0, 1.0, 4000)
    b = rng.normal(0, 1.0, 4000)
    r = L.delong_test(y, a, b)
    assert r["delta_auc"] > 0.15 and r["p_value"] < 1e-6
    same = L.delong_test(y, a, a)
    assert abs(same["delta_auc"]) < 1e-12


def test_amortised_balance_hits_zero_at_term():
    P, r, n = 10000.0, 12.0, 36
    A = P * (r / 1200) / (1 - (1 + r / 1200) ** -n)
    assert abs(float(L.amortised_balance(P, r, A, n))) < 1e-6
    assert abs(float(L.amortised_balance(P, r, A, 0)) - P) < 1e-9


def test_ecl_bounds_and_stage3():
    n = 5
    haz = np.full(90, 0.01)
    kw = dict(term=np.full(n, 36), mob=np.full(n, 10), bal=np.full(n, 1000.0), int_rate=np.full(n, 12.0),
              installment=np.full(n, 33.0), pd12=np.full(n, 0.05), lgd=0.9, haz36=haz, haz60=haz, pool_p12={36: 0.05, 60: 0.05})
    e1, p1 = L.project_ecl(stage=np.ones(n, int), **kw)
    e2, p2 = L.project_ecl(stage=np.full(n, 2), **kw)
    e3, p3 = L.project_ecl(stage=np.full(n, 3), **kw)
    assert (e2 > e1).all() and (p2 > p1).all()          # lifetime > 12 months
    assert np.allclose(e3, 900.0) and np.allclose(p3, 1.0)
    assert (e1 <= p1 * 0.9 * 1000.0 + 1e-9).all()       # ECL <= PD x LGD x opening EAD


def test_calibrator_monotone_and_ece():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 0.3, 20000)
    y = (rng.uniform(size=20000) < p * 1.4).astype(float)
    c = L.Calibrator("isotonic").fit(p, y)
    q = c.predict(np.sort(p))
    assert (np.diff(q) >= -1e-12).all()
    assert L.ece_point(y, c.predict(p)) < L.ece_point(y, p)
