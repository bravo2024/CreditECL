"""Applicant scorer: loads the saved artifacts only (no retraining). Used by app.py and tests."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import rcl_lib as L  # noqa: E402  (needed to unpickle the saved pipelines)

ART = HERE / "artifacts"

PLAIN = {
    "loan_amnt": "loan amount", "term_months": "loan term", "emp_length_years": "employment length",
    "annual_inc": "annual income", "dti": "debt-to-income ratio", "delinq_2yrs": "recent delinquencies",
    "inq_last_6mths": "credit enquiries in the last 6 months", "open_acc": "number of open accounts",
    "pub_rec": "public records", "revol_bal": "revolving balance", "revol_util": "revolving utilisation",
    "total_acc": "total accounts", "mort_acc": "mortgage accounts", "fico_mid": "FICO score",
    "purpose": "loan purpose", "addr_state": "state", "home_ownership": "home ownership",
    "verification_status": "income verification", "application_type": "application type",
}
_CACHE: dict = {}


def load_models():
    if "m" not in _CACHE:
        import joblib
        _CACHE["m"] = joblib.load(ART / "models" / "pd_models.joblib")
    return _CACHE["m"]


def grade_of(pd_cal, edges) -> int:
    return int(np.searchsorted(np.asarray(edges), pd_cal, side="right") + 1)


def _reason_codes(bundle, frame, top=3):
    pipe = bundle["woe_scorecard"]
    woe, lr = pipe.named_steps["woe"], pipe.named_steps["clf"]
    w = woe.transform(frame[woe.columns_]).iloc[0]
    factor = L.SCALING["pdo"] / math.log(2.0)
    rows = []
    for j, col in enumerate(woe.columns_):
        coef = float(lr.coef_[0][j])
        vals = np.array(list(woe.maps_[col].values()), float)
        pts = -factor * coef * float(w[col])
        best = float((-factor * coef * vals).max())
        rows.append((best - pts, col, pts))
    rows.sort(reverse=True)
    out = []
    for gap, col, pts in rows[:top]:
        val = frame[col].iloc[0]
        val = val.item() if hasattr(val, "item") else val
        out.append({"feature": col, "value": val, "points": round(pts, 1), "points_lost_vs_best_bin": round(gap, 1),
                    "text": f"Your {PLAIN.get(col, col)} ({val}) costs about {gap:.0f} scorecard points versus the best band."})
    return out


def score_applicant(applicant: dict) -> dict:
    """applicant: dict with the 19 PD_FEATURES. Returns calibrated PD, grade, scorecard points, reasons."""
    b = load_models()
    feats = b["features"]
    miss = [f for f in feats if f not in applicant]
    if miss:
        raise KeyError(f"missing features: {miss}")
    x = pd.DataFrame([{f: applicant[f] for f in feats}])
    p_raw = float(b["champion"].predict_proba(x)[:, 1][0])
    p_cal = float(b["calibrator"].predict([p_raw])[0])
    p_woe = float(b["woe_scorecard"].predict_proba(x)[:, 1][0])
    return {"pd_raw": p_raw, "pd_calibrated": p_cal, "grade": grade_of(p_cal, b["master_scale_edges"]),
            "n_grades": len(b["master_scale_edges"]) + 1, "score_points": float(L.score_from_pd(p_cal)),
            "pd_scorecard": p_woe, "champion": b["champion_name"], "reasons": _reason_codes(b, x)}


DEFAULT_APPLICANT = dict(
    loan_amnt=10000, term_months=36, emp_length_years=5, annual_inc=60000, dti=15.0, delinq_2yrs=0,
    inq_last_6mths=1, open_acc=10, pub_rec=0, revol_bal=8000, revol_util=40.0, total_acc=20, mort_acc=1,
    fico_mid=700, purpose="debt_consolidation", addr_state="CA", home_ownership="RENT",
    verification_status="Verified", application_type="Individual")
