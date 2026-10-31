"""Artifact, metric sanity, app and scorer tests (run: python -m pytest tests -q)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ART = ROOT / "artifacts"
M = json.loads((ART / "metrics.json").read_text(encoding="utf-8"))


def test_artifacts_exist():
    manifest = json.loads((ART / "MANIFEST.json").read_text())["files"]
    for k in manifest:
        assert (ART / k).exists() and (ART / k).stat().st_size > 0, k


def test_metrics_sane():
    c = M["cv"]
    assert 0.2 < c["champion_oot_gini"] < 0.7          # credit-score range, no leakage-level value
    assert c["champion_oot_auc"] < 0.95
    assert 0 < c["champion_oot_ks"] < 0.6
    assert c["champion_minus_woe_gini"] > 0
    assert 0.9 <= c["backtest_ratio_oot"] <= 1.1
    assert c["ece_oot"] < 0.02
    assert abs(sum(c["stage_shares_loans"].values()) - 1) < 1e-6
    assert 0 < c["ecl_coverage"] < 0.3
    assert c["pd_lifetime_36m_loans"] > c["pd_12m_36m_loans"]
    assert c["psi_oot"] < 0.25


def test_champion_matches_frozen_rule():
    rule = json.loads((ART / "champion_rule.json").read_text())
    assert rule["oot_outcomes_used_to_write_rule"] is False
    cand = rule["candidates_valid_gini"]
    best = max(cand, key=cand.get)
    assert rule["champion"] == M["cv"]["champion"]
    assert rule["champion"] == best or cand[best] - cand[rule["classical_benchmark"]] < 0.002


def test_calibration_fitted_on_validation():
    assert "validation" in M["calibration"]["selection_rule"]
    assert abs(M["backtest"]["overall"][0]["ratio_actual_over_pred"] - 1) < 1e-3   # calibration window is in-sample by design


def test_scorer_works():
    import scorer
    r = scorer.score_applicant(scorer.DEFAULT_APPLICANT)
    assert 0 < r["pd_calibrated"] < 1 and 1 <= r["grade"] <= r["n_grades"] and len(r["reasons"]) == 3
    bad = dict(scorer.DEFAULT_APPLICANT, fico_mid=620, dti=35.0, inq_last_6mths=5, revol_util=95.0)
    assert scorer.score_applicant(bad)["pd_calibrated"] > r["pd_calibrated"]


def test_app_imports():
    import ast
    ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
    import plotly.express  # noqa: F401
    import streamlit  # noqa: F401


def test_app_runs_headless():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not at.exception, at.exception
