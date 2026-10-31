"""RetailCreditLifecycle - Streamlit app. Loads saved artifacts only (no retraining).

    streamlit run app.py --server.port 8630
"""
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

HERE = Path(__file__).parent
ART = HERE / "artifacts"
sys.path.insert(0, str(HERE))
import scorer  # noqa: E402

st.set_page_config(page_title="RetailCreditLifecycle", layout="wide")


@st.cache_data
def metrics():
    return json.loads((ART / "metrics.json").read_text(encoding="utf-8"))


@st.cache_data
def table(name):
    return pd.read_csv(ART / "tables" / name)


def fig(name, caption):
    p = ART / "figures" / name
    if p.exists():
        st.image(str(p), caption=caption, use_container_width=True)


def pct(x, d=2):
    return f"{100 * x:.{d}f}%"


M = metrics()
C, LG, EC, ST, BT = M["cv"], M["lgd"], M["ecl"], M["staging"], M["backtest"]
st.title("RetailCreditLifecycle")
st.caption("PD, term structure, LGD, EAD, IFRS 9 staging and ECL on one public book (LendingClub accepted loans 2007-2018Q4). "
           "Portfolio project: not a validated Basel or IFRS 9 model.")

tabs = st.tabs(["Overview", "Data & Quality", "Feature Analysis", "Multicollinearity", "Logistic Ladder", "Challenger Models",
                "Model Comparison", "Scorecard & Decision", "Applicant Scorer", "Term Structure & ECL", "Monitoring", "Glossary"])

with tabs[0]:
    st.subheader("Problem")
    st.write("Predict which new loans will be 90+ days past due within 12 months, then carry that risk through lifetime PD, loss given default, "
             "exposure and an IFRS 9 expected credit loss on the live book. Everything comes from one dataset.")
    st.write(f"**Default definition.** {M['label']['definition']}")
    a = st.columns(5)
    a[0].metric("Champion", C["champion"])
    a[1].metric("OOT Gini", f"{C['champion_oot_gini']:.3f}")
    a[2].metric("OOT KS", f"{C['champion_oot_ks']:.3f}")
    a[3].metric("ECE (OOT)", f"{C['ece_oot']:.4f}")
    a[4].metric("PSI (OOT)", f"{C['psi_oot']:.4f}")
    b = st.columns(5)
    b[0].metric("Predicted vs actual 2017", f"{pct(C['backtest_predicted_oot'])} / {pct(C['backtest_actual_oot'])}")
    b[1].metric("Gini over WoE scorecard", f"{C['champion_minus_woe_gini']:+.3f}")
    b[2].metric("DeLong p", f"{C['champion_vs_woe_delong_p']:.1e}")
    b[3].metric("ECL (live book)", f"USD {EC['ecl_usd'] / 1e6:.0f}M")
    b[4].metric("Coverage", pct(EC["coverage"]))
    st.info(f"In plain English: on {C['n_labelled_12m']:,} labelled loans, trained to 2015, tuned on 2016 and tested once on 2017. "
            f"When the model says {pct(C['backtest_predicted_oot'])} will default, {pct(C['backtest_actual_oot'])} did. "
            f"No LGD model beat a simple average out of time, so the average ({C['lgd_mean_mature']:.3f}) is used.")
    st.dataframe(table("split_summary.csv"), hide_index=True)

with tabs[1]:
    st.subheader("Data and quality")
    d = M["data"]
    st.write(f"{d['rows_kept']:,} rows kept ({d['footer_rows_dropped']} footer rows dropped), duplicates removed: {d['duplicates']['duplicate_ids_removed']}, "
             f"snapshot month {d['snapshot_month']} (read from the data).")
    st.dataframe(table("data_quality_features.csv"), hide_index=True)
    st.write("Bad rate by split, corrected label vs the old (survivor-dropping) label")
    st.dataframe(table("label_new_vs_legacy_by_split.csv"), hide_index=True)
    fig("bad_rate_by_vintage_new_vs_legacy.png", "The old label dropped loans still Current, which inflated recent vintages' bad rate about 4x.")
    st.write("Leakage audit: fields dropped")
    st.dataframe(table("leakage_audit_dropped_fields.csv"), hide_index=True)
    st.dataframe(table("single_feature_auc_leak_check.csv"), hide_index=True)
    st.write("Gini and KS by issue quarter")
    st.dataframe(table("vintage_gini_ks.csv"), hide_index=True)

with tabs[2]:
    st.subheader("Weight of evidence and information value")
    iv = table("iv_table.csv").sort_values("iv", ascending=False)
    st.plotly_chart(px.bar(iv, x="iv", y="feature", orientation="h", color="strength", height=520), use_container_width=True)
    wb = table("woe_bins.csv")
    feats = sorted(wb.feature.unique())
    feat = st.selectbox("Feature", feats, index=feats.index(iv.feature.iloc[0]) if iv.feature.iloc[0] in feats else 0)
    w = wb[wb.feature == feat]
    f2 = go.Figure([go.Bar(x=w.bin.astype(str), y=w.bad_rate, name="bad rate")])
    f2.update_layout(yaxis_tickformat=".1%", height=380, xaxis_title="bin", yaxis_title="bad rate")
    st.plotly_chart(f2, use_container_width=True)
    st.dataframe(w, hide_index=True)
    fig("woe_bins_top_feature.png", "Higher WoE = riskier bin.")

with tabs[3]:
    st.subheader("Multicollinearity")
    mc = M["multicollinearity"]
    st.write(f"Pairs with |r| > 0.7: {len(mc['pairs_above_0.7'])}; max initial VIF {mc['max_vif_initial']:.2f}; features dropped: {len(mc['vif_dropped'])}.")
    st.dataframe(table("vif_table.csv"), hide_index=True)
    fig("correlation_heatmap.png", "No pair of inputs is highly correlated.")
    st.write(f"Coefficient sign check on the WoE logit: {M['scorecard']['n_wrong_sign_coefs']} counter-intuitive signs "
             f"({', '.join(M['scorecard']['wrong_sign_features'])}); converged: {M['scorecard']['logit_converged']}.")
    st.dataframe(table("statsmodels_logit_woe.csv"), hide_index=True)

with tabs[4]:
    st.subheader("Logistic ladder")
    r = pd.DataFrame(M["pd"]["roster"])
    st.dataframe(r[r.family.isin(["baseline", "classical linear", "credit scorecard"])], hide_index=True)
    st.write("Tuning (stratified CV, roc_auc)")
    st.dataframe(table("tuning_report.csv")[["model", "search", "n_candidates", "n_folds", "best_cv_roc_auc", "best_cv_gini", "best_params"]], hide_index=True)
    lg_ = table("statsmodels_logit_woe.csv")
    lg_ = lg_[lg_.term != "const"].sort_values("odds_ratio")
    f3 = px.scatter(lg_, x="odds_ratio", y="term", log_x=True, height=520, title="Odds ratios on WoE inputs")
    f3.add_vline(x=1, line_dash="dash")
    st.plotly_chart(f3, use_container_width=True)
    st.caption("Tuned C landed on 1.0 for the L2 logistic, so plain and L2 results are identical.")

with tabs[5]:
    st.subheader("Challenger models")
    r = pd.DataFrame(M["pd"]["roster"])
    st.dataframe(r[r.family == "tree ensemble"], hide_index=True)
    st.dataframe(table("tuning_report.csv")[["model", "search", "best_cv_roc_auc", "search_seconds", "best_params"]], hide_index=True)
    try:
        b_ = scorer.load_models()
        cb = b_["champion"].named_steps["clf"]
        imp = pd.DataFrame({"feature": list(b_["features"]), "importance": cb.get_feature_importance()}).sort_values("importance")
        st.plotly_chart(px.bar(imp, x="importance", y="feature", orientation="h", height=520, title="Champion feature importance (CatBoost)"),
                        use_container_width=True)
    except Exception as e:  # noqa: BLE001
        st.warning(f"Feature importance unavailable: {e}")
    st.caption("SHAP values are not stored in the artifacts; the Applicant Scorer tab gives scorecard-based reason codes instead.")

with tabs[6]:
    st.subheader("Model comparison (out of time 2017)")
    st.dataframe(pd.DataFrame(M["pd"]["roster"]), hide_index=True)
    rows = [{"comparison": k, "delta_gini": v["delong"]["delta_gini"], "DeLong p": v["delong"]["p_value"],
             "bootstrap 95% CI": str(v["bootstrap"]["ci95"])} for k, v in M["pd"]["significance"].items()]
    st.write("Paired tests on the same loans (raw scores)")
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    c1, c2 = st.columns(2)
    with c1:
        fig("roc_oot.png", "ROC curves, 2017 out-of-time.")
        st.dataframe(table("calibration_methods.csv"), hide_index=True)
    with c2:
        fig("reliability_oot.png", "Predicted vs observed default rate by decile after calibration.")
        st.dataframe(table("calibration_deciles_oot.csv"), hide_index=True)
    rule = json.loads((ART / "champion_rule.json").read_text(encoding="utf-8"))
    st.write(f"Champion rule (frozen before scoring 2017): {rule['rule']}")

with tabs[7]:
    st.subheader("Scorecard, master scale and cut-off")
    pts = table("points_table.csv")
    f4 = st.selectbox("Scorecard feature", sorted(pts.feature.unique()), key="ptsf")
    st.dataframe(pts[pts.feature == f4], hide_index=True)
    st.write("Master scale backtest (calibrated PD vs actual)")
    gb = table("master_scale_backtest.csv")
    st.dataframe(gb, hide_index=True)
    oo = gb[gb.window == "oot"]
    f5 = go.Figure([go.Bar(x=oo.grade, y=oo.predicted_pd, name="predicted"), go.Bar(x=oo.grade, y=oo.actual_default_rate, name="actual")])
    f5.update_layout(barmode="group", yaxis_tickformat=".1%", height=360, xaxis_title="grade")
    st.plotly_chart(f5, use_container_width=True)
    st.markdown("**Cut-off simulator** (assumed economics; an illustration, not a pricing model)")
    co = table("cutoff_curve_oot.csv")
    s1, s2, s3 = st.columns(3)
    lgd_in = s1.slider("LGD", 0.5, 1.0, float(round(EC["lgd_used"], 2)), 0.01)
    margin = s2.slider("Interest margin on an approved loan (share of amount)", 0.00, 0.30, 0.12, 0.01)
    appr = s3.slider("Approval rate", float(co.approval_rate.min()), float(co.approval_rate.max()), float(co.approval_rate.median()), 0.01)
    row = co.iloc[(co.approval_rate - appr).abs().argsort().iloc[0]]
    k = st.columns(4)
    k[0].metric("Approval rate", pct(row.approval_rate, 1))
    k[1].metric("Max PD approved", pct(row.max_pd))
    k[2].metric("Bad rate of approved", pct(row.bad_rate))
    k[3].metric("Profit per USD lent (simple)", f"{margin - row.bad_rate * lgd_in:.3f}")
    st.caption(f"Notebook assumptions {M['cutoff']['assumptions']}: USD {M['cutoff']['oot_profit_per_approved']:.0f} per approved loan at the validation-chosen cut-off.")
    fig("cutoff_profit.png", "Profit per approved loan vs approval rate (valid and OOT).")
    fig("grade_backtest_oot.png", "Grade-level predicted vs actual default rates, 2017.")

with tabs[8]:
    st.subheader("Applicant scorer")
    D = scorer.DEFAULT_APPLICANT
    a1, a2, a3 = st.columns(3)
    app = {}
    app["loan_amnt"] = a1.number_input("Loan amount", 500, 40000, D["loan_amnt"], 500)
    app["term_months"] = a1.selectbox("Term (months)", [36, 60])
    app["emp_length_years"] = a1.slider("Employment length (years)", 0, 10, D["emp_length_years"])
    app["annual_inc"] = a1.number_input("Annual income", 5000, 500000, D["annual_inc"], 1000)
    app["dti"] = a1.slider("Debt-to-income (%)", 0.0, 60.0, D["dti"])
    app["delinq_2yrs"] = a2.number_input("Delinquencies, last 2 years", 0, 20, D["delinq_2yrs"])
    app["inq_last_6mths"] = a2.number_input("Enquiries, last 6 months", 0, 10, D["inq_last_6mths"])
    app["open_acc"] = a2.number_input("Open accounts", 0, 50, D["open_acc"])
    app["pub_rec"] = a2.number_input("Public records", 0, 10, D["pub_rec"])
    app["revol_bal"] = a2.number_input("Revolving balance", 0, 500000, D["revol_bal"], 500)
    app["revol_util"] = a2.slider("Revolving utilisation (%)", 0.0, 120.0, D["revol_util"])
    app["total_acc"] = a3.number_input("Total accounts", 1, 100, D["total_acc"])
    app["mort_acc"] = a3.number_input("Mortgage accounts", 0, 20, D["mort_acc"])
    app["fico_mid"] = a3.slider("FICO score", 600, 850, D["fico_mid"])
    app["purpose"] = a3.selectbox("Purpose", ["debt_consolidation", "credit_card", "home_improvement", "other", "major_purchase",
                                              "small_business", "car", "medical", "moving", "vacation", "house", "wedding", "renewable_energy"])
    app["addr_state"] = a3.selectbox("State", ["CA", "NY", "TX", "FL", "IL", "NJ", "PA", "OH", "GA", "VA", "NC", "MI", "MD", "AZ", "MA", "CO", "WA", "MN", "IN", "TN"])
    app["home_ownership"] = a3.selectbox("Home ownership", ["RENT", "MORTGAGE", "OWN"])
    app["verification_status"] = a3.selectbox("Income verification", ["Verified", "Source Verified", "Not Verified"])
    app["application_type"] = "Individual"
    res = scorer.score_applicant(app)
    k = st.columns(4)
    k[0].metric("Calibrated 12m PD", pct(res["pd_calibrated"]))
    k[1].metric("Grade", f"{res['grade']} of {res['n_grades']} (1 = best)")
    k[2].metric("Score (600 @ 20:1, PDO 20)", f"{res['score_points']:.0f}")
    k[3].metric("WoE scorecard PD (uncalibrated)", pct(res["pd_scorecard"]))
    st.write("Top 3 reasons (scorecard points lost versus the best band):")
    for rs in res["reasons"]:
        st.write("- " + rs["text"])
    st.caption("Reason codes come from the WoE scorecard; the headline PD is from the boosted champion. Educational only.")

with tabs[9]:
    st.subheader("Term structure, EAD, LGD, staging and ECL")
    st.dataframe(table("pd_term_structure.csv"), hide_index=True)
    fig("pd_term_structure.png", "Cumulative PD by months on book, prepayment as a competing risk.")
    st.write(f"EAD: amortisation formula vs realised exposure at default, r = {C['ead_amortisation_corr']:.3f}, "
             f"MAE USD {C['ead_amortisation_mae_usd']:.0f}. CCF: {M['ead']['ccf']}.")
    fig("ead_validation.png", "Modelled vs observed exposure at default.")
    st.write(f"LGD: mean {C['lgd_mean_mature']:.3f}. Verdict: {LG['oot_verdict']}")
    st.dataframe(table("lgd_model_comparison.csv"), hide_index=True)
    fig("lgd_study.png", "LGD models vs the training mean.")
    st.write(f"Staging rules: SICR = {ST['sicr_rule']}; Stage 3 = {ST['stage3_rule']}.")
    st.dataframe(table("ecl_by_stage.csv"), hide_index=True)
    fig("stage_ecl.png", "Loans, exposure and ECL by stage.")
    st.write("ECL sensitivities")
    st.dataframe(table("ecl_sensitivity.csv"), hide_index=True)
    for n in EC["scope_notes"]:
        st.caption("- " + n)

with tabs[10]:
    st.subheader("Monitoring and backtest")
    st.dataframe(table("psi_score.csv"), hide_index=True)
    fig("psi_score.png", "Score PSI: green < 0.10, amber 0.10-0.25, red > 0.25.")
    st.dataframe(table("csi_features.csv"), hide_index=True)
    st.write("Backtest 2017 (overall)")
    st.dataframe(table("backtest_overall.csv"), hide_index=True)
    lo, hi = BT["recalibration_trigger_band"]
    st.write(f"Triggers: actual/predicted ratio outside [{lo}, {hi}] (2017: {C['backtest_ratio_oot']:.3f}); "
             "score PSI above 0.25; Gini more than 0.05 below validation; a grade failing the binomial test twice in a row.")

with tabs[11]:
    st.subheader("Glossary")
    for k_, v_ in {
        "PD": "Probability of default: the chance a loan goes 90+ days past due within a stated horizon.",
        "LGD": "Loss given default: the share of the exposure not recovered after default.",
        "EAD": "Exposure at default: what is still owed when the borrower defaults.",
        "ECL": "Expected credit loss, roughly PD x LGD x EAD; the provision under IFRS 9.",
        "Gini / AUC": "How well the score ranks bad above good borrowers. Gini = 2 x AUC - 1; 0 is random, 1 is perfect.",
        "KS": "Largest gap between the cumulative share of bad and good borrowers when sorted by score.",
        "WoE / IV": "Weight of evidence: log of bad-to-good share in a bin. Information value sums it into one predictor-strength number.",
        "PSI / CSI": "Population / characteristic stability index: how much the score or an input has drifted. Under 0.10 is stable.",
        "ECE": "Expected calibration error: average gap between predicted and observed default rate across bins.",
        "SICR": "Significant increase in credit risk: moves a loan from Stage 1 to Stage 2.",
        "Stage 1/2/3": "12-month ECL / lifetime ECL / credit-impaired (default).",
        "Out of time": "Tested on later vintages than the model was built on, as it would face in use.",
    }.items():
        st.markdown(f"**{k_}** - {v_}")
