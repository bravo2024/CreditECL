# Model card - LendingClub 12-month PD with LGD, EAD, staging and ECL (SR 11-7 / PRA SS1/23 style)

**Status:** portfolio / research model on public data. Not independently validated; not approved for any lending or provisioning decision.

**Purpose.** Rank-order and price 12-month default risk of newly originated unsecured consumer loans, then carry that PD through a term structure, LGD, EAD, IFRS 9 staging and a point-in-time ECL on the live book, all on one dataset.

**Data.** LendingClub accepted loans 2007-2018Q4, 2,260,668 loans, snapshot 2019-03 (read from the data). 19 application-time features; post-origination fields, LendingClub grade and rate are excluded (`tables/leakage_audit_dropped_fields.csv`); highest single-feature AUC 0.581. Duplicates removed before the split (0 found).

**Default definition and windows.** bad_12m=1: ended Charged Off/Default/Late(31-120) and last payment in month<=9 (90+ DPD within 12m); first-payment defaults included; 0: all other loans aged >=15 months incl. Current; undefined: age<15m or active loan that stopped paying inside the window. Performance window 12 months; loans younger than 15 months are unlabelled; 2,427 first-payment defaults count as bad. Labelled 1,765,426 loans, bad rate 4.13%. Split by issue date: train <=2015 (887,440), validation 2016 (434,407), out-of-time 2017 (443,579).

**Method.** Plain, L2 and WoE-L1 logistic; WoE points scorecard (PDO 20, 600 points at 20:1); random forest, XGBoost, LightGBM, CatBoost. Hyper-parameters tuned by 3-fold stratified CV (roc_auc) on training data. Champion rule frozen before scoring 2017 (`artifacts/champion_rule.json`): highest validation Gini; a lead under 0.002 over the scorecard goes to the scorecard. Champion: CatBoost (tuned). Calibration: isotonic, fitted on 2016 only, chosen by 5-fold CV ECE inside 2016.

**Performance (2017, out of time).** Gini 0.353, KS 0.256, AUC 0.676. WoE scorecard Gini 0.313; difference +0.0394 (DeLong p 2.1e-87, bootstrap 95% CI [0.03551, 0.04352]). Calibrated ECE 0.0015. Predicted 4.55% vs actual 4.40%. 7 of 8 grades pass the binomial test. Score PSI 0.0096. Quarterly Gini over 2016-17 vintages 0.324 to 0.390.

**Lifecycle components.** Term structure: monthly hazard on all loans, cumulative incidence with prepayment as a competing risk (3.87% to 14.83% for 36m loans). EAD: amortisation schedule, r 0.993, MAE USD 278. LGD: mean 0.898 on 102,428 mature charge-offs; the model adopted on validation lost to the mean out of time. Staging: SICR = PD now >= 2x origination PD and +1pp, or Late 31-120 backstop; Stage 3 = Default or Late 31-120 with >=3 months unpaid. ECL = 12m (Stage 1), lifetime (Stage 2), LGD x balance (Stage 3): USD 729M on USD 9.51B.

**Limitations.** Accepted loans only (no reject inference); one lender and period; modest discrimination; 90+ DPD is a last-payment proxy; the scorecard has 3 counter-intuitive signs (delinq_2yrs, pub_rec, application_type), documented not hidden; grade 8 over-predicts out of time; no macro scenarios; LGD is a mean; live-book PDs are partly in-sample for surviving loans; cut-off economics are assumed.

**Monitoring and triggers.** Score PSI quarterly (green <0.10, amber 0.10-0.25, red >0.25); CSI per feature (largest: revol_util, 0.213 vs 2018 vintages); predicted/actual ratio outside [0.9, 1.1] triggers recalibration (2017 ratio 0.968); Gini falling more than 0.05 below validation triggers redevelopment; a grade failing the binomial test twice in a row is reviewed.

**Reproducibility.** Seeds fixed; Kaggle run 48 min (sklearn 1.6.1, catboost 1.2.10). Generated from `artifacts/`; see `RetailCreditLifecycle.executed.ipynb`.
