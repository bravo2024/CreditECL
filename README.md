# CreditECL: Retail Credit Lifecycle

Probability of default, term structure, LGD, EAD, IFRS 9 staging and expected credit loss (ECL) on a single public retail loan book, validated out of time on the 2017 vintages.

Built: October 2026

## Business problem

A retail lender needs one consistent chain from application score to provision: a calibrated 12-month PD, a lifetime PD curve, loss severity and exposure assumptions, IFRS 9 stage allocation for the live book, and the resulting ECL. The project builds that chain on one dataset so every figure traces to the same loans, and tests each step out of time.

## Data

- Source: LendingClub accepted loans 2007-2018Q4 (Kaggle `wordsforthewise/lending-club`), 2,260,668 loans in the file, snapshot March 2019.
- Label: 90+ days past due within 12 months of origination (observed through final status and last-payment month); 1,765,426 loans are labelled.
- Split by issue date: train up to 2015 (887,440 loans, bad rate 3.72%), validation 2016 (434,407, 4.70%), out-of-time test 2017 (443,579, 4.40%).
- Live book for staging and ECL: 907,904 active loans.
- The raw files and derived parquet tables are not committed (several are over 50 MB). Download the source from Kaggle and run the notebook to rebuild them.

## Approach

1. Base models: plain and tuned L2 logistic regression on one-hot features, and a WoE logistic scorecard (tuned C and bins) as the classical benchmark.
2. Advanced models: CatBoost, XGBoost, LightGBM and Random Forest, tuned with cross-validated grid or random search on the training window.
3. Model selection rule frozen on 2016 validation Gini before any 2017 outcome was used (`artifacts/champion_rule.json`): CatBoost was selected. A lead under 0.002 Gini over the scorecard would have gone to the scorecard.
4. Calibration: isotonic regression fitted on 2016 only, method chosen by cross-validated ECE inside 2016.
5. Term structure: exposure-based hazards with prepayment as a competing risk, giving cumulative PD curves for 36- and 60-month loans.
6. LGD: fractional logit, direct boosting and a two-stage model, each compared with the training mean out of time using a paired bootstrap.
7. EAD: amortisation formula checked against realised exposure at default.
8. Staging and ECL on the live book: significant increase in credit risk (PD at least 2.0x origination PD and at least 1.0 point higher, or a Late 31-120 backstop); Stage 3 is Default or Late 31-120 with at least three months since the last payment (a 90+ DPD proxy).

## Results

Out-of-time test, 2017 vintages (all values from `artifacts/`):

| Model | Valid Gini | OOT Gini | OOT KS | OOT AUC |
|---|---|---|---|---|
| L2 logistic, one-hot (base) | 0.318 | 0.295 | 0.215 | 0.648 |
| WoE logistic scorecard (base) | 0.314 | 0.313 | 0.226 | 0.657 |
| Random Forest | 0.338 | 0.336 | 0.240 | 0.668 |
| LightGBM | 0.348 | 0.342 | 0.245 | 0.671 |
| XGBoost | 0.353 | 0.347 | 0.250 | 0.674 |
| CatBoost (advanced, selected) | 0.357 | 0.353 | 0.256 | 0.676 |

- Gain from the advanced model: CatBoost improves OOT Gini over the WoE scorecard by +0.039 (paired bootstrap 95% CI 0.0355 to 0.0435; paired DeLong p = 2e-87). Discrimination is modest in absolute terms.
- Calibration: OOT ECE 0.0015 after isotonic calibration (0.0089 raw). 2017 backtest: predicted 4.55% against actual 4.40% (ratio 0.968). Seven of eight master-scale grades pass the binomial test out of time; grade 8 does not (ratio 0.924).
- Term structure: 36-month loans reach 3.87% cumulative PD at 12 months and 14.83% over the lifetime; 60-month loans reach 5.46% and 28.77%.
- LGD: mature mean 0.898. No model beats the training mean out of time (best candidate RMSE gain -0.0006, 95% CI -0.00084 to -0.00039), so the mean is used in the ECL. Models that looked better on validation lost to the mean on 2017.
- EAD: amortisation formula against realised exposure at default, correlation 0.993, MAE USD 278.
- Staging of the 907,904 live loans: Stage 1 94.3%, Stage 2 4.7%, Stage 3 1.0%.

## Business impact

| Stage | Loans | EAD (USD M) | ECL (USD M) | Coverage |
|---|---|---|---|---|
| 1 | 855,966 | 8,946 | 565 | 6.3% |
| 2 | 42,821 | 459 | 69 | 15.1% |
| 3 | 9,117 | 105 | 94 | 89.8% |
| Total | 907,904 | 9,510 | 729 | 7.66% |

- Stage 3 holds 1.0% of loans and 13.0% of ECL; Stage 1 holds 77.5% of ECL because of its size.
- ECL sensitivity (`artifacts/tables/ecl_sensitivity.csv`): from USD 608M (PD x0.8) to USD 1,020M (PD x1.5); treating all Late 31-120 loans as Stage 3 gives USD 831M. These are sensitivities, not forecasts.
- Illustrative cut-off: a PD cut-off chosen on 2016 approves 95.6% of 2017 applicants with a 4.05% bad rate (4.40% for all). Profit per approved loan is USD 612 against USD 606 when approving everyone, under assumed funding cost 4.5% and operating cost USD 120.

## Monitoring and validation

- Score PSI: train to 2016 0.0013, train to 2017 0.0096, train to 2018 vintages 0.0309 (all stable). The largest feature CSI is revol_util at 0.094 in 2017 and 0.213 in 2018, close to the 0.25 shift threshold.
- Quarterly Gini ranges from 0.32 to 0.39.
- Leakage checks: dropped post-origination fields are listed in `artifacts/tables/leakage_audit_dropped_fields.csv` and a single-feature AUC check is in `single_feature_auc_leak_check.csv`.
- The selection rule is saved before the out-of-time scores were read, and calibration is never fitted on 2017.
- `MODEL_CARD.md` lists intended use and review triggers.

## How to run

```bash
pip install -r requirements.txt
# local run: download accepted_2007_to_2018Q4.csv from Kaggle and point RCL_DATA at it
RCL_DATA=path/to/accepted_2007_to_2018Q4.csv jupyter nbconvert --to notebook --execute RetailCreditLifecycle.executed.ipynb --inplace
python -m pytest -q
streamlit run app.py
```

The saved models and tables are in `artifacts/`, so the tests, the scorer and the app run without the raw data. The full run took about 48 minutes on Kaggle.

## Repo structure

| Path | Purpose |
|---|---|
| `RetailCreditLifecycle.executed.ipynb` | Executed pipeline notebook with outputs |
| `rcl_lib.py` | Library: labels, WoE, models, calibration, hazards, LGD, staging, ECL |
| `scorer.py`, `app.py` | Applicant scorer and Streamlit app, reading saved artifacts only |
| `artifacts/` | `metrics.json`, tables, figures, fitted models, manifest |
| `MODEL_CARD.md` | Model documentation |
| `tests/` | pytest suite (artifacts, metric sanity, library, scorer) |

## Limitations

- LendingClub publishes no delinquency history. Default is inferred from final status and last-payment month, and Stage 3 uses a 90+ DPD proxy. Looser and stricter variants are in the ECL sensitivity table.
- Discrimination is modest (OOT Gini 0.35): public, accepted-loans-only data with weak features (top IV 0.091), not a bureau-rich bank portfolio.
- No LGD model beats the mean out of time, so LGD does not vary by loan in the ECL.
- ECL has no macro-scenario weighting. Accepted loans only, one lender, no reject inference.
- The overall 2017 binomial test rejects exact calibration (n = 443,579, p = 2e-06) even though the predicted-to-actual ratio is inside a 10% band.
- Cut-off economics rest on assumed funding and operating costs and are an illustration.
- This is a portfolio project, not a validated Basel or IFRS 9 model.
