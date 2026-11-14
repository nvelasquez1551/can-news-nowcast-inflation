# Model 2 Specification: CPI Nowcast (with and without NEWS)

Status: draft for team review
Owner: project team
Inputs: `data/processed/news_index_monthly.csv` (Model 1), FRED series
Consumers: final report, Streamlit demo (nowcast vs. official CPI view)

## 1. Objective

Nowcast US month-over-month CPI inflation for month *t* at the **end of month *t*** (team decision), before the official release in mid-month *t+1*, and test whether the Model 1 news indexes improve that nowcast.

The key result is a comparison: **the same model with and without NEWS features**, on the same months, evaluated out of sample. If NEWS does not help, that is reported as the result.

## 2. Target (Y)

- `y_t = 100 * (ln CPIAUCSL_t - ln CPIAUCSL_{t-1})`, month-over-month log change in percent.
- `CPIAUCSL` is seasonally adjusted, so no month dummies are needed.
- Use the latest vintage from FRED for the target. Real-time vintages (ALFRED) are a robustness check only (Section 9).

## 3. Information set at the end of month t (leakage rule)

Every feature for month *t* must be something that was public on the last day of month *t*. This is the central rule of the project.

| Series (FRED id) | Frequency | Released | Feature for month *t* |
|---|---|---|---|
| `CPIAUCSL` (CPI) | monthly | BLS, around day 10 to 15 of the next month | lags of `y` starting at `y_{t-1}` (released mid-month *t*) |
| `CPILFESL` (core CPI) | monthly | same BLS release as CPI | value for *t-1*, as log change |
| `DCOILWTICO` (WTI oil) | daily | next business day | mean over month *t*, as log change vs. the mean over *t-1* |
| `GASREGW` (gasoline) | weekly (Mondays) | same day | mean of the weekly observations dated in month *t*, as log change vs. *t-1* |
| `MICH` (UMich 1-year expectations) | monthly | final value near the end of the month | value for *t-1* (conservative: the final value for *t* can come out on the last days of *t*) |
| `news_index_monthly.csv` | monthly | Model 1, all articles of month *t* | Model 1 columns for *t* |

Tentative series, added only after the base model works:

| Series | Feature for month *t* |
|---|---|
| `PPIACO` (PPI) | value for *t-1* (released mid-month *t*) |
| `HOUST` (housing starts), `PAYEMS` (payrolls) | value for *t-1* |
| `T10YIE` (10-year breakeven) | mean over month *t* |
| `UNRATE` (unemployment) | value for *t-1* (released the first Friday of *t*) |
| `FEDFUNDS` (Fed funds rate) | value for *t-1* |

The team notebook `02 - Transform macro data` encodes this table as `AVAILABILITY_LAG` (1 for monthly releases, 0 for oil and gasoline). Confirm each lag against the FRED release calendar before the final run.

### 3.1 CPI release dates

At the end of month *t*, CPI for *t-1* has always been published in the 2020 to 2023 test period (BLS releases it around mid-month *t*). So `y_{t-1}` is a valid feature and no release calendar or FRED API key is needed for the base model. Known exceptions outside the test period (e.g. the October 2025 CPI, never published because of the government shutdown) must be checked if the sample is extended.

## 4. Sample

- Start: the Model 1 start month *S* (`data/processed/start_month.txt`), or the first month where all macro features exist, whichever is later.
- Training: *S* to 2019-12. Test: 2020-01 to 2023-12 (48 months).
- Check that FNSPID still has enough articles in the last test months; if 2023 coverage drops, report it and end the test period earlier.
- The macro-only notebook starts in 1999. For the with/without NEWS comparison every variant must start at *S*; a macro-only run from 1999 can be reported separately.
- All feature sets in Section 5 use **exactly the same months**. Months where any NEWS column is NaN (zero articles) are dropped from every variant and logged, so the comparison stays fair.

## 5. Feature sets

| Set | Features |
|---|---|
| `macro` | CPI lags `y_{t-1}` to `y_{t-3}`, core CPI lag, oil, gasoline, MICH |
| `macro+finbert` | `macro` + `finbert_score_mean`, `finbert_net_share` |
| `macro+infla` | `macro` + `infla_score_mean`, `infla_net_share` |
| `macro+news` | `macro` + both FinBERT and InflaBERT columns above |

`news_n_articles` is not a feature (FNSPID coverage grows over time); it is only a diagnostic.

## 6. Models

| Model | Notes |
|---|---|
| AR(p) | Lags of `y` only, `p` in 1 to 6 chosen by BIC on the training sample. Benchmark that ignores all other data. |
| Lasso | Standardized features; `alpha` by time-series cross-validation. Runs on every feature set. |
| Random Forest | Runs on every feature set. |
| XGBoost | Runs on every feature set. |

- Hyperparameters are tuned once, with `TimeSeriesSplit` on the training sample (through 2019), and then kept fixed for the test period.
- Fixed random seeds. Small grids: the training sample has only a few hundred months at most.

## 7. Evaluation

- **Expanding window:** for each test month *t*, refit each model on all months from *S* to *t-1* (their CPI is public by the end of *t*), then nowcast *t*.
- Metrics per model and feature set: RMSE and MAE over the test period, plus the same metrics for 2020 to 2021 and 2022 to 2023 separately (the 2021 to 2022 inflation surge dominates the errors).
- **Diebold-Mariano test** (squared loss, horizon 1, Harvey-Leybourne-Newbold small-sample correction): each `macro+*` variant against `macro` for the same model, and every model against AR(p).
- With 48 test months the tests have low power; report p-values, not only "significant / not significant".
- Compare with InflaBERT's reported RMSE (0.0409 to 0.0407, p = 0.19) only after checking that the target units match.

## 8. Expected outputs

1. `processed_data/model_table_macro.csv` (team notebook 02): one row per month, every feature in Section 3 plus `y`.
2. `data/processed/model2_predictions.csv`: `month`, `y`, and one column per model and feature set (e.g. `xgb_macro`, `xgb_macro+infla`). This is the input for the Streamlit nowcast view.
3. `reports/model2_results.md`: metrics table, Diebold-Mariano table, months dropped, lag-2 months, plot of `y` against the best models with and without NEWS, and Lasso coefficients / feature importances for the NEWS columns.

## 9. Robustness (after the base results)

- Mid-month nowcast, `cutoff_day: 15`, comparable with InflaBERT. Requires rerunning Model 1 with that setting and the CPI release-date rule (*t-2* when *t-1* is not yet out on the 15th).
- Real-time CPI vintages from ALFRED for the lag features.
- Narrow keyword list in Model 1 (`use_narrow_lexicon: true`).
- Tentative series from Section 3.

## 10. Repo layout

```
01 ... 07 (Colab)              # team notebooks, kept out of git: FRED download (01), macro transform (02), models (07)
src/model2/evaluate.py         # shared metrics: RMSE, MAE, Diebold-Mariano, compare_forecasts
tests/test_model2.py
app/streamlit_app.py           # demo; reads data/processed/model2_predictions.csv
```

Model 2 runs on CPU (laptop or Colab); no GPU is needed.

## 11. Tests

- Oil and gasoline averages for month *t* use only observations dated in month *t*.
- Monthly releases (CPI, core CPI, PPI, MICH) enter month *t* with their *t-1* value.
- MICH for month *t* is the value for *t-1*.
- Expanding window: the training set for test month *t* never contains `y_t` or any later month.
- All feature sets are evaluated on identical months.
- Diebold-Mariano on a synthetic pair of error series gives a known statistic.
- Tests use tiny synthetic frames and do not call the FRED API. The leakage check in notebook 02 (features built from data up to *T* equal those from the full sample) stays.

## 12. Acceptance criteria

- The team notebooks run end to end from the Model 1 index and FRED and write the three outputs in Section 8.
- No feature for month *t* uses information published after the end of month *t* (Section 3).
- The report states clearly, for each model, whether adding NEWS lowers RMSE/MAE and the Diebold-Mariano p-value.
- Tests pass.

## 13. Open questions for the team

- AR(p) or Lasso as the main benchmark in the presentation?
- Include the tentative series in the base model, or only as robustness?
- End the test period in 2023-12, or earlier if FNSPID coverage drops?
