# Model 2 Specification: CPI Nowcast (with and without NEWS)

Status: draft for team review
Owner: project team
Inputs: `data/processed/news_index_monthly.csv` (Model 1), FRED series
Consumers: final report, Streamlit demo (nowcast vs. official CPI view)

## 1. Objective

Nowcast US month-over-month CPI inflation for month *t* on day `cutoff_day` (default 15) of month *t*, before the official release in mid-month *t+1*, and test whether the Model 1 news indexes improve that nowcast.

The key result is a comparison: **the same model with and without NEWS features**, on the same months, evaluated out of sample. If NEWS does not help, that is reported as the result.

## 2. Target (Y)

- `y_t = 100 * (ln CPIAUCSL_t - ln CPIAUCSL_{t-1})`, month-over-month log change in percent.
- `CPIAUCSL` is seasonally adjusted, so no month dummies are needed.
- Use the latest vintage from FRED for the target. Real-time vintages (ALFRED) are a robustness check only (Section 9).

## 3. Information set at the cutoff (leakage rule)

Every feature for month *t* must be something that was public on day `cutoff_day` of month *t*. This is the central rule of the project.

| Series (FRED id) | Frequency | Released | Feature for month *t* |
|---|---|---|---|
| `CPIAUCSL` (CPI) | monthly | BLS, around day 10 to 15 of the next month | lags of `y`: latest month released by the cutoff (Section 3.1) |
| `CPILFESL` (core CPI) | monthly | same BLS release as CPI | same lag rule as CPI, as log change |
| `DCOILWTICO` (WTI oil) | daily | next business day | mean of observations dated day 1 to `cutoff_day` of *t*, as log change vs. the same window of *t-1* |
| `GASREGW` (gasoline) | weekly (Mondays) | same day | mean of weekly observations dated day 1 to `cutoff_day` of *t*, as log change vs. the same window of *t-1* |
| `MICH` (UMich 1-year expectations) | monthly | final value at the end of the month | value for *t-1* (the value for *t* is not final until the end of *t*) |
| `news_index_monthly.csv` | monthly | Model 1, articles day 1 to `cutoff_day` | Model 1 columns for *t* |

Tentative series, added only after the base model works:

| Series | Feature for month *t* |
|---|---|
| `PPIACO` (PPI) | latest month released by the cutoff (same rule as CPI) |
| `T10YIE` (10-year breakeven) | mean of day 1 to `cutoff_day` of *t* |
| `UNRATE` (unemployment) | value for *t-1* (released the first Friday of *t*) |
| `FEDFUNDS` (Fed funds rate) | value for *t-1* |

### 3.1 CPI release dates

CPI for *t-1* is usually published between day 10 and 15 of month *t*, but not always before the cutoff.

- Get the historical BLS release dates for the CPI release from the FRED API (`fred/release/dates`, CPI release id 10). This needs a free FRED API key, read from the `FRED_API_KEY` environment variable or a Colab secret. Never commit the key.
- For each month *t*: if CPI for *t-1* was released on or before day `cutoff_day` of *t*, the latest known lag is `y_{t-1}`; otherwise it is `y_{t-2}`.
- Store this in a column `cpi_lag_used` (1 or 2) and build the lag features from the latest known month. Log how many months use lag 2.
- If no API key is available, fall back to always using `y_{t-2}` (conservative, never leaks) and say so in the report.

## 4. Sample

- Start: the Model 1 start month *S* (`data/processed/start_month.txt`), or the first month where all macro features exist, whichever is later.
- Training: *S* to 2019-12. Test: 2020-01 to 2023-12 (48 months).
- Check that FNSPID still has enough articles in the last test months; if 2023 coverage drops, report it and end the test period earlier.
- All feature sets in Section 5 use **exactly the same months**. Months where any NEWS column is NaN (zero articles) are dropped from every variant and logged, so the comparison stays fair.

## 5. Feature sets

| Set | Features |
|---|---|
| `macro` | CPI lags (1 to 3 known months), core CPI lag, oil, gasoline, MICH |
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

- **Expanding window:** for each test month *t*, refit each model on all months from *S* up to the last month whose CPI was released by the cutoff of *t*, then nowcast *t*.
- Metrics per model and feature set: RMSE and MAE over the test period, plus the same metrics for 2020 to 2021 and 2022 to 2023 separately (the 2021 to 2022 inflation surge dominates the errors).
- **Diebold-Mariano test** (squared loss, horizon 1, Harvey-Leybourne-Newbold small-sample correction): each `macro+*` variant against `macro` for the same model, and every model against AR(p).
- With 48 test months the tests have low power; report p-values, not only "significant / not significant".
- Compare with InflaBERT's reported RMSE (0.0409 to 0.0407, p = 0.19) only after checking that the target units match.

## 8. Expected outputs

1. `data/processed/macro_monthly.csv`: one row per month, every feature in Section 3 plus `y` and `cpi_lag_used`.
2. `data/processed/model2_predictions.csv`: `month`, `y`, and one column per model and feature set (e.g. `xgb_macro`, `xgb_macro+infla`). This is the input for the Streamlit nowcast view.
3. `reports/model2_results.md`: metrics table, Diebold-Mariano table, months dropped, lag-2 months, plot of `y` against the best models with and without NEWS, and Lasso coefficients / feature importances for the NEWS columns.

## 9. Robustness (after the base results)

- `cutoff_day` at the end of the month (requires rerunning Model 1 with the same setting).
- Real-time CPI vintages from ALFRED for the lag features.
- Narrow keyword list in Model 1 (`use_narrow_lexicon: true`).
- Tentative series from Section 3.

## 10. Suggested repo layout

```
config/model2.yaml          # series ids, cutoff_day, sample dates, feature sets, model grids, seeds
src/model2/fred.py          # download FRED series and CPI release dates, cache to data/raw/fred/
src/model2/features.py      # cutoff-window averages, lag rule, macro_monthly.csv
src/model2/models.py        # AR, Lasso, Random Forest, XGBoost
src/model2/evaluate.py      # expanding window, RMSE/MAE, Diebold-Mariano
src/model2/pipeline.py      # CLI: features -> fit -> evaluate -> report
tests/test_model2.py
```

Model 2 runs on CPU (laptop or Colab); no GPU is needed.

## 11. Tests

- Cutoff windows: oil and gasoline averages for month *t* use only observations dated day 1 to `cutoff_day`.
- Lag rule: on a synthetic release calendar, a month whose CPI comes out after the cutoff uses `y_{t-2}`.
- MICH for month *t* is the value for *t-1*.
- Expanding window: the training set for test month *t* never contains `y_t` or any later month.
- All feature sets are evaluated on identical months.
- Diebold-Mariano on a synthetic pair of error series gives a known statistic.
- Tests use tiny synthetic frames and do not call the FRED API.

## 12. Acceptance criteria

- `python -m src.model2.pipeline --config config/model2.yaml` runs end to end from the Model 1 index and FRED, and writes the three outputs in Section 8.
- No feature for month *t* uses information published after day `cutoff_day` of *t* (Section 3, enforced by tests).
- The report states clearly, for each model, whether adding NEWS lowers RMSE/MAE and the Diebold-Mariano p-value.
- Tests pass.

## 13. Open questions for the team

- AR(p) or Lasso as the main benchmark in the presentation?
- Get a FRED API key for the release dates (recommended), or accept the conservative `y_{t-2}` rule?
- Include the tentative series in the base model, or only as robustness?
- End the test period in 2023-12, or earlier if FNSPID coverage drops?
