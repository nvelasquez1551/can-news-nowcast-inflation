# Model 1 Specification: News Sentiment Indexes (NEWS)

Status: draft for development
Owner: project team
Consumers: Model 2 (CPI nowcast), Streamlit demo (headline classifier view)

## 1. Objective

Turn financial news headlines into **monthly numeric indexes** that summarize the tone of price-related news in each month, so Model 2 can test whether news improves a nowcast of US monthly CPI inflation.

Model 1 does **not** predict inflation. It only reads text and produces scores. Whether those scores help predict CPI is answered by Model 2 (same model with and without the news indexes, compared with RMSE/MAE and Diebold-Mariano / Giacomini-White tests).

## 2. Key design decisions

Two pretrained classifiers are run **off the shelf** (no fine-tuning, no manual labeling) and each produces its own monthly index:

| Scorer | Hugging Face model | What it measures | Labels |
|---|---|---|---|
| FinBERT | `ProsusAI/finbert` | Market sentiment (good/bad news for markets) | positive, negative, neutral |
| InflaBERT | `MAPAi/InflaBERT` | Inflation direction (prices going up/down) | inflation up, down, neutral |

- **FinBERT** measures economic *mood*, not inflation direction. Example: "Gasoline prices soar" is likely `negative` sentiment but implies inflation *up*. The sign of its relationship to CPI is left for Model 2 to learn. Do not remap labels by hand.
- **InflaBERT** (EPFL, arXiv 2410.20198) is a Fin-distill-RoBERTa model fine-tuned on ~2,000 GPT-4-labeled FNSPID articles (augmented to 6,000) with labels inflation up / down / neutral. Reported out-of-sample weighted F1: 0.57. It directly targets the signal we want.
- Comparing the two indexes in Model 2 is itself a result: market sentiment vs. inflation-specific sentiment.
- Do **not** label articles using realized CPI. That leaks the target into the features.

### Caveats to report
- InflaBERT was trained on FNSPID **summaries** (<=100 words) from **2010 to 2019**. Scoring 2020 to 2023 is out of sample for InflaBERT. Some 2010 to 2019 articles may have been in its training set, which can make its in-sample (Model 2 training period) fit look slightly better than it would be in real time. Mention this in limitations.
- We score **headlines** by default; InflaBERT saw summaries in training, so there is a small input mismatch. Optional variant: score `Lsa_summary` when it is non-empty.

## 3. Data source

- **FNSPID** (Financial News and Stock Price Integration Dataset), 15.7M news records, 1999 to 2023, about S&P 500 companies.
  - Hugging Face: https://huggingface.co/datasets/Zihan1004/FNSPID/ (folder `Stock_news/`)
  - Repo: https://github.com/Zdong104/FNSPID_Financial_News_Dataset (no longer maintained since July 2025).
- Two news files are available (sizes from the Hugging Face API):

| file | size | notes |
|---|---|---|
| `Stock_news/nasdaq_exteral_data.csv` | ~23.2 GB | Nasdaq articles, includes full `Article` text and summaries; has an extra leading index column `Unnamed: 0` |
| `Stock_news/All_external.csv` | ~5.7 GB | Benzinga and other publishers; in the rows inspected, `Article` and summary columns are mostly empty (headline-only) |

- Columns (both files): `Date, Article_title, Stock_symbol, Url, Publisher, Author, Article, Lsa_summary, Luhn_summary, Textrank_summary, Lexrank_summary`.
- `Date` format: `YYYY-MM-DD HH:MM:SS UTC`. Parse as UTC.
- **To verify on first pass:** date coverage per year in each file (pick the file, or the union, that covers 1999 to 2023 best), duplicates across files, and missing values. Record findings in `docs/data_notes.md`.
- No need to download the full files: stream them (`datasets` with `streaming=True`, `pd.read_csv("hf://datasets/Zihan1004/FNSPID/Stock_news/<file>", chunksize=...)`, or DuckDB over `hf://`). No scraping of post-2023 news is required; the study period ends in 2023.

## 4. Nowcast timing (cutoff rule)

- **Cutoff: day 15 of month t.** The nowcast for month *t* is made on the 15th of month *t* using news published from day 1 through day 15 of month *t* (UTC, inclusive). This follows InflaBERT's setup, so results are comparable.
- Keep the cutoff day as a config parameter (`cutoff_day: 15`) so the team can test end-of-month as a robustness check.
- **Note for Model 2:** every other predictor must also be known by the 15th of month *t* (e.g., oil and gasoline averaged over days 1 to 15; CPI for month *t-1* is usually published around mid-month *t*, so check release dates and use *t-2* if it is not yet out).

## 5. Pipeline

### Step 1. Stream and reduce
- Stream the CSV in chunks; keep `Date`, `Article_title`, `Stock_symbol`, `Publisher`, `Lsa_summary`.
- Drop rows with missing date or title. Drop exact duplicate titles on the same date.

### Step 2. Filter to price-related news
- Keep articles whose title (or summary) matches a price/inflation keyword list, case-insensitive, whole-word matching. Starting list (team to review; store in config):
  `inflation, cpi, core cpi, consumer price index, consumer prices, deflation, disinflation, prices, gasoline, gasoline prices, gas prices, food prices, oil, crude, energy prices, wages, wage growth, cost of living, federal reserve, fed, interest rate, rate hike, tariffs, supply chain, ppi`
  (InflaBERT's own lexicon was narrower: Inflation, Gasoline prices, Food prices, Deflation, Consumer price index, CPI, Core CPI. Optionally also build the index with that narrower list as a robustness check.)
- Log article counts per year before and after filtering.
- Output: `data/interim/news_filtered.parquet`.

### Step 3. Score with both models
- Batched GPU inference (Colab is fine), `torch.no_grad()`, truncate to 512 tokens.
- Read label order from each model's `config.id2label`; do not hard-code it.
- Per article and per scorer, store:
  - FinBERT: `finbert_p_pos`, `finbert_p_neg`, `finbert_p_neu`, `finbert_label`, `finbert_score = p_pos - p_neg`
  - InflaBERT: `infla_p_up`, `infla_p_down`, `infla_p_neu`, `infla_label`, `infla_score = p_up - p_down` (equivalent to the paper's polarity score, the expected label with up=+1, neutral=0, down=-1)
- Scores range from -1 to 1. Inference must be resumable (write per chunk; skip chunks already scored).
- Output: `data/processed/news_scored.parquet`.

### Step 4. Aggregate to monthly indexes
- Month *t* uses articles dated day 1 through `cutoff_day` of month *t*.
- One row per month, columns:

| column | definition |
|---|---|
| `month` | first day of the month (YYYY-MM-01) |
| `finbert_score_mean` | mean `finbert_score` |
| `finbert_share_pos`, `finbert_share_neg` | share of articles with that label |
| `finbert_net_share` | `share_pos - share_neg` |
| `infla_score_mean` | mean `infla_score` |
| `infla_share_up`, `infla_share_down` | share of articles with that label |
| `infla_net_share` | `share_up - share_down` |
| `news_n_articles` | number of filtered articles in the window (used for the start-date rule; diagnostic only, not a default feature, because FNSPID coverage grows over time) |

- Months with zero articles: leave NaN and log them. Do not forward-fill silently.

### Step 5. Choose the sample start date (data-driven)
- Rule: `min_articles_per_month: 30` (config). The index **start month** is the first month *S* such that *S* and the 11 months after it (12 consecutive months) each have at least 30 filtered articles in their day 1 to `cutoff_day` window.
- Drop all months before *S* from `news_index_monthly.csv` (keep the full counts table for diagnostics).
- If any month after *S* falls below 30, flag it in the diagnostics report and keep it (do not move *S* later just for isolated gaps); the team decides how to handle it.
- Report *S* and how it compares with InflaBERT's 2010 start. Model 2 uses *S* as the start of its training sample (training through 2019, test 2020 to 2023).
- Output: `data/processed/news_index_monthly.csv` (and Parquet). **This is the interface to Model 2.**

## 6. Expected outputs

1. `data/interim/news_filtered.parquet`
2. `data/processed/news_scored.parquet`
3. `data/processed/news_index_monthly.csv` (table above)
4. `src/model1/predict.py` exposing `classify_headline(text: str) -> dict` returning both models' label, probabilities, and score, for the Streamlit headline view.
5. `reports/model1_diagnostics.md` (or notebook): articles per year before/after filtering, label distributions per year for each scorer, time-series plot of `finbert_score_mean` and `infla_score_mean` with monthly CPI inflation overlaid, correlation between the two indexes, and the sanity check below.

## 7. Validation

- **Sanity check (~20 minutes):** sample 50 scored headlines; one team member marks whether each implies prices up, down, or neither; report how often each model's label agrees. Goes in the limitations section; not a training step.
- Unit tests: keyword filter (true/false examples), scores in [-1, 1], probabilities sum to 1, label order read from config, cutoff rule excludes articles after day 15, start-date rule picks the correct month on a synthetic counts series, aggregation on a tiny synthetic dataset gives known values.
- Reproducibility: pinned model names and revisions, pinned package versions, all paths and parameters in one config file.

## 8. Suggested repo layout

```
config/model1.yaml          # file choice, keyword list, cutoff_day, min_articles_per_month, batch size, model names
src/model1/load.py          # Step 1
src/model1/filter.py        # Step 2
src/model1/score.py         # Step 3
src/model1/aggregate.py     # Step 4
src/model1/predict.py       # single-headline inference for the demo
tests/test_model1.py
data/                       # gitignored
reports/
```

## 9. Acceptance criteria

- The pipeline runs end to end from the Hugging Face files and produces `news_index_monthly.csv` with one row per month across the covered period and all columns above.
- No article dated after day `cutoff_day` of month *t* contributes to month *t*.
- The index starts at the data-driven month *S* (Step 5), and *S* is documented in the diagnostics report.
- `classify_headline("Gasoline prices soar")` returns valid outputs from both models in under a second on CPU after loading.
- Diagnostics report and sanity-check results are committed. Tests pass.

## 10. Open questions for the team

- Which FNSPID file (or both) gives the best 1999 to 2023 coverage?
- Headline only, or summary when available?
- Final keyword list (broad vs. InflaBERT's narrow lexicon).
- Keep day-15 cutoff as primary, end-of-month as robustness?
- How to treat isolated months below 30 articles after the start month *S*.

## Reference

Allard, Teiletche, Zinebi (EPFL). *Enhancing Inflation Nowcasting with LLM: Sentiment Analysis on News* (InflaBERT). arXiv:2410.20198. Code: https://github.com/paultltc/InflaBERT
