# Model 1 Specification: News Sentiment Index (NEWS)

Status: draft for development
Owner: project team
Consumers: Model 2 (CPI nowcast), Streamlit demo (headline classifier view)

## 1. Objective

Turn financial news headlines into a **monthly numeric index (NEWS)** that summarizes the tone of price-related news in each month, so Model 2 can test whether news improves a nowcast of US monthly CPI inflation.

Model 1 does **not** predict inflation. It only reads text and produces sentiment scores. Whether those scores help predict CPI is answered by Model 2 (same model with and without NEWS, compared with RMSE/MAE and a Diebold-Mariano test).

## 2. Key design decision

Use **FinBERT off the shelf** (`ProsusAI/finbert` on Hugging Face) with no fine-tuning and no manual labeling.

- FinBERT outputs market sentiment: `positive`, `negative`, `neutral`.
- This measures the economic *mood* of price-related news, not inflation direction. Example: "Gasoline prices soar" is likely `negative` sentiment but implies inflation *up*. This is a known limitation and must be stated in the final report.
- The sign of the relationship between NEWS and CPI is left for Model 2 to learn. Do not flip or remap labels by hand.
- Do **not** label articles using realized CPI. That leaks the target into the features.

Future work (out of scope now): fine-tune FinBERT on LLM-generated `up` / `down` / `neutral` inflation-direction labels.

## 3. Data source

- **FNSPID** (Financial News and Stock Price Integration Dataset), 15.7M news records, 1999 to 2023, about S&P 500 companies.
  - Hugging Face: https://huggingface.co/datasets/Zihan1004/FNSPID/
  - File needed: `Stock_news/nasdaq_exteral_data.csv` (the stock price files are not needed).
  - Repo: https://github.com/Zdong104/FNSPID_Financial_News_Dataset (no longer maintained since July 2025).
- No scraping of post-2023 news is required. The study period ends in 2023.
- **To verify on first load:** exact column names (date, headline/title, summary, ticker), date format and time zone, and duplicate handling. Record findings in `docs/data_notes.md`.

## 4. Pipeline

### Step 1. Load and reduce
- Read the CSV in chunks (`pd.read_csv(..., chunksize=...)`); the full file is too large to load at once.
- Keep only: article date, headline, summary (if present), ticker (if present).
- Drop rows with missing date or headline. Drop exact duplicate headlines on the same date.
- Save the reduced table as Parquet.

### Step 2. Filter to price-related news
- Keep articles whose headline (or summary) matches a price/inflation keyword list, case-insensitive, whole-word matching. Starting list (team to review and keep in `config`):
  `inflation, cpi, consumer prices, price index, prices, deflation, disinflation, gasoline, gas prices, oil, crude, energy prices, food prices, wages, wage growth, cost of living, federal reserve, fed, interest rate, rate hike, tariffs, supply chain, ppi`
- Log how many articles survive per year. Expected order of magnitude: InflaBERT (EPFL, arXiv 2410.20198) kept ~74k of a ~500k sample with a similar filter.
- Output: `data/interim/news_filtered.parquet`.

### Step 3. Score with FinBERT
- Model: `ProsusAI/finbert`, max sequence length 512 tokens (headlines fit easily), batched inference on GPU (Colab is fine), `torch.no_grad()`.
- Input text: the headline. (Optional variant: headline + summary, if summaries exist; keep headline-only as the default.)
- For each article, store softmax probabilities `p_pos`, `p_neg`, `p_neu`, the argmax `label`, and `score = p_pos - p_neg` (range -1 to 1).
- Inference must be resumable (write results per chunk; skip chunks already scored).
- Output: `data/processed/news_scored.parquet`.

### Step 4. Aggregate to monthly NEWS index
- Assign each article to a month using its publication date.
- **Leakage rule:** for the nowcast of month *t*, only include articles published on or before the nowcast cutoff for month *t*. Default cutoff: last calendar day of month *t* (assumption to confirm with the team; keep it a config parameter).
- Monthly features, one row per month:

| column | definition |
|---|---|
| `month` | first day of the month (YYYY-MM-01) |
| `news_score_mean` | mean of `score` across the month's filtered articles |
| `news_share_pos` | share of articles with `label == positive` |
| `news_share_neg` | share of articles with `label == negative` |
| `news_net_share` | `news_share_pos - news_share_neg` |
| `news_n_articles` | number of filtered articles (diagnostic; not a default model feature, because coverage grows over time) |

- Months with zero filtered articles: leave NaN and log them. Do not forward-fill silently.
- Use means and shares, not raw counts, as model features. FNSPID coverage is much thinner in early years.
- Output: `data/processed/news_index_monthly.csv` (and Parquet), covering every month in the study period.

## 5. Expected outputs

1. `data/interim/news_filtered.parquet`: filtered articles.
2. `data/processed/news_scored.parquet`: per-article FinBERT scores.
3. `data/processed/news_index_monthly.csv`: the monthly NEWS index (table above). **This is the interface to Model 2.**
4. `src/model1/predict.py` exposing `classify_headline(text: str) -> dict` returning `{label, p_pos, p_neg, p_neu, score}` for the Streamlit headline view.
5. `reports/model1_diagnostics.md` (or notebook) with: articles per year before/after filtering, label distribution per year, a time-series plot of `news_score_mean` with monthly CPI inflation overlaid, and the sanity check below.

## 6. Validation

- **Sanity check (about 20 minutes):** sample 50 scored headlines at random, have one team member mark whether each implies prices up, down, or neither, and report how often FinBERT's label lines up with that. This goes in the limitations section; it is not a training step.
- Unit tests: keyword filter (true/false examples), `score` in [-1, 1], probabilities sum to 1, cutoff rule excludes articles after the cutoff, aggregation on a tiny synthetic dataset gives known values.
- Reproducibility: fixed model name and revision, pinned package versions, all paths and parameters in one config file.

## 7. Suggested repo layout

```
config/model1.yaml          # paths, keyword list, cutoff rule, batch size
src/model1/load.py          # Step 1
src/model1/filter.py        # Step 2
src/model1/score.py         # Step 3
src/model1/aggregate.py     # Step 4
src/model1/predict.py       # single-headline inference for the demo
tests/test_model1.py
data/                       # gitignored
reports/
```

Raw and processed data stay out of git. Document how to download FNSPID in the README.

## 8. Acceptance criteria

- Running the pipeline end to end from the downloaded CSV produces `news_index_monthly.csv` with one row per month from 1999-01 to 2023-12 (or the first/last month with data), with the columns above.
- No article dated after a month's cutoff contributes to that month.
- `classify_headline("Gasoline prices soar")` returns a valid result in under a second on CPU after the model is loaded.
- Diagnostics report and sanity-check results are committed.
- Tests pass.

## 9. Open questions for the team

- Nowcast cutoff date: end of month (default) or a mid-month date?
- Headline only, or headline + summary?
- Final keyword list.
- Whether to also try the fine-tuned inflation-direction classifier later as a comparison.
