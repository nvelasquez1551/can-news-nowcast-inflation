# can-news-nowcast-inflation

Nowcasting US monthly inflation with financial news: an NLP classifier builds a news index that feeds an ML model, compared with and without news.

Specs: [Model 1](docs/specs/model-1-news-sentiment.md) (news indexes, implemented below) · [Model 2](docs/specs/model-2-cpi-nowcast.md) (CPI nowcast, in progress).

Model 2 code: the [`nowcast`](nowcast/README.md) package (`pip install -e ".[dev]"`).

## Model 1: news sentiment indexes

Model 1 does not predict CPI. It streams FNSPID headlines, keeps price-related articles, scores them with off-the-shelf **FinBERT** (`ProsusAI/finbert`) and **InflaBERT** (`MAPAi/InflaBERT`), and writes a monthly table that Model 2 will consume:

`data/processed/news_index_monthly.csv`

Month *t* uses articles published from UTC day 1 through `cutoff_day` of month *t*. The default is **31**, the whole calendar month: the team nowcasts at the end of the month. Set `cutoff_day: 15` to reproduce InflaBERT's mid-month setup; later articles are then dropped, not rolled into month *t+1*. The sample starts at the first month **S** that begins 12 consecutive months with at least 30 in-window articles.

All paths and parameters live in [`config/model1.yaml`](config/model1.yaml). Useful knobs: `cutoff_day`, `keywords_broad` / `use_narrow_lexicon`, `exclude_title_patterns` (drops securities-offering headlines such as "XYZ prices $500M notes offering"), `prefer_lsa_summary`, `sample_rows`, `chunksize`, `batch_size`, `use_fp16`.

### Run on Google Colab (GPU)

The FNSPID files are ~5.7 GB and ~23 GB. Stream them in Colab; do not download them onto a laptop.

1. In Colab: **File > Open notebook > GitHub**, paste this repo URL, open `notebooks/run_model1_colab.ipynb` (tick **Include private repos** if needed).
2. Runtime > Change runtime type > **T4 GPU**.
3. If the repo is private, add a GitHub PAT as a Colab secret named `GITHUB_TOKEN`.
4. Set `SAMPLE_ROWS = 50000` for a smoke test, or `None` for the full run.
5. Run all cells. Outputs go to Google Drive under `MyDrive/can-news-nowcast-inflation/` (`data/` and `reports/`).

The notebook clones the repo, installs [`requirements.txt`](requirements.txt), mounts Drive, checks the GPU, runs the unit tests, then runs:

```bash
python -m src.model1.pipeline --config config/model1.yaml \
  --data-root /content/drive/MyDrive/can-news-nowcast-inflation/data \
  --reports-dir /content/drive/MyDrive/can-news-nowcast-inflation/reports
```

Add `--sample-rows 50000` for the smoke test. Chunks on Drive are skipped if already written, so a disconnected session can be restarted.

After scoring, `classify_headline("Gasoline prices soar")` (in `src/model1/predict.py`) is the Streamlit demo hook.

### Tests (no models, no dataset)

```bash
pytest tests/test_model1.py
```

Tests use tiny synthetic frames and mock `id2label` maps. They do not download FinBERT, InflaBERT, or FNSPID.

### First Colab run: what to check

- Header row and date parse on each file; extra `Unnamed: 0` on the Nasdaq file should be ignored.
- Logged `id2label` maps. Confirm InflaBERT `positive` means inflation-up on a few headlines.
- Disk use while streaming (the 23 GB file must not be fully cached).
- `torch.cuda.is_available()` after `pip install`.
- Year coverage per file before trusting the union (`sample_rows: 50000` does not answer this).
- Whether a 12-month streak at 30 articles exists, and which later months fall below 30.
- Fill `reports/sanity_sample.csv` by hand (`human_price_direction`); the pipeline leaves it blank.

## Model 2: evaluation helpers

Feature construction and model fitting for the CPI nowcast live in the team notebooks (see the [Model 2 spec](docs/specs/model-2-cpi-nowcast.md)). `src/model2/evaluate.py` holds the shared metrics:

- `rmse`, `mae`
- `diebold_mariano(errors_a, errors_b, horizon=1)`: Diebold-Mariano test with the Harvey-Leybourne-Newbold correction. A positive statistic means forecast B is more accurate.
- `compare_forecasts(frame, target="y", forecasts=[...], baseline="ar")`: one row per forecast with RMSE, MAE, and DM against the baseline, on the months all forecasts share.

```python
from src.model2.evaluate import compare_forecasts

compare_forecasts(predictions, target="y", forecasts=["xgb_macro", "xgb_macro+infla"], baseline="ar")
```

```bash
pytest tests/test_model2.py
```

## Demo (Streamlit)

```bash
streamlit run app/streamlit_app.py
```

- **Clasificador de titulares:** type a headline and see the InflaBERT and FinBERT label, score, and probabilities. The models download on first use.
- **Nowcast vs. dato oficial:** reads `data/processed/model2_predictions.csv` (`month`, `y`, one column per model) and shows the chart and the `compare_forecasts` table. Until Model 2 writes that file, the tab shows a notice.
