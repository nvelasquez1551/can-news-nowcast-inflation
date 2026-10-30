# can-news-nowcast-inflation

Nowcasting US monthly inflation with financial news: an NLP classifier builds a news index that feeds an ML model, compared with and without news.

## Model 1: news sentiment indexes

Model 1 does not predict CPI. It streams FNSPID headlines, keeps price-related articles, scores them with off-the-shelf **FinBERT** (`ProsusAI/finbert`) and **InflaBERT** (`MAPAi/InflaBERT`), and writes a monthly table that Model 2 will consume:

`data/processed/news_index_monthly.csv`

Month *t* uses only articles published from UTC day 1 through `cutoff_day` (default **15**) of month *t*. Later articles in that month are dropped; they are not rolled into month *t+1*. The sample starts at the first month **S** that begins 12 consecutive months with at least 30 in-window articles.

All paths and parameters live in [`config/model1.yaml`](config/model1.yaml). Useful knobs: `cutoff_day`, `keywords_broad` / `use_narrow_lexicon`, `prefer_lsa_summary`, `sample_rows`, `chunksize`, `batch_size`.

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
