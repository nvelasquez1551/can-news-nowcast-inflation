# Agent prompt: build Model 1 (code only)

Paste everything below the line into a Cursor agent chat opened on this repo.

---

Implement Model 1 for this data science bootcamp project, following the specification in `docs/specs/model-1-news-sentiment.md`. Read the whole spec first; it is the source of truth for scope, outputs, column names, config parameters, tests, and acceptance criteria.

Context: the project nowcasts US monthly CPI inflation. Model 1 turns FNSPID financial news headlines (Hugging Face dataset `Zihan1004/FNSPID`, folder `Stock_news/`) into monthly news indexes using two off-the-shelf classifiers, `ProsusAI/finbert` and `MAPAi/InflaBERT`, with a day-15 cutoff per month and a data-driven start date (first month beginning 12 consecutive months with >=30 filtered articles). Model 2 (the CPI nowcast) is out of scope; just produce the `news_index_monthly.csv` interface it will consume.

**Write code only. Do not install packages, create virtual environments, download models or data, or run the pipeline or tests on this machine.** Everything will be executed later on Google Colab. If you need to know something you would normally check by running code (exact column names, each model's `id2label` order), do not hard-code assumptions: read it at runtime (e.g. from `model.config.id2label`) and add an assertion or a clear log line so the Colab run surfaces it.

What to deliver:
- Pipeline code under `src/model1/` (stream and reduce, keyword filter, scoring with both models, monthly aggregation, start-date rule) driven by a single config file `config/model1.yaml`.
- `src/model1/predict.py` with `classify_headline(text: str) -> dict` for the Streamlit demo.
- `requirements.txt` with pinned versions (written, not installed).
- Unit tests in `tests/test_model1.py` that use tiny synthetic data and do not need the models or the dataset (mock the scorers).
- `notebooks/run_model1_colab.ipynb` as described in the spec's Colab section: clone, install, Google Drive outputs, GPU check, resumable steps, smoke-test mode, then run the tests and the pipeline.
- A short README section with setup and run instructions.

Constraints:
- Streaming only; never assume the full FNSPID files (~5.7 GB and ~23 GB) are on disk.
- No data files committed; keep `data/` gitignored.
- Leakage rule: no article after day `cutoff_day` of month t may contribute to month t.
- Keep code simple and readable for a bootcamp team; comments where logic is non-obvious.

Done when: all files above exist, the code is consistent with the spec, and you've listed anything in the spec you found ambiguous plus what the team should check on the first Colab run.
