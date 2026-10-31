# FNSPID data notes (Model 1)

This file is overwritten by the Model 1 pipeline on the first Colab run (`src.model1.diagnostics`). Until then it is a placeholder.

Expected contents:

- Header columns found in each FNSPID file (the Nasdaq file should have extra `Unnamed: 0`, ignored via `usecols`)
- Rows dropped for missing date/title and exact `(date, title)` duplicates
- Year coverage per file before and after the keyword filter
- Data-driven start month **S** compared with InflaBERT's 2010-01 start

A `sample_rows` smoke test does **not** answer 1999–2023 coverage. Use a full run for that.

bye
