# can-news-nowcast-inflation

Nowcasting US monthly inflation with financial news: an NLP classifier builds a news index that feeds an ML model, compared with and without news.

Guía en español (qué hace cada parte y qué se ha hecho): [docs/GUIA_DEL_PROYECTO.md](docs/GUIA_DEL_PROYECTO.md).

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

## Local app: API + frontend + model

Streamlit calls a FastAPI backend; the backend loads the latest model from `models/` and the data from `data/`. Everything runs on your machine.

```
app/streamlit_app.py  ──HTTP──>  nowcast/api/fast.py  ──>  models/ (registry) + data/raw, data/processed
     (frontend)                       (backend)
```

**First time** (from the repo root, in the project environment):

1. `make install` — dependencies plus the `nowcast` package. For the headline classifier you also need PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.
2. Copy `.env.example` to `.env` and set `FRED_API_KEY` (free at fred.stlouisfed.org > My Account > API Keys).
3. `make data` — downloads FRED into `data/raw/macro_monthly_raw.csv`.
4. `make train` — trains the linear model and XGBoost and saves them in `models/`.
5. `make train_news` — same models with the Model 1 news index (`linear_news`, `xgb_news`). Needs `data/processed/news_index_union_rescaled.csv`. For the headlines, `make headlines` builds `data/processed/news_scored.parquet` from `news_scored_fnspid.parquet` and `news_scored_alpaca.parquet` (same source per month as the index). The API only serves `linear` and `linear_news`; the final saved models are trained on all months (2007-01..2026-10).

**Every time**, in two terminals:

```bash
make run_api        # http://localhost:8000 (docs at /docs)
make run_frontend   # http://localhost:8501
```

What the frontend shows:

- **Nowcast del IPC:** pick a month. Line chart of official CPI vs. nowcast, and a bar chart that splits the change in the nowcast versus the previous month by driver (past inflation, oil, gasoline, expectations, seasonality, news...), plus a one-line summary of why it moved. Contributions add up exactly to the nowcast: coefficient × scaled value for the linear model, XGBoost's own SHAP values for XGBoost.
- **Titulares del mes:** when `data/processed/news_scored.parquet` (Model 1 output) is present, the headlines of the month with the most inflationary and disinflationary InflaBERT scores.
- **Busca un titular:** type a few words (whole words) and pick one of the matching headlines from the Model 1 data to see its InflaBERT label and source.
- **Prueba un titular:** type any new headline to get its raw InflaBERT label and score (`POST /classify`; needs PyTorch and the model weights).

- **¿Ayudan las noticias?:** out-of-sample RMSE of the linear model with and without news, both refit live on 2007-01..2020-12 and scored on 2021-01..2026-10 (68 months; 2025-10 and 2025-11 have no official CPI), with a Diebold-Mariano p-value.

When the news model exists, the app uses `linear_news` by default and the bars include a **Noticias** driver; the macro-only `linear` can be picked under *Ajustes*. If the news model or the news index for a month is missing, the API falls back to `linear` and says so (`fallback`).

API endpoints: `GET /`, `/models`, `/months`, `/history`, `/nowcast?month=YYYY-MM&model=linear|linear_news`, `/comparison`, `/headlines?month=YYYY-MM`, `/headlines/search?q=...&month=YYYY-MM`, `/headlines/keywords`, `POST /classify`. The frontend reads the API address from the `API_URL` environment variable or, if it is not set, from the Streamlit secret `API_URL` (`make run_frontend` sets it to `http://localhost:8000`). That address is what changes when the API moves to the cloud.

### Frontend against a remote API (Streamlit Cloud)

The app always needs an API: it reads `API_URL` from the environment or from Streamlit secrets, and if neither is set it shows an error and stops. There is no embedded mode (the old one that trained on `demo_data/` was removed). With `API_URL` set, the app only imports `streamlit`, `altair`, `pandas` and `requests`; [`app/requirements.txt`](app/requirements.txt) pins those plus `numpy` (no PyTorch, no scikit-learn).

On Streamlit Community Cloud: repository `nvelasquez1551/can-news-nowcast-inflation`, branch `main`, main file `app/streamlit_app.py`, and in the app's secrets:

```toml
API_URL = "https://nowcast-api-551929414015.us-central1.run.app"
```

Locally, the same secret can go in `.streamlit/secrets.toml` (ignored by git). See «Despliegue en la nube» below.

## Despliegue en la nube

### Arquitectura

```
Streamlit Cloud (app/streamlit_app.py)  ──HTTP, API_URL──>  Google Cloud Run (nowcast-api)
                                                               imagen en Artifact Registry
```

- **API:** un contenedor en **Google Cloud Run**. La imagen se construye con [`Dockerfile`](Dockerfile) (API ligera, sin clasificador) o con [`Dockerfile.classifier`](Dockerfile.classifier) (la misma API más PyTorch CPU, transformers y los pesos de InflaBERT en la revisión fijada, para `POST /classify`). La imagen se guarda en **Artifact Registry**.
- **App:** **Streamlit Cloud** ejecuta `app/streamlit_app.py`, que llama a la API mediante `API_URL` (secret de Streamlit).

### URLs de la demo pública

Son **temporales** y pueden dejar de funcionar:

- API: https://nowcast-api-551929414015.us-central1.run.app
- App: https://can-news-nowcast-inflation-mizduqnxjvzpp8cfet5nyb.streamlit.app/

### Qué NO está en la nube

- **Titulares:** la imagen corre con `HEADLINES_ENABLED=0`, porque la licencia para redistribuir el texto de los titulares está pendiente (ver «Datos y licencias»). `/headlines*` responden `enabled: false` y la app muestra un aviso.
- La imagen **no lleva** archivos `.parquet`, datos crudos (solo `data/raw/macro_monthly_raw.csv` y `data/processed/news_index_union_rescaled.csv`), `.env` ni modelos XGBoost. El build falla si encuentra un `.parquet`, un `xgb*` o un `.env` dentro de `/app`.
- El **clasificador sí está activo** (`CLASSIFY_ENABLED=1`, imagen de `Dockerfile.classifier`). Dentro de esa imagen FinBERT está apagado: `POST /classify` solo devuelve InflaBERT, que es lo que muestra la app, y no se descarga nada en ejecución.

### Comandos usados (en orden)

Con `GCP_PROJECT` y `GCP_REGION` definidas en la terminal:

```bash
gcloud auth configure-docker $GCP_REGION-docker.pkg.dev
gcloud artifacts repositories create nowcast --repository-format=docker \
  --location=$GCP_REGION --description="Nowcast API images"

docker build -f Dockerfile.classifier --platform linux/amd64 \
  -t $GCP_REGION-docker.pkg.dev/$GCP_PROJECT/nowcast/nowcast-api:prod-clf .
docker push $GCP_REGION-docker.pkg.dev/$GCP_PROJECT/nowcast/nowcast-api:prod-clf

gcloud run deploy nowcast-api \
  --image $GCP_REGION-docker.pkg.dev/$GCP_PROJECT/nowcast/nowcast-api:prod-clf \
  --region $GCP_REGION --memory 2Gi --cpu 1 --cpu-boost --timeout 300 \
  --max-instances 1 --allow-unauthenticated
```

La API ligera se desplegó antes igual, con `docker build --platform linux/amd64 -t .../nowcast-api:prod .` (sin `-f`) y `--memory 1Gi`.

### Cómo actualizar después de reentrenar

Los modelos van **dentro de la imagen**. Después de reentrenar (ver «Cómo se actualiza el modelo cada mes»):

1. Cambiar los nombres de los archivos de `models/` en `Dockerfile` y `.dockerignore` (o en `Dockerfile.classifier` y `Dockerfile.classifier.dockerignore`).
2. Volver a construir la imagen (`docker build ...`), subirla (`docker push ...`) y volver a desplegar (`gcloud run deploy ...`), con los mismos comandos de arriba.

El reentrenamiento en la nube **no está implementado** todavía.

### Límites conocidos

- **Arranque en frío:** tras un rato sin tráfico, Cloud Run apaga la instancia y la primera petición tarda más. La primera llamada a `/classify` además carga InflaBERT.
- **Streamlit Cloud gratuito:** las apps se duermen tras un tiempo sin visitas y hay que despertarlas.
- **Una sola instancia** como máximo (`--max-instances 1`).

### Cómo borrar todo

```bash
gcloud run services delete nowcast-api --region $GCP_REGION
gcloud artifacts repositories delete nowcast --location=$GCP_REGION   # las imágenes guardadas
```

O borrar el proyecto de GCP completo (`gcloud projects delete $GCP_PROJECT`). La app de Streamlit Cloud se borra desde su panel.

## Qué elige el usuario y qué recibe

El usuario elige **un mes**. La app devuelve la estimación (nowcast) de la inflación mensual del IPC de EE. UU. de ese mes, en %; el dato oficial, cuando existe; los factores que movieron la estimación frente al mes anterior; la comparación del modelo lineal con y sin noticias (fuera de muestra); los titulares del mes que más empujaron el índice de noticias al alza y a la baja, con un buscador; y una caja «Prueba un titular» que clasifica con InflaBERT un titular escrito por el usuario.

## Cómo se actualiza el modelo cada mes

Hoy el ciclo es **manual**: no hay ningún job programado. Automatizarlo (un job mensual) y monitorear el error del nowcast frente al IPC oficial son **trabajo futuro**. Pasos, en orden, cuando el BLS publica un mes nuevo de IPC (ejemplo: 2026-11):

1. **Descargar FRED.** Manual: cambiar `FRED_END` en [`nowcast/params.py`](nowcast/params.py) a la fecha de descarga. Luego `make data` (escribe `data/raw/macro_monthly_raw.csv`; necesita `FRED_API_KEY` en `.env`).
2. **Descargar el mes nuevo de Alpaca.** `python -m src.news_index.download_alpaca --only 2026-11` (necesita `ALPACA_API_KEY_ID` y `ALPACA_API_SECRET_KEY` en `.env`; guarda `data/raw/alpaca/2026-11.parquet`, ya prefiltrado con `keywords_broad`).
3. **Filtrar.** Manual: subir el fin de `ALPACA_RANGE` en [`src/news_index/final_filter.py`](src/news_index/final_filter.py) al mes nuevo. Luego `python -m src.news_index.final_filter` (escribe `data/interim/final_filter_alpaca.parquet`).
4. **Puntuar con InflaBERT.** `python -m src.news_index.score_final --source alpaca` (mejor con GPU, p. ej. en Colab). El script no es incremental: si la entrada creció, aborta por el `manifest.json` y hay que borrar `data/processed/news_scored_alpaca_chunks/` antes, así que vuelve a puntuar todo Alpaca.
5. **Reconstruir el índice mensual.** Manual: agregar al final de `data/processed/news_index_union_rescaled.csv` una fila por mes nuevo con la media de `infla_score` de ese mes (`infla_score_mean`), el número de titulares (`news_n_articles`), `source = alpaca` y `below_min_flag = news_n_articles < 30`. **No recalcular el reescalado**: solo los meses FNSPID (2007-2014) están reescalados, con la media y el desvío de la ventana de solape 2015-2016, que no cambian; las filas de Alpaca son la media sin reescalar. Re-ejecutar el notebook [`notebooks/model1/01_news_index.ipynb`](notebooks/model1/01_news_index.ipynb) no sirve tal cual: tiene el rango de Alpaca fijo hasta 2026-10. Después, `make headlines` reconstruye `data/processed/news_scored.parquet` para los titulares (comprueba que los conteos por mes coinciden con el índice).
6. **Reentrenar y guardar los modelos.** `make train` y `make train_news` ajustan en 2007-01..2020-12 (CV) y guardan en `models/`. Los modelos finales que sirve la API se entrenaron con **todos** los meses (2007-01..2026-10, sin 2025-10 y 2025-11): ese ajuste no tiene comando en el repo todavía (manual: ajustar `build_model("linear", cols)` sobre toda la tabla y guardarlo con `save_model` de `nowcast/ml_logic/registry.py`, como `linear` y `linear_news`). `TEST_END` en `params.py` define la evaluación y no debe moverse para reentrenar.
7. **Revisar la API y Streamlit.** `make run_api` y `make run_frontend`; comprobar que el mes nuevo aparece en `/months`, que `/nowcast?month=2026-11` responde sin `fallback` y que `/headlines?month=2026-11` trae titulares. `make test` antes de dar el mes por cerrado.

## Resultados

Test 2021-01..2026-10 (68 meses; sin 2025-10 y 2025-11), evaluado **una sola vez** con la configuración fijada por CV. RMSE en puntos de inflación mensual (%):

| Modelo | Expanding refit | Fijo (entrenado hasta 2020-12) |
|---|---|---|
| Lineal, sin noticias (`linear`) | 0.1607 | 0.1640 |
| Lineal, con noticias (`linear_news`) | 0.1607 | 0.1648 |
| XGBoost, sin noticias (`xgb`) | 0.1782 | 0.2286 |
| XGBoost, con noticias (`xgb_news`) | 0.1862 | 0.2444 |

Diebold-Mariano (pérdida cuadrática, h=1, corrección HLN), con noticias frente a sin noticias: lineal p = 0.985 (expanding) y p = 0.367 (fijo); XGBoost p = 0.424 (expanding) y p = 0.020 (fijo, con noticias **peor**).

El índice de noticias no mejoró el error de forma medible: en la lineal el RMSE es prácticamente igual con y sin noticias, y en XGBoost es algo peor con noticias.

## Limitaciones conocidas

- **Dos ventanas de entrenamiento.** La evaluación usa train 2007-01..2020-12; el modelo guardado que sirve la API se entrenó con 2007-01..2026-10, así que todos los meses de la app hasta 2026-10 son *in-sample* para ese modelo (la app lo marca). La comparación con y sin noticias de la app se re-ajusta en vivo con train hasta 2020-12.
- **Test ya visto en parte.** Los meses 2021-2023 del test se vieron en versiones anteriores (v0 y v1) del proyecto; el test completo se evaluó una vez con esta versión, pero esos meses no son del todo nuevos.
- **IPC de octubre de 2025.** El BLS no lo publicó (cierre del gobierno). Se interpola solo para construir features (rezagos y medias móviles); el target de 2025-10 y 2025-11 queda vacío, y no hay nowcast para 2025-10, 2025-11 ni 2025-12 (2025-12 no tiene mes anterior con dato para explicar el cambio).
- **Cambio de fuente en 2015.** Las noticias vienen de FNSPID hasta 2014-12 y de Alpaca desde 2015-01. La regla de compatibilidad fijada de antemano (correlación ≥ 0.7 en el solape 2015-2016) **falló** (r = 0.448), lo que indicaba usar solo Alpaca; aun así se eligió, por decisión explícita, la unión con FNSPID reescalado a la media y el desvío de Alpaca (`news_index_union_rescaled.csv`). El índice puede tener un quiebre en 2015.
- **Dominio de InflaBERT.** InflaBERT se entrenó con otro tipo de noticias; sus puntajes en estos titulares son ruidosos y se usan sin reentrenar. Revisión manual de 50 titulares del índice elegidos al azar, con un solo etiquetador y una regla escrita antes de etiquetar: InflaBERT acierta el 32 % (IC 95 % 21-46 %), mientras que decir siempre «neutral» acierta el 42 %; kappa 0,02. En los 29 titulares con dirección de precios coincide en la dirección el 55 % (IC 95 % 38-72 %). El 42 % de los titulares (21 de 50) no tenía dirección de precios, y el modelo casi nunca predice «neutral». Con 50 titulares y un solo etiquetador los intervalos son amplios; el resultado no demuestra que el modelo sea peor que el azar, solo que no se observa un acuerdo claro.
- **La app necesita la API.** Sin `API_URL` (variable de entorno o secret de Streamlit), la app muestra un aviso y se detiene; ya no hay modo embebido.
- **Dos etiquetas fuera de la regla.** Dos titulares sobre subidas de tasas se etiquetaron «down» con una interpretación distinta de la regla escrita, y no se corrigieron después de ver los resultados.
- **Sin actualización automática.** Ver «Cómo se actualiza el modelo cada mes».

## Datos y licencias

- Los titulares vienen de **FNSPID** (2007-2014) y de **Alpaca (Benzinga)** (2015-2026). La licencia para **redistribuir** el texto de los titulares está **pendiente de confirmar** para las dos fuentes.
- El archivo local `data/processed/news_scored.parquet` (titulares con su puntaje; lo arma `make headlines`) **no está en el repo**: `data/` está en `.gitignore`.
- `demo_data/processed/news_scored.parquet` (archivo del equipo, sí versionado) **contiene texto de titulares**; su licencia también está pendiente.
- Para un despliegue público, dos variables de entorno apagan esas funciones con un aviso claro: `HEADLINES_ENABLED=0` apaga `/headlines`, `/headlines/search`, `/headlines/keywords` y las secciones de titulares de Streamlit; `CLASSIFY_ENABLED=0` apaga `POST /classify` y la caja «Prueba un titular». Las dos están encendidas por defecto.
