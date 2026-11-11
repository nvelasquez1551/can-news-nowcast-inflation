# nowcast: nowcast del IPC (Modelo 2)

Nowcast de la inflación mensual de EE.UU. (`y = 100 * diff(log(CPIAUCSL))`) con datos macro de FRED disponibles al final del mes *t*. Es el refactor de los notebooks `01`, `02`, `03_train_macro_models_v1` y `03b_expanding_refit_v1`, que siguen siendo la fuente de verdad: el experimento no cambia.

- **Modelo principal (declarado de antemano):** regresión lineal.
- **Modelo secundario:** XGBoost.
- **Evaluación principal:** expanding refit sobre 2020-01..2023-12.

## Estructura

```
nowcast/
  params.py            # todas las rutas y parámetros (nada escrito a mano en otro lado)
  ml_logic/
    data.py            # descarga FRED, carga del CSV raw y del índice de noticias, split por fecha
    features.py        # target y features (rezagos, medias móviles, rezagos de disponibilidad)
    preprocessor.py    # lo único que se ajusta con train: StandardScaler + OneHotEncoder(month)
    model.py           # pipelines, CV, grid search, evaluación, Diebold-Mariano, expanding refit
    registry.py        # guardar/cargar modelos (joblib) + metadata (json)
  interface/
    main.py            # train(), evaluate(), predict(): solo orquesta
```

## Instalación

Desde la raíz del repo, en el entorno `can_news_predict_inflation`:

```bash
pip install -e ".[dev]"
```

`xgboost` está fijado en 3.2.0, la versión con la que corrió el notebook 03 v1.

## Datos

| Ruta | Contenido |
|---|---|
| `data/raw/macro_monthly_raw.csv` | series FRED mensuales (notebook 01) |
| `data/processed/` | salidas del paquete (`nowcast_*.csv`) y archivos de referencia v1 |
| `models/` | modelos guardados por `train()` |

Las tres carpetas están en `.gitignore`. Para descargar de FRED hace falta `FRED_API_KEY` en `.env`. `download_fred()` no sobrescribe un CSV raw existente (FRED revisa datos históricos y cambiaría los números de v1).

## Uso

```bash
python -m nowcast.interface.main             # train + predict (no toca el test)
RUN_TEST=1 python -m nowcast.interface.main  # además evalúa en 2020-2023
```

```python
from nowcast.interface.main import train, evaluate, predict

train()      # CV en 1999-2019, ajuste final, guarda en models/
evaluate()   # solo con RUN_TEST=1: evaluación fija + expanding refit
predict()    # DataFrame (month, nowcast) con el modelo principal
```

- **`train()`** usa solo 1999-02..2019-12: CV con `TimeSeriesSplit(5)` y grid search de XGBoost.
- **`evaluate()`**
  - Evaluación fija: modelos de `train()` sobre 2020-01..2023-12. Sirve para verificar el refactor contra v1.
  - Expanding refit (resultado principal): cada mes *t* se re-ajusta con todas las filas anteriores a *t*. XGBoost vuelve a buscar hiperparámetros cada 12 meses (2020-01, 2021-01, 2022-01, 2023-01).
  - Baselines: `naive_lag1` (= `y_lag1`) y la media de train. Es fija en la evaluación fija y es la media de las filas anteriores a *t* en el expanding.
  - Diebold-Mariano (h=1, corrección HLN): linear vs naive, xgb vs naive, xgb vs linear.

Las decisiones (features, `USE_MONTH`, hiperparámetros) se toman solo con CV hasta 2019, nunca mirando el test.

## ⚠️ El test 2020-2023 ya se vio varias veces

Los meses 2020-01..2023-12 se usaron en v0, en v1, en la verificación de este refactor y en el expanding refit. **Ya no son un test "fresco"**: cualquier resultado sobre ese periodo debe reportarse con esta advertencia. Por eso `evaluate()` solo corre con `RUN_TEST=1`.

## Noticias

`USE_NEWS = False` por defecto: la corrida macro usa la muestra completa, idéntica a v1. Con `USE_NEWS = True` y `data/processed/news_index_monthly.csv` presente (columnas `month`, `infla_score_mean`, `infla_net_share`), se eliminan los meses sin noticias y la muestra empieza en el primer mes con noticias. Si el archivo no existe, todo corre solo con datos macro.

## Tests

```bash
pytest tests/
```

Usan datos sintéticos: no llaman a FRED, no leen `data/` ni `models/` y nunca usan los meses reales de test. Verifican, entre otras cosas, que las features en *t* solo usan información disponible en *t*, que el expanding refit nunca entrena con *t* o meses posteriores, y que el scaler aprende la media de train.
