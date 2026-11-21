# Imagen de la API del nowcast (solo FastAPI; Streamlit no va aquí).
#
#   docker build -t nowcast-api .
#   docker run --rm -p 8080:8080 nowcast-api
#
# Lleva los modelos guardados linear y linear_news y los dos archivos de datos que
# necesitan /nowcast, /history, /months y /comparison. No lleva titulares: la
# imagen apaga /headlines* y /classify (HEADLINES_ENABLED=0, CLASSIFY_ENABLED=0),
# así que no necesita torch ni transformers.
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HEADLINES_ENABLED=0 \
    CLASSIFY_ENABLED=0 \
    PORT=8080

WORKDIR /app

# Mismas versiones que el entorno donde se guardaron los modelos (joblib de
# scikit-learn). xgboost-cpu: el mismo módulo xgboost sin las librerías CUDA; la API
# lo importa (nowcast.ml_logic.model) pero no sirve modelos XGBoost.
RUN pip install \
        fastapi==0.162.5 uvicorn==0.54.0 pydantic==2.14.3 \
        numpy==2.2.6 pandas==2.3.3 scipy==1.15.3 scikit-learn==1.7.2 joblib==1.6.0 \
        xgboost-cpu==3.2.0 pyarrow==25.0.1 pyyaml==6.0.3 \
        fredapi==0.5.2 python-dotenv==1.2.4

# Código (sin tests, notebooks, src/ del Modelo 1 ni config/)
COPY nowcast/ nowcast/

# Modelos finales (entrenados con 2007-01..2026-10). Al reentrenar, cambiar los
# nombres aquí y en .dockerignore.
COPY models/linear_20261117-195644-556120.joblib \
     models/linear_20261117-195644-556120.json \
     models/linear_news_20261117-195644-580389.joblib \
     models/linear_news_20261117-195644-580389.json \
     models/

# Datos: macro de FRED e índice mensual de noticias (sin texto de titulares)
COPY data/raw/macro_monthly_raw.csv data/raw/
COPY data/processed/news_index_union_rescaled.csv data/processed/

# Control: ningún parquet (titulares), xgb ni .env dentro de la imagen
RUN test -z "$(find /app -name '*.parquet' -o -name 'xgb*' -o -name '.env')" \
    && useradd --create-home --uid 1000 app
USER app

EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn nowcast.api.fast:app --host 0.0.0.0 --port ${PORT:-8080}"]
