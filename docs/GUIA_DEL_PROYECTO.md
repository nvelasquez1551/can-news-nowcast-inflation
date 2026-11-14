# Guía del proyecto: ¿pueden las noticias anticipar la inflación?

Esta guía explica, sin tecnicismos, qué hace el proyecto, para qué sirve cada parte y qué se ha hecho hasta ahora. El README tiene los comandos; esta guía, el porqué.

## 1. La pregunta

El dato oficial de inflación de EE. UU. (el IPC) se publica unas dos semanas después de terminar el mes. Mientras tanto, las noticias ya cuentan qué pasa con los precios. El proyecto responde:

> **¿Leer las noticias ayuda a estimar la inflación del mes antes de que salga el dato oficial?**

## 2. La respuesta

**No.** En 2020–2023, meses que el modelo no vio al entrenarse, la estimación se equivoca en promedio lo mismo con y sin noticias (unos 0,19 puntos de inflación mensual), siempre que los dos modelos aprendan con los mismos años (2007–2019). Con XGBoost, añadir las noticias empeora el resultado.

Las noticias sí se mueven con la inflación (correlación de 0,46), pero no aportan información que el modelo no tenga ya con la gasolina, el petróleo y la inflación de meses anteriores. El estudio de referencia (InflaBERT, EPFL) llegó a la misma conclusión. Como decía el pitch: si las noticias no mejoran la estimación, se muestra igual, porque la comparación es el resultado.

## 3. Cómo funciona: dos modelos encadenados

```
15,5 M de noticias ──> MODELO 1 ──> nota mensual ──┐
(FNSPID)              (InflaBERT)   de las noticias  │
                                                     ├──> MODELO 2 ──> estimación
datos de precios ───────────────────────────────────┘    (regresión)    de la inflación
(FRED: gasolina, petróleo, IPC...)                                          │
                                                                            v
                                                      API ──> app web (Streamlit)
```

- **Modelo 1 (lee noticias).** InflaBERT, un modelo de lenguaje, lee cada titular y dice si apunta a **más inflación**, **menos inflación** o **nada claro**, con un puntaje de -1 a +1. La media de cada mes es la "nota de las noticias". No predice la inflación: solo convierte texto en números.
- **Modelo 2 (estima la inflación).** Un modelo estadístico estima cuánto subieron los precios en el mes con datos que ya se conocen a fin de mes: gasolina, petróleo, inflación de meses anteriores, precios al productor, empleo, vivienda, expectativas y la época del año. En la versión "con noticias" también usa la nota del Modelo 1.
- **La prueba clave:** el mismo Modelo 2 con y sin la nota de noticias, comparando cuánto se equivoca.

## 4. La app, parte por parte

| Parte | Qué muestra | Cómo leerla |
|---|---|---|
| **Tarjeta azul** | La estimación del mes, el dato oficial y la diferencia | Cuanto menor la diferencia, mejor acertó |
| **Qué movió la estimación** | Por qué cambió la estimación frente al mes anterior | Cada barra es un motivo. Ámbar: empujó hacia arriba; azul: hacia abajo. Sumadas dan el cambio total |
| **Estimación frente al dato oficial** | Dos líneas mes a mes | Si van juntas, el modelo acierta |
| **Las noticias del mes** | Cuántas noticias de precios hubo, qué porcentaje apuntaba a más inflación y cuánto movieron la estimación | Debajo, los titulares más inflacionarios y desinflacionarios |
| **¿Ayudan las noticias?** | El error medio con y sin noticias | Es el resultado del proyecto |
| **Busca un titular** | Las 25 palabras clave del Modelo 1 | Elige una (o escribe otras) y luego un titular para ver qué dice InflaBERT |

### Ajustes: los cuatro modelos

| Opción | Qué es |
|---|---|
| **Regresión lineal · sin noticias** | El modelo principal, solo con datos de precios. Aprende desde 1999 |
| **XGBoost · sin noticias** | Otro tipo de modelo (árboles de decisión), solo con datos de precios. Sirve para comprobar que el resultado no depende del tipo de modelo |
| **Regresión lineal · con noticias** | El modelo principal más la nota de las noticias. Aprende desde 2007, porque antes no hay noticias. **Es el que se usa por defecto** |
| **XGBoost · con noticias** | XGBoost más la nota de las noticias |

La regresión lineal es el modelo principal porque se puede explicar: cada barra de "Qué movió la estimación" es exactamente el peso de un factor. XGBoost es más flexible, pero con tan pocos meses de datos no acierta más.

## 5. Para qué sirve cada carpeta

| Carpeta o archivo | Para qué sirve |
|---|---|
| `src/model1/` | **Modelo 1.** Descarga las noticias de FNSPID (Hugging Face), filtra las de precios, las puntúa con FinBERT e InflaBERT y calcula la nota mensual |
| `config/model1.yaml` | Ajustes del Modelo 1: palabras clave, corte a fin de mes, tamaño de los bloques |
| `notebooks/run_model1_colab.ipynb` | Ejecuta el Modelo 1 en Google Colab con GPU. Se usa con **Entorno de ejecución → Ejecutar todo** |
| `nowcast/` | **Modelo 2** (paquete de Nicolás): datos de FRED (`ml_logic/data.py`), variables (`features.py`), modelos (`model.py`), guardado de modelos (`registry.py`) y entrenamiento (`interface/main.py`) |
| `nowcast/ml_logic/explain.py` | Calcula cuánto aporta cada factor a la estimación y por qué cambió frente al mes anterior |
| `nowcast/params.py` | Todas las decisiones del Modelo 2 (fechas, variables, rutas) en un solo sitio |
| `nowcast/api/fast.py` | **API.** Recibe preguntas de la app ("¿cuál es la estimación de junio 2022?") y responde con los datos |
| `app/streamlit_app.py` | **La app web** que ve el usuario |
| `.streamlit/config.toml` | Colores y estilo de la app |
| `demo_data/` | Copia pequeña de los datos (precios, nota mensual de noticias y titulares) para que la app funcione en la web sin el resto del proyecto |
| `src/model2/evaluate.py` | Medidas de error (RMSE, MAE) y la prueba de Diebold-Mariano, que dice si la diferencia entre dos modelos puede ser casualidad |
| `notebooks/01…03b` | Notebooks de Nicolás: descarga de FRED, transformación y entrenamiento del Modelo 2 |
| `docs/specs/` | Especificaciones técnicas del Modelo 1 y del Modelo 2 |
| `tests/` | Pruebas automáticas (`make test`). Comprueban, entre otras cosas, que ningún dato del futuro se cuele en la estimación |
| `Makefile` | Atajos de comandos (`make train`, `make run_api`…) |
| `data/`, `models/` | Datos y modelos entrenados en tu ordenador. No se suben a GitHub |

## 6. Cómo usarlo

**En tu ordenador** (desde la carpeta del proyecto):

```bash
make install       # una vez: instala lo necesario
make train         # entrena los modelos sin noticias
make train_news    # entrena los modelos con noticias
make run_api       # terminal 1: la API
make run_frontend  # terminal 2: la app, en http://localhost:8501
```

**En la web (Streamlit Community Cloud):** la app se publica desde `arturo22isla/can-news-nowcast-inflation`, rama `main`, archivo `app/streamlit_app.py`. En la web no hay API aparte: la app la lleva dentro y entrena los modelos la primera vez que se abre (más o menos un minuto). Cuando cambie `main` en el repositorio de Nicolás, en la copia `arturo22isla` hay que pulsar **Sync fork** para que la web se actualice.

**Modelo 1 en Colab:** abrir `notebooks/run_model1_colab.ipynb` desde GitHub, elegir **T4 GPU** y **Ejecutar todo**. Si se desconecta, se vuelve a ejecutar y sigue donde iba. Al terminar, los resultados quedan en Google Drive (`can-news-nowcast-inflation/data/processed`).

## 7. Qué se hizo

1. **Pitch y especificaciones.** Pregunta, datos (FNSPID y FRED) y método de los dos modelos.
2. **Modelo 1.** Código para leer las noticias en streaming, filtrarlas y puntuarlas. Mejoras: corte a fin de mes (decisión del equipo), filtro de titulares de emisiones de bonos ("XYZ prices $500M notes offering"), menos memoria y reanudación tras desconexiones de Colab.
3. **Ejecución en Colab.** 15,5 millones de noticias leídas, 422.756 sobre precios y nota mensual desde enero de 2007 (antes casi no hay noticias de precios en FNSPID).
4. **Modelo 2** (Nicolás). Variables de FRED disponibles a fin de mes, regresión lineal y XGBoost, prueba con reentrenamiento mes a mes en 2020–2023.
5. **Modelos con noticias.** `linear_news` y `xgb_news`, guardados junto a los modelos sin noticias.
6. **Explicación de la estimación.** Cuánto aporta cada factor, de forma exacta: los aportes suman la estimación.
7. **API y app.** Estimación, motivos del cambio, noticias del mes, comparación justa con y sin noticias, y búsqueda de titulares por palabra clave.
8. **Versión web.** La app funciona sola en Streamlit Community Cloud con `demo_data/`.

## 8. Límites que conviene contar

- **InflaBERT acierta unas 6 de cada 10 veces.** Algunos titulares salen mal clasificados; por ejemplo, "Gasoline Prices Take a Dive" sale como "más inflación".
- **FNSPID trata sobre empresas, no sobre economía.** Muchas noticias de "petróleo" son de empresas petroleras.
- **Los años 2020–2023 se han mirado muchas veces** durante el proyecto, así que ya no son una prueba "fresca".
- **El COVID** (primavera de 2020) concentra buena parte del error de todos los modelos.
- **La web usa una copia de los datos** (`demo_data/`). Si se vuelve a ejecutar el Modelo 1 o se descargan datos nuevos de FRED, hay que actualizarla.

## 9. Siguientes pasos

- Llevar la API a Google Cloud (GCP) y que la app web la use mediante `API_URL`.
- Que Nicolás dé permiso de administrador en su repositorio, para publicar desde ahí y no desde la copia.
- Probar otras formas de usar las noticias, por ejemplo solo las que hablan de inflación directamente.
