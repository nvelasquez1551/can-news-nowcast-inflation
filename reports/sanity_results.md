# InflaBERT frente a etiquetas humanas (muestra de control)

Muestra: 60 titulares de los que entran en el índice (`news_index_union_rescaled.csv`), semilla 42:
25 al azar de FNSPID (2007-01..2014-12), 25 al azar de Alpaca (2015-01..2026-10) y 10 extra que el
modelo etiquetó neutral (5 por fuente). Etiquetados a ciegas por una sola persona (sin ver puntaje,
etiqueta del modelo ni fuente) con `scripts/label_sanity.py`. Números completos en
`sanity_results.csv`. Este archivo no contiene texto de titulares.

## Muestra al azar (50 titulares)

| Métrica | Valor |
|---|---|
| Accuracy | 0.32 (16/50), IC 95 % Wilson 0.21-0.46 |
| Línea base «siempre neu» (etiqueta humana más frecuente) | 0.42 (21/50) |
| Kappa de Cohen | 0.02 |
| Accuracy FNSPID / Alpaca | 0.32 (8/25) / 0.32 (8/25) |

Matriz de confusión (filas = humano, columnas = modelo):

| | up | down | neu |
|---|---|---|---|
| **up** | 12 | 4 | 0 |
| **down** | 9 | 4 | 0 |
| **neu** | 17 | 4 | 0 |

Precisión / recall: up 0.32 / 0.75; down 0.33 / 0.31; neu sin definir (el modelo no predijo
ningún neutral en la muestra al azar) / 0.00.

## Dirección

Entre los 29 titulares al azar que la persona etiquetó up o down, el modelo da la misma dirección
en 16 (0.55, IC 95 % 0.38-0.72) y la opuesta en 13.

## Neutrales extra (10 titulares que el modelo llamó neutral)

La persona también los etiquetó neu en 8 de 10 (precisión de la etiqueta neutral del modelo 0.80,
IC 95 % 0.49-0.94). Esto no dice nada del recall: en la muestra al azar el modelo no marcó como
neutral ninguno de los 21 titulares que la persona consideró neutrales.

## Lectura

En estos titulares InflaBERT no supera a una regla trivial (siempre «neutral») y su acuerdo con la
persona es casi el del azar (kappa 0.02). La mayor parte del error viene de titulares sin dirección de
precios (sobre todo de la Fed) que el modelo marca como «más inflación», y aun en los que sí tienen
dirección acierta el sentido apenas algo más que una moneda al aire. Con 50 titulares y un solo
etiquetador los intervalos son anchos, pero el resultado es coherente con que el índice de noticias
no mejore el nowcast.
