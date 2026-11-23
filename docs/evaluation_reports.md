# Evaluación de Qwen3-ASR con contexto de entidades

## Resumen ejecutivo

En las 1,000 muestras de WorldSpeech (`en_us`), añadir entidades extraídas con spaCy redujo el corpus WER de 12.70% a 12.32% (−0.379 puntos porcentuales; −2.99% relativo) y el corpus CER de 8.12% a 7.85% (−0.272 pp; −3.35% relativo). La tasa de transcripciones perfectas subió de 22.2% a 23.5%. Es una mejora agregada pequeña pero consistente; no es universal: el WER mejoró en 88 muestras, quedó igual en 867 y empeoró en 45. La configuración usa entidades del texto de referencia, por lo que mide contexto oracle y no un sistema NER+ASR desplegable.

## Configuración experimental

- **Modelo:** `Qwen/Qwen3-ASR-1.7B-hf`, idioma `en`, máximo de 256 tokens nuevos; comparación pareada sin prompt y con prompt de entidades.
- **Dataset:** `disco-eth/WorldSpeech`, subset `en_us`, split `test`; `metrics_summary.json` declara `samples: 1000` y el CSV contiene 1,000 filas de predicción.
- **Selección y filtros:** máximo de 1,000 muestras, semilla 42 y manifiesto congelado `data/sample_ids.json`; transcript no vacío, calidad `dnsmos_ovr >= 3.0` y duración inclusiva de 3 a 15 segundos. El audio se convierte a mono y se remuestrea a 16 kHz.
- **NER y contexto:** `src/model.py` carga `en_core_web_trf` y, si no está disponible, intenta `en_core_web_sm`. Extrae entidades únicas en orden de aparición del transcript de referencia y las pasa como prompt de la segunda inferencia. El resultado guardado no identifica cuál de los dos modelos spaCy estaba instalado; el contexto procede del texto de referencia, no del audio.
- **Evaluación:** normalización en minúsculas, eliminación de puntuación y colapso de espacios (`lab6`); WER/CER por muestra y métricas de corpus. Los conteos de errores proceden de alineación de palabras.

## Resultados cuantitativos (Sin contexto vs. Con contexto NER)

| Métrica | Sin contexto | Con contexto NER | Cambio |
|---|---:|---:|---:|
| Muestras | 1,000 | 1,000 | — |
| Corpus WER | 0.127018 (12.7018%) | 0.123226 (12.3226%) | −0.003792 (−0.3792 pp; −2.99% relativo) |
| Mean WER | 0.169639 (16.9639%) | 0.165780 (16.5780%) | −0.003859 |
| Median WER | 0.089087 (8.9087%) | 0.084524 (8.4524%) | −0.004563 |
| Corpus CER | 0.081234 (8.1234%) | 0.078509 (7.8509%) | −0.002725 (−0.2725 pp; −3.35% relativo) |
| Mean CER | 0.126429 (12.6429%) | 0.124369 (12.4369%) | −0.002060 |
| Median CER | 0.043597 (4.3597%) | 0.041921 (4.1921%) | −0.001676 |
| `perfect_wer_ratio` | 0.222 (222; 22.2%) | 0.235 (235; 23.5%) | +0.013 (+13 muestras; +1.3 pp) |
| Sustituciones | 1,225 | 1,117 | −108 |
| Deleciones | 653 | 651 | −2 |
| Inserciones | 1,371 | 1,384 | +13 |
| Aciertos (`hits`) | 23,701 | 23,811 | +110 |

El número de palabras de referencia alineadas (`N`) es 25,579 en ambas condiciones. El saldo es una reducción de 97 errores de palabra (sustituciones + deleciones + inserciones), explicada principalmente por menos sustituciones; las inserciones aumentan ligeramente. En el análisis pareado del CSV, el WER mejoró en 88/1,000 casos (8.8%), empeoró en 45 (4.5%) y no cambió en 867 (86.7%). Para CER, mejoró en 94, empeoró en 44 y quedó igual en 862.

## Análisis de errores y casos reales

Los textos y errores siguientes están normalizados como aparecen en `predictions.csv`.

**Entidades corregidas o favorecidas por el contexto**

- `us_en_20201117_0884` — **Referencia:** “missouri representative huffman representative thompson representative lamalfa from california”. **Sin entidades:** “of missouri representative huffman representative thompson representative la la mafa of california”. **Con entidades:** “missouri representative huffman representative thompson representative la lamalfa of california”. **Entidades:** `Missouri, Representative Huffman, LaMalfa, California`. WER baja de 0.5556 a 0.2222: recupera el apellido, aunque no corrige todo.
- `us_en_20171010_0356` — **Referencia:** “of student loans out there nearly 900 billion of that 13 trillion”. **Sin entidades:** “of student loans out there nearly nine hundred billion of that one point three trillion”. **Con entidades:** “of student loans out there nearly 900 billion of that 13 trillion”. **Entidades:** `Nearly $900 billion, $1.3 trillion`. WER baja de 0.4167 a 0.0.
- `us_en_20250714_0461` — **Referencia:** “s 1596 to rename the anahuac national wildlife refuge located in the state of texas as the jocelyn nungaray national wildlife refuge”. **Sin entidades:** “senate 1596 an act to rename the anahuac national wildlife refuge located in the state of texas as the jocelyn nungarrete national wildlife refuge”. **Con entidades:** “senate one thousand five hundred and ninety six an act to rename the anahuac national wildlife refuge located in the state of texas as the jocelyn nungaray national wildlife refuge”. **Entidades:** `1596, the Anahuac National Wildlife Refuge, the State of Texas, the ``Jocelyn Nungaray National Wildlife Refuge''`. El apellido queda corregido, pero expandir el número y añadir palabras empeora el WER total de 0.1818 a 0.4545.

**Fallos fonéticos y regresiones que persisten**

- `us_en_20170428_0404` — **Referencia:** “ago washington got weaker they were able to get ahold of the doctor who was one of george washingtons best friends in the world dr craik”. **Sin entidades:** “and washington got weaker well they were able to get a hold of a doctor that was one of george washingtons best friends in the world doctor craik craik”. **Con entidades:** “and washington got weaker well they were able to get a hold of a doctor that was one of george washingtons best friends in the world doctor craik c r a i k”. **Entidades:** `Washington, one, George Washington's, Craik`. El deletreo fonético del apellido sigue sin coincidir con la referencia; WER sube de 0.3077 a 0.4615.
- `us_en_20220713_2240` — **Referencia:** “377 378 379 380 381 and 382 printed in part a of house report 117405 offered by mr smith of washington”. **Sin entidades:** “377 378 379 380 381 and 382 printed in part a of house report number 117405 offered by mr smith of washington”. **Con entidades:** “377 378 379 380 381 house 117 smith washington”. **Entidades:** `377, 378, 379, 380, 381, 382, House, 117, Smith, Washington`. Una salida casi correcta pierde gran parte de la frase con el contexto: WER sube de 0.0476 a 0.6190.

**Outliers**

- `us_en_20180314_0345` tiene la referencia corta “for 5 minutes”. Sin entidades, el modelo genera una respuesta extensa que empieza “well thank you mr speaker…” (WER 11.6667); con el contexto `5 minutes`, antepone “for five minutes” pero conserva el resto extenso (WER 12.3333). La longitud de la referencia hace que pequeñas salidas desalineadas produzcan WER extremos; el contexto no resuelve el desajuste.
- `us_en_20211209_0522` también presenta WER 10.5 en ambas condiciones: la referencia es “the recognition”, frente a una transcripción mucho más larga que comienza “seek recognition designated by miss underwood…”. Estos casos son desajustes respecto de la referencia; sin revisar el audio no se puede atribuirlos con certeza a alucinación del modelo.

## Conclusiones técnicas y próximos pasos

El contexto de entidades mejora modestamente las métricas globales y la transcripción exacta, sobre todo al reducir sustituciones de nombres y cifras. A la vez, puede inducir regresiones importantes y no evita errores fonéticos, inserciones ni salidas desproporcionadas para referencias breves. No debe tratarse como una mejora garantizada por muestra.

Para la fase de fine-tuning pre-condicionado:

1. Mantener esta evaluación pareada como baseline y evaluar el modelo ajustado con los mismos IDs, filtros y normalización.
2. Separar el efecto del fine-tuning del del prompt mediante ablaciones sin contexto y con contexto; comparar el NER oracle de referencia con entidades obtenidas de una fuente disponible en inferencia real.
3. Reportar resultados por tipo y frecuencia de entidad, duración y longitud de referencia; medir menciones corregidas, conservadas y degradadas con entidades estructuradas, no texto delimitado por comas.
4. Inspeccionar manualmente las regresiones de gran magnitud y referencias cortas; registrar latencia y procedencia del modelo spaCy.

## Artefactos analizados

- `results/step2_baseline/baseline_en_qwen3_asr/metrics_summary.json` — metadato `samples: 1000` y métricas agregadas.
- `results/step2_baseline/baseline_en_qwen3_asr/predictions.csv` — 1,000 predicciones pareadas, entidades, referencias y WER/CER por muestra.
- `configs/eval_config.yaml`, `src/dataset.py`, `src/model.py` y `src/evaluate.py` — configuración y comportamiento del pipeline.

No se encontraron archivos separados de entidades o errores; las entidades y los ejemplos se analizaron desde las columnas del CSV.
