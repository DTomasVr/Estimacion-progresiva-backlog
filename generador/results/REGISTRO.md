# Registro de resultados

Cada cifra con su archivo de origen. Generado por src/32_registro_resultados.py

## datos

| medida | valor | fuente | nota |
|---|---|---|---|
| incidencias unificadas | 2845066 | `data/interim/nodes.parquet` | TAWOS + Public Jira, src/03 |
| aristas padre-hijo | 244677 | `data/interim/edges.parquet` | 100753 epic->hijo, 143924 padre->subtarea |
| proyectos | 1289 | `data/processed/split_projects.parquet` | train 1003 / val 122 / test 164 |
| ejemplos abajo (train) | 11202 | `data/processed/s2s_hist_train.parquet` | historia -> tareas |
| ejemplos abajo (val) | 1382 | `data/processed/s2s_hist_val.parquet` | historia -> tareas |
| ejemplos abajo (test) | 1751 | `data/processed/s2s_hist_test.parquet` | historia -> tareas |
| pares One2One arriba (train) | 71817 | `data/processed/s2s_o2o_train.parquet` | 9624 en formato historia |
| incidencias con formato de historia | 16172 | `data/interim/historias_formato.parquet` | 6502 con el formato en el titulo; src/12 |

## sbert

| medida | valor | fuente | nota |
|---|---|---|---|
| nDCG@10 consultas en | 0.3531 -> 0.3923 | `results/07_sbert_test.csv` | delta 0.0392 IC95 [0.0302, 0.0481] |
| nDCG@10 consultas es | 0.2294 -> 0.3167 | `results/07_sbert_test.csv` | delta 0.0873 IC95 [0.0781, 0.096] |

## abajo

| medida | valor | fuente | nota |
|---|---|---|---|
| un modelo (val) | sF1 0.4936, cobertura 0.4444 | `results/23_cobertura.csv` |  |
| prueba: solo_recuperacion | sF1 0.3182, cobertura 0.3169 | `results/11_test_metricas.csv` | n=2478 |
| prueba: mt5-small | sF1 0.4684, cobertura 0.4289 | `results/11_test_metricas.csv` | n=2478 |
| prueba: mt5-base | sF1 0.464, cobertura 0.4266 | `results/11_test_metricas.csv` | n=2478 |
| conjunto de dos modelos (prueba) | sF1 0.4852, cobertura 0.4673 | `logs/23 + comparacion pareada` | +0.012 sF1 [0.009, 0.015]; +0.029 cobertura [0.026, 0.033] |

## arriba

| medida | valor | fuente | nota |
|---|---|---|---|
| dalpiaz_test(4 proyectos) [solo recuperacion] | sF1 0.2241, cobertura 0.179, formato 0.25 | `results/18_test_o2o.csv` | n_gen 3.0 |
| dalpiaz_test(4 proyectos) [modelo] | sF1 0.4436, cobertura 0.4136, formato 1.0 | `results/18_test_o2o.csv` | n_gen 20.0 |
| o2o_test_historias [solo recuperacion] | sF1 0.2937, cobertura 0.304, formato 0.039 | `results/18_test_o2o.csv` | n_gen 3.5 |
| o2o_test_historias [modelo] | sF1 0.4013, cobertura 0.4616, formato 0.924 | `results/18_test_o2o.csv` | n_gen 17.2 |
| variante actual (val) | sF1 0.4139, fuga 0.1185 | `results/24_calidad_arriba.csv` |  |
| variante sin_ejemplos (val) | sF1 0.417, fuga 0.0 | `results/24_calidad_arriba.csv` |  |
| variante umbral0.5 (val) | sF1 0.4223, fuga 0.0898 | `results/24_calidad_arriba.csv` |  |
| variante umbral0.55 (val) | sF1 0.4132, fuga 0.0333 | `results/24_calidad_arriba.csv` |  |
| variante k4 (val) | sF1 0.414, fuga 0.1701 | `results/24_calidad_arriba.csv` |  |
| variante relevancia (val) | sF1 0.4335, fuga 0.105 | `results/24_calidad_arriba.csv` |  |

## espanol

| medida | valor | fuente | nota |
|---|---|---|---|
| arriba antes/en | sF1 0.4548 | `results/26_eval_es.csv` |  |
| arriba antes/es | sF1 0.3768 | `results/26_eval_es.csv` |  |
| arriba despues/en | sF1 0.4228 | `results/26_eval_es.csv` |  |
| arriba despues/es | sF1 0.3743 | `results/26_eval_es.csv` |  |
| entrada es traducida al ingles | sF1 0.370 -> 0.416 | `medicion directa 2026-09-23` | +0.045 IC95 [0.027, 0.065]; 175 ms; ida y vuelta, numero optimista |
| salida traducida (historias) | fidelidad 0.9034 (piso 0.1993), terminos 1.0 | `results/27_traduccion_salida.csv` | 12.4225 ms/item |
| salida traducida (tareas) | fidelidad 0.8806 (piso 0.1453), terminos 0.9945 | `results/27_traduccion_salida.csv` | 4.2927 ms/item |

## dominio

| medida | valor | fuente | nota |
|---|---|---|---|
| dominio 2 fuera | fuera 0.4569 vs dentro 0.4463 | `results/29_lodo_v2.csv` | diferencia 0.0106 |
| dominio 3 fuera | fuera 0.4811 vs dentro 0.4584 | `results/29_lodo_v2.csv` | diferencia 0.0227 |
| dominio 4 fuera | fuera 0.4591 vs dentro 0.4521 | `results/29_lodo_v2.csv` | diferencia 0.007 |

## banco

| medida | valor | fuente | nota |
|---|---|---|---|
| proyectos medidos | 64 | `results/20_banco.jsonl` | con los 4 sistemas presentes |
| pbe - | 4.17 s (p90 4.6) | `results/20_banco.jsonl` | n=64, tareas 49 |
| gemma2:9b cero | 35.73 s (p90 45.3) | `results/20_banco.jsonl` | n=64, tareas 110 |
| gemma2:9b ejemplos | 34.40 s (p90 45.0) | `results/20_banco.jsonl` | n=64, tareas 108 |
| llama3.1:8b cero | 21.63 s (p90 25.2) | `results/20_banco.jsonl` | n=64, tareas 122 |
| llama3.1:8b ejemplos | 21.05 s (p90 27.3) | `results/20_banco.jsonl` | n=64, tareas 125 |
| mistral:7b cero | 39.44 s (p90 43.3) | `results/20_banco.jsonl` | n=64, tareas 203 |
| mistral:7b ejemplos | 39.93 s (p90 43.3) | `results/20_banco.jsonl` | n=64, tareas 198 |
| fidelidad nuestra (64 proyectos) | sF1 0.4720 | `results/22_banco_pareado.csv` | media por proyecto, mismos casos que los LLM |
| fidelidad llama3.1:8b cero | sF1 0.4634 | `results/22_banco_pareado.csv` | delta -0.0087 IC95 [-0.0353, +0.0173] pareado n=64; empate: el IC cruza cero |
| fidelidad mistral:7b cero | sF1 0.4732 | `results/22_banco_pareado.csv` | delta +0.0012 IC95 [-0.0228, +0.0264] pareado n=64; empate: el IC cruza cero |
| fidelidad gemma2:9b cero | sF1 0.4748 | `results/22_banco_pareado.csv` | delta +0.0027 IC95 [-0.0258, +0.0313] pareado n=64; empate: el IC cruza cero |
| fidelidad llama3.1:8b ejemplos | sF1 0.4794 | `results/22_banco_pareado.csv` | delta +0.0073 IC95 [-0.0163, +0.0329] pareado n=64; empate: el IC cruza cero |
| fidelidad mistral:7b ejemplos | sF1 0.4855 | `results/22_banco_pareado.csv` | delta +0.0135 IC95 [-0.0092, +0.0359] pareado n=64; empate: el IC cruza cero |
| fidelidad gemma2:9b ejemplos | sF1 0.4867 | `results/22_banco_pareado.csv` | delta +0.0147 IC95 [-0.0097, +0.0404] pareado n=64; empate: el IC cruza cero |
| tokens de salida por backlog (LLM) | 1412 | `results/20_banco.jsonl` | base del costo por API |

## costo

| medida | valor | fuente | nota |
|---|---|---|---|
| API tipo Groq por proyecto | ~0.0005 USD | `precios publicos 2026` | ~1000 tok entrada + ~1400 salida; GPT OSS 20B 0.075/0.30 por millon |
| PC propia 24/7 | ~58 kWh/mes = ~S/ 40 | `medicion nvidia-smi + tarifa OSINERGMIN` | GPU 14 W en reposo, 80 W generando |
| serverless GPU (L4) | 0.10 centavos caliente / 0.44 frio | `Modal 0.000222 USD/s` | carga de modelos 10-17 s es el costo dominante en frio |
