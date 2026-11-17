# Parámetros citados en el artículo

## Selección de historias (Ec. 1)

Implementación: `seleccion_por_rasgos` en `generador/src/19_pipeline.py`. Es voraz: en cada paso
añade la candidata $j$ que maximiza

$$
\Delta(j \mid S) = \sum_{r \in R} \big[\max(c_r, s_{rj}) - c_r\big]
\;+\; \underbrace{0{,}05\,\bar{s}_{j} + \mu\,\mathbb{1}[\rho_j \notin \rho(S)] + \mu_c\,\mathbb{1}[\kappa_j \notin \kappa(S)] + \lambda\,b_j}_{B_j}
\;-\; \underbrace{\nu\,\mathbb{1}[\beta_j \in \beta(S)] + \textstyle\sum_{m=1}^{6} \nu_m\, d_{mj}}_{C_j}
$$

**Símbolos**

| Símbolo | Significado |
|---|---|
| $\bar s_j$ | relevancia media de $j$ a los rasgos |
| $\rho_j$ | rol de $j$ |
| $\kappa_j$ | clase de función de $j$ |
| $\beta_j$ | beneficio literal de $j$ |
| $b_j$ | probabilidad de beneficio coherente (reordenador logístico) |

**Pesos:** $\mu = 0{,}15$, $\mu_c = 0{,}12$, $\lambda = 0{,}15$ y $\nu = 0{,}10$.

**Defectos de forma** $d_{mj} \in \{0, 1\}$ y sus pesos $\nu_m$:

| # | Defecto | $\nu_m$ |
|---|---|---|
| 1 | circular: el beneficio repite la capacidad | 0,50 |
| 2 | capacidad de más de 18 palabras | 0,20 |
| 3 | autorreferente | 0,40 |
| 4 | eco de la capacidad | 0,40 |
| 5 | rol y función incompatibles | 0,45 |
| 6 | beneficio vacío | 0,45 |

**Cómo se aplican los castigos:** se restan dentro de la ganancia, no se usan como filtro previo.

**Otros parámetros de la generación**

| Parámetro | Valor |
|---|---|
| Muestreo de núcleo | $p = 0{,}95$ |
| Candidatas por backlog | mediana 36, entre 28 y 48 |
| Historias elegidas | $k = 12$ |
| Historias ancla | entran primero, hasta $k/2$ |

## Entrenamiento (mT5-small)

Script: `generador/src/09_train_mt5.py`. Configuración común:

- **Optimizador:** Adafactor con tasa fija, sin escala relativa.
- **Precisión:** pesos fp32 con autocast bf16.
- **GPU:** RTX 5060 de 8 GB.
- **Selección:** se conserva la época de menor pérdida de validación.

| Modelo | Lote | Tasa | Épocas | Tiempo | VRAM pico |
|---|---|---|---|---|---|
| Etapa de historias | 32 | 5·10⁻⁴ | 4 | 12,4 min | 6,14 GB |
| Tareas, modelo 1 (reales y sintéticos) | 16 | 5·10⁻⁴ | 4 | 24,6 min | 6,16 GB |
| Tareas, modelo 2 (solo reales) | 8 | 9,2·10⁻⁴ | 12 | 43,0 min | 3,39 GB |

Los dos primeros continúan desde un punto de control anterior. Los minutos son los de la última
fase, no los de toda la cadena.

## Estimación

| Parámetro | Valor |
|---|---|
| Anclaje | $\lambda_c = 12$ |
| Con estimaciones del usuario | $\lambda_u = 12$ y $\lambda'_c = 30$ |
| Intervalo | nivel del 80 %, grupos por número de cerradas (0, 3, 5, 10, 20) y tamaño del proyecto (cortes en 20, 60 y 200 tareas) |
| Evaluación | 60 sorteos de cierre al azar; `GroupKFold` por proyecto completo |

## Modelos de lenguaje comparados

| Uso | Modelos | Ejecución |
|---|---|---|
| Estimación | `openai/gpt-oss-120b`, `openai/gpt-oss-20b` y `qwen/qwen3.8-27b` | API de Groq, temperatura 0, semilla 0 |
| Generación | `gemma2:9b`, `llama3.1:8b` y `mistral:7b` | Ollama, temperatura 0, en la misma máquina |

Para la generación se usaron las etiquetas por defecto de Ollama. La cuantización no quedó
registrada.
