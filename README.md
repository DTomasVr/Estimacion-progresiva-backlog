# Generación de backlogs y estimación de esfuerzo progresiva

Código, experimentos y resultados del artículo *Generación de backlogs y estimación de esfuerzo
progresiva en proyectos de software ágiles con modelos de lenguaje pequeños*.

El modelo convierte la descripción de un producto en un backlog de historias de usuario con sus
tareas. Luego estima las horas de cada tarea y da un total del proyecto con un intervalo del 80 %,
que se afina a medida que se cierran tareas y que el equipo aporta lo que sabe.

## Reproducir el artículo

```bash
python articulo/reproducir.py            # tablas del artículo desde los resultados guardados (segundos)
python articulo/reproducir.py --todo     # además, vuelve a correr los experimentos del total (minutos)
```

- **Sin opciones:** regenera las tablas I, III y IV en `estimador/results/cuadros_paper/tex/`, tal como
  aparecen en el artículo. Solo usa los CSV de `estimador/results/`.
- **Con `--todo`:** antes vuelve a correr los experimentos del total (E-265, 268, 275, 277 y 278). Para
  eso necesita `estimador/data/processed/pbe_effort_v4.parquet`, que no se versiona por tamaño y
  licencias; ver «Datos».

Dependencias: Python 3.12 con `numpy`, `pandas` y `pyarrow`.

## Dónde está cada cifra del artículo

Las rutas son relativas a `estimador/`, salvo las que empiezan por `generador/`.

| En el artículo | Script | Resultado |
|---|---|---|
| **Tabla I**: PRED(25), ×2 y MdMRE del total según las tareas cerradas (39,5 % → 62,7 %) | `src/275_curva_anclaje_dos_criterios.py` | `results/curva_anclaje_dos_criterios/resumen.csv` |
| Tabla I: anchura y cobertura del intervalo (×5,57 → ×2,14 en los 114; ×2,21 en los 81) | `src/161_banda_proyecto.py`, `src/233_banda_que_cierra_antes.py` | `results/banda_proyecto/resumen.csv`, `results/banda_que_cierra_antes/resumen.csv` |
| Cobertura por tamaño del proyecto (0,72–0,96 con una cerrada; 0,78–0,82 con 10 o más) | `src/233_banda_que_cierra_antes.py` | `results/banda_que_cierra_antes/cobertura_por_tramo.csv` |
| Extrapolación simple y sin contracción, λc = 0 (12,5 / 45,9 / 59,2 % con 1, 10 y 20 cerradas; IC por proyectos de la diferencia) | `src/277_extrapolacion_simple.py` | `results/extrapolacion_simple/resumen.csv`, `results/extrapolacion_simple/diferencia_ic.csv` |
| Orden real de cierre (de 42,1 % a 50,0 %, frente a 71,3 % al azar; 38 proyectos) | `src/265_avance_orden_real.py` | `results/avance_orden_real/resumen.csv` |
| Entradas del usuario (46,3 → 68,3 %; 62,0 → 82,9 %; 41 proyectos) | `src/268_tres_estados.py` | `results/tres_estados/resumen.csv` |
| Meta del trabajo futuro: orden real con estimaciones (18 proyectos) | `src/278_orden_real_con_estimaciones.py` | `results/orden_real_con_estimaciones/resumen.csv` |
| **Tabla II**: Deep-SE y GPT2SP (reimplementación propia) | `src/46_baselines_literatura.py` | `results/baselines_literatura.csv` |
| Tabla II: propuesta y constante de 2,45 h | `src/211_rejilla_de_valores_reales.py` | `results/rejilla_valores/contra_literatura.csv` |
| **Tabla III**: estimación frente a LLM (601 tareas, Groq, temperatura 0) | `src/259_banco_api.py`, `src/261_tablas_llm_api.py` | `results/llm_bench/api_resumen.csv` |
| 13,7 ms por tarea (estimador) y 5,2 GB de RAM y 3,3 GB de disco (sistema completo, generador y estimador, en CPU) | estimador desplegado | `results/estimador_v7/servicio.json` (RAM); el disco es el tamaño del paquete de producción (3,1 GiB) |
| **Tabla IV**: generación frente a LLM (64 proyectos) | `src/272_banco_generacion.py`, `src/274_redundancia_tareas.py` | `results/banco_generacion/resumen.csv`, `results/redundancia_tareas/resumen.csv` |
| sF1 de 0,669 frente a 328 historias humanas | `generador/src/97_eval_humano.py` | `generador/results/49_real_vigente.csv` |
| Efecto de las historias ancla: 0,669 con anclas, 0,635 sin ellas, +0,035 (IC por proyectos) | `generador/src/103_ablacion_anclas.py` | `generador/results/103_ablacion_anclas.csv` |
| Escritura de las tablas en LaTeX | `src/273_cuadros_paper.py` | `results/cuadros_paper/` |

Los pesos de la selección de historias (Ec. 1) y los hiperparámetros de entrenamiento están en
[PARAMETROS.md](PARAMETROS.md).

## Estructura

| Carpeta | Qué contiene |
|---|---|
| `articulo/` | `reproducir.py`, el punto de entrada para regenerar el artículo. |
| `estimador/` | Datos derivados, experimentos y modelo de estimación de esfuerzo. |
| `generador/` | Pipeline de generación de historias y tareas (`src/19_pipeline.py`), sus experimentos y resultados. |

Cada experimento es un script numerado (`estimador/src/NNN_*.py`, `generador/src/NN_*.py`) que
escribe sus resultados en `results/`. Para el artículo solo hacen falta los scripts de la tabla
anterior.

Los demás scripts son el historial completo de experimentos, incluidos los descartados. Se conservan
por transparencia: `estimador/INFORME_PROYECTO.md` y `generador/DISENO_GENERACION.md` cuentan qué se
probó y por qué se descartó.

## Datos

Los conjuntos de partida son públicos y se usan según sus licencias. Hay que descargarlos de sus
fuentes en `estimador/data/raw/` (estimación) y en la carpeta de datos del generador.

| Conjunto | Uso en el artículo | Licencia | Fuente |
|---|---|---|---|
| TAWOS (Tawosi et al., MSR 2022) | horas de tareas; pares historia → tareas | Apache 2.0 | github.com/SOLAR-group/tawos |
| The Public Jira Dataset (Montgomery et al., MSR 2022) | horas de tareas; pares historia → tareas | CC BY 4.0 | doi:10.5281/zenodo.5882881 |
| SiP (Jones y Cullum, 2019) | horas y estimaciones de tareas | propia: investigación y estudio no comercial | github.com/Derek-Jones/SiP_dataset |
| Requirements Data Sets (User Stories) (Dalpiaz, 2018) | historias de entrenamiento (15 proyectos) y 4 proyectos retenidos | CC BY 4.0 | doi:10.17632/7zbk8zsd8y.1 |
| Material de Bragilovski et al. (RE 2024) | las 328 historias humanas de evaluación | CC BY 4.0 | doi:10.5281/zenodo.12740518 |

- **Tabla unificada:** se construye con la cadena `estimador/src/01_*` a `09_*` y
  `estimador/src/52_build_v4.py`.
- **Modelos entrenados** (unos 3 GB): no están en este repositorio.
- **Predicciones del total** que usan los experimentos del anclaje: sí están
  (`estimador/results/modelo_total/pred_total.npy`).

## In English (summary)

Code, experiments and results of the paper *Backlog generation and progressive effort estimation
for agile software projects with small language models* (submitted in Spanish).

- **Reproduce the paper's tables:** `python articulo/reproducir.py` (from the saved CSVs, seconds);
  add `--todo` to rerun the project-total experiments (needs the processed data, see «Datos»).
- **Where each number comes from:** the table «Dónde está cada cifra del artículo» maps every figure
  of the paper to its script and result file.
- **Layout:** `estimador/` (effort estimation), `generador/` (backlog generation), `articulo/`
  (entry point). Numbered scripts are the full experiment history; only those in the table are
  needed for the paper.
- **Data and licences:** see the table in «Datos». TAWOS (Apache 2.0); Public Jira Dataset,
  Dalpiaz and Bragilovski et al. (CC BY 4.0; the Dalpiaz curator states no known licence agreements
  on the stories); SiP (research and non-commercial use).
- **Parameters** (selection weights, training, LLM versions): `PARAMETROS.md`.

## Licencia

GPL-3.0 (ver `LICENSE`).
