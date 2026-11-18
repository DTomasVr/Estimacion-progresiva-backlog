# Backlog generation and progressive effort estimation

Code, experiments and results of the paper *Backlog generation and progressive effort estimation in
agile software projects with small language models*. A Spanish version of this README is in
[README.es.md](README.es.md).

The model turns a product description into a backlog of user stories with their tasks. It then
estimates the hours of each task and gives a project total with an 80 % interval. The total is
refined as tasks are closed and as the team provides what it knows.

## Reproducing the paper

```bash
python articulo/reproducir.py            # paper tables from the saved results (seconds)
python articulo/reproducir.py --todo     # also reruns the project-total experiments (minutes)
```

- **No options:** regenerates Tables I, III and IV in `estimador/results/cuadros_paper/tex/`, exactly
  as they appear in the paper. It only uses the CSV files in `estimador/results/`.
- **With `--todo`:** first reruns the project-total experiments (E-265, 268, 275, 277 and 278). This
  needs `estimador/data/processed/pbe_effort_v4.parquet`, which is not versioned because of its size
  and the dataset licences (see «Data»).

Requirements: Python 3.12 with `numpy`, `pandas` and `pyarrow`.

## Where each number of the paper comes from

Paths are relative to `estimador/` unless they start with `generador/`. Decimal commas follow the
Spanish paper.

| In the paper | Script | Result |
|---|---|---|
| **Table I**: PRED(25), ×2 and MdMRE of the total by closed tasks (39,5 % → 62,7 %) | `src/275_curva_anclaje_dos_criterios.py` | `results/curva_anclaje_dos_criterios/resumen.csv` |
| Table I: interval width and coverage (×5,57 → ×2,14 on the 114 projects; ×2,21 and 0,81 on the 81) | `src/161_banda_proyecto.py`, `src/233_banda_que_cierra_antes.py` | `results/banda_proyecto/resumen.csv`, `results/banda_que_cierra_antes/resumen.csv` |
| Coverage by project size (0,72–0,96 with one closed task; 0,78–0,82 with 10 or more) | `src/233_banda_que_cierra_antes.py` | `results/banda_que_cierra_antes/cobertura_por_tramo.csv` |
| Simple extrapolation and no shrinkage, λc = 0 (12,5 / 45,9 / 59,2 % with 1, 10 and 20 closed tasks) and the anchoring advantage with project-level CIs (+25,1 [18,1; 32,2], +7,9 [2,5; 12,8], +3,5 [−2,1; 8,4]) | `src/277_extrapolacion_simple.py` | `results/extrapolacion_simple/resumen.csv`, `results/extrapolacion_simple/diferencia_ic.csv` |
| Real closing order (42,1 % → 50,0 %, against 71,3 % at random; 38 projects) | `src/265_avance_orden_real.py` | `results/avance_orden_real/resumen.csv` |
| User inputs (46,3 → 68,3 %; 62,0 → 82,9 %; 41 projects) | `src/268_tres_estados.py` | `results/tres_estados/resumen.csv` |
| Future-work goal: real order with user estimates (18 projects) | `src/278_orden_real_con_estimaciones.py` | `results/orden_real_con_estimaciones/resumen.csv` |
| **Table II**: Deep-SE and GPT2SP (own reimplementation) | `src/46_baselines_literatura.py` | `results/baselines_literatura.csv` |
| Table II: proposal and the 2,45 h constant | `src/211_rejilla_de_valores_reales.py` | `results/rejilla_valores/contra_literatura.csv` |
| **Table III**: estimation against LLMs (601 tasks, Groq, temperature 0) | `src/259_banco_api.py`, `src/261_tablas_llm_api.py` | `results/llm_bench/api_resumen.csv` |
| 13,7 ms per task (estimator); 5,2 GB of RAM and 3,3 GB of disk (whole system, generator and estimator, on CPU) | deployed estimator | `results/estimador_v7/servicio.json` (RAM); the disk figure is the size of the production package (3,1 GiB) |
| **Table IV**: generation against LLMs (64 projects) | `src/272_banco_generacion.py`, `src/274_redundancia_tareas.py` | `results/banco_generacion/resumen.csv`, `results/redundancia_tareas/resumen.csv` |
| LLM advantage on the 4 Dalpiaz projects (+0,027 to +0,049, no CI) | `src/272_banco_generacion.py` | `results/banco_generacion/por_proyecto.csv` |
| sF1 of 0,669 against 328 human stories | `generador/src/97_eval_humano.py` | `generador/results/49_real_vigente.csv` |
| Effect of the anchor stories: 0,669 with anchors, 0,635 without, +0,035 (project-level CI) | `generador/src/103_ablacion_anclas.py` | `generador/results/103_ablacion_anclas.csv` |
| LaTeX tables | `src/273_cuadros_paper.py` | `results/cuadros_paper/` |

The weights of the story selection (Eq. 1) and the training hyperparameters are in
[PARAMETROS.md](PARAMETROS.md).

## Generator version behind each number

All generator numbers come from the same version. The pipeline and the three mT5 models are
identical in development and production (checked with MD5).

| Component | MD5 |
|---|---|
| `generador/src/19_pipeline.py` | `cc293264bd5132d51af79e785145666d` |
| mT5, story stage (v9bt) | `7d7f6c771d7751e81c69826798cd1b52` |
| mT5, task stage 1 (v5) | `3355d766e13dd562df566a5ee03260f3` |
| mT5, task stage 2 (bs8, lr 9,2e-4) | `85225d221b0dd753c2dbc14ffbade51e` |

| Number in the paper | Where | Script | Result |
|---|---|---|---|
| sF1 0,669 with anchors and 0,635 without; +0,035 [0,026; 0,044] | PI3 | `generador/src/103_ablacion_anclas.py` | `generador/results/103_ablacion_anclas.csv` |
| 0,387 · 43,7 % · 35 tasks · 10,8 s | Table IV | `estimador/src/272_banco_generacion.py` | `estimador/results/banco_generacion/` |
| 11,8 % near-duplicate tasks | Table IV | `estimador/src/274_redundancia_tareas.py` | `estimador/results/redundancia_tareas/` |
| Differences with the LLMs (+0,076 to +0,088; CI +0,055 to +0,116) and +0,027 to +0,049 on the 4 Dalpiaz projects | PI3 | `estimador/src/272_banco_generacion.py` | `estimador/results/banco_generacion/` |

**Control rerun** (18/11, not in the paper): `generador/src/104_repetir_e272.py` →
`estimador/results/banco_generacion_rerun_18nov/`.

- **sF1:** 0,391 against 0,387. The difference is +0,004, CI95 [−0,004; +0,011]: sampling noise, since
  E-272 does not fix a seed.
- **Near-duplicate tasks:** vary between runs, from 9,6 % to 11,8 %.

## User-story collections

**Dalpiaz, *Requirements Data Sets (User Stories)*, v1 (2018)**, 22 collections:

| Use | Collections |
|---|---|
| Training (15) | g03, g04, g08, g10, g11, g12, g13, g14, g17, g18, g21, g22, g24, g27, g28 |
| Validation (3) | g05, g23, g26 |
| Held out, Table IV (4) | g02, g16, g19, g25 |

**Evaluation: 328 stories from the material of Bragilovski et al. (RE 2024)**

- They are 6 projects written by master's students: Fish&Chips, Grocery, School, Sports, Supermarket
  and Ticket.
- That material has 9 collections. The other 3 (camperplus = g12, planningpoker = g13 and
  recycling = g04) come from Dalpiaz and are in our training data, so they are excluded from the
  evaluation.

## Result files: column names

The code, comments and the column names and labels of the paper's result files are in English.
Three things keep their Spanish names: file and folder names (they are cited in the paper and the
thesis), the raw LLM run logs (`*.jsonl`, whose fields such as `modelo`, `condicion`, `valido`,
`horas`, `segundos`, `caso`, `sistema`, `tareas` are kept exactly as recorded), and the outputs of
scripts the paper does not use. The paper's LaTeX tables are in Spanish because the paper is.

Common columns:

| Column | Meaning |
|---|---|
| `PRED25` | share of projects (or tasks) whose estimate is within ±25 % of the real value |
| `MdMRE` | median of the magnitude of relative error, abs(estimate − real) / real |
| `factor2` | share within a factor 2 of the real value |
| `projects`, `n` | number of projects (or tasks, or backlogs) behind the row |
| `k`, `closed` | number of tasks already closed in the project |
| `ci95_lo`, `ci95_hi` | 95 % bootstrap interval of the difference in the row |
| `coverage` | estimation: share of real totals inside the interval; generation: share of real stories matched |
| `sF1` | set F1 of the generated stories against the real ones (generation) |

Files used by the paper (paths relative to `estimador/`):

| File | Columns and values |
|---|---|
| `results/curva_anclaje_dos_criterios/resumen.csv` | `criterion`: `all` (the 114 projects), `larger` (only projects with more than k tasks), `k or 20 % of the backlog`; `PRED25_cold`: PRED25 of the same projects at kick-off |
| `results/banda_proyecto/resumen.csv` | `scenario` (`cold`, `3 closed tasks`, `10 closed tasks`), `level`, `method` (`single`, `by size band`), `width_x`: median ratio of the interval's upper to lower bound |
| `results/banda_que_cierra_antes/resumen.csv` | `band`: interval variant (`A2 · by k and size band (deployed)` is the one in production, `D · shrunk size band M=…` the one used with few closed tasks); `nominal_level`, `median_width`, `width_p25`, `width_p75`, and the same for `larger` projects |
| `results/extrapolacion_simple/resumen.csv`, `diferencia_ic.csv` | `rule`: `anchoring lambda=12`, `no shrinkage (lambda=0)`, `simple extrapolation`; `diff_PRED25` anchoring minus extrapolation, `projects_anchoring_better`, `projects_extrapolation_better` |
| `results/avance_orden_real/resumen.csv` | `cut`: `cold`, `real order · X % closed`, `random · X % closed`; `closed_tasks_median`, `closed_fraction_median`; size-band columns (`8-15` … `over 400`) give PRED25 by project size |
| `results/tres_estados/resumen.csv` | `user_estimates` (share of the backlog the user estimates), `lambda_u`, `lambda_c`, `PRED25_clean` (only over what the user did not give) |
| `results/orden_real_con_estimaciones/resumen.csv` | `user_estimates` (`none`, `all`), `closed_fraction`, `PRED25_real_order`, `PRED25_random` |
| `results/baselines_literatura.csv` | `model`, `beats_baseline` (projects where the model beats the project median), `RMSE_log` |
| `results/rejilla_valores/contra_literatura.csv` | `model`: `ours, not snapped`, `ours, SNAPPED to the grid`, `best constant`; `MAE_h` (hours), `SA`, `spearman` |
| `results/llm_bench/api_resumen.csv` | `condition` (`A` text only, `B` with examples, `-` reference), `model` (`PBE v7 (ours)` is the proposal), `provider`, `version` (`raw`, `calibrated`), `valid` (share of parseable answers), `sec_per_task`, `factor` (calibration factor), `diff_MdMRE` (CI95 of LLM minus PBE), `verdict` (`LLM better`, `PBE better`, `tie`), `v7_same_tasks_*` |
| `results/banco_generacion/resumen.csv` | `system` (`PBE (current)` is the proposal; `… zero-shot` the LLMs), `sec_median`, `stories`, `tasks`, `format` (share in the "As a…" form), `recognises` (share of projects the LLM claims to recognise), `diff_sF1` (other minus PBE), `verdict` (`PBE wins`, `other wins`, `tie`) |
| `results/banco_generacion/por_proyecto.csv` | one row per system and project: `case`, `source`, `seconds`, `n_tasks` |
| `results/redundancia_tareas/resumen.csv` | `near_duplicates_090` (share of tasks with a cosine ≥ 0,90 to another task of the same backlog), `near_duplicates_080`, `mean_similarity`, `distinct_tasks`, `diff_dup90` (LLM minus PBE) |
| `generador/results/49_real_vigente.csv`, `103_ablacion_anclas.csv` | `seed`, `project`, `n_ref`, `n_gen`, `anchors` (anchor stories used), `anchors_on` (1 with anchors), `precision`, `seconds` |

## Layout

| Folder | Contents |
|---|---|
| `articulo/` | `reproducir.py`, the entry point to regenerate the paper. |
| `estimador/` | Derived data, experiments and the effort-estimation model. |
| `generador/` | Story and task generation pipeline (`src/19_pipeline.py`), its experiments and results. |

Each experiment is a numbered script (`estimador/src/NNN_*.py`, `generador/src/NN_*.py`) that
writes its results to `results/`. Only the scripts in the table above are needed for the paper.

The other scripts are the full experiment history, including discarded ones. They are kept for
transparency; `estimador/INFORME_PROYECTO.md` and `generador/DISENO_GENERACION.md` (in Spanish)
explain what was tried and why it was discarded.

## Data

The source datasets are public and are used under their licences. Download them from their sources
into `estimador/data/raw/` (estimation) and into the generator's data folder.

| Dataset | Use in the paper | Licence | Source |
|---|---|---|---|
| TAWOS (Tawosi et al., MSR 2022) | task hours; story → tasks pairs | Apache 2.0 | github.com/SOLAR-group/tawos |
| The Public Jira Dataset (Montgomery et al., MSR 2022) | task hours; story → tasks pairs | CC BY 4.0 | doi:10.5281/zenodo.5882881 |
| SiP (Jones and Cullum, 2019) | task hours and estimates | CC BY-NC-ND 4.0 (no commercial use, no modified versions): this repository does not include its rows; the scripts read them from the original | github.com/Derek-Jones/SiP_dataset |
| Requirements Data Sets (User Stories) (Dalpiaz, 2018), v1 | training stories (15 collections), validation (3) and 4 held-out projects | the curator states no known licence agreements or rights on the stories (the record lists CC BY 4.0) | doi:10.17632/7zbk8zsd8y.1 |
| Material of Bragilovski et al. (RE 2024) | the 328 human evaluation stories | CC BY 4.0 | doi:10.5281/zenodo.12740518 |

- **Unified table:** built with the chain `estimador/src/01_*` to `09_*` and
  `estimador/src/52_build_v4.py`.
- **Trained models** (about 3 GB): not in this repository.
- **Total predictions** used by the anchoring experiments: included
  (`estimador/results/modelo_total/pred_total.npy`).

## Licence

GPL-3.0 (see `LICENSE`).
