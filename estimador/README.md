# Agix — conjunto de datos de esfuerzo real por tarea

Tubería de preparación de datos para el estimador de esfuerzo de Agix: predice
las **horas reales** que toma una tarea de software a partir de su nombre y
descripción.

## Resultado

`data/processed/agix_effort_clean.parquet` — 18 593 filas, **17 457 tareas de
software** con texto libre y **horas efectivamente cronometradas**. Ninguna
etiqueta procede de una conversión de puntos de historia.

| Fuente | Tareas | Texto | Objetivo | Mediana |
|---|---|---|---|---|
| TAWOS | 9 631 | título + descripción | `Timespent / 60` | 2,00 h |
| SiP | 7 826 | resumen + categoría | `HoursActual` | 2,20 h |

Particiones incluidas y verificadas: `split` (70/15/15 agrupado por texto y
casi-duplicados, 0 grupos con fuga) y `split_project` (proyectos completos
retenidos, 15,1 %).

## Línea base

Ridge sobre TF-IDF, objetivo logarítmico: MdMRE 0,702 y PRED(25) 0,184, un 12 %
mejor que predecir siempre la mediana. La curva de aprendizaje está plana desde
las 8 000 filas: el volumen no es la restricción, lo es el ruido de etiqueta.

El intervalo conforme sí resulta utilizable: cobertura empírica de 49,2 / 78,6 /
89,4 % frente a niveles nominales de 50 / 80 / 90 %, con bandas un 15 % más
estrechas que ignorando el texto. En proyectos nunca vistos el rendimiento no se
degrada (MdMRE 0,678), lo que respalda su uso como *prior* de arranque en frío.

## Hallazgos que condicionan la metodología

1. `Timespent` de TAWOS está en **minutos**, no en segundos. Prueba en
   `src/05_timespent_units.py`.
2. Los puntos de historia **no predicen** las horas reales: R² conjunto de
   0,058, y un convertidor calibrado por proyecto no supera a una constante.
   Por eso se descartó el cruce con Deep-SE. Evidencia en
   `src/07_sp_conversion_quality.py`.
3. Solo el 2,23 % de los issues de TAWOS lleva registro horario, y esa muestra
   está sesgada hacia tareas pequeñas, asignadas y completadas.
   `src/08_worklog_bias.py`.

Informe completo: `reports/auditoria_datos.html`.

## Ejecución

```
python -m venv .venv
.venv/Scripts/python.exe -m pip install pandas numpy scikit-learn scipy pyarrow matplotlib
```

Descargar a `data/raw/`:

- `TAWOS.sql.zip` — https://doi.org/10.5522/04/21308124 (Apache 2.0, 638 MB)
- Conjuntos de Derek Jones — https://github.com/Derek-Jones/Software-estimation-datasets

Luego ejecutar `src/01` … `src/13` en orden. El volcado de 4,31 GB se procesa
en flujo desde el zip; no requiere servidor MySQL ni espacio en disco.

## Fuentes

- Tawosi, V., Al-Subaihin, A., Moussa, R. y Sarro, F. (2022). *A Versatile
  Dataset of Agile Open Source Software Projects*. MSR 2022.
  doi:10.1145/3524842.3528029
- Jones, D. M. y Cullum, S. (2019). *A conversation around the analysis of the
  SiP effort estimation dataset*. arXiv:1901.01621
- Jones, D. M. (2023). *Small team estimating in story points; a project
  dataset*.
