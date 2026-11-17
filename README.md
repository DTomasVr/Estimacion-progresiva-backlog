# Generación de backlogs y estimación de esfuerzo progresiva

Código, experimentos y resultados del artículo *Generación de backlogs y estimación de esfuerzo
progresiva en proyectos de software ágiles con modelos de lenguaje pequeños*.

El modelo propuesto convierte la descripción de un producto en un backlog de historias de usuario
con sus tareas, estima las horas de cada tarea y da un total del proyecto con un intervalo del
80 % que se afina a medida que se cierran tareas y que el equipo aporta lo que sabe.

## Estructura

| carpeta | qué contiene |
|---|---|
| `estimador/` | Datos derivados, experimentos y modelo de estimación de esfuerzo. Conserva el historial completo de su desarrollo (desde el 30/08/2026). |
| `generador/` | Pipeline de generación de historias y tareas, sus experimentos y resultados. |

Cada experimento es un script numerado (`estimador/src/NNN_*.py`, `generador/src/NN_*.py`) que
escribe sus resultados en `results/`. Las cifras del artículo se leen de esos archivos, y sus
tablas se generan con `estimador/src/273_cuadros_paper.py`.

## Tablas del artículo y su origen

| tabla | script | resultado |
|---|---|---|
| I · el total según las tareas cerradas | `estimador/src/275_curva_anclaje_dos_criterios.py` | `estimador/results/curva_anclaje_dos_criterios/` |
| II · estimaciones del usuario y tareas cerradas | `estimador/src/268_tres_estados.py` | `estimador/results/tres_estados/` |
| III · estimación por tarea | experimentos del bloque de prueba | `estimador/results/baselines_literatura.csv` |
| IV · estimación frente a LLM | `estimador/src/259_banco_api.py`, `estimador/src/261_tablas_llm_api.py` | `estimador/results/llm_bench/` |
| V · generación frente a LLM | `estimador/src/272_banco_generacion.py` | `estimador/results/banco_generacion/` |

## Datos

Los conjuntos de partida son públicos: TAWOS, The Public Jira Dataset y SiP. Los modelos
entrenados (unos 3 GB) y los datos procesados más pesados no están en este repositorio.

## Licencia

GPL-3.0 (ver `LICENSE`).
