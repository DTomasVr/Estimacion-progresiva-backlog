"""Assemble the delivery folder: datasets, scripts, and two executed notebooks.

The notebooks are executed here rather than shipped empty. A reader who never
runs a cell still sees every table and figure, and -- more to the point -- the
build fails if any documented step stops reproducing against the files being
delivered, so the narrative cannot quietly drift away from the data.

Execution order is fixed: the v3 notebook writes the column dictionary that the
v4 notebook reads to inherit its descriptions.
"""
import shutil
import sys
import time
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

sys.path.insert(0, "src")

RAIZ = Path("entrega")
DEST = {
    "datos": RAIZ / "datos",
    "inter": RAIZ / "datos" / "intermedios",
    "nb": RAIZ / "notebooks",
    "scripts": RAIZ / "scripts",
    "figuras": RAIZ / "figuras",
    "tablas": RAIZ / "tablas",
    "origen": RAIZ / "tablas_origen",
}

# script de origen -> subcarpeta de la entrega, en orden de tuberia
SCRIPTS = [
    ("src/tawos_sql.py", "00_lectores"),
    ("src/jira_archive.py", "00_lectores"),
    ("src/01_scan_schema.py", "01_extraccion"),
    ("src/02_extract_tables.py", "01_extraccion"),
    ("src/05_timespent_units.py", "01_extraccion"),
    ("src/06_build_gold.py", "01_extraccion"),
    ("src/07_sp_conversion_quality.py", "01_extraccion"),
    ("src/08_worklog_bias.py", "01_extraccion"),
    ("src/09_build_unified.py", "01_extraccion"),
    ("src/50_extract_jira.py", "01_extraccion"),
    ("src/51_perfil_jira.py", "01_extraccion"),
    ("src/10_data_quality_report.py", "02_limpieza"),
    ("src/12_clean_dataset.py", "02_limpieza"),
    ("src/25_review_target.py", "02_limpieza"),
    ("src/26_clean_v2.py", "02_limpieza"),
    ("src/39_build_v3.py", "02_limpieza"),
    ("src/52_build_v4.py", "02_limpieza"),
    ("src/41_split3.py", "03_particiones"),
    ("src/15_encode_sbert.py", "04_codificacion"),
    ("src/53_encode_v4.py", "04_codificacion"),
    ("src/exp_common.py", "05_evaluacion"),
    ("src/features.py", "05_evaluacion"),
]

DATASETS = [
    ("data/processed/agix_effort_v3.parquet", "datos"),
    ("data/processed/agix_effort_v3.csv", "datos"),
    ("data/processed/agix_effort_v4.parquet", "datos"),
]
INTERMEDIOS = [
    "data/processed/agix_effort_dataset.parquet",
    "data/processed/agix_effort_clean.parquet",
    "data/processed/agix_effort_v2.parquet",
    "data/processed/tawos_gold.parquet",
    "data/processed/tawos_sp_hours.parquet",
    "data/raw/jones/SiP/Sip-task-info.csv",
    "data/raw/jones/SiP/est-act-dates.csv",
]
ORIGEN = [
    "results/jira_publico_cobertura.csv",
    "results/jira_publico_conteos.json",
    "data/interim/tawos_table_sizes.json",
]


def copiar():
    for d in DEST.values():
        d.mkdir(parents=True, exist_ok=True)
    n = 0
    for origen, destino in DATASETS:
        p = Path(origen)
        if p.exists():
            shutil.copy2(p, DEST[destino] / p.name)
            n += 1
        else:
            print(f"  AVISO: falta {origen}")
    for origen in INTERMEDIOS:
        p = Path(origen)
        if p.exists():
            shutil.copy2(p, DEST["inter"] / p.name)
            n += 1
        else:
            print(f"  AVISO: falta {origen}")
    jp_src = Path("data/interim/jira_publico")
    if jp_src.exists():
        jp_dst = DEST["inter"] / "jira_publico"
        jp_dst.mkdir(exist_ok=True)
        for p in jp_src.glob("*.parquet"):
            shutil.copy2(p, jp_dst / p.name)
            n += 1
    for origen in ORIGEN:
        p = Path(origen)
        if p.exists():
            shutil.copy2(p, DEST["origen"] / p.name)
            n += 1
    for origen, sub in SCRIPTS:
        p = Path(origen)
        if not p.exists():
            print(f"  AVISO: falta {origen}")
            continue
        d = DEST["scripts"] / sub
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, d / p.name)
        n += 1
    print(f"  {n} ficheros copiados")


def csv_v4():
    """v4 as CSV too, gzipped: the plain file is ~250 MB and nobody wants that."""
    import pandas as pd
    destino = DEST["datos"] / "agix_effort_v4.csv.gz"
    if destino.exists():
        return
    df = pd.read_parquet("data/processed/agix_effort_v4.parquet")
    df.to_csv(destino, index=False, compression="gzip", encoding="utf-8")
    print(f"  escrito {destino.name} "
          f"({destino.stat().st_size/1e6:.0f} MB comprimido)")


def ejecutar(nb, nombre):
    destino = DEST["nb"] / nombre
    nbf.write(nb, destino)
    print(f"\n  ejecutando {nombre} ...", flush=True)
    t0 = time.time()
    cliente = NotebookClient(nb, timeout=1800, kernel_name="python3",
                            resources={"metadata": {"path": str(DEST["nb"])}})
    cliente.execute()
    nbf.write(nb, destino)
    errores = [c for c in nb.cells
               if any(o.get("output_type") == "error"
                      for o in c.get("outputs", []))]
    estado = "OK" if not errores else f"{len(errores)} CELDAS CON ERROR"
    print(f"  {nombre}: {estado}  ({time.time()-t0:.0f}s)")
    for c in errores:
        o = next(o for o in c.outputs if o.get("output_type") == "error")
        print(f"    {o.get('ename')}: {str(o.get('evalue'))[:200]}")
        for l in o.get("traceback", [])[-4:]:
            print("      " + l[:200])
    return not errores


LEEME = """# Entrega de datos · componente de estimación de esfuerzo de Agix

Conjuntos **v3** y **v4**, los scripts que los producen, y dos cuadernos que
documentan y **verifican** cada paso del preprocesamiento.

## Qué hay aquí

```
entrega/
├── notebooks/
│   ├── 01_dataset_v3.ipynb      TAWOS + SiP: adquisición, limpieza D1–D10 y E1–E4, particiones
│   └── 02_dataset_v4.ipynb      Public Jira: adquisición, descarte D1–D6, criterios de mezcla
├── datos/
│   ├── agix_effort_v3.parquet   18 474 filas × 49 columnas  (17 350 tareas de software)
│   ├── agix_effort_v3.csv
│   ├── agix_effort_v4.parquet   90 975 filas × 52 columnas  (89 851 tareas de software)
│   ├── agix_effort_v4.csv.gz
│   ├── diccionario_columnas_v3.csv
│   ├── diccionario_columnas_v4.csv
│   └── intermedios/             v0, v1, v2, TAWOS gold, CSV de SiP, bloque crudo de Jira
├── scripts/
│   ├── 00_lectores/             lectura en flujo del mysqldump y del mongodump
│   ├── 01_extraccion/           extracción y perfilado de cada fuente
│   ├── 02_limpieza/             v1, v2, v3 y v4
│   ├── 03_particiones/          partición cronológica en tres bloques
│   ├── 04_codificacion/         codificación SBERT
│   └── 05_evaluacion/           utilidades comunes de métricas y rasgos
├── figuras/                     todas las figuras en PNG (las generan los cuadernos)
├── tablas/                      todas las tablas en CSV (las generan los cuadernos)
└── tablas_origen/               tablas de apoyo producidas por la tubería original
```

## Cómo leerlo

Los dos cuadernos **ya están ejecutados**: se pueden leer sin instalar nada. Las
figuras y tablas también quedan sueltas en `figuras/` y `tablas/` para citarlas
en el documento de tesis.

Para volver a ejecutarlos:

```bash
pip install pandas pyarrow numpy matplotlib jupyter
jupyter lab notebooks/
```

Ejecutar **primero** `01_dataset_v3.ipynb`: escribe el diccionario de columnas
que `02_dataset_v4.ipynb` reutiliza.

## El objetivo

`effort_hours` es **siempre tiempo de reloj**, nunca una conversión desde puntos
de historia. Procede de `Timespent/60` en TAWOS (la unidad se demuestra en el
cuaderno v3, §3), de `HoursActual` en SiP, y de `timespent/3600` en el corpus
público de Jira.

## Resumen de descartes

| paso | entrada | salida | perdidas |
|---|---|---|---|
| v0 → v1 | 20 248 | 18 593 | 1 655 |
| v1 → v2 | 18 593 | 18 474 | 119 |
| v2 → v3 | 18 474 | 18 474 | 0 (solo añade 21 columnas) |
| Jira público, crudo → limpio | 78 634 | 72 501 | 6 133 |
| v3 + Jira público → v4 | 18 474 | 90 975 | — |

Los criterios están en el cuaderno v3 (tablas 8 y 12) y en el v4 (tabla 23),
cada uno con su motivo y su verificación contra el fichero entregado.

## Criterio de mezcla que gobierna v4

**El bloque de prueba de v3 queda intacto, fila por fila.** El material nuevo
solo entra en entrenamiento, en validación, o en un bloque *externo* de 150
proyectos retenidos enteros. Si al añadir datos se cambiara también el examen,
la comparación mediría la dificultad de dos exámenes distintos en vez de la
calidad de dos modelos. Las seis comprobaciones que lo garantizan están en la
tabla 28 del cuaderno v4, y el cuaderno falla si alguna no pasa.

## Advertencia de uso

El bloque público tiene una mediana de esfuerzo de 0,83 h frente a 2,00 h de
TAWOS y SiP. Entrenar mezclando sin más desplaza las predicciones hacia abajo.
Quien reutilice v4 debe decidir explícitamente qué hace con esa diferencia.

## Licencias de las fuentes

- **TAWOS** — Tawosi et al., MSR 2022. CC BY 4.0.
- **SiP** — Jones, 2020. CC BY 4.0.
- **The Public Jira Dataset** — Montgomery, Lüders y Maalej, MSR 2022.
  Datos CC BY 4.0, código MIT. Zenodo 15719919.
"""


def main():
    print("=" * 70)
    print("ENTREGA · copiando ficheros")
    print("=" * 70)
    copiar()
    csv_v4()
    (RAIZ / "LEEME.md").write_text(LEEME, encoding="utf-8")
    print(f"  escrito {RAIZ / 'LEEME.md'}")

    print("\n" + "=" * 70)
    print("ENTREGA · generando y ejecutando cuadernos")
    print("=" * 70)
    import entrega_nb_v3
    import entrega_nb_v4
    ok3 = ejecutar(entrega_nb_v3.notebook(), "01_dataset_v3.ipynb")
    ok4 = ejecutar(entrega_nb_v4.notebook(), "02_dataset_v4.ipynb")

    print("\n" + "=" * 70)
    total = sum(p.stat().st_size for p in RAIZ.rglob("*") if p.is_file())
    print(f"carpeta: {RAIZ.resolve()}")
    print(f"tamaño : {total/1e6:,.0f} MB en "
          f"{sum(1 for p in RAIZ.rglob('*') if p.is_file()):,} ficheros")
    print(f"figuras: {len(list(DEST['figuras'].glob('*.png')))}")
    print(f"tablas : {len(list(DEST['tablas'].glob('*.csv')))}")
    if not (ok3 and ok4):
        sys.exit("algun cuaderno fallo al ejecutarse")
    print("ambos cuadernos ejecutaron sin errores")


if __name__ == "__main__":
    main()
