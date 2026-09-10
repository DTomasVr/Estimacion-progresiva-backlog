"""Profile the extracted public-Jira rows before deciding what is usable.

The questions that decide the merge: is timespent really seconds, how much of
it is quantised to whole Jira days, how much is a container rolling up its
children's work, and how much of it we already have through TAWOS.
"""
import glob
import sys

import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
IN = "data/interim/jira_publico"


def cargar():
    fr = []
    for p in sorted(glob.glob(f"{IN}/*.parquet")):
        d = pd.read_parquet(p)
        d["fuente"] = p.split("\\")[-1].split("/")[-1].replace(".parquet", "")
        fr.append(d)
    return pd.concat(fr, ignore_index=True)


def main():
    df = cargar()
    print(f"filas {len(df):,}  columnas {len(df.columns)}")
    print(df.columns.tolist(), "\n")

    df["horas"] = df.timespent / 3600.0
    print("--- timespent crudo (unidades) ---")
    print(df.timespent.describe([.01, .05, .25, .5, .75, .95, .99]).round(1).to_string())
    print("\nvalores mas frecuentes de timespent (seg):")
    print(df.timespent.value_counts().head(15).to_string())

    print("\n--- horas ---")
    print(df.horas.describe([.01, .05, .25, .5, .75, .95, .99]).round(3).to_string())
    for lo, hi in [(0, .05), (.05, .25), (.25, 1), (1, 8), (8, 40), (40, 200), (200, 1e9)]:
        n = int(((df.horas >= lo) & (df.horas < hi)).sum())
        print(f"  [{lo:>6} , {hi:>6}) h : {n:>7,}  {100*n/len(df):5.2f}%")

    print("\n--- cuantizacion ---")
    for etiqueta, seg in [("1h", 3600), ("4h (medio dia)", 14400),
                          ("8h (1 dia jira)", 28800), ("1 semana (5d)", 144000)]:
        print(f"  exactamente {etiqueta:16s}: {int((df.timespent == seg).sum()):>6,}")
    mult = (df.timespent % 28800 == 0).sum()
    print(f"  multiplo exacto de 8h : {mult:,} ({100*mult/len(df):.1f}%)")
    mult1 = (df.timespent % 3600 == 0).sum()
    print(f"  multiplo exacto de 1h : {mult1:,} ({100*mult1/len(df):.1f}%)")

    print("\n--- contenedores (agregado > propio) ---")
    a = df.agg_timespent.fillna(0)
    print(f"  agg > timespent  : {int((a > df.timespent).sum()):,}")
    print(f"  subtareas > 0    : {int((df.n_subtareas.fillna(0) > 0).sum()):,}")
    print(f"  ambos            : {int(((a > df.timespent) & (df.n_subtareas.fillna(0) > 0)).sum()):,}")

    print("\n--- tipo de issue ---")
    print(df.issuetype.value_counts().head(20).to_string())

    print("\n--- texto ---")
    print(f"  sin summary     : {int(df.summary.isna().sum()):,}")
    print(f"  sin description : {int(df.description.isna().sum()):,}")
    long = df.summary.fillna("").str.split().str.len()
    print(f"  palabras summary: mediana {long.median():.0f}  p95 {long.quantile(.95):.0f}")

    print("\n--- con estimacion humana ---")
    amb = df[df.estimate_orig.notna() & (df.estimate_orig > 0)]
    print(f"  filas con estimacion y tiempo: {len(amb):,}")
    print(amb.fuente.value_counts().to_string())
    r = amb.timespent / amb.estimate_orig
    print(f"\n  razon real/estimado: mediana {r.median():.3f}  "
          f"p25 {r.quantile(.25):.3f}  p75 {r.quantile(.75):.3f}")
    mre = (amb.timespent - amb.estimate_orig).abs() / amb.timespent
    print(f"  MdMRE de la estimacion humana: {mre.median():.4f}  "
          f"PRED25 {float((mre <= .25).mean()):.4f}")

    print("\n--- solape con TAWOS ---")
    v3 = pd.read_parquet("data/processed/agix_effort_v3.parquet",
                         columns=["task_id", "source", "text_hash"])
    claves = set(v3[v3.source == "TAWOS"].task_id.str.replace("TAWOS-", "", regex=False))
    dup = df.key.isin(claves)
    print(f"  claves TAWOS en v3        : {len(claves):,}")
    print(f"  filas jira que ya estan   : {int(dup.sum()):,}")
    if dup.any():
        print(df[dup].fuente.value_counts().to_string())

    print("\n--- proyectos ---")
    print(f"  proyectos distintos: {df.project_key.nunique():,}")
    vc = df.project_key.value_counts()
    print(f"  con >=50 filas     : {int((vc >= 50).sum()):,}  "
          f"cubren {int(vc[vc >= 50].sum()):,} filas")
    print(vc.head(15).to_string())


if __name__ == "__main__":
    main()
