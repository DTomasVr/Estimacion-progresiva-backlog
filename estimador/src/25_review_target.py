"""Full review of the target variable against Agix's actual use case.

Agix measures ONE developer's active time on ONE task, via a tracker they start
and stop. Every row in the training data must mean the same thing, or the model
learns a quantity it will never be asked to predict.

Checks:
  1. multi-developer aggregation (SiP has the breakdown; TAWOS does not)
  2. rows that are not a single task at all (epics, projects, plans)
  3. quantisation of the target -- are the hours real measurements or estimates
     typed in round numbers?
  4. temporal and project concentration
  5. distribution mismatch between the two sources
"""
import sys

import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
CLEAN = "data/processed/agix_effort_clean.parquet"

CONTAINER = (r"\b(project|phase|programme|initiative|roadmap|plan|planning|"
             r"meeting|workshop|kick.?off|various|misc|miscellaneous|"
             r"catch.?all|assorted|month.?end|period close|ongoing|routine|"
             r"housekeeping)\b")


def head(n, t):
    print(f"\n{'='*78}\n{n}. {t}\n{'='*78}")


def main():
    d = pd.read_parquet(CLEAN)
    sw = d[d.is_software_task == 1].copy()
    sw["banda"] = pd.cut(sw.effort_hours, [0, 1, 4, 16, 40, np.inf],
                         labels=["<1h", "1-4h", "4-16h", "16-40h", ">40h"])
    print(f"filas de software: {len(sw):,}")

    # ---------------------------------------------------------------
    head(1, "AGREGACION POR MULTIPLES DESARROLLADORES")
    t = pd.read_csv("data/raw/jones/SiP/Sip-task-info.csv", encoding="cp1252")
    g = t.groupby("TaskNumber")
    ndev = g.DeveloperID.nunique().rename("n_dev")
    dmax = g.DeveloperHoursActual.max().rename("dev_max_h")
    sip = sw[sw.source == "SiP"].copy()
    sip["tn"] = sip.task_id.str.replace("SIP-", "", regex=False).astype(int)
    sip = sip.join(ndev, on="tn").join(dmax, on="tn")
    multi = sip[sip.n_dev > 1]
    print(f"  SiP: {len(multi):,} de {len(sip):,} filas con mas de un "
          f"desarrollador ({len(multi)/len(sip)*100:.1f}%)")
    print(f"  factor de inflacion mediano (total / mayor contribuidor): "
          f"{(multi.effort_hours/multi.dev_max_h).median():.2f}x")
    print("\n  si se usara el mayor contribuidor en vez del total:")
    print(f"    mediana  {sip.effort_hours.median():.2f} h -> "
          f"{sip.dev_max_h.median():.2f} h")
    print(f"    p95      {sip.effort_hours.quantile(.95):.2f} h -> "
          f"{sip.dev_max_h.quantile(.95):.2f} h")
    print(f"    maximo   {sip.effort_hours.max():.2f} h -> "
          f"{sip.dev_max_h.max():.2f} h")
    print(f"    desv.tip.log {np.std(np.log(sip.effort_hours)):.3f} -> "
          f"{np.std(np.log(sip.dev_max_h.clip(lower=.01))):.3f}")
    print("\n  TAWOS: el volcado solo publica Timespent agregado por incidencia,")
    print("         no el desglose por desarrollador. La misma inflacion existe")
    print("         pero no es observable ni corregible.")

    # ---------------------------------------------------------------
    head(2, "FILAS QUE NO SON UNA TAREA")
    low = sw.text.str.lower()
    is_epic = sw.raw_kind.eq("Epic")
    has_cont = low.str.contains(CONTAINER, regex=True, na=False)
    for name, m in [("Epic (TAWOS)", is_epic),
                    ("lenguaje de contenedor", has_cont),
                    ("contenedor Y > 40 h", has_cont & (sw.effort_hours > 40)),
                    ("contenedor Y > 16 h", has_cont & (sw.effort_hours > 16))]:
        print(f"  {name:26s} {int(m.sum()):5,}  mediana "
              f"{sw.loc[m, 'effort_hours'].median() if m.any() else float('nan'):6.2f} h")

    # ---------------------------------------------------------------
    head(3, "CUANTIZACION DEL OBJETIVO")
    for src in ["TAWOS", "SiP"]:
        e = sw[sw.source == src].effort_hours
        print(f"  {src}:")
        for step, lbl in [(1.0, "hora entera"), (0.5, "media hora"),
                          (0.25, "cuarto de hora")]:
            frac = float(np.mean(np.abs(e / step - np.round(e / step)) < 1e-6))
            print(f"    multiplo de {lbl:16s}: {frac*100:5.1f}%")
        print(f"    valores distintos: {e.nunique():,} en {len(e):,} filas")
        print(f"    los 6 mas frecuentes: "
              f"{', '.join(f'{v}h x{c}' for v, c in e.value_counts().head(6).items())}")

    # ---------------------------------------------------------------
    head(4, "CONCENTRACION")
    top = sw.project.value_counts()
    print(f"  proyectos: {len(top)}   top-1 {top.index[0]} "
          f"{top.iloc[0]/len(sw)*100:.1f}%   top-5 {top.head(5).sum()/len(sw)*100:.1f}%")
    yr = sw.created.dt.year.value_counts().sort_index()
    yr = yr[yr > 200]
    print(f"  por ano (>200 filas): "
          f"{', '.join(f'{int(k)}:{v}' for k, v in yr.items())}")

    # ---------------------------------------------------------------
    head(5, "DIFERENCIA ENTRE FUENTES")
    st = sw.groupby("source").effort_hours.describe(
        percentiles=[.25, .5, .75, .95]).round(2)
    print(st.to_string())
    print("\n  distribucion por banda (%):")
    print((pd.crosstab(sw.banda, sw.source, normalize="columns") * 100)
          .round(1).to_string())


if __name__ == "__main__":
    main()
