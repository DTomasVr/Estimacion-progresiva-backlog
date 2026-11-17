"""E76 - ¿Sirve que el equipo nos diga horas de tareas ANTERIORES, de otros proyectos?

El anclaje medido hasta ahora usa tareas cerradas DEL PROPIO proyecto, asi que solo
sirve cuando el proyecto ya arranco. La pregunta de producto es otra: si al crear el
backlog le pedimos al jefe de proyecto «dime cuanto tardo tu equipo en estas tareas
de otros proyectos», ¿eso ya mejora la estimacion en frio?

No es lo mismo y no estaba medido. Lo que el anclaje recupera es la ESCALA del equipo
(horas por tarea), y la escala podria ser una propiedad del equipo mas que del
proyecto. Si lo es, las horas de otros proyectos del mismo equipo valen; si no, no.

  A  en frio                                  lo que hay
  B  k tareas cerradas DEL PROPIO proyecto    el mecanismo ya medido (E67)
  C  k tareas de OTRO proyecto de la MISMA     <- lo que se propone
     organizacion
  D  k tareas de otra organizacion al azar     control: si C == D, la organizacion
                                               no aporta nada y es un efecto espurio

Se evalua sobre los proyectos que tienen al menos un hermano en su organizacion,
para que los cuatro brazos corran sobre el mismo conjunto. Se promedia la METRICA
sobre 60 sorteos, no la prediccion.

En B las k tareas se SABEN y salen de la estimacion; en C y D no se sabe nada de
este proyecto, asi que se estiman todas sus tareas y se reescalan.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
OUT = Path("results/horas_del_equipo")
S = "=" * 104
LAMBDA, SORTEOS, KS = 12, 60, (1, 3, 5, 10, 20)


def mt(real, pred):
    real, pred = np.asarray(real, float), np.clip(np.asarray(pred, float), 1e-6, None)
    r = np.abs(pred - real) / real
    m = float(np.median(np.log(pred / real)))
    return dict(MdMRE=float(np.median(r)), PRED25=float(np.mean(r <= .25)),
                factor2=float(np.mean(np.maximum(pred / real, real / pred) <= 2)),
                MAE_h=float(np.abs(pred - real).mean()),
                SSPB_pc=float(np.sign(m) * (np.exp(abs(m)) - 1) * 100))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet("data/processed/pbe_effort_v4.parquet")
    sw = df[df.is_software_task == 1]
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    pred = np.clip(np.load("results/modelo_total/pred_total.npy"), 1e-6, None)
    assert len(pred) == len(sw)
    tam = sw.groupby("project").size()
    m = sw.project.isin(tam[tam >= 8].index).to_numpy()
    sw, pred = sw[m].reset_index(drop=True), pred[m]

    proy = sw.project.to_numpy()
    real = sw.effort_hours.to_numpy(float)
    nom = list(pd.unique(proy))
    idx = {p: np.flatnonzero(proy == p) for p in nom}
    org = {p: (p.split(":")[0] if ":" in p else p) for p in nom}
    por_org = {}
    for p in nom:
        por_org.setdefault(org[p], []).append(p)

    # solo los que tienen hermano en su organizacion
    ev = [p for p in nom if len(por_org[org[p]]) >= 2]
    real_tot = np.array([real[idx[p]].sum() for p in ev])
    print(f"{len(nom)} proyectos · {len(ev)} con hermano en su organizacion · "
          f"{len(set(org[p] for p in ev))} organizaciones · {time.time() - t0:.0f} s")

    # fuentes de tareas ajenas
    hermanas = {p: np.concatenate([idx[q] for q in por_org[org[p]] if q != p])
                for p in ev}
    ajenas = {p: np.concatenate([idx[q] for q in nom if org[q] != org[p]])
              for p in ev}

    def total_B(p, k, rng):
        i = idx[p]
        if len(i) <= k:
            return pred[i].sum()
        sel = rng.choice(i, k, replace=False)
        resto = np.setdiff1d(i, sel)
        off = np.log(real[sel].sum() / pred[sel].sum()) * k / (k + LAMBDA)
        return real[sel].sum() + (pred[resto] * np.exp(off)).sum()

    def total_ajeno(p, k, rng, fuente):
        f = fuente[p]
        sel = rng.choice(f, min(k, len(f)), replace=False)
        off = np.log(real[sel].sum() / pred[sel].sum()) * k / (k + LAMBDA)
        return (pred[idx[p]] * np.exp(off)).sum()

    filas = [dict(k=0, brazo="A · en frio",
                  **mt(real_tot, [pred[idx[p]].sum() for p in ev]))]
    for k in KS:
        for etq, fn in (("B · tareas cerradas DEL PROPIO proyecto", total_B),
                        ("C · tareas de otro proyecto del MISMO equipo",
                         lambda p, k, r: total_ajeno(p, k, r, hermanas)),
                        ("D · tareas de otra organizacion (control)",
                         lambda p, k, r: total_ajeno(p, k, r, ajenas))):
            ms = []
            for s in range(SORTEOS):
                rng = np.random.default_rng(1000 * k + s)
                ms.append(mt(real_tot, [fn(p, k, rng) for p in ev]))
            filas.append(dict(k=k, brazo=etq,
                              **{c: float(np.mean([x[c] for x in ms])) for c in ms[0]}))
        print(f"  k={k} listo · {time.time() - t0:.0f} s", flush=True)

    t = pd.DataFrame(filas)
    print(f"\n{S}\n¿SIRVEN LAS HORAS DE OTROS PROYECTOS DEL MISMO EQUIPO? "
          f"({len(ev)} proyectos)\n{S}")
    for c in ("PRED25", "MdMRE"):
        print(f"\n{c}:")
        piv = t[t.k > 0].pivot(index="brazo", columns="k", values=c)
        piv.loc["A · en frio"] = float(t.iloc[0][c])
        print(piv.round(4).to_string())

    a = float(t.iloc[0].PRED25)
    b5 = float(t[(t.k == 5) & t.brazo.str.startswith("B")].PRED25.iloc[0])
    c5 = float(t[(t.k == 5) & t.brazo.str.startswith("C")].PRED25.iloc[0])
    d5 = float(t[(t.k == 5) & t.brazo.str.startswith("D")].PRED25.iloc[0])
    print(f"\n{S}\nLECTURA (con 5 tareas)\n{S}")
    print(f"  en frio                              ±25 % {a:.1%}")
    print(f"  5 del propio proyecto                ±25 % {b5:.1%}  ({b5 - a:+.1%})")
    print(f"  5 de otro proyecto del mismo equipo  ±25 % {c5:.1%}  ({c5 - a:+.1%})")
    print(f"  5 de otra organizacion (control)     ±25 % {d5:.1%}  ({d5 - a:+.1%})")
    # La comparacion que decide es contra A (no hacer nada), no contra D. Que C le
    # gane al control solo dice que la organizacion lleva ALGO de senal; si C no le
    # gana a A, esa senal no compensa y la respuesta de producto es que no se pida.
    print()
    if c5 > a:
        print(f"  SIRVE: las horas de otros proyectos del equipo suben el ±25 % "
              f"{(c5 - a) * 100:+.1f} puntos sobre el frio")
    else:
        print(f"  NO SIRVE: las horas de otros proyectos del equipo DEJAN PEOR que "
              f"no hacer nada ({(c5 - a) * 100:+.1f} puntos)")
        print(f"  (la organizacion lleva algo de senal -C le gana al control D por "
              f"{(c5 - d5) * 100:.1f} puntos- pero no compensa reescalar con ella)")
    print(f"  lo que si sirve son las tareas del PROPIO proyecto: {(b5 - a) * 100:+.1f} "
          f"puntos")
    t.to_csv(OUT / "resumen.csv", index=False)
    json.dump({"proyectos": len(ev), "sorteos": SORTEOS, "lambda": LAMBDA},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/resumen.csv · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
