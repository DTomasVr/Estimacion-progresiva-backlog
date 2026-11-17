"""E44 - Subir la metrica del proyecto: lo que queda es el desplazamiento comun.

E43 deja el total del proyecto en MdMRE 0,452 y R2 0,813 en frio, y la tabla por
tamano dice donde esta el limite: el error NO cae como 1 / raiz(n). Los proyectos
de mas de 400 tareas se quedan en MdMRE 0,251, no en cero. Eso significa que lo
que sobrevive a la suma no es el ruido de cada tarea -- ese si se cancela -- sino
un error COMUN a todo el proyecto: el modelo lo situa entero demasiado alto o
demasiado bajo. Es el mismo desplazamiento que E30 midio valiendo 0,115 de R2.

Un desplazamiento comun no se arregla con mas texto. Se arregla con UN dato del
proyecto. Y en PBE ese dato aparece solo: en cuanto el equipo cierra las
primeras tareas, el cronometro da horas reales. Asi que la pregunta util para el
producto no es "cuanto acierta en frio" sino CUANTAS TAREAS CERRADAS HACEN FALTA
para que el total sea fiable.

  0  constante                    el control
  1  frio con smearing            E43, el punto de partida
  2  con k tareas cerradas        se estima el desplazamiento del proyecto con k
                                  tareas reales y contraccion de James-Stein, que
                                  es imprescindible: con k pequeno el
                                  desplazamiento crudo es ruido y EMPEORA
  3  desplazamiento real          la mediana real de los residuos del proyecto.
                                  OJO: no es un techo. Es el desplazamiento que
                                  centra las tareas una a una, y centrar tareas no
                                  es lo mismo que acertar su suma. De hecho el
                                  brazo con 20 tareas cerradas lo supera, porque
                                  la contraccion deja un desplazamiento menor y
                                  para el TOTAL eso resulta mejor

Se prueba k = 1, 2, 3, 5, 10, 20 y se promedian 40 sorteos de que tareas son las
que resultaron cerradas primero, porque con k pequeno cual toque importa mucho.
Ademas se prueba un smearing que dependa del tamano predicho, por si el factor
global de 2,254 esta mal para los extremos.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 210)
OUT = Path("results/mejorar_total")
S = "=" * 118
MIN_TAREAS = 8
SORTEOS = 40
LAMBDA = 12          # contraccion de James-Stein, la misma que el artefacto v3


def mt(real, pred):
    real = np.asarray(real, float)
    pred = np.clip(np.asarray(pred, float), 1e-6, None)
    r = np.abs(pred - real) / real
    q = np.log(pred / real)
    m = float(np.median(q))
    lr = np.log(real)
    return dict(MAE_h=float(np.abs(pred - real).mean()),
                RMSE_h=float(np.sqrt(np.mean((pred - real) ** 2))),
                MdMRE=float(np.median(r)),
                R2_log=float(1 - ((lr - np.log(pred)) ** 2).mean() / lr.var()),
                PRED25=float(np.mean(r <= .25)), PRED50=float(np.mean(r <= .50)),
                factor2=float(np.mean(np.maximum(pred / real, real / pred) <= 2)),
                SSPB_pc=float(np.sign(m) * (np.exp(abs(m)) - 1) * 100))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet("data/processed/pbe_effort_v4.parquet")
    sw = df[df.is_software_task == 1].copy()
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    y = sw.effort_hours.to_numpy()
    ly = np.log(y)
    proy = sw.project.to_numpy()
    lp = np.load("results/analogia_cv/pred.npy")[:, 0]

    e = np.exp(ly - lp)
    g = pd.DataFrame({"p": proy, "e": e}).groupby("p").e.agg(["sum", "count"])
    fac = ((e.sum() - g["sum"]) / (len(e) - g["count"]))      # smearing sin el propio proyecto
    med = float(np.median(y))

    # smearing que depende del tamano predicho, por si el global falla en los extremos
    tramo = pd.qcut(lp, 6, labels=False, duplicates="drop")
    ft = pd.DataFrame({"t": tramo, "e": e}).groupby("t").e.mean()
    print("factor de smearing por tramo de tamano predicho: " +
          " · ".join(f"{v:.2f}" for v in ft.values))

    d = pd.DataFrame({"p": proy, "real": y, "lp": lp, "ly": ly,
                      "sm_t": ft.reindex(tramo).to_numpy()})
    tam = d.groupby("p").size()
    validos = tam[tam >= MIN_TAREAS].index
    d = d[d.p.isin(validos)].reset_index(drop=True)
    print(f"{len(validos)} proyectos · {len(d):,} tareas · "
          f"smearing global mediano {fac.median():.3f}")

    base = d.groupby("p").apply(
        lambda q: pd.Series(dict(real=q.real.sum(), n=len(q),
                                 frio=float((np.exp(q.lp) * fac[q.name]).sum()),
                                 frio_t=float((np.exp(q.lp) * q.sm_t).sum()),
                                 cte=len(q) * med)))
    # oraculo: el desplazamiento real del proyecto
    off_or = d.groupby("p").apply(lambda q: float(np.median(q.ly - q.lp)))
    base["oraculo"] = d.groupby("p").apply(
        lambda q: float((np.exp(q.lp + off_or[q.name]) * fac[q.name]).sum()))

    # k tareas cerradas, con contraccion
    rng = np.random.default_rng(0)
    idx = {p: q.index.to_numpy() for p, q in d.groupby("p")}
    for k in (1, 2, 3, 5, 10, 20):
        acum = np.zeros(len(base))
        for s in range(SORTEOS):
            tot = []
            for p in base.index:
                i = idx[p]
                if len(i) <= k:
                    tot.append(base.loc[p, "frio"])
                    continue
                sel = rng.choice(i, k, replace=False)
                bruto = float(np.median(d.ly[sel] - d.lp[sel]))
                off = bruto * k / (k + LAMBDA)
                tot.append(float((np.exp(d.lp[i] + off) * fac[p]).sum()))
            acum += np.array(tot)
        base[f"k{k}"] = acum / SORTEOS

    brazos = {"0 · constante (n x mediana)": "cte",
              "1 · frio, smearing global": "frio",
              "1b · frio, smearing por tramo": "frio_t",
              "2 · con 1 tarea cerrada": "k1",
              "2 · con 2 tareas cerradas": "k2",
              "2 · con 3 tareas cerradas": "k3",
              "2 · con 5 tareas cerradas": "k5",
              "2 · con 10 tareas cerradas": "k10",
              "2 · con 20 tareas cerradas": "k20",
              "3 · desplazamiento mediano real del proyecto": "oraculo"}
    r = pd.DataFrame([dict(brazo=n, **mt(base.real, base[c])) for n, c in brazos.items()])
    print(f"\n{S}\nTOTAL DEL PROYECTO SEGUN CUANTO HISTORIAL HAY  ({len(base)} proyectos)\n{S}")
    print(r[["brazo", "MAE_h", "RMSE_h", "MdMRE", "R2_log", "PRED25", "PRED50",
             "factor2", "SSPB_pc"]].round(4).to_string(index=False))
    f = r.set_index("brazo")
    print(f"\n  en frio {f.loc['1 · frio, smearing global', 'PRED25']:.1%} de los proyectos "
          f"caen dentro del +-25 %; con 5 tareas cerradas "
          f"{f.loc['2 · con 5 tareas cerradas', 'PRED25']:.1%} y con 20, "
          f"{f.loc['2 · con 20 tareas cerradas', 'PRED25']:.1%}")
    print("  el brazo 3 NO es un techo: con 20 tareas cerradas se le gana, porque la")
    print("  contraccion deja un desplazamiento mas pequeno y para un TOTAL eso va mejor")
    print("  que centrar cada tarea por separado")
    r.to_csv(OUT / "resumen.csv", index=False)
    base.to_csv(OUT / "totales.csv")
    json.dump({"lambda": LAMBDA, "sorteos": SORTEOS, "min_tareas": MIN_TAREAS},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/resumen.csv")


if __name__ == "__main__":
    main()
