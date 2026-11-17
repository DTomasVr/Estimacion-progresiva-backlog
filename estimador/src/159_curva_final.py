"""E53 - La curva que va al deck: cuantas tareas cerradas hacen falta.

Cuando el proyecto ya tiene tareas cerradas, el cronometro da horas reales y se
puede corregir el desplazamiento del proyecto. Se mide cuanto mejora el total
segun cuantas tareas lleve cerradas el equipo.

TRES COSAS QUE HAY QUE HACER BIEN, Y LAS TRES SE HICIERON MAL PRIMERO:

  1 EL ESTADISTICO DEL ANCLAJE. El desplazamiento no se estima con la mediana de
    los cocientes. Este estimador predice la MEDIA condicional, y la mediana de
    log(real/estimado) es sistematicamente negativa con una distribucion
    asimetrica: al anclar con ella, cuantas mas tareas cerradas mas se hundia la
    estimacion (con 20 tareas el sesgo llegaba a -45,7 %). Lo que corresponde
    para un total es el cociente de SUMAS.

  2 LAS TAREAS YA CERRADAS SE SABEN, NO SE ESTIMAN. Si el equipo cerro tres
    tareas, sus horas son un dato, no una prediccion. El total es la suma de lo
    conocido mas la estimacion corregida de lo que queda. Sumar la estimacion de
    las tareas cerradas es tirar informacion.

  3 SE PROMEDIA LA METRICA, NO LA PREDICCION. Cuales son las tres tareas que
    resultaron cerrarse primero es un sorteo, asi que hay que repetirlo. Pero
    promediar las PREDICCIONES de 40 sorteos y medir sobre esa media da una cifra
    falsamente buena, porque el promedio cancela la varianza del sorteo: sale lo
    que se conseguiria con 40 x k tareas, no con k. Lo que corresponde es medir
    en cada sorteo y promediar las METRICAS, y dar tambien su dispersion.
"""
import json
import os
import sys
import warnings
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "10"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 215)
OUT = Path("results/curva_final")
S = "=" * 122
MIN_TAREAS, SORTEOS, LAMBDA = 8, 60, 12


def mt(real, pred):
    real = np.asarray(real, float)
    pred = np.clip(np.asarray(pred, float), 1e-6, None)
    r = np.abs(pred - real) / real
    q = np.log(pred / real)
    m = float(np.median(q))
    lr = np.log(real)
    return dict(MAE_h=float(np.abs(pred - real).mean()),
                RMSE_h=float(np.sqrt(np.mean((pred - real) ** 2))),
                MdMRE=float(np.median(r)), MMRE=float(r.mean()),
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
    proy = sw.project.to_numpy()
    est = np.clip(np.load("results/modelo_total/pred_total.npy"), 1e-6, None)
    d = pd.DataFrame({"p": proy, "real": y, "est": est})
    tam = d.groupby("p").size()
    d = d[d.p.isin(tam[tam >= MIN_TAREAS].index)].reset_index(drop=True)
    nom = list(pd.unique(d.p))
    idx = {p: np.flatnonzero(d.p.to_numpy() == p) for p in nom}
    re_, es_ = d.real.to_numpy(), d.est.to_numpy()
    g = d.groupby("p")
    real = g.real.sum().reindex(nom).to_numpy()
    n_t = g.size().reindex(nom).to_numpy()
    med = float(np.median(y))
    print(f"{len(nom)} proyectos · {len(d):,} tareas · {SORTEOS} sorteos por escenario")

    def sortear(k, lam=LAMBDA):
        """Devuelve la matriz (sorteos x proyectos) de totales estimados."""
        rng = np.random.default_rng(1000 + k)
        M = np.zeros((SORTEOS, len(nom)))
        for s in range(SORTEOS):
            for j, p in enumerate(nom):
                i = idx[p]
                if len(i) <= k:
                    M[s, j] = es_[i].sum()
                    continue
                sel = rng.choice(i, k, replace=False)
                resto = np.setdiff1d(i, sel, assume_unique=False)
                off = np.log(re_[sel].sum() / es_[sel].sum()) * k / (k + lam)
                # lo cerrado se SABE; solo se estima lo que queda
                M[s, j] = re_[sel].sum() + np.exp(np.log(es_[resto]) + off).sum()
        return M

    filas = []
    filas.append(dict(brazo="0 · constante (n x mediana)", sorteos=1,
                      **mt(real, n_t * med), sd_PRED25=0.0, sd_MdMRE=0.0))
    filas.append(dict(brazo="1 · en frio", sorteos=1,
                      **mt(real, g.est.sum().reindex(nom).to_numpy()),
                      sd_PRED25=0.0, sd_MdMRE=0.0))
    for k in (1, 2, 3, 5, 10, 20):
        M = sortear(k)
        ms = [mt(real, M[s]) for s in range(SORTEOS)]
        prom = {c: float(np.mean([m[c] for m in ms])) for c in ms[0]}
        eti = f"{k} tarea{'s' if k > 1 else ''} cerrada{'s' if k > 1 else ''}"
        filas.append(dict(brazo=f"2 · con {eti}", sorteos=SORTEOS, **prom,
                          sd_PRED25=float(np.std([m["PRED25"] for m in ms])),
                          sd_MdMRE=float(np.std([m["MdMRE"] for m in ms]))))
        print(f"  k={k} listo", flush=True)
    t = pd.DataFrame(filas)
    print(f"\n{S}\nTOTAL DEL PROYECTO SEGUN CUANTO HISTORIAL HAY\n{S}")
    print(t[["brazo", "MAE_h", "RMSE_h", "MdMRE", "sd_MdMRE", "R2_log", "PRED25",
             "sd_PRED25", "PRED50", "factor2", "SSPB_pc"]].round(4).to_string(index=False))
    print("  sd_X es la dispersion de X entre sorteos: cuanto depende de QUE tareas")
    print("  fueron las que se cerraron primero")
    f = t.set_index("brazo")
    print(f"\n  en frio: {f.loc['1 · en frio', 'PRED25']:.1%} dentro del +-25 % y "
          f"{f.loc['1 · en frio', 'factor2']:.1%} dentro del factor 2")
    print(f"  con 5 cerradas: {f.loc['2 · con 5 tareas cerradas', 'PRED25']:.1%} "
          f"y {f.loc['2 · con 5 tareas cerradas', 'factor2']:.1%}")

    frio = g.est.sum().reindex(nom).to_numpy()
    tot = pd.DataFrame({"real": real, "frio": frio, "cte": n_t * med, "n": n_t},
                       index=nom)
    tot["tramo"] = pd.cut(tot.n, [7, 15, 40, 120, 400, 100000],
                          labels=["8-15", "16-40", "41-120", "121-400", "mas de 400"])
    fs = []
    for tr, gg in tot.groupby("tramo", observed=True):
        if len(gg) < 3:
            continue
        a, b = mt(gg.real, gg.frio), mt(gg.real, gg.cte)
        fs.append(dict(tareas=tr, proyectos=len(gg), MdMRE=a["MdMRE"],
                       PRED25=a["PRED25"], factor2=a["factor2"],
                       MdMRE_cte=b["MdMRE"], PRED25_cte=b["PRED25"]))
    e = pd.DataFrame(fs)
    print(f"\n{S}\nPOR TAMANO DEL PROYECTO, EN FRIO\n{S}")
    print(e.round(4).to_string(index=False))
    t.to_csv(OUT / "resumen.csv", index=False)
    e.to_csv(OUT / "por_tamano.csv", index=False)
    tot.to_csv(OUT / "totales.csv")
    json.dump({"lambda": LAMBDA, "sorteos": SORTEOS, "min_tareas": MIN_TAREAS},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/resumen.csv")


if __name__ == "__main__":
    main()
