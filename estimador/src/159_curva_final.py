"""E53 - La curva que va al deck, ya con el modelo entrenado para el total.

E44 midio cuantas tareas cerradas hacen falta para que el total sea fiable, pero
sobre el estimador viejo (mediana empujada con smearing). E52 lo ha mejorado en
casi todo, asi que la curva hay que rehacerla: es la tabla que va en la
exposicion y tiene que salir del mejor estimador, no del que habia.

Mismo protocolo que E44, para que sea comparable:

  - el desplazamiento del proyecto se estima con k tareas cerradas y contraccion
    de James-Stein (lambda 12), que con k pequeno es imprescindible
  - se promedian 40 sorteos de cuales son las tareas que resultaron cerrarse
    primero, porque con k pequeno cual toque importa mucho
  - se compara siempre contra el predictor constante, que no lee el texto
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
S = "=" * 118
MIN_TAREAS, SORTEOS, LAMBDA = 8, 40, 12


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
    df = pd.read_parquet("data/processed/agix_effort_v4.parquet")
    sw = df[df.is_software_task == 1].copy()
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    y = sw.effort_hours.to_numpy()
    proy = sw.project.to_numpy()
    est = np.clip(np.load("results/modelo_total/pred_total.npy"), 1e-6, None)
    d = pd.DataFrame({"p": proy, "real": y, "est": est})
    tam = d.groupby("p").size()
    d = d[d.p.isin(tam[tam >= MIN_TAREAS].index)].reset_index(drop=True)
    le = np.log(d.est.to_numpy())
    lr = np.log(d.real.to_numpy())
    nombres = list(pd.unique(d.p))
    idx = {p: np.flatnonzero(d.p.to_numpy() == p) for p in nombres}
    g = d.groupby("p")
    real = g.real.sum().reindex(nombres).to_numpy()
    med = float(np.median(y))
    base = {"0 · constante (n x mediana)": g.size().reindex(nombres).to_numpy() * med,
            "1 · en frio": g.est.sum().reindex(nombres).to_numpy()}
    print(f"{len(nombres)} proyectos · {len(d):,} tareas")

    # El desplazamiento de un TOTAL no se estima con la mediana de los cocientes.
    # Este estimador predice la MEDIA condicional, y la mediana de log(real/est)
    # es sistematicamente negativa con una distribucion asimetrica: al anclar con
    # ella, cuantas mas tareas cerradas mas se hunde la estimacion (con 20 tareas
    # el sesgo llegaba a -40 %). Lo que corresponde es el cociente de SUMAS, que
    # es el estadistico que hace cuadrar un total. Se dejan los dos para verlo.
    rng = np.random.default_rng(0)
    re_, es_ = d.real.to_numpy(), d.est.to_numpy()
    for modo in ("cociente de sumas", "mediana de cocientes"):
        for k in (1, 2, 3, 5, 10, 20):
            ac = np.zeros(len(nombres))
            for _ in range(SORTEOS):
                tot = []
                for p in nombres:
                    i = idx[p]
                    if len(i) <= k:
                        tot.append(float(es_[i].sum()))
                        continue
                    s = rng.choice(i, k, replace=False)
                    bruto = (np.log(re_[s].sum() / es_[s].sum())
                             if modo == "cociente de sumas"
                             else float(np.median(lr[s] - le[s])))
                    off = bruto * k / (k + LAMBDA)
                    tot.append(float(np.exp(le[i] + off).sum()))
                ac += np.array(tot)
            eti = f"{k} tarea{'s' if k > 1 else ''} cerrada{'s' if k > 1 else ''}"
            n = 2 if modo == "cociente de sumas" else 3
            base[f"{n} · con {eti} ({modo})"] = ac / SORTEOS

    t = pd.DataFrame([dict(brazo=n, **mt(real, v)) for n, v in base.items()])
    print(f"\n{S}\nTOTAL DEL PROYECTO SEGUN CUANTO HISTORIAL HAY\n{S}")
    print(t[["brazo", "MAE_h", "RMSE_h", "MdMRE", "R2_log", "PRED25", "PRED50",
             "factor2", "SSPB_pc"]].round(4).to_string(index=False))
    f = t.set_index("brazo")
    print(f"\n  en frio: {f.loc['1 · en frio', 'PRED25']:.1%} dentro del +-25 % y "
          f"{f.loc['1 · en frio', 'factor2']:.1%} dentro del factor 2")
    print(f"  con 5 cerradas: {f.loc['2 · con 5 tareas cerradas (cociente de sumas)', 'PRED25']:.1%} "
          f"y {f.loc['2 · con 5 tareas cerradas (cociente de sumas)', 'factor2']:.1%}")

    # por tamano de proyecto, que es lo que pregunta cualquiera
    d["n"] = d.groupby("p").p.transform("size")
    tot = pd.DataFrame({"real": real, "frio": base["1 · en frio"],
                        "cte": base["0 · constante (n x mediana)"],
                        "n": g.size().reindex(nombres).to_numpy()}, index=nombres)
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
    json.dump({"lambda": LAMBDA, "sorteos": SORTEOS}, open(OUT / "meta.json", "w"),
              indent=1)
    print(f"\nescrito {OUT}/resumen.csv")


if __name__ == "__main__":
    main()
