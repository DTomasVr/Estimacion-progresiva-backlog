"""E60 - La regla de Bayes para la ventana, aplicada al TOTAL del proyecto.

El +-25 % del total lleva clavado en el 36,8 % pase lo que pase: lo han dejado
igual el modelo del total, el canal lineal y el codificador por pliegue. Merece
la pena atacarlo directamente, porque no es un error, es una TASA DE ACIERTO
dentro de una ventana, y para eso el estimador optimo no es la media ni la
mediana sino el centro de la ventana que mas masa recoge (Gneiting, 2011). A
nivel de tarea eso subio el PRED25 de 0,167 a 0,183; aqui no se ha probado.

La ventana del +-25 % es asimetrica en logaritmos. Con q = log(real / estimado),
acertar significa

    q  dentro de  [log c - log 1,25 ,  log c + log(1/0,75)]

o sea una ventana de anchura 0,511 que no esta centrada: 0,223 por abajo y 0,288
por arriba. Asi que hay un desplazamiento c que recoge mas masa que c = 1, y se
puede encontrar deslizando la ventana sobre la distribucion de los residuos de
LOS DEMAS proyectos.

  0  sin desplazar
  1  desplazamiento global elegido para el +-25 %
  2  desplazamiento por tramo de tamano
  3  lo mismo para el factor 2

El riesgo es el de siempre: son 114 proyectos y deslizar una ventana sobre ellos
puede aprender los proyectos concretos. Por eso el desplazamiento de cada
proyecto se elige SIN mirar ese proyecto, y se reporta lo que se gana en la
metrica que se optimiza y lo que se pierde en las demas.
"""
import json
import os
import sys
import warnings
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "4"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 215)
OUT = Path("results/bayes_total")
S = "=" * 118
MIN_TAREAS = 8


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


def mejor_c(q, lo, hi):
    """Desplazamiento que mete mas residuos dentro de su ventana.

    q son los residuos log(real/estimado) de los proyectos de referencia; la
    ventana en torno a log c va de log c + lo a log c + hi."""
    cand = np.linspace(-1.2, 1.2, 481)
    masa = [(np.mean((q >= c + lo) & (q <= c + hi)), c) for c in cand]
    return float(max(masa)[1])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet("data/processed/pbe_effort_v4.parquet")
    sw = df[df.is_software_task == 1].copy()
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    y = sw.effort_hours.to_numpy()
    est = np.clip(np.load("results/modelo_total/pred_total.npy"), 1e-6, None)
    d = pd.DataFrame({"p": sw.project.to_numpy(), "real": y, "est": est})
    tam = d.groupby("p").size()
    d = d[d.p.isin(tam[tam >= MIN_TAREAS].index)]
    g = d.groupby("p")
    real = g.real.sum().to_numpy()
    pred = g.est.sum().to_numpy()
    n_t = g.size().to_numpy()
    q = np.log(real / pred)
    estr = np.digitize(n_t, [21, 61, 201])
    print(f"{len(real)} proyectos · residuo mediano x{np.exp(np.median(q)):.3f}")

    V = {"pred25": (-np.log(1.25), np.log(1 / .75)),
         "factor2": (-np.log(2), np.log(2))}
    brazos = {"0 · sin desplazar": pred}
    for etq, (lo, hi) in V.items():
        glob = np.zeros(len(real))
        tramo = np.zeros(len(real))
        for i in range(len(real)):
            k = np.ones(len(real), bool)
            k[i] = False
            glob[i] = pred[i] * np.exp(mejor_c(q[k], lo, hi))
            kk = k & (estr == estr[i])
            tramo[i] = pred[i] * np.exp(mejor_c(q[kk] if kk.sum() >= 15 else q[k], lo, hi))
        brazos[f"1 · desplazamiento global para {etq}"] = glob
        brazos[f"2 · desplazamiento por tramo para {etq}"] = tramo
    t = pd.DataFrame([dict(brazo=n, **mt(real, v)) for n, v in brazos.items()])
    print(f"\n{S}\nTOTAL DEL PROYECTO · deslizar la ventana\n{S}")
    print(t[["brazo", "MAE_h", "RMSE_h", "MdMRE", "R2_log", "PRED25", "PRED50",
             "factor2", "SSPB_pc"]].round(4).to_string(index=False))
    f = t.set_index("brazo")
    b = f.loc["0 · sin desplazar"]
    m25 = f.loc["1 · desplazamiento global para pred25"]
    print(f"\n  el desplazamiento tipico para el +-25 % es "
          f"x{np.exp(mejor_c(q, *V['pred25'])):.3f}")
    print(f"  +-25 %: {b.PRED25:.1%} -> {m25.PRED25:.1%} · "
          f"y el MdMRE pasa de {b.MdMRE:.4f} a {m25.MdMRE:.4f}")
    t.to_csv(OUT / "resumen.csv", index=False)
    json.dump({"c_pred25": float(np.exp(mejor_c(q, *V["pred25"]))),
               "c_factor2": float(np.exp(mejor_c(q, *V["factor2"])))},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/resumen.csv")


if __name__ == "__main__":
    main()
