"""E58 - Que aporta pedirle la estimacion al usuario, medido a nivel de PROYECTO.

Todo el trabajo sobre la estimacion del usuario esta medido tarea a tarea, donde
es la palanca mas grande del proyecto (R2 0,063 -> 0,340 en proyectos nunca
vistos). Pero la decision de producto -- si el formulario de Agix pide o no una
estimacion al crear cada tarea -- se toma mirando lo que cambia en el TOTAL del
proyecto, que es lo que el usuario ve.

Hay un motivo para esperar que aporte MENOS aqui que por tarea: el error del
usuario tambien se cancela al sumar, igual que el del modelo, asi que la ventaja
relativa se estrecha. Y otro para esperar que aporte igual: si el usuario es
optimista de forma sistematica -- y lo es -- ese sesgo NO se cancela, se acumula.
Cual de los dos manda es una pregunta empirica.

  0  constante
  1  el modelo en frio
  2  la estimacion del usuario en crudo
  3  la estimacion corregida por el modelo
  4  media geometrica de la corregida y la de en frio

El corrector se ajusta con validacion cruzada por proyecto entero, igual que todo
lo demas. Se restringe a los proyectos con estimacion en al menos el 80 % de sus
tareas; en los demas la mezcla de tareas con y sin estimacion enturbiaria la
lectura. En las pocas tareas sin estimacion de esos proyectos se usa el modelo en
frio.
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "6"

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 215)
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/usuario_proyecto")
S = "=" * 120
MIN_TAREAS, PLIEGUES, COBERTURA = 8, 5, .8


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
    t0 = time.time()
    df = pd.read_parquet("data/processed/agix_effort_v4.parquet")
    sw = df[df.is_software_task == 1].copy()
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    y = sw.effort_hours.to_numpy()
    ly = np.log(y)
    proy = sw.project.to_numpy()
    E5 = np.load(CACHE / "e5-large.npy")
    frio = np.clip(np.load("results/modelo_total/pred_total.npy"), 1e-6, None)

    est = np.array(sw.HoursEstimate.where(sw.HoursEstimate > 0), dtype=float)
    pt = sw[(sw.source == "TAWOS") & (sw.story_point > 0)]
    if len(pt):
        pe, co = np.polyfit(np.log(pt.story_point.to_numpy(float)),
                            np.log(pt.effort_hours.to_numpy(float)), 1)
        est[pt.index.to_numpy()] = np.exp(co) * pt.story_point.to_numpy(float) ** pe
        print(f"puntos de historia convertidos con horas = {np.exp(co):.2f} x "
              f"puntos^{pe:.3f}")
    hay = ~np.isnan(est)

    d = pd.DataFrame({"p": proy, "real": y, "frio": frio, "est": est, "hay": hay})
    tam = d.groupby("p").size()
    cob = d.groupby("p").hay.mean()
    val = [p for p in tam.index if tam[p] >= MIN_TAREAS and cob[p] >= COBERTURA]
    k = d.p.isin(val).to_numpy()
    print(f"{len(val)} proyectos con al menos {MIN_TAREAS} tareas y estimacion en el "
          f"{COBERTURA:.0%} o mas · {int(k.sum()):,} tareas")

    # corrector fuera de muestra, por proyecto entero
    # Dos correctores. El de siempre aprende log(real/estimacion) con error
    # cuadratico, o sea la MEDIANA del factor de correccion; sirve para una tarea
    # suelta y mete sesgo al sumar. El segundo predice E[horas] directamente con
    # perdida de Poisson, que es lo que necesita un total. Es el mismo fallo que
    # aparecio con el smearing y con el anclaje, por tercera vez.
    corr = np.full(len(sw), np.nan)
    corr_p = np.full(len(sw), np.nan)
    m = hay & k
    Xc = np.hstack([E5, np.log(np.where(hay, est, 1))[:, None], np.log(frio)[:, None]])
    obj = ly - np.log(np.where(hay, est, 1))
    ii = np.flatnonzero(m)
    for nf, (a, b) in enumerate(GroupKFold(PLIEGUES).split(Xc[ii], obj[ii], proy[ii])):
        r = HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=.06,
                                          l2_regularization=1., random_state=0)
        r.fit(Xc[ii[a]], obj[ii[a]])
        corr[ii[b]] = est[ii[b]] * np.exp(r.predict(Xc[ii[b]]))
        rp = HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=.06,
                                           l2_regularization=1., loss="poisson",
                                           random_state=0)
        rp.fit(Xc[ii[a]], y[ii[a]])
        corr_p[ii[b]] = rp.predict(Xc[ii[b]])
        print(f"  pliegue {nf + 1}/{PLIEGUES} · {time.time() - t0:.0f} s", flush=True)

    dd = d[k].copy()
    dd["crudo"] = np.where(dd.hay, dd.est, dd.frio)
    cc = corr[k]
    cp = corr_p[k]
    dd["corregida"] = np.where(dd.hay & ~np.isnan(cc), cc, dd.frio)
    dd["corr_poisson"] = np.where(dd.hay & ~np.isnan(cp), cp, dd.frio)
    dd["mezcla"] = np.sqrt(dd.corr_poisson * dd.frio)
    g = dd.groupby("p")
    real = g.real.sum()
    med = float(np.median(y))
    brazos = {"0 · constante (n x mediana)": g.size() * med,
              "1 · el modelo en frio": g.frio.sum(),
              "2 · estimacion del usuario en crudo": g.crudo.sum(),
              "3 · corregida con error cuadratico (mediana)": g.corregida.sum(),
              "4 · corregida con Poisson (media)": g.corr_poisson.sum(),
              "5 · media geometrica de la de Poisson y el frio": g.mezcla.sum()}
    t = pd.DataFrame([dict(brazo=n, **mt(real, v)) for n, v in brazos.items()])
    print(f"\n{S}\nTOTAL DEL PROYECTO · {len(real)} proyectos con estimacion de usuario\n{S}")
    print(t[["brazo", "MAE_h", "RMSE_h", "MdMRE", "MMRE", "R2_log", "PRED25",
             "PRED50", "factor2", "SSPB_pc"]].round(4).to_string(index=False))
    f = t.set_index("brazo")
    print(f"\n  el usuario en crudo se queda a un sesgo del "
          f"{f.loc['2 · estimacion del usuario en crudo', 'SSPB_pc']:+.0f} %; "
          f"corregido, {f.loc['4 · corregida con Poisson (media)', 'SSPB_pc']:+.0f} %")
    print(f"  +-25 %: en frio {f.loc['1 · el modelo en frio', 'PRED25']:.1%} · "
          f"usuario crudo {f.loc['2 · estimacion del usuario en crudo', 'PRED25']:.1%} · "
          f"corregida con Poisson {f.loc['4 · corregida con Poisson (media)', 'PRED25']:.1%}")
    t.to_csv(OUT / "resumen.csv", index=False)
    json.dump({"proyectos": len(real), "cobertura_min": COBERTURA},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/resumen.csv · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
