"""E25 · ¿Cómo de mal puede estimar el usuario y que siga compensando preguntarle?

La recomendacion de la noche es pedirle la estimacion al usuario: multiplica el
R² por cinco en proyectos nunca vistos. Pero esa cifra se midio con las
estimaciones que hay en el corpus, y no sabemos si los usuarios de PBE estimaran
igual de bien. En SiP la razon real/estimacion tiene sd 0,77 en log; en el bloque
publico de Jira, 1,37. Casi el doble de ruido.

Asi que hay que saber donde esta el punto de equilibrio: a partir de que nivel de
ruido en la estimacion del usuario deja de compensar preguntarle.

Se degrada la estimacion real anadiendo ruido lognormal de sd creciente y se
vuelve a medir todo: la estimacion cruda, el corrector y el modelo en frio. El
corrector se reajusta con la estimacion ya degradada, porque en produccion
aprenderia de los usuarios que tenga, no de otros.

Tambien se prueba un sesgo sistematico -- usuarios optimistas que estiman a la
baja de forma constante -- porque es el patron que describe la literatura de la
falacia de la planificacion y es distinto del ruido puro.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("m71", "src/71_estimacion_usuario.py")
M71 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M71)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
DATA = "data/processed/pbe_effort_v4.parquet"
CACHE = Path("data/processed/embeddings/codificadores")
CV = Path("results/palancas_cv/predicciones.npy")
OUT = Path("results/sensibilidad")
SEP = "=" * 112
RUIDOS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
SESGOS = [0.0, -0.35, -0.7]          # log: -0,7 es estimar la mitad de lo real


def metricas(y, p):
    p = np.clip(np.asarray(p, float), .01, None)
    y = np.asarray(y, float)
    m = E.point_metrics(y, p)
    ly, lp = np.log(y), np.log(p)
    return dict(MdMRE=m["MdMRE"], MdAElog=float(np.median(np.abs(lp - ly))),
                PRED25=m["PRED25"], MAE_h=m["MAE_h"],
                R2=float(1 - ((ly - lp) ** 2).mean() / ly.var()),
                sesgo_pred=float(np.exp(np.median(lp - ly))), SA=M71.sa(y, p))


def hgb():
    return HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=.06,
                                         l2_regularization=1., random_state=0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet(DATA)
    sw = df[df.is_software_task == 1].copy()
    sw["apache"] = sw.fuente_jira.eq("Apache")
    sw = sw[~sw.apache].reset_index(drop=True)
    y = sw.effort_hours.to_numpy()
    ly = np.log(y)
    proy = sw.project.to_numpy()
    E5 = np.load(CACHE / "e5-large.npy")
    lp = np.load(CV)[0]
    sw["est"] = sw.HoursEstimate.where(sw.HoursEstimate > 0)
    pt = sw[(sw.source == "TAWOS") & (sw.story_point > 0)]
    pend, corte = np.polyfit(np.log(pt.story_point.to_numpy(float)),
                             np.log(pt.effort_hours.to_numpy(float)), 1)
    sw.loc[pt.index, "est"] = np.exp(corte) * pt.story_point.to_numpy(float) ** pend
    k = np.flatnonzero(sw.est.notna().to_numpy())
    le0 = np.log(sw.est.to_numpy(float)[k])
    print(f"{len(k):,} tareas con estimación · ruido actual de la razón: "
          f"sd {float((ly[k] - le0).std()):.3f}")
    frio = metricas(y[k], np.exp(lp[k]))
    print(f"  el modelo en frío sobre esas mismas tareas: R² {frio['R2']:+.4f} · "
          f"MdMRE {frio['MdMRE']:.4f} · SA {frio['SA']:.4f}")

    gkf = GroupKFold(n_splits=5)
    filas = []
    for sesgo in SESGOS:
        for ruido in RUIDOS:
            rng = np.random.default_rng(int(ruido * 100) + int(abs(sesgo) * 1000))
            le = le0 + sesgo + rng.normal(0, ruido, len(k))
            razon = ly[k] - le
            X = np.hstack([E5[k], le[:, None], lp[k][:, None]])
            p = np.zeros(len(k))
            for tr, te in gkf.split(X, razon, proy[k]):
                p[te] = hgb().fit(X[tr], razon[tr]).predict(X[te])
            for etq, pred in (("estimación cruda", np.exp(le)),
                              ("corregida", np.exp(le + p))):
                filas.append(dict(sesgo=sesgo, ruido=ruido, brazo=etq,
                                  **metricas(y[k], pred)))
            print(f"  sesgo {sesgo:+.2f} · ruido {ruido:.2f} -> cruda R² "
                  f"{filas[-2]['R2']:+.4f} · corregida R² {filas[-1]['R2']:+.4f}"
                  f"  ({time.time() - t0:.0f} s)", flush=True)
    filas.append(dict(sesgo=np.nan, ruido=np.nan, brazo="el modelo en frío", **frio))
    t = pd.DataFrame(filas)
    print(f"\n{SEP}\nSENSIBILIDAD A LA CALIDAD DE LA ESTIMACIÓN DEL USUARIO\n{SEP}")
    for sesgo in SESGOS:
        v = t[t.sesgo == sesgo]
        print(f"\nsesgo {sesgo:+.2f} en log "
              f"(el usuario estima ×{np.exp(sesgo):.2f} de lo real)")
        print(v[["ruido", "brazo", "MdMRE", "MdAElog", "PRED25", "R2", "SA",
                 "sesgo_pred"]]
              .round(4).to_string(index=False))
    print(f"\n  referencia · el modelo en frío: R² {frio['R2']:+.4f} · "
          f"MdMRE {frio['MdMRE']:.4f} · SA {frio['SA']:.4f}")
    # punto de equilibrio
    print(f"\n{SEP}\nPUNTO DE EQUILIBRIO\n{SEP}")
    for sesgo in SESGOS:
        v = t[(t.sesgo == sesgo) & (t.brazo == "corregida")].sort_values("ruido")
        peor = v[v.R2 <= frio["R2"]]
        lim = "nunca deja de compensar en el rango probado" if not len(peor) else \
            f"deja de compensar a partir de sd ≈ {peor.ruido.iloc[0]:.2f}"
        print(f"  sesgo {sesgo:+.2f}: {lim}")
    t.to_csv(OUT / "resumen.csv", index=False)
    json.dump({"ruidos": RUIDOS, "sesgos": SESGOS, "frio": frio},
              open(OUT / "meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nescrito {OUT}/resumen.csv")


if __name__ == "__main__":
    main()
