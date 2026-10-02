"""E78 - Nuevo protocolo: el usuario aporta historial del proyecto, y se APRENDE a usarlo.

CAMBIO DE PROTOCOLO. Hasta ahora la metrica media el total del proyecto ENTERO, y las
tareas ya cerradas entraban en ese total con sus horas reales -- lo que infla la cifra,
porque parte del total deja de estimarse. Aqui se mide lo que de verdad hay que
predecir: el error sobre las tareas QUE QUEDAN por estimar. Las k historicas solo
sirven para calibrar y NO cuentan en el error.

Es la simulacion honesta del producto: «dame k tareas tuyas con sus horas reales y yo
te estimo el resto del backlog».

CAMBIO DE DISENO. Hoy el historial se usa con una regla fija:

    off = log( suma real / suma predicha )  x  k/(k+12)

El k/(k+12) es credibilidad de libro, pero el 12 es un numero que se barrio una vez y
la regla ignora todo lo demas que el historial dice: si las k tareas concuerdan entre
si o no, como de grandes son, cuantas quedan por estimar. Aqui se entrena un corrector
que recibe todo eso como rasgos y aprende que desplazamiento aplicar.

  A  en frio                     no se usa el historial
  B  regla fija k/(k+12)         el mecanismo actual
  C  corrector aprendido         lo que se propone
  O  oraculo                     el desplazamiento perfecto, como cota de lo alcanzable

LA REGLA QUE NO SE ROMPE: el proyecto que se evalua no aparece NUNCA en el
entrenamiento del corrector. Las k tareas historicas se usan solo en inferencia.

Se evalua sobre los proyectos con 25 tareas o mas para que el MISMO conjunto sirva
para toda la curva, y se promedia la METRICA sobre los sorteos, no la prediccion.
"""
import json
import sys
import zlib
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 210)
OUT = Path("results/historial_aprendido")
S = "=" * 110
LAMBDA, SORTEOS, KS, MIN_TAREAS = 12, 30, (1, 3, 5, 10, 20), 25


def mt(real, pred):
    real, pred = np.asarray(real, float), np.clip(np.asarray(pred, float), 1e-6, None)
    r = np.abs(pred - real) / real
    m = float(np.median(np.log(pred / real)))
    return dict(PRED25=float(np.mean(r <= .25)), MdMRE=float(np.median(r)),
                factor2=float(np.mean(np.maximum(pred / real, real / pred) <= 2)),
                MAE_h=float(np.abs(pred - real).mean()),
                SSPB_pc=float(np.sign(m) * (np.exp(abs(m)) - 1) * 100))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet("data/processed/agix_effort_v4.parquet")
    sw = df[df.is_software_task == 1]
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    pred = np.clip(np.load("results/modelo_total/pred_total.npy"), 1e-6, None)
    assert len(pred) == len(sw)
    tam = sw.groupby("project").size()
    nom = list(tam[tam >= MIN_TAREAS].index)
    proy = sw.project.to_numpy()
    real = sw.effort_hours.to_numpy(float)
    idx = {p: np.flatnonzero(proy == p) for p in nom}
    print(f"{len(nom)} proyectos con {MIN_TAREAS}+ tareas · "
          f"{sum(len(idx[p]) for p in nom):,} tareas · {time.time() - t0:.0f} s",
          flush=True)

    # ---- construir todas las filas (proyecto, k, sorteo) -------------------
    filas, meta = [], []
    for p in nom:
        i = idx[p]
        for k in KS:
            for s in range(SORTEOS):
                rng = np.random.default_rng(
                    zlib.crc32(f"{p}|{k}|{s}".encode()))
                sel = rng.choice(len(i), k, replace=False)
                resto = np.setdiff1d(np.arange(len(i)), sel)
                rs, ps = real[i][sel], pred[i][sel]
                rr, pr = real[i][resto], pred[i][resto]
                coc = np.log(rs.sum() / ps.sum())          # la senal cruda
                porc = np.log(rs / ps)                      # cociente tarea a tarea
                filas.append([
                    coc,                                    # 0 cociente de sumas
                    float(np.median(porc)),                 # 1 mediana de cocientes
                    float(np.std(porc)) if k > 1 else 0.,   # 2 ¿concuerdan entre si?
                    float(k), float(np.log(k)),             # 3,4 cuanto historial
                    float(np.log(len(resto))),              # 5 cuanto queda
                    float(np.mean(np.log(rs))),             # 6 escala observada
                    float(np.std(np.log(rs))) if k > 1 else 0.,
                    float(np.mean(np.log(ps))),             # 8 escala predicha del hist
                    float(np.mean(np.log(pr))),             # 9 escala predicha del resto
                    float(np.std(np.log(pr))),
                ])
                meta.append((p, k, s, float(rr.sum()), float(pr.sum()),
                             float(np.log(rr.sum() / pr.sum())),
                             float(np.log(real[i].sum() / pred[i].sum()))))
    X = np.array(filas, float)
    M = pd.DataFrame(meta, columns=["p", "k", "s", "real_resto", "pred_resto",
                               "y", "off_escala"])
    y = M.y.to_numpy()
    print(f"  {len(X):,} filas · {X.shape[1]} rasgos · {time.time() - t0:.0f} s",
          flush=True)

    # ---- corrector aprendido, dejando fuera el proyecto evaluado ----------
    off_C = np.full(len(X), np.nan)
    off_H = np.full(len(X), np.nan)
    pp = M.p.to_numpy()
    for j, p in enumerate(nom):
        te = pp == p
        tr = ~te
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        r = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit((X[tr] - mu) / sd, y[tr])
        off_C[te] = r.predict((X[te] - mu) / sd)
        h = HistGradientBoostingRegressor(max_iter=160, max_depth=4, learning_rate=.07,
                                          l2_regularization=1., random_state=0)
        h.fit(X[tr], y[tr])
        off_H[te] = h.predict(X[te])
        if (j + 1) % 20 == 0:
            print(f"  corrector: {j + 1}/{len(nom)} · {time.time() - t0:.0f} s",
                  flush=True)

    # ---- evaluar: metrica por (k, brazo), promediando la METRICA ----------
    M["off_fijo"] = X[:, 0] * M.k / (M.k + LAMBDA)
    M["off_C"], M["off_H"] = off_C, off_H
    brazos = [("A · en frio (sin usar el historial)", lambda d: d.pred_resto),
              ("B · regla fija k/(k+12)", lambda d: d.pred_resto * np.exp(d.off_fijo)),
              ("C · corrector aprendido (lineal)", lambda d: d.pred_resto * np.exp(d.off_C)),
              ("D · corrector aprendido (arboles)", lambda d: d.pred_resto * np.exp(d.off_H)),
              # conocer la escala exacta del proyecto ENTERO: es lo mas que el
              # historial podria revelar. No da 100 % porque el resto no tiene
              # exactamente la escala media del proyecto.
              ("O · escala exacta del proyecto (cota)",
               lambda d: d.pred_resto * np.exp(d.off_escala))]
    out = []
    for k in KS:
        dk = M[M.k == k]
        for etq, fn in brazos:
            ms = [mt(g.real_resto, fn(g)) for _, g in dk.groupby("s")]
            out.append(dict(k=k, brazo=etq,
                            **{c: float(np.mean([m[c] for m in ms])) for c in ms[0]}))
    t = pd.DataFrame(out)

    print(f"\n{S}\nERROR SOBRE LAS TAREAS QUE QUEDAN POR ESTIMAR ({len(nom)} proyectos)"
          f"\n{S}")
    for c in ("PRED25", "MdMRE"):
        print(f"\n{c}:")
        print(t.pivot(index="brazo", columns="k", values=c).round(4).to_string())

    print(f"\n{S}\nLECTURA\n{S}")
    for k in (3, 5, 10):
        a = float(t[(t.k == k) & t.brazo.str.startswith("A")].PRED25.iloc[0])
        b = float(t[(t.k == k) & t.brazo.str.startswith("B")].PRED25.iloc[0])
        c_ = float(t[(t.k == k) & t.brazo.str.startswith("C")].PRED25.iloc[0])
        d_ = float(t[(t.k == k) & t.brazo.str.startswith("D")].PRED25.iloc[0])
        o = float(t[(t.k == k) & t.brazo.str.startswith("O")].PRED25.iloc[0])
        mejor = max(c_, d_)
        print(f"  k={k:2d}  frio {a:.1%} · regla fija {b:.1%} · aprendido "
              f"{mejor:.1%} · oraculo {o:.1%}"
              f"   {'APRENDIDO GANA' if mejor > b + .005 else 'empate o peor'}"
              f" ({(mejor - b) * 100:+.1f} pts)")
    t.to_csv(OUT / "resumen.csv", index=False)
    json.dump({"proyectos": len(nom), "sorteos": SORTEOS, "min_tareas": MIN_TAREAS,
               "filas": int(len(X))}, open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/resumen.csv · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
