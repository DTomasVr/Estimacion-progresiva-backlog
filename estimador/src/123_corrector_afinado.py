"""E24 · Afinar la unica palanca grande que queda: el corrector de la estimacion.

E23 dejo claro que, en proyectos nunca vistos, pedirle la estimacion al usuario
y corregirla multiplica el R² por cinco (0,063 -> 0,340) mientras que las anclas
solo lo llevan a 0,110. Si hay una cosa que merece exprimirse, es esa.

El corrector actual es el brazo C2 del experimento 77: un HGB sobre
[log(estimacion), prediccion en frio]. Se probaron variantes en aquel momento
sobre el corpus viejo y con el codificador viejo. Aqui se repiten sobre el banco
grande -- validacion cruzada por proyecto entero, 204 proyectos -- con las
predicciones en frio del modelo final, que ya estan calculadas fuera de
proyecto.

  C0  la estimacion tal cual, sin corregir              (referencia)
  C1  + el texto de la tarea
  C2  estimacion + frio                                  (el actual)
  C3  + la familiaridad con el historial
  C4  mezcla geometrica estimacion^w x frio^(1-w), w en validacion
  C5  C2 pero prediciendo el residuo con perdida absoluta
  C6  C2 con la razon acotada: el corrector no puede mover mas de un factor k

C6 sale de una sospecha: si el corrector se equivoca, conviene que se equivoque
poco. Acotar cuanto puede corregir es barato y limita el daño.
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
pd.set_option("display.width", 210)
DATA = "data/processed/pbe_effort_v4.parquet"
CACHE = Path("data/processed/embeddings/codificadores")
CV = Path("results/palancas_cv/predicciones.npy")
OUT = Path("results/corrector")
SEP = "=" * 116


def metricas(y, p):
    p = np.clip(np.asarray(p, float), .01, None)
    y = np.asarray(y, float)
    m = E.point_metrics(y, p)
    ly, lp = np.log(y), np.log(p)
    return dict(MdMRE=m["MdMRE"], MdAElog=float(np.median(np.abs(lp - ly))),
                PRED25=m["PRED25"], PRED50=m["PRED50"], MAE_h=m["MAE_h"],
                R2=float(1 - ((ly - lp) ** 2).mean() / ly.var()),
                sesgo=float(np.exp(np.median(lp - ly))), SA=M71.sa(y, p),
                spearman=float(pd.Series(y).corr(pd.Series(p), method="spearman")))


def hgb(**kw):
    return HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=.06,
                                         l2_regularization=1., random_state=0, **kw)


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
    Xn = E5 / np.linalg.norm(E5, axis=1, keepdims=True)
    lp = np.load(CV)[0]                      # frio fuera de proyecto, del script 121

    sw["est"] = sw.HoursEstimate.where(sw.HoursEstimate > 0)
    pt = sw[(sw.source == "TAWOS") & (sw.story_point > 0)]
    pend, corte = np.polyfit(np.log(pt.story_point.to_numpy(float)),
                             np.log(pt.effort_hours.to_numpy(float)), 1)
    sw.loc[pt.index, "est"] = np.exp(corte) * pt.story_point.to_numpy(float) ** pend
    est = sw.est.to_numpy(float)
    hay = ~np.isnan(est)
    k = np.flatnonzero(hay)
    le = np.log(est[k])
    print(f"{hay.sum():,} tareas con estimación de {len(sw):,} · "
          f"{len(np.unique(proy[k]))} proyectos")

    razon = ly[k] - le
    print(f"  razón real/estimación: mediana ×{np.exp(np.median(razon)):.2f} · "
          f"sd {razon.std():.3f} · iguales {float(np.mean(np.abs(razon) < 1e-9)):.1%}")

    gkf = GroupKFold(n_splits=5)
    preds = {"C0 · la estimación tal cual": est[k]}
    conj = {
        "C1 · + texto": np.hstack([E5[k], le[:, None], lp[k][:, None]]),
        "C2 · estimación + frío (el actual)": np.column_stack([le, lp[k]]),
        "C3 · + familiaridad": None,
        "C5 · pérdida absoluta": np.column_stack([le, lp[k]]),
    }
    # familiaridad: coseno maximo con las tareas de OTROS pliegues del mismo proyecto
    fam = np.zeros(len(k))
    for p in np.unique(proy[k]):
        i = np.flatnonzero(proy[k] == p)
        if len(i) < 2:
            continue
        s = Xn[k][i] @ Xn[k][i].T
        np.fill_diagonal(s, -1)
        fam[i] = s.max(1)
    conj["C3 · + familiaridad"] = np.column_stack([le, lp[k], fam])

    for nombre, X in conj.items():
        p = np.zeros(len(k))
        for tr, te in gkf.split(X, razon, proy[k]):
            m = hgb(loss="absolute_error") if "absoluta" in nombre else hgb()
            p[te] = m.fit(X[tr], razon[tr]).predict(X[te])
        preds[nombre] = np.exp(le + p)
        print(f"  {nombre} listo · {time.time() - t0:.0f} s", flush=True)

    # C4 · mezcla geometrica, w por pliegue sobre el propio entrenamiento
    p4 = np.zeros(len(k))
    for tr, te in gkf.split(le, razon, proy[k]):
        ws = np.arange(0, 1.01, .05)
        mejor = max(ws, key=lambda w: M71.sa(y[k][tr], np.exp(w * le[tr] + (1 - w) * lp[k][tr])))
        p4[te] = np.exp(mejor * le[te] + (1 - mejor) * lp[k][te])
    preds["C4 · mezcla geométrica"] = p4

    # C6 · C2 con la correccion acotada
    base = np.log(preds["C2 · estimación + frío (el actual)"]) - le
    for tope in (0.5, 1.0):
        preds[f"C6 · C2 acotado a ×{np.exp(tope):.1f}"] = np.exp(
            le + np.clip(base, -tope, tope))

    preds["referencia · el modelo en frío"] = np.exp(lp[k])
    filas = [dict(brazo=n, n=len(k), **metricas(y[k], p)) for n, p in preds.items()]
    t = pd.DataFrame(filas)
    print(f"\n{SEP}\nCORRECTOR DE LA ESTIMACIÓN · proyectos nunca vistos\n{SEP}")
    print(t.sort_values("R2", ascending=False).round(4).to_string(index=False))

    # subconjunto donde la estimacion NO acierta sola
    dist = np.abs(razon) > 1e-9
    print(f"\n{SEP}\nSOLO DONDE LO REAL ≠ LO ESTIMADO (n={int(dist.sum()):,})\n{SEP}")
    t2 = pd.DataFrame([dict(brazo=n, n=int(dist.sum()), **metricas(y[k][dist], p[dist]))
                       for n, p in preds.items()])
    print(t2.sort_values("R2", ascending=False).round(4).to_string(index=False))

    t.to_csv(OUT / "resumen.csv", index=False)
    t2.to_csv(OUT / "distintas.csv", index=False)
    json.dump({"n": int(hay.sum())}, open(OUT / "meta.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\nescrito {OUT}/resumen.csv y distintas.csv")


if __name__ == "__main__":
    main()
