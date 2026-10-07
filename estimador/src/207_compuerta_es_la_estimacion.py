"""E92 - La estimacion del usuario SI es una compuerta valida. ¿Y es necesaria la mezcla?

E90 y E91 dejaron el problema acotado con un numero: la mezcla de expertos
necesita un clasificador de regimen con AUC >= 0,80 para empatar con el global,
y el titulo de la tarea solo da 0,646. Tres compuertas distintas -el
clasificador, la propia prediccion del global y las dos juntas- caen todas en
0,644-0,649, lo que sugiere que es un techo del texto y no del clasificador.

Pero hay una fuente que no es el texto y que el producto YA pide: la estimacion
en horas del usuario (HU-46). Como compuerta da AUC 0,915, y enrutando con ella
el ±25 % pasa de 15,1 % a 19,4 % sobre las 11.546 tareas que la traen.

Antes de darlo por bueno hay que descartar lo evidente: si uno tiene la
estimacion, lo normal no es montar una mezcla, es meter la estimacion como una
variable mas del regresor. Si eso solo ya gana, la mezcla no aporta nada y hay
que decirlo.

  A  global, solo texto                      la referencia
  B  global + la estimacion como variable    EL CONTROL que decide
  C  enrutado por la estimacion a expertos de solo texto
  D  enrutado por la estimacion a expertos que tambien ven la estimacion
  E  regla directa: estimacion <= 2 h -> experto corto
  F  oraculo                                 la cota

Todo fuera de pliegue con GroupKFold por proyecto entero, sobre el subconjunto
que trae estimacion. El objetivo es log(horas) con error cuadratico: la cabeza
que se MUESTRA por tarea, que es donde duele la compresion.
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "10"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/compuerta_estimacion")
S = "=" * 100
PLIEGUES, CORTE = 5, 2.0
PAR = dict(max_iter=250, max_depth=6, learning_rate=.06, l2_regularization=1.,
           random_state=0)


def mide(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, None)
    r = np.abs(p - y) / y
    return dict(PRED25=float(np.mean(r <= .25)), MdMRE=float(np.median(r)),
                factor2=float(np.mean(np.maximum(p / y, y / p) <= 2)),
                MAE_h=float(np.abs(p - y).mean()),
                sesgo=float(np.median(p / y)),
                spearman=float(spearmanr(p, y).statistic))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet("data/processed/agix_effort_v4.parquet")
    todo = df[df.is_software_task == 1].reset_index(drop=True)
    V = np.load(CACHE / "e5-large-todo.npy")
    noap = (~todo.fuente_jira.eq("Apache")).to_numpy()
    sw, V = todo[noap].reset_index(drop=True), V[noap].astype(np.float32)
    est = sw.HoursEstimate.where(sw.HoursEstimate > 0).to_numpy(float)
    hay = ~np.isnan(est)
    sw, V, est = sw[hay].reset_index(drop=True), V[hay], est[hay]
    y = sw.effort_hours.to_numpy(float)
    proy = sw.project.to_numpy()
    ly, le = np.log(y), np.log(est)
    corto = y <= CORTE
    print(f"{len(y):,} tareas con estimacion · {sw.project.nunique()} proyectos · "
          f"cortas {corto.mean():.1%} · {time.time() - t0:.0f} s", flush=True)
    print(f"  la estimacion como compuerta: AUC "
          f"{roc_auc_score(corto, -est):.4f}", flush=True)

    Xe = np.hstack([V, le[:, None]]).astype(np.float32)
    p_txt = np.zeros(len(y))      # global solo texto
    p_est = np.zeros(len(y))      # global + estimacion  <- el control
    p_cor = np.zeros(len(y))      # experto corto, solo texto
    p_lar = np.zeros(len(y))
    p_cor_e = np.zeros(len(y))    # experto corto, con estimacion
    p_lar_e = np.zeros(len(y))
    for nf, (a, b) in enumerate(GroupKFold(PLIEGUES).split(V, y, proy)):
        ac, al = a[corto[a]], a[~corto[a]]
        p_txt[b] = np.exp(HistGradientBoostingRegressor(**PAR)
                          .fit(V[a], ly[a]).predict(V[b]))
        p_est[b] = np.exp(HistGradientBoostingRegressor(**PAR)
                          .fit(Xe[a], ly[a]).predict(Xe[b]))
        p_cor[b] = np.exp(HistGradientBoostingRegressor(**PAR)
                          .fit(V[ac], ly[ac]).predict(V[b]))
        p_lar[b] = np.exp(HistGradientBoostingRegressor(**PAR)
                          .fit(V[al], ly[al]).predict(V[b]))
        p_cor_e[b] = np.exp(HistGradientBoostingRegressor(**PAR)
                            .fit(Xe[ac], ly[ac]).predict(Xe[b]))
        p_lar_e[b] = np.exp(HistGradientBoostingRegressor(**PAR)
                            .fit(Xe[al], ly[al]).predict(Xe[b]))
        print(f"  pliegue {nf + 1}/{PLIEGUES} · {time.time() - t0:.0f} s", flush=True)

    ruta = est <= CORTE
    brazos = {
        "A · global, solo texto": p_txt,
        "B · global + la estimacion (CONTROL)": p_est,
        "C · enrutado por la estimacion": np.where(ruta, p_cor, p_lar),
        "D · enrutado, expertos con estimacion": np.where(ruta, p_cor_e, p_lar_e),
        "E · la estimacion del usuario en crudo": est,
        "F · oraculo, expertos con estimacion": np.where(corto, p_cor_e, p_lar_e),
    }
    filas = []
    for k, v in brazos.items():
        filas.append(dict(brazo=k, ambito="todas", **mide(y, v)))
        filas.append(dict(brazo=k, ambito="solo cortas", **mide(y[corto], v[corto])))
    t = pd.DataFrame(filas)
    cols = ["brazo", "PRED25", "MdMRE", "factor2", "MAE_h", "sesgo", "spearman"]
    for amb in ("todas", "solo cortas"):
        print(f"\n{S}\n{amb.upper()} ({int((corto if amb != 'todas' else np.ones(len(y), bool)).sum()):,} tareas)\n{S}")
        print(t[t.ambito == amb][cols].round(4).to_string(index=False))

    print(f"\n{S}\nLO QUE DECIDE: ¿APORTA LA MEZCLA SOBRE EL CONTROL?\n{S}")
    g = lambda n: float(t[(t.brazo.str.startswith(n)) & (t.ambito == "todas")].PRED25.iloc[0])  # noqa: E731
    a, b_, c, d, f = g("A"), g("B"), g("C"), g("D"), g("F")
    print(f"  A solo texto                      {a:.1%}")
    print(f"  B + estimacion como variable      {b_:.1%}  ({(b_ - a) * 100:+.1f} sobre A)")
    print(f"  D enrutado con expertos           {d:.1%}  ({(d - b_) * 100:+.1f} sobre B)")
    print(f"  F oraculo                         {f:.1%}")
    if d - b_ > .005:
        print(f"\n  La mezcla APORTA {(d - b_) * 100:+.1f} puntos sobre meter la estimacion")
        print("  como una variable mas. Especializar por regimen hace algo que el")
        print("  regresor unico no consigue aunque vea el mismo dato.")
    else:
        print(f"\n  La mezcla NO aporta sobre el control ({(d - b_) * 100:+.1f} puntos).")
        print("  Basta con meter la estimacion como variable; el regresor ya")
        print("  aprende solo el cambio de regimen y no hace falta enrutar.")
    t.to_csv(OUT / "resumen.csv", index=False)
    json.dump({"n": int(len(y)), "auc_estimacion": float(roc_auc_score(corto, -est)),
               "A_texto": a, "B_control": b_, "C_enrutado": c, "D_enrutado_est": d,
               "F_oraculo": f}, open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/resumen.csv · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
