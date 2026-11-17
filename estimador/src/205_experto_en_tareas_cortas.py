"""E90 - Un modelo especialista en tareas cortas, y si se puede ensamblar.

EL PROBLEMA, medido sobre las 24.848 tareas no Apache:

    tramo     n      real med   pred med   sesgo
    <=1 h    6.102      0,50       4,15     x11,45
    1-2 h    4.792      1,00       4,40     x 3,80
    4-8 h    3.983      5,00       5,56     x 1,08
    >16 h    2.301     25,58       7,83     x 0,27

El modelo predice entre 4 y 8 horas para TODO. La realidad va de 0,5 a 25,6.
La prediccion abarca menos de x2 donde la realidad abarca x51.

LA IDEA (literatura): los modelos LOCALES, entrenados en regiones homogeneas,
le ganan al global cuando el global promedia regimenes distintos -Menzies et
al., "Local vs. global models for effort estimation and defect prediction"-.
Y para regresion desbalanceada, la familia que se usa es mezcla de expertos con
una compuerta que reparte, donde algun experto se especializa en las regiones
poco pobladas.

LO QUE YA SE PROBO Y PERDIO (E-91 clasificador_tramo): clasificar el tramo y
aplicar su mediana. El clasificador saca F1 0,32 con cuatro clases y recall
0,057 en >16 h. Esto es DISTINTO: no se clasifica para aplicar una constante,
se entrena un REGRESOR especializado en cada regimen.

  A  global                     lo actual
  B  enrutado duro por compuerta
  C  mezcla blanda por la probabilidad de la compuerta
  D  enrutado ORACULO           <- la cota superior de toda la familia
  E  diagnostico: el experto corto EN tareas cortas, contra el global ahi mismo

El brazo D es el que decide. Si conociendo de antemano a que regimen pertenece
cada tarea la mezcla no mejora, no hay compuerta posible que lo arregle y la
via entera esta muerta. Por eso se mide primero.

Protocolo: GroupKFold por proyecto entero, todo fuera de pliegue, objetivo
log(horas) con error cuadratico, que es la cabeza que se MUESTRA por tarea.
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
from sklearn.ensemble import (HistGradientBoostingClassifier,  # noqa: E402
                              HistGradientBoostingRegressor)
from sklearn.model_selection import GroupKFold  # noqa: E402

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 210)
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/experto_cortas")
S = "=" * 104
PLIEGUES = 5
CORTE = 2.0          # la mediana de horas del corpus
PAR = dict(max_iter=250, max_depth=6, learning_rate=.06, l2_regularization=1.,
           random_state=0)


def mt(y, p):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-6, None)
    r = np.abs(p - y) / y
    from scipy.stats import spearmanr
    return dict(PRED25=float(np.mean(r <= .25)), MdMRE=float(np.median(r)),
                factor2=float(np.mean(np.maximum(p / y, y / p) <= 2)),
                MAE_h=float(np.abs(p - y).mean()),
                sesgo=float(np.median(p / y)),
                spearman=float(spearmanr(p, y).statistic),
                sd_log_pred=float(np.std(np.log(p))))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet("data/processed/pbe_effort_v4.parquet")
    todo = df[df.is_software_task == 1].reset_index(drop=True)
    V = np.load(CACHE / "e5-large-todo.npy")
    noap = (~todo.fuente_jira.eq("Apache")).to_numpy()
    sw, V = todo[noap].reset_index(drop=True), V[noap].astype(np.float32)
    y = sw.effort_hours.to_numpy(float)
    proy = sw.project.to_numpy()
    corto = y <= CORTE
    ly = np.log(y)
    print(f"{len(sw):,} tareas · {sw.project.nunique()} proyectos · "
          f"cortas (<={CORTE:g} h) {corto.mean():.1%} · "
          f"{corto.sum() / len(sw) * 100:.0f}% del recuento pero "
          f"{y[corto].sum() / y.sum() * 100:.1f}% de las horas · "
          f"{time.time() - t0:.0f} s", flush=True)

    p_glob = np.zeros(len(sw))
    p_cor = np.zeros(len(sw))
    p_lar = np.zeros(len(sw))
    p_gate = np.zeros(len(sw))
    for nf, (a, b) in enumerate(GroupKFold(PLIEGUES).split(V, y, proy)):
        g = HistGradientBoostingRegressor(**PAR).fit(V[a], ly[a])
        p_glob[b] = np.exp(g.predict(V[b]))
        ac = a[corto[a]]
        al = a[~corto[a]]
        mc = HistGradientBoostingRegressor(**PAR).fit(V[ac], ly[ac])
        ml = HistGradientBoostingRegressor(**PAR).fit(V[al], ly[al])
        p_cor[b] = np.exp(mc.predict(V[b]))
        p_lar[b] = np.exp(ml.predict(V[b]))
        cl = HistGradientBoostingClassifier(**PAR).fit(V[a], corto[a])
        p_gate[b] = cl.predict_proba(V[b])[:, 1]
        print(f"  pliegue {nf + 1}/{PLIEGUES} · entreno corto n={len(ac):,} · "
              f"largo n={len(al):,} · {time.time() - t0:.0f} s", flush=True)

    # --------- calidad de la compuerta ----------
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(corto, p_gate)
    dura = p_gate >= .5
    print(f"\n{S}\nLA COMPUERTA: ¿SE PUEDE SABER POR EL TEXTO SI UNA TAREA ES "
          f"CORTA?\n{S}")
    print(f"  AUC {auc:.4f} · acierto {np.mean(dura == corto):.1%} · "
          f"base (decir siempre la clase mayoritaria) "
          f"{max(corto.mean(), 1 - corto.mean()):.1%}")

    mezcla = {
        "A · global (lo actual)": p_glob,
        "B · enrutado duro por compuerta": np.where(dura, p_cor, p_lar),
        "C · mezcla blanda (prob. compuerta)":
            np.exp(p_gate * np.log(p_cor) + (1 - p_gate) * np.log(p_lar)),
        "D · enrutado ORACULO (cota superior)": np.where(corto, p_cor, p_lar),
    }
    filas = [dict(brazo=k, ambito="todas", **mt(y, v)) for k, v in mezcla.items()]
    for k, v in mezcla.items():
        filas.append(dict(brazo=k, ambito="solo cortas", **mt(y[corto], v[corto])))
        filas.append(dict(brazo=k, ambito="solo largas",
                          **mt(y[~corto], v[~corto])))
    # E: el experto corto contra el global, los dos EN tareas cortas
    filas.append(dict(brazo="E · experto corto", ambito="solo cortas",
                      **mt(y[corto], p_cor[corto])))
    filas.append(dict(brazo="E · experto largo", ambito="solo largas",
                      **mt(y[~corto], p_lar[~corto])))
    t = pd.DataFrame(filas)
    cols = ["brazo", "ambito", "PRED25", "MdMRE", "factor2", "MAE_h", "sesgo",
            "spearman", "sd_log_pred"]

    for amb in ("todas", "solo cortas", "solo largas"):
        print(f"\n{S}\n{amb.upper()}\n{S}")
        print(t[t.ambito == amb][cols].round(4).to_string(index=False))

    print(f"\n{S}\nLO QUE DECIDE\n{S}")
    g25 = float(t[(t.brazo.str.startswith("A")) & (t.ambito == "todas")].PRED25.iloc[0])
    o25 = float(t[(t.brazo.str.startswith("D")) & (t.ambito == "todas")].PRED25.iloc[0])
    print(f"  global {g25:.1%} · ORACULO {o25:.1%} ({(o25 - g25) * 100:+.1f} pts)")
    if o25 - g25 < .01:
        print("  El oraculo NO mejora: ninguna compuerta puede arreglarlo y la via")
        print("  de mezcla de expertos esta muerta, se sepa o no el regimen.")
    else:
        print("  El oraculo SI mejora: hay techo, y lo que falta es la compuerta")
        print(f"  (AUC actual {auc:.3f}).")
    gc = t[(t.brazo.str.startswith("A")) & (t.ambito == "solo cortas")].iloc[0]
    ec = t[(t.brazo.str.startswith("E · experto corto"))].iloc[0]
    print(f"\n  EN TAREAS CORTAS, el experto contra el global:")
    print(f"    sesgo      x{gc.sesgo:.2f} -> x{ec.sesgo:.2f}  (1,0 es lo justo)")
    print(f"    ±25 %      {gc.PRED25:.1%} -> {ec.PRED25:.1%}")
    print(f"    spearman   {gc.spearman:.3f} -> {ec.spearman:.3f}  "
          f"(¿ordena DENTRO del regimen?)")
    if ec.spearman < .15:
        print("    El experto corrige la ESCALA pero no ordena: dentro del regimen")
        print("    corto el texto no dice cual tarea cuesta mas.")
    t.to_csv(OUT / "resumen.csv", index=False)
    json.dump({"n": int(len(sw)), "corte": CORTE, "auc_compuerta": float(auc),
               "pred25_global": g25, "pred25_oraculo": o25},
              open(OUT / "meta.json", "w"), indent=1)
    np.savez_compressed(OUT / "predicciones.npz", p_glob=p_glob, p_cor=p_cor,
                        p_lar=p_lar, p_gate=p_gate, y=y, corto=corto)
    print(f"\nescrito {OUT}/resumen.csv · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
