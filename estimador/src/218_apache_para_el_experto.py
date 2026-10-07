"""E103 - La curva no esta plana y hay 65.003 tareas sin usar. Antes de inventar
datos, gastemos los que ya tenemos.

E102 midio tres cosas sobre el experto en tareas cortas:

  sobreajusta            brecha 0,3064 en log; con iter=800 el error de
                         entrenamiento cae a 0,07 y el de fuera no se mueve
  la curva NO esta plana del 50 % al 100 % de los proyectos el error baja
                         0,0171 y el spearman se DOBLA, de 0,050 a 0,100
  regularizar ayuda poco l2=10 mejora 0,004

Las dos primeras dicen lo mismo: al experto corto le faltan datos. Y antes de
generarlos sinteticamente -donde la etiqueta tendria que salir de algun sitio-
hay 65.003 tareas de Apache que el proyecto aparta a proposito.

Apache se excluyo por una razon buena: son el 72 % del corpus y el modelo
acabaria aprendiendo a hacer Apache. Pero esa razon vale para el modelo que se
EVALUA, no necesariamente para un experto de regimen que solo tiene que
aprender como se escriben las tareas de pocas horas.

La prueba es limpia porque la evaluacion no cambia: siempre fuera de pliegue
sobre proyectos no Apache nunca vistos. Apache solo entra al ENTRENAMIENTO.

  A  experto corto, solo no Apache            lo de E102
  B  + Apache                                 4x datos
  C  + Apache, con peso 1/4                   por si domina demasiado
  D  + Apache, solo la mitad                  punto intermedio de la curva

Si B sube el spearman como predice la curva, no hacen falta datos sinteticos.
Si no sube, quiere decir que las tareas cortas de Apache no se parecen a las
nuestras, y eso tambien es una respuesta: los sinteticos tendrian el mismo
problema, solo que peor.
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
from sklearn.model_selection import GroupKFold  # noqa: E402

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/apache_para_el_experto")
S = "=" * 100
PLIEGUES, CORTE = 3, 2.0
PAR = dict(max_iter=250, max_depth=6, learning_rate=.06, l2_regularization=1.,
           random_state=0)


def met(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, None)
    r = np.abs(p - y) / y
    return dict(err_log=float(np.median(np.abs(np.log(p) - np.log(y)))),
                rho=float(spearmanr(p, y).statistic),
                PRED25=float(np.mean(r <= .25)),
                sesgo=float(np.median(p / y)),
                sd_log_pred=float(np.std(np.log(p))))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet("data/processed/agix_effort_v4.parquet")
    todo = df[df.is_software_task == 1].reset_index(drop=True)
    V = np.load(CACHE / "e5-large-todo.npy").astype(np.float32)
    ap = todo.fuente_jira.eq("Apache").to_numpy()
    y_all = todo.effort_hours.to_numpy(float)
    corto_all = y_all <= CORTE
    # las cortas de Apache, que es lo unico que se va a anadir
    i_ap = np.flatnonzero(ap & corto_all)
    print(f"no Apache: {int((~ap).sum()):,} tareas · "
          f"Apache: {int(ap.sum()):,} · cortas de Apache: {len(i_ap):,} · "
          f"{time.time() - t0:.0f} s", flush=True)
    print(f"  horas medianas · no Apache {np.median(y_all[~ap & corto_all]):.2f} h"
          f" · Apache {np.median(y_all[i_ap]):.2f} h", flush=True)

    sw = todo[~ap].reset_index(drop=True)
    Vn = V[~ap]
    y = sw.effort_hours.to_numpy(float)
    ly = np.log(y)
    proy = sw.project.to_numpy()
    corto = y <= CORTE

    # Nada de sample_weight: esta medido que hace a HGB 12 veces mas lento. Para
    # dar menos peso a Apache se SUBMUESTREA, que consigue lo mismo y es gratis.
    brazos = {"A · solo no Apache": 0,
              "B · + Apache entero": len(i_ap),
              "C · + un cuarto de Apache": len(i_ap) // 4,
              "D · + media Apache": len(i_ap) // 2}
    filas = []
    for etq, n_ap in brazos.items():
        ms = []
        for nf, (a, b) in enumerate(GroupKFold(PLIEGUES).split(Vn, y, proy)):
            tr = a[corto[a]]
            X, Y = Vn[tr], ly[tr]
            if n_ap:
                rng = np.random.default_rng(1700 + nf)
                sel = rng.choice(i_ap, n_ap, replace=False)
                X = np.vstack([X, V[sel]])
                Y = np.concatenate([Y, np.log(y_all[sel])])
            mo = HistGradientBoostingRegressor(**PAR).fit(X, Y)
            bb = b[corto[b]]
            ms.append(met(y[bb], np.exp(mo.predict(Vn[bb]))))
            print(f"  {etq} pliegue {nf + 1}/{PLIEGUES} · n={len(X):,} · "
                  f"{time.time() - t0:.0f} s", flush=True)
        filas.append(dict(brazo=etq, n_entreno=int(len(X)),
                          **{c: float(np.mean([x[c] for x in ms]))
                             for c in ms[0]}))
    t = pd.DataFrame(filas)
    print(f"\n{S}\nEL EXPERTO CORTO, EVALUADO SIEMPRE EN PROYECTOS NO APACHE "
          f"NUNCA VISTOS\n{S}")
    print(t.round(4).to_string(index=False))

    print(f"\n{S}\nLECTURA\n{S}")
    a = t[t.brazo.str.startswith("A")].iloc[0]
    b_ = t[t.brazo.str.startswith("B")].iloc[0]
    print(f"  spearman  {a.rho:.4f} -> {b_.rho:.4f}  ({b_.rho - a.rho:+.4f})")
    print(f"  error log {a.err_log:.4f} -> {b_.err_log:.4f}  "
          f"({b_.err_log - a.err_log:+.4f})")
    print(f"  ±25 %     {a.PRED25:.1%} -> {b_.PRED25:.1%}")
    print(f"  n entreno {int(a.n_entreno):,} -> {int(b_.n_entreno):,}")
    if b_.rho - a.rho > .02:
        print("\n  Apache SIRVE. La curva de E102 no mentia y no hacen falta datos")
        print("  sinteticos: los datos ya estaban en el repositorio.")
    elif b_.err_log < a.err_log - .005:
        print("\n  Apache baja el error pero no mejora el orden. Sirve para la")
        print("  escala, no para distinguir dentro del regimen.")
    else:
        print("\n  Apache NO sirve. Sus tareas cortas no se parecen a las nuestras,")
        print("  y eso es un aviso para lo sintetico: mas texto del tipo que no")
        print("  toca no arregla nada.")
    t.to_csv(OUT / "resumen.csv", index=False)
    json.dump({"cortas_apache": int(len(i_ap)),
               "rho_sin": float(a.rho), "rho_con": float(b_.rho),
               "err_sin": float(a.err_log), "err_con": float(b_.err_log)},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/resumen.csv · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
