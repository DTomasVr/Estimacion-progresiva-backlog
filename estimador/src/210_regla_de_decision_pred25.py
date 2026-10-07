"""E95 - Estabamos optimizando la funcion equivocada.

EL HALLAZGO QUE LO DESTAPA: predecir la constante 2,45 h para TODAS las tareas
da un +-25 % del 18,20 %. Nuestro modelo, con e5-large y un apilado, da 15,06 %.
Un numero fijo le gana por 3,1 puntos.

No es un fallo del modelo. Es que entrenamos con error cuadratico sobre log,
cuyo minimizador es la MEDIANA condicional, y despues reportamos PRED(25), que
es otra cosa completamente distinta. Shepperd y Kitchenham lo dejaron escrito:
MMRE y PRED(x) son medidas de la dispersion y la CURTOSIS de z = estimado/real,
asi que premian concentrar z cerca de 1.

Y la distribucion del esfuerzo es un peine, no una campana: diez valores -1 h,
0,5 h, 2 h, 4 h, 3 h...- cubren el 46,8 % del corpus. La mediana condicional
cae ENTRE dientes; una constante bien puesta cae ENCIMA de uno.

LA REGLA CORRECTA para PRED(25) es de teoria de la decision:

    p*(x) = argmax_p  P( y en [p/1,25 , p/0,75] | x )

o sea, el centro de la ventana relativa de maxima masa. No la mediana, no la
media: la MODA en escala relativa.

Para estimar P(y|x) se reusa el canal de analogia que ya existe: la
distribucion empirica de las horas de los K vecinos mas parecidos en el espacio
del codificador. Se contrae hacia la distribucion global, porque K vecinos son
pocos, con un peso que se elige en validacion.

  A  lo actual (mediana condicional)
  B  la mejor constante                      la vergonzosa
  C  regla de decision sobre la marginal     debe dar ~B, es el control
  D  regla de decision sobre los K vecinos   <- lo nuevo
  E  D contraido hacia la marginal
  F  ajustar lo actual a la rejilla de valores frecuentes

Se reportan TAMBIEN MAE, SA y sesgo, porque la regla busca la moda y eso puede
costar en las metricas de magnitud. Si se gana en PRED(25) perdiendo en MAE hay
que decirlo, no esconderlo.

OJO: esto cambia el valor que se MUESTRA por tarea. El total del proyecto sigue
sumando la cabeza de la media, que es otra cosa (HU-65-3, HU-66-4).
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
from sklearn.model_selection import GroupKFold  # noqa: E402

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/regla_decision")
S = "=" * 104
PLIEGUES, K_VEC = 5, 200
TOL = .25
PESOS = (0., .1, .25, .5, .75)      # contraccion hacia la marginal
LOTE = 2048


def metricas(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, None)
    r = np.abs(p - y) / y
    mar = np.abs(np.median(y) - y)          # la referencia de SA: predecir la mediana
    return dict(PRED25=float(np.mean(r <= TOL)), MdMRE=float(np.median(r)),
                factor2=float(np.mean(np.maximum(p / y, y / p) <= 2)),
                MAE_h=float(np.abs(p - y).mean()),
                SA=float(1 - np.abs(p - y).mean() / mar.mean()),
                sesgo=float(np.median(p / y)),
                spearman=float(spearmanr(p, y).statistic))


def mejor_punto(masa, rej, cand_idx, lo_idx, hi_idx):
    """argmax_p de la masa en [p/1,25 , p/0,75], con la ventana precalculada."""
    acum = np.concatenate([np.zeros((len(masa), 1)), np.cumsum(masa, 1)], 1)
    ganancia = acum[:, hi_idx] - acum[:, lo_idx]
    return rej[cand_idx[np.argmax(ganancia, 1)]]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet("data/processed/agix_effort_v4.parquet")
    todo = df[df.is_software_task == 1].reset_index(drop=True)
    V = np.load(CACHE / "e5-large-todo.npy")
    noap = (~todo.fuente_jira.eq("Apache")).to_numpy()
    sw, V = todo[noap].reset_index(drop=True), V[noap].astype(np.float32)
    y = sw.effort_hours.to_numpy(float)
    proy = sw.project.to_numpy()
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    z = np.load("results/experto_cortas/predicciones.npz")
    p_act = z["p_glob"]
    assert len(p_act) == len(y)

    # rejilla de soporte: los valores que la gente registra de verdad
    rej = np.unique(y)
    rej = rej[(rej > 0) & (rej <= np.quantile(y, .999))]
    idx_val = np.searchsorted(rej, y)
    idx_val = np.clip(idx_val, 0, len(rej) - 1)
    # candidatos y su ventana relativa, precalculados una vez
    cand_idx = np.arange(len(rej))
    lo_idx = np.searchsorted(rej, rej / (1 + TOL), side="left")
    hi_idx = np.searchsorted(rej, rej / (1 - TOL), side="right")
    print(f"{len(y):,} tareas · {len(rej)} valores distintos en la rejilla · "
          f"K={K_VEC} vecinos · {time.time() - t0:.0f} s", flush=True)

    p_marg = np.zeros(len(y))
    p_vec = np.zeros(len(y))
    p_mix = {w: np.zeros(len(y)) for w in PESOS}
    for nf, (a, b) in enumerate(GroupKFold(PLIEGUES).split(V, y, proy)):
        # distribucion marginal del pliegue de entrenamiento
        marg = np.bincount(idx_val[a], minlength=len(rej)).astype(float)
        marg /= marg.sum()
        p_marg[b] = mejor_punto(np.repeat(marg[None], len(b), 0), rej,
                                cand_idx, lo_idx, hi_idx)
        # distribucion de los K vecinos, por lotes para no reventar la memoria
        Va = V[a]
        for i in range(0, len(b), LOTE):
            bb = b[i:i + LOTE]
            sim = V[bb] @ Va.T
            vec = np.argpartition(-sim, K_VEC, axis=1)[:, :K_VEC]
            masa = np.zeros((len(bb), len(rej)))
            filas = np.repeat(np.arange(len(bb)), K_VEC)
            np.add.at(masa, (filas, idx_val[a][vec].ravel()), 1.0)
            masa /= masa.sum(1, keepdims=True)
            p_vec[bb] = mejor_punto(masa, rej, cand_idx, lo_idx, hi_idx)
            for w in PESOS:
                p_mix[w][bb] = mejor_punto((1 - w) * masa + w * marg[None], rej,
                                           cand_idx, lo_idx, hi_idx)
        print(f"  pliegue {nf + 1}/{PLIEGUES} · {time.time() - t0:.0f} s", flush=True)

    # F: ajustar la prediccion actual al valor frecuente mas cercano
    frec = rej[np.argsort(-np.bincount(idx_val, minlength=len(rej)))[:12]]
    p_snap = frec[np.argmin(np.abs(np.log(p_act[:, None] / frec[None])), 1)]
    # B: la mejor constante, elegida fuera de pliegue
    cte = np.zeros(len(y))
    for a, b in GroupKFold(PLIEGUES).split(V, y, proy):
        sc = [np.mean(np.abs(c - y[a]) / y[a] <= TOL) for c in rej]
        cte[b] = rej[int(np.argmax(sc))]

    brazos = {
        "A · lo actual (mediana condicional)": p_act,
        "B · la mejor constante": cte,
        "C · regla de decision, marginal": p_marg,
        "D · regla de decision, K vecinos": p_vec,
        "F · ajustar lo actual a la rejilla": p_snap,
    }
    for w in PESOS:
        if w:
            brazos[f"E · vecinos contraidos w={w:g}"] = p_mix[w]
    t = pd.DataFrame([dict(brazo=k, **metricas(y, v)) for k, v in brazos.items()])
    cols = ["brazo", "PRED25", "MdMRE", "factor2", "MAE_h", "SA", "sesgo", "spearman"]
    print(f"\n{S}\nLA REGLA DE DECISION ({len(y):,} tareas, fuera de pliegue)\n{S}")
    print(t[cols].round(4).to_string(index=False))

    print(f"\n{S}\nLECTURA\n{S}")
    a = float(t[t.brazo.str.startswith("A")].PRED25.iloc[0])
    bb_ = float(t[t.brazo.str.startswith("B")].PRED25.iloc[0])
    best = t.loc[t.PRED25.idxmax()]
    print(f"  lo actual          {a:.2%}")
    print(f"  la mejor constante {bb_:.2%}  ({(bb_ - a) * 100:+.1f} puntos)")
    print(f"  LO MEJOR           {best.PRED25:.2%}  ({best.brazo})")
    print(f"                     {(best.PRED25 - a) * 100:+.1f} puntos sobre lo actual, "
          f"{(best.PRED25 - bb_) * 100:+.1f} sobre la constante")
    am = float(t[t.brazo.str.startswith("A")].MAE_h.iloc[0])
    print(f"\n  el precio: MAE {am:.2f} h -> {best.MAE_h:.2f} h · "
          f"SA {float(t[t.brazo.str.startswith('A')].SA.iloc[0]):.3f} -> {best.SA:.3f}")
    if best.MAE_h > am:
        print("  Se gana en PRED(25) y se PIERDE en MAE. Hay que reportar las dos:")
        print("  la regla busca la moda, que no es el minimizador del error absoluto.")
    t.to_csv(OUT / "resumen.csv", index=False)
    json.dump({"n": int(len(y)), "K": K_VEC, "actual": a, "constante": bb_,
               "mejor": float(best.PRED25), "mejor_brazo": best.brazo},
              open(OUT / "meta.json", "w"), indent=1)
    np.save(OUT / "pred_regla.npy", brazos[best.brazo])
    print(f"\nescrito {OUT}/resumen.csv · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
