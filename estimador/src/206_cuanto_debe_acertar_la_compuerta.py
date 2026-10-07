"""E91 - El experto corto funciona. ¿Que compuerta haria falta para aprovecharlo?

E90 dejo tres cosas medidas:

  el experto corto ARREGLA la escala en su regimen   sesgo x2,46 -> x0,85
  el enrutado ORACULO vale +5,8 puntos de ±25 %      15,1 % -> 20,9 %
  la compuerta real no llega                          AUC 0,646

Es el primer sitio del proyecto donde hay techo de verdad. Toda la ganancia
esta en SABER a que regimen pertenece la tarea, que es informacion que el texto
da a medias. Asi que la pregunta util no es si la mezcla sirve -sirve- sino
cuanto tendria que acertar la compuerta para que compense, y si se puede
arrancar una parte sin una compuerta perfecta.

  1  COMPUERTAS ALTERNATIVAS   la prediccion del propio modelo global ordena
     tareas (spearman 0,318). ¿Es mejor compuerta que el clasificador? ¿Y las
     dos juntas?
  2  LA CURVA                  compuertas simuladas de AUC creciente, para ver
     donde esta el punto de equilibrio y cuanto se saca en cada nivel
  3  ABSTENCION                enrutar SOLO cuando la compuerta esta segura y
     caer al global en el resto. Es lo unico de aqui que se puede desplegar hoy

No se reentrena nada: se leen las predicciones fuera de pliegue de E90.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
OUT = Path("results/compuerta_necesaria")
ENT = Path("results/experto_cortas")
S = "=" * 100
PLIEGUES = 5


def pred25(y, p):
    return float(np.mean(np.abs(np.clip(p, 1e-6, None) - y) / y <= .25))


def mide(y, p):
    r = np.abs(np.clip(p, 1e-6, None) - y) / y
    return dict(PRED25=float(np.mean(r <= .25)), MdMRE=float(np.median(r)),
                MAE_h=float(np.abs(np.clip(p, 1e-6, None) - y).mean()))


def compuerta_simulada(corto, auc_obj, rng):
    """Una puntuacion con el AUC pedido respecto de `corto`.

    Se construye mezclando la verdad con ruido gaussiano: para dos normales de
    igual varianza separadas d, AUC = Phi(d/sqrt(2)), asi que d sale de invertir.
    """
    from scipy.stats import norm
    if auc_obj >= .9999:
        return corto.astype(float)
    d = norm.ppf(auc_obj) * np.sqrt(2)
    return corto * d + rng.normal(0, 1, len(corto))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    z = np.load(ENT / "predicciones.npz")
    y, corto = z["y"], z["corto"].astype(bool)
    p_glob, p_cor, p_lar, p_gate = z["p_glob"], z["p_cor"], z["p_lar"], z["p_gate"]
    df = pd.read_parquet("data/processed/agix_effort_v4.parquet")
    todo = df[df.is_software_task == 1].reset_index(drop=True)
    sw = todo[~todo.fuente_jira.eq("Apache")].reset_index(drop=True)
    proy = sw.project.to_numpy()
    assert len(proy) == len(y)
    base = mide(y, p_glob)
    orac = mide(y, np.where(corto, p_cor, p_lar))
    print(f"{len(y):,} tareas · global ±25 % {base['PRED25']:.1%} · "
          f"oraculo {orac['PRED25']:.1%} · {time.time() - t0:.0f} s", flush=True)

    # ---------- 1. compuertas alternativas ----------
    # la prediccion del global, como puntuacion de "esto es corto" (al reves)
    s_pred = -np.log(np.clip(p_glob, 1e-6, None))
    # las dos juntas, ajustado fuera de pliegue por proyecto
    X = np.column_stack([p_gate, np.log(np.clip(p_glob, 1e-6, None))])
    s_dos = np.zeros(len(y))
    for a, b in GroupKFold(PLIEGUES).split(X, corto, proy):
        lr = LogisticRegression(max_iter=1000).fit(X[a], corto[a])
        s_dos[b] = lr.predict_proba(X[b])[:, 1]
    cands = {"clasificador (E90)": p_gate,
             "la prediccion del global": s_pred,
             "las dos juntas": s_dos}
    print(f"\n{S}\n1. ¿HAY UNA COMPUERTA MEJOR?\n{S}")
    filas_c = []
    for k, s in cands.items():
        a = float(roc_auc_score(corto, s))
        # umbral que maximiza el acierto, para dar la mejor version de cada una
        u = np.quantile(s, 1 - corto.mean())
        p = np.where(s >= u, p_cor, p_lar)
        m = mide(y, p)
        filas_c.append(dict(compuerta=k, AUC=a, acierto=float(np.mean((s >= u) == corto)),
                            **m))
        print(f"  {k:26s} AUC {a:.4f} · acierto {np.mean((s >= u) == corto):.1%} · "
              f"±25 % enrutando {m['PRED25']:.1%}")
    print(f"  {'(sin enrutar: el global)':26s} {'':10s}   {'':17s} "
          f"±25 % {base['PRED25']:.1%}")

    # ---------- 2. la curva: cuanto AUC hace falta ----------
    print(f"\n{S}\n2. ¿CUANTO TENDRIA QUE ACERTAR LA COMPUERTA?\n{S}")
    filas_s = []
    for auc_obj in (.65, .70, .75, .80, .85, .90, .95, 1.0):
        ms = []
        for s_ in range(20):
            rng = np.random.default_rng(2000 + s_)
            sc = compuerta_simulada(corto, auc_obj, rng)
            u = np.quantile(sc, 1 - corto.mean())
            ms.append(mide(y, np.where(sc >= u, p_cor, p_lar)))
        d = {c: float(np.mean([x[c] for x in ms])) for c in ms[0]}
        filas_s.append(dict(AUC=auc_obj, **d))
        marca = "  <- el global" if abs(d["PRED25"] - base["PRED25"]) < .004 else ""
        print(f"  AUC {auc_obj:.2f} · ±25 % {d['PRED25']:.1%} · "
              f"MdMRE {d['MdMRE']:.3f} · MAE {d['MAE_h']:.2f} h{marca}")
    eq = [f["AUC"] for f in filas_s if f["PRED25"] >= base["PRED25"]]
    print(f"\n  el global esta en {base['PRED25']:.1%}; enrutar empata o gana a "
          f"partir de AUC ~{min(eq):.2f}" if eq else
          "\n  ninguna compuerta simulada alcanza al global")
    print(f"  la mejor compuerta real que tenemos: "
          f"AUC {max(f['AUC'] for f in filas_c):.3f}")

    # ---------- 3. abstencion: enrutar solo cuando se esta seguro ----------
    print(f"\n{S}\n3. ENRUTAR SOLO CUANDO LA COMPUERTA ESTA SEGURA\n{S}")
    mejor = max(filas_c, key=lambda f: f["AUC"])["compuerta"]
    s = cands[mejor]
    r = rankdata(s) / len(s)
    filas_a = []
    print(f"  compuerta: {mejor}")
    for frac in (0, .05, .10, .20, .30, .50, 1.0):
        if frac == 0:
            p = p_glob.copy()
        else:
            lo, hi = frac / 2, 1 - frac / 2
            p = p_glob.copy()
            p[r <= lo] = p_lar[r <= lo]
            p[r >= hi] = p_cor[r >= hi]
        m = mide(y, p)
        # que acierto tiene la compuerta en la franja que SI enruta
        if frac == 0:
            ac = np.nan
        else:
            z_ = (r <= frac / 2) | (r >= 1 - frac / 2)
            ac = float(np.mean(((r >= 1 - frac / 2)[z_]) == corto[z_]))
        filas_a.append(dict(fraccion_enrutada=frac, acierto_en_la_franja=ac, **m))
        print(f"  se enruta el {frac:5.0%} mas seguro · acierto ahi "
              f"{ac if ac == ac else float('nan'):.1%} · ±25 % {m['PRED25']:.1%} · "
              f"MdMRE {m['MdMRE']:.3f}")

    mejor_a = max(filas_a, key=lambda f: f["PRED25"])
    print(f"\n{S}\nLECTURA\n{S}")
    print(f"  global {base['PRED25']:.1%} · oraculo {orac['PRED25']:.1%} · "
          f"mejor abstencion {mejor_a['PRED25']:.1%} "
          f"(enrutando el {mejor_a['fraccion_enrutada']:.0%})")
    if mejor_a["PRED25"] - base["PRED25"] > .005:
        print(f"  SE PUEDE sacar {(mejor_a['PRED25'] - base['PRED25']) * 100:+.1f} "
              f"puntos hoy, enrutando solo donde la compuerta esta segura.")
        print(f"  Es el {(mejor_a['PRED25'] - base['PRED25']) / (orac['PRED25'] - base['PRED25']):.0%} "
              f"de lo que daria el oraculo.")
    else:
        print("  La abstencion no rescata nada: la compuerta no es fiable ni en")
        print("  sus propios extremos.")
    pd.DataFrame(filas_c).to_csv(OUT / "compuertas.csv", index=False)
    pd.DataFrame(filas_s).to_csv(OUT / "curva_auc.csv", index=False)
    pd.DataFrame(filas_a).to_csv(OUT / "abstencion.csv", index=False)
    json.dump({"pred25_global": base["PRED25"], "pred25_oraculo": orac["PRED25"],
               "mejor_auc_real": max(f["AUC"] for f in filas_c),
               "mejor_abstencion": mejor_a}, open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/ · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
