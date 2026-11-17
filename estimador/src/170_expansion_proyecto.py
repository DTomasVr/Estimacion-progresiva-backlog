"""E63 - Si el modelo comprime la escala del proyecto, expandir deberia ayudar.

E61 deja el sesgo del total ordenado con el tamano tipico de tarea del proyecto:

    menos de 0,5 h por tarea   x5,36
    1-2 h                      x1,31
    2-4 h                      x0,89
    mas de 4 h                 x0,61

O sea que a los proyectos pequenos les predice de mas y a los grandes de menos:
compresion, la misma que se ve por tarea. Y la compresion tiene un arreglo
conocido -- expandir alrededor del centro -- SI la pendiente lo pide.

La prueba es la regresion de log(real) sobre log(estimado) a nivel de proyecto:

    pendiente > 1   el estimado esta comprimido y expandir ayuda
    pendiente < 1   el estimado ya esta demasiado extendido para lo que sabe, y
                    expandir empeora

A nivel de TAREA esa pendiente salio 0,898, o sea menor que uno, y por eso todos
los intentos de expandir fracasaron: el sesgo por decil del valor REAL era
regresion a la media, no un sesgo corregible. A nivel de proyecto la pregunta
esta abierta, porque al sumar se cancela ruido y la senal puede quedar mas limpia.

  0  sin tocar
  1  pendiente global, estimada dejando fuera cada proyecto
  2  pendiente y corte (recalibracion lineal en log)
  3  isotonica, por si la relacion no es una recta
"""
import json
import os
import sys
import warnings
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "6"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 215)
OUT = Path("results/expansion_proyecto")
S = "=" * 116
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
                MdMRE=float(np.median(r)),
                R2_log=float(1 - ((lr - np.log(pred)) ** 2).mean() / lr.var()),
                PRED25=float(np.mean(r <= .25)), PRED50=float(np.mean(r <= .50)),
                factor2=float(np.mean(np.maximum(pred / real, real / pred) <= 2)),
                SSPB_pc=float(np.sign(m) * (np.exp(abs(m)) - 1) * 100))


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
    lr, lp = np.log(real), np.log(pred)
    b, a = np.polyfit(lp, lr, 1)
    print(f"{len(real)} proyectos")
    print(f"  log(real) = {a:+.3f} {b:+.3f} x log(estimado)  ->  pendiente {b:.3f}")
    print(f"  sd del estimado {lp.std():.3f} · sd del real {lr.std():.3f} · "
          f"razon {lp.std() / lr.std():.3f}")
    print(f"  {'la pendiente pide EXPANDIR' if b > 1.02 else ('pide CONTRAER' if b < .98 else 'no pide nada')}")

    # el mismo diagnostico por TAMANO TIPICO DE TAREA predicho, que es lo unico
    # observable en frio; si ahi tambien hay pendiente, es corregible
    tt = np.log(g.est.median().to_numpy())
    b2, a2 = np.polyfit(tt, np.log(g.real.median().to_numpy()), 1)
    print(f"\n  y sobre el tamano tipico de TAREA: pendiente {b2:.3f} "
          f"(sd predicho {tt.std():.3f} · sd real {np.log(g.real.median()).std():.3f})")

    brazos = {"0 · sin tocar": pred}
    p1 = np.zeros(len(real))
    p2 = np.zeros(len(real))
    p3 = np.zeros(len(real))
    for i in range(len(real)):
        k = np.ones(len(real), bool)
        k[i] = False
        bb = float(np.polyfit(lp[k], lr[k], 1)[0])
        c = float(np.mean(lp[k]))
        p1[i] = np.exp(c + bb * (lp[i] - c))                 # solo pendiente
        bb2, aa2 = np.polyfit(lp[k], lr[k], 1)
        p2[i] = np.exp(aa2 + bb2 * lp[i])                    # pendiente y corte
        iso = IsotonicRegression(out_of_bounds="clip").fit(lp[k], lr[k])
        p3[i] = np.exp(float(iso.predict([lp[i]])[0]))
    brazos["1 · solo pendiente"] = p1
    brazos["2 · recalibracion lineal en log"] = p2
    brazos["3 · isotonica"] = p3
    t = pd.DataFrame([dict(brazo=n, **mt(real, v)) for n, v in brazos.items()])
    print(f"\n{S}\nTOTAL DEL PROYECTO\n{S}")
    print(t[["brazo", "MAE_h", "RMSE_h", "MdMRE", "R2_log", "PRED25", "PRED50",
             "factor2", "SSPB_pc"]].round(4).to_string(index=False))
    t.to_csv(OUT / "resumen.csv", index=False)
    json.dump({"pendiente": float(b), "corte": float(a),
               "pendiente_tamano_tarea": float(b2)},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/resumen.csv")


if __name__ == "__main__":
    main()


def descomponer():
    """log(total) = log(numero de tareas) + log(tamano tipico).

    El numero de tareas se conoce EXACTO y el tamano tipico sale comprimido
    (pendiente 1,54). Al sumar, el componente exacto diluye al comprimido y la
    pendiente del total sale 1,0; pero eso no significa que no haya nada que
    corregir, significa que hay que corregir el componente que toca.

    Se ajusta  log(real) = c0 + c1 log(n) + c2 log(tamano predicho)  dejando
    fuera cada proyecto. Si c2 sale claramente mayor que 1, expandir el tamano
    -- y solo el tamano -- deberia ayudar.
    """
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
    n = g.size().to_numpy().astype(float)
    ts = pred / n                       # tamano tipico predicho
    X = np.column_stack([np.ones(len(n)), np.log(n), np.log(ts)])
    c = np.linalg.lstsq(X, np.log(real), rcond=None)[0]
    print(f"\n{S}\nDESCOMPOSICION EN NUMERO DE TAREAS Y TAMANO TIPICO\n{S}")
    print(f"  log(real) = {c[0]:+.3f} {c[1]:+.3f} log(n) {c[2]:+.3f} log(tamano)")
    print(f"  si el coeficiente del tamano fuese > 1 claramente, expandirlo ayudaria")
    p4 = np.zeros(len(real))
    for i in range(len(real)):
        k = np.ones(len(real), bool)
        k[i] = False
        cc = np.linalg.lstsq(X[k], np.log(real[k]), rcond=None)[0]
        p4[i] = float(np.exp(X[i] @ cc))
    t = pd.DataFrame([dict(brazo="0 · sin tocar", **mt(real, pred)),
                      dict(brazo="4 · n y tamano por separado", **mt(real, p4))])
    print()
    print(t[["brazo", "MAE_h", "RMSE_h", "MdMRE", "R2_log", "PRED25", "PRED50",
             "factor2", "SSPB_pc"]].round(4).to_string(index=False))
    t.to_csv(OUT / "descomposicion.csv", index=False)


if __name__ == "__main__":
    descomponer()
