"""E47 - Corregir el total con rasgos del PROYECTO, no solo sumando tareas.

Hasta ahora el total es una suma ciega: se estiman las tareas una a una y se
suman. Eso ignora que el proyecto, mirado entero, dice cosas sobre si mismo. Un
proyecto de 300 tareas con titulos de tres palabras y vecinos que no concuerdan
no se parece a uno de 30 tareas bien descritas, y no hay razon para que el error
de la suma sea el mismo en los dos.

La idea es aprender el factor de correccion del total a partir de rasgos
agregados del proyecto:

    log(real / estimado)  ~  numero de tareas, dispersion de lo estimado,
                             cuanto se parece a lo que ya se ha visto, longitud
                             media de los titulos, acuerdo entre vecinos...

Hay un peligro evidente y hay que tratarlo en serio: son 114 proyectos. Con esa
muestra, un modelo con muchos rasgos aprende los proyectos concretos y no el
patron. Por eso se usa un ridge con pocos rasgos, se valida dejando fuera UN
PROYECTO cada vez, y se compara contra el brazo de no corregir nada. Si la
correccion no gana por bastante, no vale: la ganancia aparente vendria de haber
mirado los mismos proyectos con los que se mide.
"""
import json
import os
import sys
import warnings
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "4"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.model_selection import LeaveOneOut  # noqa: E402

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 215)
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/correccion_proyecto")
S = "=" * 118
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
                MdMRE=float(np.median(r)), MMRE=float(r.mean()),
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
    ly = np.log(y)
    proy = sw.project.to_numpy()
    P = np.load("results/analogia_cv/pred.npy")
    lp, disp = P[:, 0], P[:, 2]
    E5 = np.load(CACHE / "e5-large.npy")
    Vn = E5 / np.maximum(np.linalg.norm(E5, axis=1, keepdims=True), 1e-9)

    e = np.exp(ly - lp)
    tramo = pd.qcut(lp, 6, labels=False, duplicates="drop")
    ft = pd.DataFrame({"t": tramo, "e": e}).groupby("t").e.mean()
    est = np.exp(lp) * ft.reindex(tramo).to_numpy()

    d = pd.DataFrame({"p": proy, "real": y, "est": est, "lp": lp, "disp": disp,
                      "nw": sw.title.fillna("").str.split().str.len().to_numpy()})
    tam = d.groupby("p").size()
    d = d[d.p.isin(tam[tam >= MIN_TAREAS].index)].reset_index(drop=True)

    g = d.groupby("p")
    X = pd.DataFrame({
        "log_n": np.log(g.size()),
        "log_est": np.log(g.est.sum()),
        "sd_lp": g.lp.std(),
        "iqr_lp": g.lp.quantile(.75) - g.lp.quantile(.25),
        "disp_media": g.disp.mean(),
        "nw_media": g.nw.mean(),
        "frac_corta": g.apply(lambda q: float((q.nw <= 5).mean())),
        "frac_grande": g.apply(lambda q: float((q.est >= 8).mean())),
    })
    real = g.real.sum()
    est_t = g.est.sum()
    obj = np.log(real / est_t).to_numpy()
    print(f"{len(X)} proyectos · {X.shape[1]} rasgos agregados")
    print(f"el factor que habria que aplicar va de x{np.exp(obj).min():.2f} a "
          f"x{np.exp(obj).max():.2f}, mediana x{np.exp(np.median(obj)):.2f}")

    Xs = ((X - X.mean()) / (X.std() + 1e-9)).to_numpy()
    pred = np.zeros(len(X))
    for a, b in LeaveOneOut().split(Xs):
        m = RidgeCV(alphas=np.logspace(-1, 4, 30)).fit(Xs[a], obj[a])
        pred[b] = m.predict(Xs[b])
    m = RidgeCV(alphas=np.logspace(-1, 4, 30)).fit(Xs, obj)
    pesos = pd.Series(m.coef_, index=X.columns).sort_values(key=abs, ascending=False)
    print("\npesos del corrector (ajustado con todo, solo para leerlo):")
    print("  " + " · ".join(f"{k} {v:+.3f}" for k, v in pesos.items()))
    r2 = 1 - ((obj - pred) ** 2).sum() / ((obj - obj.mean()) ** 2).sum()
    print(f"\nR2 del corrector, dejando fuera un proyecto cada vez: {r2:+.4f}")
    print("  si sale negativo, el corrector es peor que no corregir nada y no vale")

    brazos = {"0 · sin corregir": est_t.to_numpy(),
              "1 · corregido con rasgos del proyecto": est_t.to_numpy() * np.exp(pred),
              "2 · corregido a medias (mitad del factor)":
                  est_t.to_numpy() * np.exp(pred / 2)}
    t = pd.DataFrame([dict(brazo=n, **mt(real.to_numpy(), v)) for n, v in brazos.items()])
    print(f"\n{S}\nTOTAL DEL PROYECTO\n{S}")
    print(t[["brazo", "MAE_h", "RMSE_h", "MdMRE", "MMRE", "R2_log", "PRED25",
             "PRED50", "factor2", "SSPB_pc"]].round(4).to_string(index=False))
    t.to_csv(OUT / "resumen.csv", index=False)
    json.dump({"r2_corrector": float(r2), "n_proyectos": int(len(X)),
               "pesos": pesos.to_dict()},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/resumen.csv")


if __name__ == "__main__":
    main()
