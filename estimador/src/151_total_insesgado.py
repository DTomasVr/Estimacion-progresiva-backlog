"""E46 - Quitarle el sesgo al mejor estimador del total, que es lo unico que le falta.

E45 deja el reparto asi sobre los totales de los 114 proyectos:

    Poisson directa           MdMRE 0,368 · RMSE 1021 · PRED25 0,316 · sesgo -15,3 %
    mediana x smearing tramo  MdMRE 0,438 · RMSE 1238 · PRED25 0,377 · sesgo  -0,4 %

O sea que la Poisson ordena y escala mejor -- mejor mediana del error y mejor
error cuadratico -- pero se queda corta un 15 % de forma sistematica, y eso le
cuesta el PRED(25), porque una ventana del +-25 % no perdona 15 puntos de sesgo.

Un sesgo sistematico y constante es lo mas facil de arreglar que hay: se estima
un factor y se multiplica. Lo unico importante es estimarlo SIN mirar el proyecto
que se evalua, o el numero sale bonito y falso. Se hace dejando fuera cada
proyecto, igual que con el smearing.

Se prueban dos formas de elegir ese factor, que no dan lo mismo:

  por la MEDIANA de las razones   centra el error en escala logaritmica. Es lo que
                                  conviene a MdMRE y a PRED(25)
  por la RAZON de los totales     hace que la suma de todo cuadre. Es lo que
                                  conviene al MAE y al sesgo agregado

Y se prueba la combinacion de la Poisson con el smearing, por si cada una acierta
en proyectos distintos.
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "6"

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 215)
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/total_insesgado")
S = "=" * 122
MIN_TAREAS, PLIEGUES = 8, 5


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


def fuera_de_proyecto(tot, col, modo):
    """Factor multiplicativo estimado sin mirar el proyecto que se corrige."""
    q = np.log(tot[col].to_numpy() / tot.real.to_numpy())
    out = np.zeros(len(tot))
    for i in range(len(tot)):
        k = np.ones(len(tot), bool)
        k[i] = False
        if modo == "mediana":
            out[i] = np.exp(-np.median(q[k]))
        else:
            out[i] = tot.real.to_numpy()[k].sum() / tot[col].to_numpy()[k].sum()
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    f = joblib.load("models/estimador_agix_v3.joblib")
    df = pd.read_parquet("data/processed/agix_effort_v4.parquet")
    sw = df[df.is_software_task == 1].copy()
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    tit = sw.title.fillna("").astype(str)
    y = sw.effort_hours.to_numpy()
    ly = np.log(y)
    proy = sw.project.to_numpy()
    X = np.hstack([np.load(CACHE / "e5-large.npy")] +
                  [sv.transform(tf.transform(tit)).astype(np.float32)
                   for tf, sv in zip((f["tfc"], f["tfw"]), f["svds"])])
    lp = np.load("results/analogia_cv/pred.npy")[:, 0]
    print(f"{len(sw):,} tareas · {time.time() - t0:.0f} s", flush=True)

    poi = np.zeros(len(sw))
    for nf, (a, b) in enumerate(GroupKFold(PLIEGUES).split(X, y, proy)):
        poi[b] = HistGradientBoostingRegressor(
            max_iter=250, max_depth=6, learning_rate=.06, l2_regularization=1.,
            random_state=0, loss="poisson").fit(X[a], y[a]).predict(X[b])
        print(f"  pliegue {nf + 1}/{PLIEGUES} · {time.time() - t0:.0f} s", flush=True)

    e = np.exp(ly - lp)
    tramo = pd.qcut(lp, 6, labels=False, duplicates="drop")
    ft = pd.DataFrame({"t": tramo, "e": e}).groupby("t").e.mean()
    sm = np.exp(lp) * ft.reindex(tramo).to_numpy()

    d = pd.DataFrame({"p": proy, "real": y, "poisson": np.clip(poi, 1e-6, None),
                      "smearing": np.clip(sm, 1e-6, None)})
    tam = d.groupby("p").size()
    d = d[d.p.isin(tam[tam >= MIN_TAREAS].index)]
    tot = d.groupby("p").sum(numeric_only=True)
    print(f"{len(tot)} proyectos")

    brazos = {"0 · mediana x smearing por tramo": tot.smearing.to_numpy(),
              "1 · Poisson directa": tot.poisson.to_numpy()}
    for modo in ("mediana", "totales"):
        for col in ("poisson", "smearing"):
            k = fuera_de_proyecto(tot, col, modo)
            brazos[f"2 · {col} sin sesgo, por {modo}"] = tot[col].to_numpy() * k
    kp = fuera_de_proyecto(tot, "poisson", "mediana")
    ks = fuera_de_proyecto(tot, "smearing", "mediana")
    brazos["3 · media geometrica de las dos sin sesgo"] = np.sqrt(
        tot.poisson.to_numpy() * kp * tot.smearing.to_numpy() * ks)

    r = pd.DataFrame([dict(brazo=n, **mt(tot.real, v)) for n, v in brazos.items()])
    r = r.sort_values("MdMRE")
    print(f"\n{S}\nTOTAL DEL PROYECTO, QUITANDO EL SESGO SIN MIRAR EL PROYECTO\n{S}")
    print(r[["brazo", "MAE_h", "RMSE_h", "MdMRE", "MMRE", "R2_log", "PRED25",
             "PRED50", "factor2", "SSPB_pc"]].round(4).to_string(index=False))
    b = r.iloc[0]
    print(f"\n  factor de correccion tipico: Poisson x{np.median(kp):.3f} · "
          f"smearing x{np.median(ks):.3f}")
    print(f"  mejor por MdMRE: {b.brazo} -> MdMRE {b.MdMRE:.4f} · R2 {b.R2_log:.4f} · "
          f"PRED25 {b.PRED25:.4f} · factor 2 {b.factor2:.4f} · sesgo {b.SSPB_pc:+.1f} %")
    r.to_csv(OUT / "resumen.csv", index=False)
    tot.to_csv(OUT / "totales.csv")
    json.dump({"factor_poisson": float(np.median(kp)),
               "factor_smearing": float(np.median(ks))},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/resumen.csv · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
