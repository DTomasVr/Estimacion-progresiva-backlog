"""E50 - Juntar todos los estimadores del total que han sobrevivido.

A estas alturas hay cinco formas distintas de estimar las horas de un proyecto, y
cada una gana una metrica distinta:

    ascendente con smearing     el +-25 % y el sesgo        (E43-E44)
    ascendente Poisson          el MdMRE y el RMSE          (E45)
    ascendente Poisson sin sesgo  el MdMRE con sesgo cero   (E46)
    ascendente por analogia     el factor 2                 (E41-E43)
    descendente por proyectos parecidos  casi todo, pero con MAE peor  (E48)

Que cada una gane una metrica distinta es la firma de que fallan en sitios
distintos, y eso es exactamente lo que una combinacion puede aprovechar. La
media geometrica de dos de ellas ya mejoro a las dos en E48.

Se combinan las cinco con pesos sobre el simplice, eligiendo los pesos DEJANDO
FUERA cada proyecto, y se comparan contra la mejor individual. Con 114 proyectos
y cinco pesos el riesgo de sobreajustar es real, asi que:

  - los pesos de cada proyecto se eligen sin mirar ese proyecto
  - se incluye el brazo de pesos IGUALES, que no elige nada, como control. Si la
    busqueda de pesos no le gana, la ganancia venia de mirar los datos
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
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 215)
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/combinar_total")
S = "=" * 122
MIN_TAREAS, PLIEGUES, K_PROY, K_VEC = 8, 5, 5, 25


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
    df = pd.read_parquet("data/processed/agix_effort_v4.parquet")
    sw = df[df.is_software_task == 1].copy()
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    tit = sw.title.fillna("").astype(str)
    y = sw.effort_hours.to_numpy()
    ly = np.log(y)
    proy = sw.project.to_numpy()
    P = np.load("results/analogia_cv/pred.npy")
    lp, lana = P[:, 0], P[:, 1]
    E5 = np.load(CACHE / "e5-large.npy")
    Vn = E5 / np.maximum(np.linalg.norm(E5, axis=1, keepdims=True), 1e-9)
    import joblib
    f = joblib.load("models/estimador_agix_v3.joblib")
    X = np.hstack([E5] + [sv.transform(tf.transform(tit)).astype(np.float32)
                          for tf, sv in zip((f["tfc"], f["tfw"]), f["svds"])])

    poi = np.zeros(len(sw))
    for a, b in GroupKFold(PLIEGUES).split(X, y, proy):
        poi[b] = HistGradientBoostingRegressor(
            max_iter=250, max_depth=6, learning_rate=.06, l2_regularization=1.,
            random_state=0, loss="poisson").fit(X[a], y[a]).predict(X[b])
    print("Poisson fuera de muestra lista", flush=True)

    def smear(lpx):
        e = np.exp(ly - lpx)
        tr = pd.qcut(lpx, 6, labels=False, duplicates="drop")
        return np.exp(lpx) * pd.DataFrame({"t": tr, "e": e}).groupby("t").e.mean(
        ).reindex(tr).to_numpy()

    tot_t = np.load("results/modelo_total/pred_total.npy")
    d = pd.DataFrame({"p": proy, "real": y, "sm": smear(lp), "poi": np.clip(poi, 1e-6, None),
                      "ana": smear(lana), "lp": lp, "tot": np.clip(tot_t, 1e-6, None)})
    tam = d.groupby("p").size()
    d = d[d.p.isin(tam[tam >= MIN_TAREAS].index)].reset_index(drop=True)
    nombres = list(pd.unique(d.p))
    g = d.groupby("p")
    real = g.real.sum().reindex(nombres).to_numpy()
    n_t = g.size().reindex(nombres).to_numpy()
    base = {"ascendente modelo del total": g.tot.sum().reindex(nombres).to_numpy(),
            "ascendente smearing": g.sm.sum().reindex(nombres).to_numpy(),
            "ascendente Poisson": g.poi.sum().reindex(nombres).to_numpy(),
            "ascendente analogia": g.ana.sum().reindex(nombres).to_numpy()}

    sel = sw.project.isin(nombres).to_numpy()
    Vp = pd.DataFrame(Vn[sel]).groupby(sw.project[sel].to_numpy()).mean().reindex(
        nombres).to_numpy()
    Vp = Vp / np.maximum(np.linalg.norm(Vp, axis=1, keepdims=True), 1e-9)
    Sim = Vp @ Vp.T
    np.fill_diagonal(Sim, -2)
    v = np.argsort(-Sim, axis=1)[:, :K_PROY]
    hpt = real / n_t
    tam_tip = np.exp(g.lp.mean().reindex(nombres).to_numpy())
    base["descendente k=5"] = (n_t * np.exp(np.median(np.log(hpt)[v], axis=1)) *
                               (tam_tip / np.exp(np.median(np.log(tam_tip)[v], axis=1))))
    # Poisson sin sesgo, dejando fuera el propio proyecto
    q = np.log(base["ascendente Poisson"] / real)
    k = np.array([np.exp(-np.median(np.delete(q, i))) for i in range(len(real))])
    base["ascendente Poisson sin sesgo"] = base["ascendente Poisson"] * k
    nombres_e = list(base)
    L = np.column_stack([np.log(base[k]) for k in nombres_e])
    print(f"{len(real)} proyectos · {len(nombres_e)} estimadores", flush=True)

    # indices de los estimadores que NO necesitan historial de otros proyectos:
    # en el arranque de Agix, sin proyectos anteriores entregados, solo se puede
    # usar la via ascendente, asi que hay que dar tambien esa cifra por separado
    solo_asc = [i for i, k in enumerate(nombres_e) if k.startswith("ascendente")]

    def busca(mask, criterio, cols=None):
        r = np.random.default_rng(3)
        cols = list(range(L.shape[1])) if cols is None else cols
        mejor_w = np.zeros(L.shape[1])
        mejor_w[cols] = 1 / len(cols)

        def val(w):
            p = np.exp(L[mask] @ w)
            rr = np.abs(p - real[mask]) / real[mask]
            return np.median(rr) if criterio == "MdMRE" else -np.mean(rr <= .25)
        mejor = val(mejor_w)
        for _ in range(1500):
            w = np.zeros(L.shape[1])
            w[cols] = r.dirichlet(np.ones(len(cols)) * .7)
            if (u := val(w)) < mejor:
                mejor, mejor_w = u, w
        return mejor_w

    comb = {}
    for criterio, cols, etq in (("MdMRE", None, "combinada, pesos para MdMRE"),
                                ("PRED25", None, "combinada, pesos para PRED25"),
                                ("MdMRE", solo_asc,
                                 "combinada SOLO ascendente (sin historial)")):
        pred = np.zeros(len(real))
        for i in range(len(real)):
            m = np.ones(len(real), bool)
            m[i] = False
            pred[i] = float(np.exp(L[i] @ busca(m, criterio, cols)))
        comb[etq] = pred
        print(f"  {etq} lista", flush=True)
    comb["combinada, pesos IGUALES (control)"] = np.exp(L.mean(1))

    filas = [dict(brazo=k, **mt(real, v)) for k, v in base.items()]
    filas += [dict(brazo=k, **mt(real, v)) for k, v in comb.items()]
    t = pd.DataFrame(filas).sort_values("MdMRE")
    print(f"\n{S}\nTOTAL DEL PROYECTO · individuales y combinaciones\n{S}")
    print(t[["brazo", "MAE_h", "RMSE_h", "MdMRE", "MMRE", "R2_log", "PRED25",
             "PRED50", "factor2", "SSPB_pc"]].round(4).to_string(index=False))
    w = busca(np.ones(len(real), bool), "MdMRE")
    print("\n  pesos con todos los proyectos (solo para leerlos): " +
          " · ".join(f"{a} {b:.2f}" for a, b in zip(nombres_e, w)))
    t.to_csv(OUT / "resumen.csv", index=False)
    json.dump({"estimadores": nombres_e, "pesos_lectura": w.tolist()},
              open(OUT / "meta.json", "w"), ensure_ascii=False, indent=1)
    print(f"\nescrito {OUT}/resumen.csv")


if __name__ == "__main__":
    main()
