"""E59 - Ampliar el modelo del total con los canales que le faltan.

E52 monto el modelo del total con tres canales: bolsa sobre e5-large + TF-IDF
reducido, un HGB sobre el TF-IDF, y la analogia. Le faltan dos cosas que en el
modelo por tarea si aportan:

  LINEAL DISPERSO   un modelo lineal sobre la matriz de TF-IDF ENTERA, sin
                    reducir. En el apilado por tarea pesa 0,22, y aporta porque
                    se fija en palabras concretas que la reduccion se come. Para
                    una media hace falta la version de Poisson, que en sklearn es
                    PoissonRegressor
  DESCRIPCION       E31 la dejo en +0,015 de R2 donde existe, y esta en el 57 %
                    de las tareas

Y, si ya estan listos, los vectores del codificador ajustado POR PLIEGUE (E54),
que es el unico canal que hasta ahora no se podia meter en la metrica de proyecto
sin fuga.

Se compara contra el modelo del total tal cual, sobre los mismos 114 proyectos y
la misma validacion cruzada por proyecto entero.
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
from scipy import sparse  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.linear_model import PoissonRegressor  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 215)
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/total_ampliado")
S = "=" * 122
MIN_TAREAS, PLIEGUES, N_BOLSA, K_VEC = 8, 5, 5, 25


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


def poi(semilla=0, **kw):
    p = dict(max_iter=250, max_depth=6, learning_rate=.06, l2_regularization=1.,
             loss="poisson", random_state=semilla)
    p.update(kw)
    return HistGradientBoostingRegressor(**p)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    f = joblib.load("models/estimador_agix_v3.joblib")
    df = pd.read_parquet("data/processed/agix_effort_v4.parquet")
    sw = df[df.is_software_task == 1].copy()
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    tit = sw.title.fillna("").astype(str)
    y = sw.effort_hours.to_numpy()
    proy = sw.project.to_numpy()
    E5 = np.load(CACHE / "e5-large.npy")
    Vn = E5 / np.maximum(np.linalg.norm(E5, axis=1, keepdims=True), 1e-9)
    SV = np.hstack([sv.transform(tf.transform(tit)).astype(np.float32)
                    for tf, sv in zip((f["tfc"], f["tfw"]), f["svds"])])
    X = np.hstack([E5, SV])
    Xs = sparse.hstack([f["tfc"].transform(tit), f["tfw"].transform(tit)]).tocsr()
    D = np.load(CACHE / "e5-large-desc.npy")
    td = sw.description.fillna("").astype(str).str.split().str.len().gt(0).to_numpy()
    D = np.where(td[:, None], D, 0).astype(np.float32)
    fp = CACHE / "e5-ajustado-C-porpliegue.npy"
    AJ = np.load(fp) if fp.exists() else None
    print(f"{len(sw):,} tareas · codificador por pliegue: "
          f"{'si' if AJ is not None else 'todavia no'} · {time.time() - t0:.0f} s",
          flush=True)

    pred = {k: np.zeros(len(sw)) for k in ("base", "mas_lineal", "mas_desc", "todo")}
    for nf, (a, b) in enumerate(GroupKFold(PLIEGUES).split(X, y, proy)):
        rng = np.random.default_rng(nf)
        pr = np.unique(proy[a])
        pin = set(rng.choice(pr, max(2, int(.2 * len(pr))), replace=False))
        ain = a[np.array([p in pin for p in proy[a]])]
        afu = a[np.array([p not in pin for p in proy[a]])]
        bolsa = []
        for s in range(N_BOLSA):
            sub = rng.choice(afu, int(.85 * len(afu)), replace=False)
            bolsa.append(poi(s, max_features=.8).fit(X[sub], y[sub]))
        m_sv = poi().fit(SV[afu], y[afu])
        A, yv = Vn[afu], y[afu]

        def vecino(sel):
            out = np.zeros(len(sel))
            for i in range(0, len(sel), 2048):
                s = Vn[sel[i:i + 2048]] @ A.T
                j = np.argpartition(-s, K_VEC, axis=1)[:, :K_VEC]
                out[i:i + 2048] = np.median(yv[j], axis=1)
            return out
        an = {k: vecino(v) for k, v in (("fu", afu), ("in", ain), ("b", b))}
        m_an = poi().fit(an["fu"][:, None], y[afu])
        m_li = PoissonRegressor(alpha=1e-3, max_iter=400).fit(Xs[afu], y[afu])
        m_de = poi().fit(D[afu], y[afu])
        m_aj = poi().fit(AJ[afu], y[afu]) if AJ is not None else None

        def cana(sel, an_, cuales):
            c = [np.mean([m.predict(X[sel]) for m in bolsa], 0),
                 m_sv.predict(SV[sel]), m_an.predict(an_[:, None])]
            if "lineal" in cuales:
                c.append(m_li.predict(Xs[sel]))
            if "desc" in cuales:
                c.append(m_de.predict(D[sel]))
            if "aj" in cuales and m_aj is not None:
                c.append(m_aj.predict(AJ[sel]))
            return np.column_stack(c)
        for etq, cuales in (("base", ()), ("mas_lineal", ("lineal",)),
                            ("mas_desc", ("desc",)),
                            ("todo", ("lineal", "desc", "aj"))):
            ap = poi(max_iter=120, max_depth=3).fit(cana(ain, an["in"], cuales), y[ain])
            pred[etq][b] = ap.predict(cana(b, an["b"], cuales))
        print(f"  pliegue {nf + 1}/{PLIEGUES} · {time.time() - t0:.0f} s", flush=True)

    d = pd.DataFrame({"p": proy, "real": y})
    for k, v in pred.items():
        d[k] = np.clip(v, 1e-6, None)
    tam = d.groupby("p").size()
    d = d[d.p.isin(tam[tam >= MIN_TAREAS].index)]
    tot = d.groupby("p").sum(numeric_only=True)
    nom = {"base": "0 · los tres canales de E52",
           "mas_lineal": "1 · + lineal disperso de Poisson",
           "mas_desc": "2 · + la descripcion",
           "todo": "3 · todo" + (" (con el codificador por pliegue)"
                                 if AJ is not None else " (sin el codificador)")}
    t = pd.DataFrame([dict(brazo=nom[k], **mt(tot.real, tot[k])) for k in pred])
    print(f"\n{S}\nTOTAL DEL PROYECTO ({len(tot)} proyectos)\n{S}")
    print(t[["brazo", "MAE_h", "RMSE_h", "MdMRE", "MMRE", "R2_log", "PRED25",
             "PRED50", "factor2", "SSPB_pc"]].round(4).to_string(index=False))
    np.save(OUT / "pred.npy", np.column_stack([pred[k] for k in
                                               ("base", "mas_lineal", "mas_desc", "todo")]))
    t.to_csv(OUT / "resumen.csv", index=False)
    json.dump({"con_codificador_por_pliegue": AJ is not None},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/resumen.csv · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
