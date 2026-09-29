"""E8 · ¿Cuantas tareas hace falta anclar para que el proyecto entero cuadre?

Todo lo medido esta noche apunta al mismo sitio: el texto ordena razonablemente
las tareas DENTRO de un proyecto (R²_intra 0,10-0,14, ρ_intra 0,29-0,36) y no
tiene ni idea de la escala del proyecto, que no es semantica y no se puede
predecir desde ningun texto disponible (E1, E2, E3 los tres negativos).

Si la parte que falta es UN numero por proyecto, no hace falta pedirsela al
usuario tarea por tarea. Basta con anclar: el usuario da las horas de k tareas
-- o simplemente cierra las primeras k -- y con eso se fija el desplazamiento de
todo el backlog.

    desplazamiento = mediana( log(real) - log(predicho) ) sobre las k ancladas
    prediccion final = predicho x exp(desplazamiento)

Se mide sobre las tareas NO ancladas, que es lo unico honesto, con 20 sorteos de
ancla por punto. k=0 es el arranque en frio de hoy.

Esto no es lo mismo que la calibracion en linea que ya medimos: alli hacian
falta tareas CERRADAS, aqui basta con que el usuario estime unas pocas al crear
el proyecto, que es justo lo que el artefacto puede pedir sin friccion.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
DATA = "data/processed/agix_effort_v4.parquet"
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/anclas")
SEP = "=" * 108
ANCLAS = [0, 1, 2, 3, 5, 10, 20]
SORTEOS = 20


def metricas(y, p):
    p = np.clip(np.asarray(p, float), .01, None)
    y = np.asarray(y, float)
    m = E.point_metrics(y, p)
    ly = np.log(y)
    return dict(MdMRE=m["MdMRE"], MdAElog=float(np.median(np.abs(np.log(p) - ly))),
                PRED25=m["PRED25"], MAE_h=m["MAE_h"],
                R2=float(1 - ((ly - np.log(p)) ** 2).mean() / ly.var()),
                sesgo=float(np.exp(np.median(np.log(p) - ly))),
                spearman=float(pd.Series(y).corr(pd.Series(p), method="spearman")))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA)
    sw = df[df.is_software_task == 1].copy()
    sw["apache"] = sw.fuente_jira.eq("Apache")
    sw = sw[~sw.apache].reset_index(drop=True)
    tit = sw.title.fillna("").astype(str)
    tr_m = (sw.split_t3 == "train").to_numpy()

    X = [np.load(CACHE / "e5-large.npy")]
    for kw in (dict(analyzer="char_wb", ngram_range=(3, 5)), dict(ngram_range=(1, 2))):
        tf = TfidfVectorizer(min_df=3, max_features=200_000, sublinear_tf=True,
                             **kw).fit(tit[tr_m])
        M = tf.transform(tit)
        X.append(TruncatedSVD(256, random_state=0).fit(M[tr_m]).transform(M).astype(np.float32))
    X = np.hstack(X)
    y = sw.effort_hours.to_numpy()
    ly = np.log(y)
    print(f"vista e5-large + tf-idf caracteres + tf-idf palabras: {X.shape}")

    m = HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=.06,
                                      l2_regularization=1., random_state=0).fit(
        X[tr_m], ly[tr_m])
    lp_todo = m.predict(X)

    bloques = {"prueba fija": (sw.split_t3 == "test").to_numpy(),
               "prueba SiP": ((sw.split_t3 == "test") & (sw.source == "SiP")).to_numpy(),
               "externo no Apache": (sw.split_t3 == "externo").to_numpy()}
    filas = []
    for b, k in bloques.items():
        d = sw[k]
        lpb, lyb = lp_todo[k], ly[k]
        proy = d.project.to_numpy()
        idx_por_proy = {p: np.flatnonzero(proy == p) for p in np.unique(proy)}
        print(f"\n{b}: {len(d):,} tareas en {len(idx_por_proy)} proyectos "
              f"(mediana {np.median([len(v) for v in idx_por_proy.values()]):.0f} por proyecto)")
        for n_anc in ANCLAS:
            acum = []
            for s in range(SORTEOS if n_anc else 1):
                rng = np.random.default_rng(s)
                pred, real = [], []
                for p, idx in idx_por_proy.items():
                    if n_anc and len(idx) > n_anc:
                        anc = rng.choice(idx, n_anc, replace=False)
                        resto = np.setdiff1d(idx, anc)
                        desp = float(np.median(lyb[anc] - lpb[anc]))
                    elif n_anc:
                        continue                      # proyecto demasiado pequeño
                    else:
                        resto, desp = idx, 0.0
                    pred.append(np.exp(lpb[resto] + desp))
                    real.append(y[k][resto])
                if not pred:
                    continue
                acum.append(metricas(np.concatenate(real), np.concatenate(pred)))
            if not acum:
                continue
            prom = {kk: float(np.mean([a[kk] for a in acum])) for kk in acum[0]}
            filas.append(dict(bloque=b, anclas=n_anc, n=int(sum(
                len(np.setdiff1d(v, [])) for v in idx_por_proy.values())), **prom))
            print(f"  {n_anc:2d} anclas -> MdMRE {prom['MdMRE']:.4f} · "
                  f"MdAElog {prom['MdAElog']:.4f} · PRED25 {prom['PRED25']:.4f} · "
                  f"R² {prom['R2']:+.4f} · sesgo {prom['sesgo']:.3f}", flush=True)

    t = pd.DataFrame(filas)
    print(f"\n{SEP}\nCUÁNTO VALE CADA ANCLA\n{SEP}")
    for b in bloques:
        v = t[t.bloque == b]
        if not len(v):
            continue
        base = v[v.anclas == 0].iloc[0]
        print(f"\n{b}")
        for _, r in v.iterrows():
            print(f"  {int(r.anclas):2d} anclas · MdMRE {r.MdMRE:.4f} "
                  f"({r.MdMRE - base.MdMRE:+.4f}) · R² {r.R2:+.4f} "
                  f"({r.R2 - base.R2:+.4f}) · PRED25 {r.PRED25:.3f}")
    t.to_csv(OUT / "resumen.csv", index=False)
    json.dump({"anclas": ANCLAS, "sorteos": SORTEOS},
              open(OUT / "meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nescrito {OUT}/resumen.csv")


if __name__ == "__main__":
    main()
