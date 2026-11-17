"""E23 · Las palancas, medidas en proyectos nunca vistos y con potencia.

E14 midio las palancas sobre el bloque externo no-Apache: 792 tareas de 49
proyectos. E22 mostro que ese banco no tiene potencia ni para ver la mejora del
modelo. Las mismas palancas, medidas con validacion cruzada dejando proyectos
enteros fuera sobre las 24.848 tareas, dan la tabla que la memoria necesita:
cada tarea predicha por un modelo que nunca vio su proyecto, y 204 proyectos en
vez de 49.

  0 / 3 / 10 / 20 anclas contraidas (lambda = 12)
  con y sin la estimacion del usuario corregida

El corrector tambien se ajusta dentro de cada pliegue, con la prediccion en frio
del propio pliegue, para que no haya fuga de ninguna clase.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GroupKFold, KFold
from sklearn.svm import LinearSVR

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("m71", "src/71_estimacion_usuario.py")
M71 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M71)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 215)
DATA = "data/processed/pbe_effort_v4.parquet"
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/palancas_cv")
SEP = "=" * 124
LAMBDA, PLIEGUES, SORTEOS = 12.0, 5, 20


def metricas(y, p, proyecto=None):
    p = np.clip(np.asarray(p, float), .01, None)
    y = np.asarray(y, float)
    m = E.point_metrics(y, p)
    ly, lp = np.log(y), np.log(p)
    d = dict(MdMRE=m["MdMRE"], MdAElog=float(np.median(np.abs(lp - ly))),
             PRED25=m["PRED25"], MAE_h=m["MAE_h"],
             R2=float(1 - ((ly - lp) ** 2).mean() / ly.var()),
             sesgo=float(np.exp(np.median(lp - ly))), SA=M71.sa(y, p),
             spearman=float(pd.Series(y).corr(pd.Series(p), method="spearman")))
    if proyecto is not None:
        g = pd.DataFrame({"p": proyecto, "ly": ly, "lp": lp})
        dy = ly - g.groupby("p").ly.transform("mean").to_numpy()
        dp = lp - g.groupby("p").lp.transform("mean").to_numpy()
        d["rho_intra"] = float(pd.Series(dy).corr(pd.Series(dp), method="spearman"))
    return d


def hgb(s=0, **kw):
    return HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=.06,
                                         l2_regularization=1., random_state=s, **kw)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet(DATA)
    sw = df[df.is_software_task == 1].copy()
    sw["apache"] = sw.fuente_jira.eq("Apache")
    sw = sw[~sw.apache].reset_index(drop=True)
    tit = sw.title.fillna("").astype(str)
    y = sw.effort_hours.to_numpy()
    ly = np.log(y)
    proy = sw.project.to_numpy()
    E5 = np.load(CACHE / "e5-large.npy")
    sw["est"] = sw.HoursEstimate.where(sw.HoursEstimate > 0)
    pt = sw[(sw.source == "TAWOS") & (sw.story_point > 0)]
    pend, corte = np.polyfit(np.log(pt.story_point.to_numpy(float)),
                             np.log(pt.effort_hours.to_numpy(float)), 1)
    sw.loc[pt.index, "est"] = np.exp(corte) * pt.story_point.to_numpy(float) ** pend
    est = sw.est.to_numpy(float)
    hay = ~np.isnan(est)
    print(f"{len(sw):,} tareas · {len(np.unique(proy))} proyectos · "
          f"con estimación {hay.mean():.1%}")

    lp = np.zeros(len(sw))
    corr = np.zeros(len(sw))
    gkf = GroupKFold(n_splits=PLIEGUES)
    for f, (tr, te) in enumerate(gkf.split(E5, ly, proy), 1):
        tfc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                              sublinear_tf=True).fit(tit.iloc[tr])
        tfw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(tit.iloc[tr])
        Ms = sparse.hstack([tfc.transform(tit), tfw.transform(tit)]).tocsr()
        Xd = np.hstack([E5] + [TruncatedSVD(256, random_state=0).fit(
            tf.transform(tit)[tr]).transform(tf.transform(tit)).astype(np.float32)
            for tf in (tfc, tfw)])
        # 20 % del entrenamiento del pliegue, por proyectos, para los pesos del apilado
        pr_tr = np.unique(proy[tr])
        rng = np.random.default_rng(f)
        pr_ap = set(rng.choice(pr_tr, max(2, len(pr_tr) // 5), replace=False))
        ap_m = np.array([p in pr_ap for p in proy[tr]])
        aj, apil = tr[~ap_m], tr[ap_m]
        acum = [hgb(s, max_features=.8).fit(
            *[a[rng.choice(aj, int(.85 * len(aj)), replace=False)] for a in (Xd, ly)]
        ) for s in range(5)]
        col = lambda i: np.column_stack([np.mean([m.predict(Xd[i]) for m in acum], 0),
                                         rid.predict(Ms[i]), svr.predict(Ms[i])])
        rid = Ridge(alpha=2.0).fit(Ms[aj], ly[aj])
        svr = LinearSVR(C=.3, max_iter=4000, random_state=0).fit(Ms[aj], ly[aj])
        mezcla = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(col(apil), ly[apil])
        lp[te] = mezcla.predict(col(te))
        # corrector de la estimacion, con frio fuera de pliegue dentro del pliegue
        oof = np.zeros(len(tr))
        for a, b in KFold(4, shuffle=True, random_state=0).split(tr):
            oof[b] = hgb().fit(Xd[tr[a]], ly[tr[a]]).predict(Xd[tr[b]])
        k = hay[tr]
        if k.sum() > 300:
            Zc = np.column_stack([np.log(np.where(hay[tr], est[tr], 1)), oof])
            c = hgb().fit(Zc[k], (ly[tr] - np.log(np.where(hay[tr], est[tr], 1)))[k])
            Zt = np.column_stack([np.log(np.where(hay[te], est[te], 1)), lp[te]])
            corr[te] = np.where(hay[te], np.log(np.where(hay[te], est[te], 1)) + c.predict(Zt),
                                lp[te])
        else:
            corr[te] = lp[te]
        print(f"  pliegue {f}/{PLIEGUES} · {time.time() - t0:.0f} s", flush=True)

    def anclar(base, n_anc, semilla):
        if n_anc == 0:
            return base, np.ones(len(base), bool)
        r = np.random.default_rng(semilla)
        out, usar = base.copy(), np.zeros(len(base), bool)
        for p in np.unique(proy):
            i = np.flatnonzero(proy == p)
            if len(i) <= n_anc:
                continue
            anc = r.choice(i, n_anc, replace=False)
            resto = np.setdiff1d(i, anc)
            out[resto] += float(np.median(ly[anc] - base[anc])) * n_anc / (n_anc + LAMBDA)
            usar[resto] = True
        return out, usar

    filas = []
    for n_anc in (0, 3, 10, 20):
        for con_est in (False, True):
            acum = []
            for s in range(SORTEOS if n_anc else 1):
                base = corr if con_est else lp
                la, usar = anclar(lp, n_anc, s)
                p = np.exp(np.where(hay, corr, la) if con_est else la)
                if con_est and n_anc:
                    p = np.where(hay, np.exp(corr + (la - lp)), np.exp(la))
                acum.append(metricas(y[usar], p[usar], proy[usar]))
            filas.append(dict(anclas=n_anc, estimacion=con_est,
                              **{k: float(np.mean([a[k] for a in acum])) for k in acum[0]}))
            print(f"  {n_anc:2d} anclas · estimación {str(con_est):5s} -> "
                  f"MdMRE {filas[-1]['MdMRE']:.4f} · R² {filas[-1]['R2']:+.4f}", flush=True)
    t = pd.DataFrame(filas)
    cols = ["anclas", "estimacion", "MdMRE", "MdAElog", "PRED25", "MAE_h", "R2", "SA",
            "sesgo", "spearman", "rho_intra"]
    print(f"\n{SEP}\nPALANCAS EN PROYECTOS NUNCA VISTOS · {len(sw):,} tareas, "
          f"{len(np.unique(proy))} proyectos\n{SEP}")
    print(t[cols].round(4).to_string(index=False))
    t.to_csv(OUT / "resumen.csv", index=False)
    np.save(OUT / "predicciones.npy", np.vstack([lp, corr]))
    json.dump({"lambda": LAMBDA, "pliegues": PLIEGUES, "n": len(sw)},
              open(OUT / "meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nescrito {OUT}/resumen.csv")


if __name__ == "__main__":
    main()
