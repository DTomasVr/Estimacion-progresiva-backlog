"""Full algorithm sweep on v3, temporal split.

Selection is done on VALIDATION. Test numbers are printed alongside as a
descriptive comparison and are explicitly NOT used to choose anything -- the
pre-registered protocol result (ridge, chosen in 43_select_on_val.py) remains
the headline. Reporting the test column for models that were never selected is
post-hoc and is labelled as such.

Beyond the standard families this adds a regime-gated hybrid: a classifier
splits tasks into small and large, and a separate regressor is fitted to each
side. It is the one combination that directly targets the compression problem,
since the gate can commit to "this is a large task" where a single regressor
hedges.
"""
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import (ExtraTreesRegressor, HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor,
                              RandomForestRegressor, StackingRegressor,
                              VotingRegressor)
from sklearn.linear_model import BayesianRidge, ElasticNet, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 260)
DATA = "data/processed/agix_effort_v3.parquet"
NUEVAS = ["n_links", "n_links_estructurales", "n_componentes", "en_sprint",
          "tiene_asignado", "Title_Changed_After_Estimation",
          "Story_Point_Changed_After_Estimation"]
CATS = ["task_kind", "source", "desarrollador"]


def datos():
    df = pd.read_parquet(DATA)
    sw = df[df.is_software_task == 1].copy()
    emb = np.load("data/processed/embeddings/"
                  "paraphrase-multilingual-mpnet-base-v2.npy")
    ids = pd.read_parquet("data/processed/agix_effort_clean.parquet",
                          columns=["task_id"]).task_id.to_numpy()
    pos = {t: i for i, t in enumerate(ids)}
    tr, va, te = (sw[sw.split_t3 == s] for s in ("train", "val", "test"))
    rows = lambda d: emb[[pos[t] for t in d.task_id]]
    oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                       min_frequency=5)
    sc = StandardScaler()
    X = [np.hstack([rows(tr), oh.fit_transform(tr[CATS].astype(str)),
                    sc.fit_transform(np.log1p(tr[NUEVAS].to_numpy(float)))])]
    for d in (va, te):
        X.append(np.hstack([rows(d), oh.transform(d[CATS].astype(str)),
                            sc.transform(np.log1p(d[NUEVAS].to_numpy(float)))]))
    return tr, va, te, X[0], X[1], X[2]


def modelos():
    return {
        "arbol": DecisionTreeRegressor(max_depth=12, min_samples_leaf=20,
                                       random_state=0),
        "random_forest": RandomForestRegressor(n_estimators=400,
                                               min_samples_leaf=2,
                                               max_features="sqrt", n_jobs=-1,
                                               random_state=0),
        "rf_profundo": RandomForestRegressor(n_estimators=500, min_samples_leaf=5,
                                             max_features=0.3, n_jobs=-1,
                                             random_state=0),
        "extra_trees": ExtraTreesRegressor(n_estimators=400, min_samples_leaf=2,
                                           max_features="sqrt", n_jobs=-1,
                                           random_state=0),
        "hgb": HistGradientBoostingRegressor(max_iter=200, max_depth=4,
                                             learning_rate=0.06,
                                             l2_regularization=1.0,
                                             random_state=0),
        "ridge": Ridge(alpha=1.0),
        "lasso": Lasso(alpha=0.001, max_iter=5000),
        "elastic": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000),
        "bayes_ridge": BayesianRidge(),
        "knn": KNeighborsRegressor(n_neighbors=10, metric="cosine",
                                   weights="distance"),
        "svr_lineal": LinearSVR(C=0.5, max_iter=5000, random_state=0),
        "mlp": MLPRegressor(hidden_layer_sizes=(256, 64), alpha=1e-3,
                            max_iter=150, early_stopping=True, random_state=0),
    }


def evaluar(nombre, p_va, p_te, va, te, tr, filas, segundos):
    y_va, y_te = va.effort_hours.to_numpy(), te.effort_hours.to_numpy()
    mv, mt = E.point_metrics(y_va, p_va), E.point_metrics(y_te, p_te)
    gana = 0, 0
    n_ok = 0
    for proj in te.project.unique():
        s = (te.project == proj).to_numpy()
        if s.sum() < 12:
            continue
        b = float(tr[tr.project == proj].effort_hours.median())
        if not np.isfinite(b):
            continue
        n_ok += 1
        if (E.point_metrics(y_te[s], p_te[s])["MdMRE"]
                < E.point_metrics(y_te[s], np.full(int(s.sum()), b))["MdMRE"]):
            gana = (gana[0] + 1, 0)
    filas.append(dict(
        modelo=nombre, MdMRE_val=round(mv["MdMRE"], 4),
        MdMRE_test=round(mt["MdMRE"], 4), PRED25_test=round(mt["PRED25"], 4),
        RMSE_log_test=round(mt["RMSE_log"], 4),
        compresion=round(float(np.std(np.log(np.clip(p_te, .01, None)))
                               / np.std(np.log(y_te))), 3),
        spearman=round(float(pd.Series(y_te).corr(pd.Series(p_te),
                                                  method="spearman")), 3),
        gana_base=f"{gana[0]}/{n_ok}", seg=round(segundos, 1)))
    E.record("sweep_v3", nombre, "split_t3",
             {**mt, "MdMRE_val": mv["MdMRE"]})


def main():
    tr, va, te, Xtr, Xva, Xte = datos()
    y_tr = np.log(tr.effort_hours.to_numpy())
    print(f"train {len(tr):,}  val {len(va):,}  test {len(te):,}  "
          f"caracteristicas {Xtr.shape[1]}\n")
    filas = []

    base_va = va.project.map(tr.groupby("project").effort_hours.median())
    base_te = te.project.map(tr.groupby("project").effort_hours.median())
    g = tr.effort_hours.median()
    evaluar("linea base (mediana)", base_va.fillna(g).to_numpy(),
            base_te.fillna(g).to_numpy(), va, te, tr, filas, 0.0)

    ajustados = {}
    for nombre, mdl in modelos().items():
        t0 = time.time()
        mdl.fit(Xtr, y_tr)
        p_va, p_te = np.exp(mdl.predict(Xva)), np.exp(mdl.predict(Xte))
        evaluar(nombre, p_va, p_te, va, te, tr, filas, time.time() - t0)
        ajustados[nombre] = mdl
        print(f"  {nombre:16s} val {filas[-1]['MdMRE_val']:.4f}  "
              f"test {filas[-1]['MdMRE_test']:.4f}  {filas[-1]['seg']:6.1f}s")

    # SVR radial sobre componentes principales
    t0 = time.time()
    pca = PCA(n_components=128, random_state=0).fit(Xtr)
    s = SVR(C=2.0, epsilon=0.1).fit(pca.transform(Xtr), y_tr)
    evaluar("svr_rbf_pca", np.exp(s.predict(pca.transform(Xva))),
            np.exp(s.predict(pca.transform(Xte))), va, te, tr, filas,
            time.time() - t0)
    print(f"  {'svr_rbf_pca':16s} val {filas[-1]['MdMRE_val']:.4f}  "
          f"test {filas[-1]['MdMRE_test']:.4f}")

    # gradient boosting en GPU
    try:
        from xgboost import XGBRegressor
        t0 = time.time()
        x = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.6, reg_lambda=2.0,
                         device="cuda", tree_method="hist", random_state=0)
        x.fit(Xtr, y_tr)
        evaluar("xgboost_gpu", np.exp(x.predict(Xva)), np.exp(x.predict(Xte)),
                va, te, tr, filas, time.time() - t0)
        print(f"  {'xgboost_gpu':16s} val {filas[-1]['MdMRE_val']:.4f}  "
              f"test {filas[-1]['MdMRE_test']:.4f}")
    except Exception as e:
        print("  xgboost no disponible:", e)

    # conjuntos
    print("\n  --- conjuntos ---")
    picks = ["ridge", "hgb", "knn", "extra_trees"]
    for nombre, ens in [
        ("voting", VotingRegressor([(n, modelos()[n]) for n in picks])),
        ("stacking", StackingRegressor([(n, modelos()[n]) for n in picks],
                                       final_estimator=Ridge(alpha=1.0), cv=3)),
    ]:
        t0 = time.time()
        ens.fit(Xtr, y_tr)
        evaluar(nombre, np.exp(ens.predict(Xva)), np.exp(ens.predict(Xte)),
                va, te, tr, filas, time.time() - t0)
        print(f"  {nombre:16s} val {filas[-1]['MdMRE_val']:.4f}  "
              f"test {filas[-1]['MdMRE_test']:.4f}")

    prom_va = np.mean([np.exp(ajustados[n].predict(Xva)) for n in picks], axis=0)
    prom_te = np.mean([np.exp(ajustados[n].predict(Xte)) for n in picks], axis=0)
    evaluar("media_simple", prom_va, prom_te, va, te, tr, filas, 0.0)
    print(f"  {'media_simple':16s} val {filas[-1]['MdMRE_val']:.4f}  "
          f"test {filas[-1]['MdMRE_test']:.4f}")

    # hibrido con puerta de regimen: clasificar grande/pequeno y regresar aparte
    print("\n  --- hibrido con puerta de regimen ---")
    UMBRAL = 8.0
    t0 = time.time()
    gate = HistGradientBoostingClassifier(max_iter=250, max_depth=5,
                                          learning_rate=0.06, random_state=0)
    gate.fit(Xtr, (tr.effort_hours > UMBRAL).astype(int).to_numpy())
    grande = tr.effort_hours.to_numpy() > UMBRAL
    m_lo = Ridge(alpha=1.0).fit(Xtr[~grande], y_tr[~grande])
    m_hi = Ridge(alpha=1.0).fit(Xtr[grande], y_tr[grande])
    for etiqueta, X, frame in [("va", Xva, va), ("te", Xte, te)]:
        pr = gate.predict_proba(X)[:, 1]
        mezcla = np.exp((1 - pr) * m_lo.predict(X) + pr * m_hi.predict(X))
        if etiqueta == "va":
            p_va = mezcla
        else:
            p_te = mezcla
    evaluar(f"hibrido_puerta_{int(UMBRAL)}h", p_va, p_te, va, te, tr, filas,
            time.time() - t0)
    print(f"  hibrido puerta   val {filas[-1]['MdMRE_val']:.4f}  "
          f"test {filas[-1]['MdMRE_test']:.4f}  "
          f"compresion {filas[-1]['compresion']}")

    r = pd.DataFrame(filas).sort_values("MdMRE_val")
    print(f"\n{'='*118}\nORDENADO POR VALIDACION (la seleccion honesta)\n{'='*118}")
    print(r.to_string(index=False))
    print("\nLa columna de prueba es descriptiva: ningun modelo fue elegido con ella.")
    r.to_csv("results/sweep_v3.csv", index=False)
    print("escrito results/sweep_v3.csv")


if __name__ == "__main__":
    main()
