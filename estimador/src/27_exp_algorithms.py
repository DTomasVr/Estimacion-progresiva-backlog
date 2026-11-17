"""Algorithm sweep over the families used in the reviewed literature.

Every learner sees the same features (SBERT embedding + task kind + source),
the same splits and the same log target, so differences are attributable to the
algorithm alone.

Families and the papers that motivate them:
  arboles        Rodriguez Sanchez et al. 2023 (decision trees + story points)
  bosques        Random Forest / Extra Trees -- the standard ensemble baseline
  boosting       Yalciner et al. 2024 (SBERT + gradient boosted trees)
  lineales       ridge / lasso / elastic net
  bayesiano      Turic et al. 2023 (Bayesian effort estimation)
  vecinos        the thesis's own TF-IDF weighted KNN
  svm            support vector regression
  red            multilayer perceptron
  conjuntos      Voting and Stacking (Ahmad & Kuo 2025, Alsaadi & Saeedi 2024,
                 Fatima et al. 2026 stacking ensemble)
"""
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import (ExtraTreesRegressor,
                              HistGradientBoostingRegressor,
                              RandomForestRegressor, StackingRegressor,
                              VotingRegressor)
from sklearn.linear_model import (BayesianRidge, ElasticNet, Lasso, Ridge)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR, SVR
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402
import features as F  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
E.DATA = "data/processed/pbe_effort_v2.parquet"


def base_learners():
    return {
        "arbol":        DecisionTreeRegressor(max_depth=12, min_samples_leaf=20,
                                              random_state=0),
        "rf":           RandomForestRegressor(n_estimators=400, min_samples_leaf=2,
                                              max_features="sqrt", n_jobs=-1,
                                              random_state=0),
        "rf_prof":      RandomForestRegressor(n_estimators=600, min_samples_leaf=5,
                                              max_features=0.3, n_jobs=-1,
                                              random_state=0),
        "extra_trees":  ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2,
                                            max_features="sqrt", n_jobs=-1,
                                            random_state=0),
        "hgb":          HistGradientBoostingRegressor(max_iter=400, max_depth=6,
                                                      learning_rate=0.06,
                                                      random_state=0),
        "ridge":        Ridge(alpha=1.0),
        "lasso":        Lasso(alpha=0.001, max_iter=5000),
        "elastic":      ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000),
        "bayes_ridge":  BayesianRidge(),
        "knn":          KNeighborsRegressor(n_neighbors=10, metric="cosine",
                                            weights="distance"),
        "svr_lineal":   LinearSVR(C=0.5, max_iter=5000, random_state=0),
        "mlp":          MLPRegressor(hidden_layer_sizes=(256, 64), alpha=1e-3,
                                     learning_rate_init=1e-3, max_iter=120,
                                     early_stopping=True, random_state=0),
    }


def run_split(split_col, emb, pos, results):
    tr, va, te = E.load(split_col)
    fb = F.FeatureBuilder(use_kind=True, use_source=True, use_priority=False,
                          use_numeric=False, use_project=False)
    Xtr, Xva, Xte = fb.fit_transform(tr, [va, te], emb, pos)
    ytr = np.log(tr.effort_hours.to_numpy())
    y_te, y_va = te.effort_hours.to_numpy(), va.effort_hours.to_numpy()
    print(f"\n{'='*78}\nsplit={split_col}  train {len(tr):,}  test {len(te):,}  "
          f"features {Xtr.shape[1]}\n{'='*78}")

    # PCA view for the kernel method, which does not scale to 783 dims
    pca = PCA(n_components=128, random_state=0).fit(Xtr)
    Ptr, Pva, Pte = pca.transform(Xtr), pca.transform(Xva), pca.transform(Xte)
    print(f"  PCA 128 componentes: {pca.explained_variance_ratio_.sum()*100:.1f}% "
          f"de la varianza")

    fitted = {}
    for name, mdl in base_learners().items():
        t0 = time.time()
        mdl.fit(Xtr, ytr)
        lp_te, lp_va = mdl.predict(Xte), mdl.predict(Xva)
        m = E.point_metrics(y_te, np.exp(lp_te))
        m.update(E.interval_metrics(lp_va, y_va, lp_te, y_te))
        m["compresion"] = float(np.std(lp_te) / np.std(np.log(y_te)))
        m["spearman"] = float(pd.Series(y_te).corr(pd.Series(np.exp(lp_te)),
                                                   method="spearman"))
        E.record("algoritmos", name, split_col, m)
        results.append(dict(split=split_col, modelo=name, **m,
                            segundos=round(time.time() - t0, 1)))
        print(f"  {name:12s} MdMRE {m['MdMRE']:.4f}  PRED25 {m['PRED25']:.4f}  "
              f"comp {m['compresion']:.3f}  rho {m['spearman']:.3f}  "
              f"{time.time()-t0:5.1f}s")
        fitted[name] = mdl

    # SVR with an RBF kernel, on the reduced space
    t0 = time.time()
    svr = SVR(C=2.0, epsilon=0.1, kernel="rbf").fit(Ptr, ytr)
    lp_te, lp_va = svr.predict(Pte), svr.predict(Pva)
    m = E.point_metrics(y_te, np.exp(lp_te))
    m.update(E.interval_metrics(lp_va, y_va, lp_te, y_te))
    m["compresion"] = float(np.std(lp_te) / np.std(np.log(y_te)))
    m["spearman"] = float(pd.Series(y_te).corr(pd.Series(np.exp(lp_te)),
                                               method="spearman"))
    E.record("algoritmos", "svr_rbf_pca", split_col, m)
    results.append(dict(split=split_col, modelo="svr_rbf_pca", **m,
                        segundos=round(time.time() - t0, 1)))
    print(f"  {'svr_rbf_pca':12s} MdMRE {m['MdMRE']:.4f}  "
          f"PRED25 {m['PRED25']:.4f}  comp {m['compresion']:.3f}  "
          f"rho {m['spearman']:.3f}  {time.time()-t0:5.1f}s")

    # ---------------- ensembles ----------------
    print("\n  --- conjuntos ---")
    picks = ["ridge", "hgb", "knn", "extra_trees"]
    # lighter members: stacking refits each of them `cv` times
    ENS_BASE = dict(base_learners())
    ENS_BASE["extra_trees"] = ExtraTreesRegressor(
        n_estimators=200, min_samples_leaf=2, max_features="sqrt",
        n_jobs=-1, random_state=0)
    ens = {
        "voting": VotingRegressor([(n, ENS_BASE[n]) for n in picks],
                                  n_jobs=1),
        "stacking": StackingRegressor(
            [(n, ENS_BASE[n]) for n in picks],
            final_estimator=Ridge(alpha=1.0), cv=3, n_jobs=1),
    }
    for name, mdl in ens.items():
        t0 = time.time()
        mdl.fit(Xtr, ytr)
        lp_te, lp_va = mdl.predict(Xte), mdl.predict(Xva)
        m = E.point_metrics(y_te, np.exp(lp_te))
        m.update(E.interval_metrics(lp_va, y_va, lp_te, y_te))
        m["compresion"] = float(np.std(lp_te) / np.std(np.log(y_te)))
        m["spearman"] = float(pd.Series(y_te).corr(pd.Series(np.exp(lp_te)),
                                                   method="spearman"))
        E.record("algoritmos", name, split_col, m,
                 config={"base": picks, "meta": "ridge" if name == "stacking" else None})
        results.append(dict(split=split_col, modelo=name, **m,
                            segundos=round(time.time() - t0, 1)))
        print(f"  {name:12s} MdMRE {m['MdMRE']:.4f}  PRED25 {m['PRED25']:.4f}  "
              f"comp {m['compresion']:.3f}  rho {m['spearman']:.3f}  "
              f"{time.time()-t0:5.1f}s")

    # simple average of the log predictions of the best families
    avg = np.mean([fitted[n].predict(Xte) for n in picks], axis=0)
    avg_va = np.mean([fitted[n].predict(Xva) for n in picks], axis=0)
    m = E.point_metrics(y_te, np.exp(avg))
    m.update(E.interval_metrics(avg_va, y_va, avg, y_te))
    m["compresion"] = float(np.std(avg) / np.std(np.log(y_te)))
    m["spearman"] = float(pd.Series(y_te).corr(pd.Series(np.exp(avg)),
                                               method="spearman"))
    E.record("algoritmos", "media_log", split_col, m, config={"base": picks})
    results.append(dict(split=split_col, modelo="media_log", **m, segundos=0.0))
    print(f"  {'media_log':12s} MdMRE {m['MdMRE']:.4f}  PRED25 {m['PRED25']:.4f}  "
          f"comp {m['compresion']:.3f}  rho {m['spearman']:.3f}")


def main():
    emb, pos = F.load_embeddings_and_pos()
    results = []
    for sc in ["split", "split_project"]:
        run_split(sc, emb, pos, results)

    df = pd.DataFrame(results)
    cols = ["modelo", "MdMRE", "PRED25", "PRED50", "RMSE_log", "compresion",
            "spearman", "cov80", "segundos"]
    for sc in ["split", "split_project"]:
        print(f"\n{'='*78}\nclasificacion final - {sc}\n{'='*78}")
        print(df[df.split == sc].sort_values("MdMRE")[cols].round(4)
              .to_string(index=False))
    df.to_csv("results/algoritmos.csv", index=False)
    print("\nescrito results/algoritmos.csv")


if __name__ == "__main__":
    main()
