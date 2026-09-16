"""Train many algorithms on the flags dataset alone and compare with the vector.

Input is ONLY data/processed/flags/agix_flags.parquet: hours and binary flags.
The 768-d vectors are loaded solely to build the two reference models the flags
must be compared against:
  vector 768 · HGB         the text vector alone, no source, no numeric features
                           -> the like-for-like test: does compressing the vector
                              into flags lose information?
  modelo actual · HGB      vector + category + source + numeric, the current model

Selection is made on the validation block only, under two criteria:
  MdMRE    the headline metric used throughout the thesis
  MdAElog  median absolute log error, which is symmetric
Both are reported because MdMRE rewards under-estimation -- the relative error of
predicting too little can never exceed 1 -- so a near-constant model shifted
downwards can win on MdMRE without ordering anything. Bias (median pred/actual)
and compression (spread of predictions over spread of actuals) are printed for
every model so such degenerate winners are visible, not hidden.

The two test blocks are read once; each winner gets paired bootstrap verdicts
against both references on both metrics.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesRegressor, HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 260)
FLAGS = "data/processed/flags/agix_flags.parquet"
V4 = "data/processed/agix_effort_v4.parquet"
EMB = "data/processed/embeddings/v4_mpnet_congelado"
RES = Path("results/flags")
NUM = ["n_links", "n_links_estructurales", "n_componentes", "tiene_asignado"]
FAMILIAS = {
    "F1 · grupos k-means": ("grupo_",),
    "F2 · categoría predicha": ("catpred_",),
    "F3 · flags semánticos": ("tipo_", "area_"),
    "F2+F3": ("catpred_", "tipo_", "area_"),
    "F1+F2+F3 · todo lo derivado del vector": ("grupo_", "catpred_", "tipo_", "area_"),
    "F4 · categoría real (referencia)": ("catreal_",),
}
BLOQUES = ("val", "prueba_d1", "externo")
REF_V = ("referencia", "vector 768 · HGB")
REF_C = ("referencia", "modelo actual (vector+cat+fuente+num) · HGB")
SEP = "=" * 130


class MedianaPorCombinacion:
    """Median log-hours of every distinct flag combination seen >= 5 times in train."""

    def fit(self, F, ylog):
        claves = [r.tobytes() for r in F]
        s = pd.Series(ylog).groupby(claves)
        self.tabla = s.median()[s.size() >= 5].to_dict()
        self.glob = float(np.median(ylog))
        return self

    def predict(self, F):
        return np.array([self.tabla.get(r.tobytes(), self.glob) for r in F])


def algoritmos():
    a = {
        "Mediana por combinación": MedianaPorCombinacion(),
        "Ridge": Ridge(alpha=1.0),
        "Árbol de decisión": DecisionTreeRegressor(max_depth=10, min_samples_leaf=50, random_state=0),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=20, n_jobs=-1, random_state=0),
        "Extra Trees": ExtraTreesRegressor(n_estimators=300, min_samples_leaf=20, n_jobs=-1, random_state=0),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=.06,
                                                              l2_regularization=1., random_state=0),
        "KNN (k=50)": KNeighborsRegressor(n_neighbors=50, n_jobs=-1),
        "MLP": MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=200,
                            early_stopping=True, random_state=0),
        "SVR lineal": LinearSVR(C=0.5, max_iter=5000, random_state=0),
    }
    try:
        from xgboost import XGBRegressor
        a["XGBoost (GPU)"] = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=.05,
                                          subsample=.8, colsample_bytree=.8, device="cuda",
                                          tree_method="hist", random_state=0)
    except Exception as e:                                   # se informa, no se oculta
        print("XGBoost no disponible:", e)
    return a


def sa(y, p, corridas=300):
    rng, n = np.random.default_rng(0), len(y)
    base = []
    for _ in range(corridas):
        i = rng.integers(0, n - 1, n)
        i += i >= np.arange(n)
        base.append(np.mean(np.abs(y - y[i])))
    return 1 - np.mean(np.abs(y - p)) / np.mean(base)


def metricas(y, p):
    p = np.clip(np.asarray(p, float), .01, None)
    m = E.point_metrics(y, p)
    ly, lp = np.log(y), np.log(p)
    return dict(MdMRE=m["MdMRE"], PRED25=m["PRED25"], MAE_h=m["MAE_h"],
                MdAElog=float(np.median(np.abs(ly - lp))),
                sesgo=float(np.median(p / y)),
                compresion=float(np.std(lp) / np.std(ly)),
                spearman=float(pd.Series(y).corr(pd.Series(p), method="spearman")),
                SA=sa(y, p))


def veredicto(y, a, b, n=2000):
    """Paired bootstrap verdict of a against b, on MdMRE and on the symmetric log error."""
    out = []
    for nombre, fa, fb in (("MdMRE", np.abs(y - a) / y, np.abs(y - b) / y),
                           ("errlog", np.abs(np.log(y / a)), np.abs(np.log(y / b)))):
        rng = np.random.default_rng(1)
        d = [np.median(fa[i]) - np.median(fb[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
        lo, hi = np.percentile(d, [2.5, 97.5])
        v = "flags mejor" if hi < 0 else "flags peor" if lo > 0 else "empate"
        out.append(f"{nombre}: {v} [{lo:+.3f}, {hi:+.3f}]")
    return " · ".join(out)


def main():
    RES.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(FLAGS)
    y = d.effort_hours.to_numpy()
    m = {b: (d.bloque == b).to_numpy() for b in ("train",) + BLOQUES}
    ylog_tr = np.log(y[m["train"]])
    filas, preds = [], {}

    def evaluar(familia, algoritmo, n_rasgos, p_por_bloque, seg):
        f = dict(familia=familia, algoritmo=algoritmo, n_rasgos=n_rasgos, seg=round(seg, 1))
        for b in BLOQUES:
            for k, v in metricas(y[m[b]], p_por_bloque[b]).items():
                f[f"{b}_{k}"] = v
        filas.append(f)
        preds[(familia, algoritmo)] = p_por_bloque
        print(f"  {familia:42s} {algoritmo:26s} val {f['val_MdMRE']:.4f} errlog {f['val_MdAElog']:.3f} "
              f"sesgo {f['val_sesgo']:.2f} compr {f['val_compresion']:.2f}  "
              f"prueba {f['prueba_d1_MdMRE']:.4f}  externo {f['externo_MdMRE']:.4f}  {seg:6.1f}s", flush=True)

    # ------------------------------------------------ referencias desde el vector
    v4 = pd.read_parquet(V4).set_index("task_id").loc[d.task_id]
    emb = np.load(f"{EMB}.npy")
    ids = np.load(f"{EMB}.ids.npy", allow_pickle=True)
    pos = {t: i for i, t in enumerate(ids)}
    X = emb[[pos[t] for t in d.task_id]]
    oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=25).fit(
        v4.loc[m["train"], ["task_kind", "source"]].astype(str))
    sc = StandardScaler().fit(np.log1p(v4.loc[m["train"], NUM].to_numpy(float)))
    completo = np.hstack([X, oh.transform(v4[["task_kind", "source"]].astype(str)),
                          sc.transform(np.log1p(v4[NUM].to_numpy(float)))])
    print("REFERENCIAS DESDE EL VECTOR")
    for (_, nombre), M in ((REF_V, X), (REF_C, completo)):
        t0 = time.time()
        r = HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=.06,
                                          l2_regularization=1., random_state=0).fit(M[m["train"]], ylog_tr)
        evaluar("referencia", nombre, M.shape[1], {b: np.exp(r.predict(M[m[b]])) for b in BLOQUES},
                time.time() - t0)
    evaluar("referencia", "constante (mediana global)", 0,
            {b: np.full(int(m[b].sum()), float(np.median(y[m["train"]]))) for b in BLOQUES}, 0)

    # ------------------------------------------------ solo flags
    print("\nSOLO FLAGS + HORAS")
    for familia, prefijos in FAMILIAS.items():
        cols = [c for c in d.columns if c.startswith(prefijos)]
        F = d[cols].to_numpy(np.float32)
        for nombre, alg in algoritmos().items():
            t0 = time.time()
            alg.fit(F[m["train"]], ylog_tr)
            evaluar(familia, nombre, len(cols), {b: np.exp(alg.predict(F[m[b]])) for b in BLOQUES},
                    time.time() - t0)

    r = pd.DataFrame(filas)
    r.to_csv(RES / "resumen.csv", index=False)
    tablas = []
    for b in BLOQUES:
        t = pd.DataFrame({"task_id": d.task_id[m[b]].to_numpy(), "bloque": b, "real": y[m[b]]})
        for (fam, alg), pb in preds.items():
            t[f"{fam} | {alg}"] = pb[b]
        tablas.append(t)
    pd.concat(tablas).to_parquet(RES / "predicciones.parquet", index=False)

    solo = r[r.familia != "referencia"]
    seleccion = {}
    for criterio in ("val_MdMRE", "val_MdAElog"):
        g = solo.loc[solo[criterio].idxmin()]
        print(f"\n{SEP}\nGANADOR EN VALIDACIÓN por {criterio}: {g.familia} · {g.algoritmo} ({g.n_rasgos} flags)"
              f" · sesgo val {g.val_sesgo:.2f} · compresión val {g.val_compresion:.2f}\n{SEP}")
        sel = {"familia": g.familia, "algoritmo": g.algoritmo, "n_flags": int(g.n_rasgos)}
        for b in ("prueba_d1", "externo"):
            yy, pg = y[m[b]], preds[(g.familia, g.algoritmo)][b]
            vv, vc = veredicto(yy, pg, preds[REF_V][b]), veredicto(yy, pg, preds[REF_C][b])
            sel[b] = {k: float(g[f"{b}_{k}"]) for k in
                      ("MdMRE", "PRED25", "MdAElog", "sesgo", "compresion", "spearman", "SA")}
            sel[b].update(vs_vector=vv, vs_modelo_actual=vc)
            print(f"  {b:10s} MdMRE {g[f'{b}_MdMRE']:.4f}  errlog {g[f'{b}_MdAElog']:.3f}  "
                  f"sesgo {g[f'{b}_sesgo']:.2f}  compr {g[f'{b}_compresion']:.2f}  "
                  f"rho {g[f'{b}_spearman']:+.3f}  SA {g[f'{b}_SA']:.3f}")
            print(f"             vs vector 768:    {vv}")
            print(f"             vs modelo actual: {vc}")
        seleccion[criterio] = sel
    json.dump(seleccion, open(RES / "ganador.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    cols = ["familia", "algoritmo", "n_rasgos", "val_MdMRE", "val_MdAElog", "val_sesgo", "val_compresion",
            "prueba_d1_MdMRE", "prueba_d1_MdAElog", "prueba_d1_SA", "externo_MdMRE", "externo_MdAElog",
            "externo_SA", "externo_spearman"]
    print(f"\n{SEP}\nREFERENCIAS\n{SEP}")
    print(r[r.familia == "referencia"][cols].round(4).to_string(index=False))
    print(f"\n{SEP}\nMEJOR ALGORITMO POR FAMILIA (elegido en validación por error log simétrico)\n{SEP}")
    mejores = solo.loc[solo.groupby("familia").val_MdAElog.idxmin()].sort_values("val_MdAElog")
    print(mejores[cols].round(4).to_string(index=False))
    for metrica, dec in (("val_MdMRE", 4), ("val_MdAElog", 3), ("val_sesgo", 2), ("val_compresion", 2)):
        print(f"\n{SEP}\n{metrica.upper()} POR ALGORITMO Y FAMILIA\n{SEP}")
        print(solo.pivot(index="algoritmo", columns="familia", values=metrica).round(dec).to_string())
    print(f"\nescrito {RES}/resumen.csv, predicciones.parquet, ganador.json")


if __name__ == "__main__":
    main()
