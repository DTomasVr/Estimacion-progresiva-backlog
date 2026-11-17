"""Steps 2-4: choose features, hyperparameters and algorithm on VALIDATION.

This script must never read split_t3 == 'test'. It loads only train and val,
and asserts as much before starting, so the final evaluation stays honest.

Selected here and frozen for the final run:
  * encoder arm      frozen vs fine-tuned-on-temporal-train
  * feature block    B0..B4
  * hyperparameters  a small grid per algorithm
  * algorithm        ridge / HGB / stacking
"""
import itertools
import json
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402

pd.set_option("display.width", 250)
DATA = "data/processed/pbe_effort_v3.parquet"
FROZEN = "data/processed/embeddings/paraphrase-multilingual-mpnet-base-v2.npy"
FINETUNED = "data/processed/embeddings/mpnet_ft_temporal.npy"

NUEVAS = ["n_links", "n_links_estructurales", "n_componentes", "en_sprint",
          "tiene_asignado", "Title_Changed_After_Estimation",
          "Story_Point_Changed_After_Estimation"]
BLOQUES = {
    "B0_emb":            dict(cat=[], num=False),
    "B1_emb_tipo":       dict(cat=["task_kind", "source"], num=False),
    "B2_emb_tipo_dev":   dict(cat=["task_kind", "source", "desarrollador"], num=False),
    "B3_emb_tipo_num":   dict(cat=["task_kind", "source"], num=True),
    "B4_todo":           dict(cat=["task_kind", "source", "desarrollador"], num=True),
}


def cargar():
    df = pd.read_parquet(DATA)
    sw = df[df.is_software_task == 1].copy()
    tr = sw[sw.split_t3 == "train"]
    va = sw[sw.split_t3 == "val"]
    assert "test" not in set(tr.split_t3) | set(va.split_t3)
    ids = df.task_id.to_numpy()
    pos = {t: i for i, t in enumerate(ids)}
    embs = {}
    base = pd.read_parquet("data/processed/pbe_effort_clean.parquet",
                           columns=["task_id"]).task_id.to_numpy()
    pos_frozen = {t: i for i, t in enumerate(base)}
    embs["congelado"] = (np.load(FROZEN), pos_frozen)
    try:
        embs["ajustado"] = (np.load(FINETUNED), pos)
    except FileNotFoundError:
        print("  (aun no existe el codificador ajustado; solo brazo congelado)")
    return tr, va, embs


def construir(tr, va, emb, pos, cfg):
    def rows(d):
        return emb[[pos[t] for t in d.task_id]]
    A, B = [rows(tr)], [rows(va)]
    if cfg["cat"]:
        oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                           min_frequency=5)
        A.append(oh.fit_transform(tr[cfg["cat"]].astype(str)))
        B.append(oh.transform(va[cfg["cat"]].astype(str)))
    if cfg["num"]:
        sc = StandardScaler()
        A.append(sc.fit_transform(np.log1p(tr[NUEVAS].to_numpy(float))))
        B.append(sc.transform(np.log1p(va[NUEVAS].to_numpy(float))))
    return np.hstack(A), np.hstack(B)


def por_proyecto(tr, va, emb, pos, cfg, make):
    """Fit inside each project, score on that project's validation block."""
    md, p25, pesos = [], [], []
    for proj in sorted(va.project.unique()):
        t, v = tr[tr.project == proj], va[va.project == proj]
        if len(t) < 50 or len(v) < 12:
            continue
        A, B = construir(t, v, emb, pos, cfg)
        y = np.log(t.effort_hours.to_numpy())
        p = np.exp(make().fit(A, y).predict(B))
        m = E.point_metrics(v.effort_hours.to_numpy(), p)
        md.append(m["MdMRE"]); p25.append(m["PRED25"]); pesos.append(len(v))
    pesos = np.array(pesos, float)
    return (float(np.average(md, weights=pesos)),
            float(np.average(p25, weights=pesos)), len(md))


def main():
    tr, va, embs = cargar()
    print(f"train {len(tr):,}  val {len(va):,}  "
          f"proyectos en validacion: {va.project.nunique()}")
    print("NOTA: este script no abre el bloque de prueba.\n")

    filas = []
    for arm, (emb, pos) in embs.items():
        for nombre, cfg in BLOQUES.items():
            m, p, n = por_proyecto(tr, va, emb, pos, cfg,
                                   lambda: Ridge(alpha=1.0))
            filas.append(dict(codificador=arm, bloque=nombre, algoritmo="ridge",
                              MdMRE_val=round(m, 4), PRED25_val=round(p, 4),
                              proyectos=n))
            print(f"  {arm:10s} {nombre:18s} ridge   MdMRE {m:.4f}  PRED25 {p:.4f}")
    r = pd.DataFrame(filas)
    mejor = r.sort_values("MdMRE_val").iloc[0]
    print(f"\nmejor combinacion codificador+bloque: {mejor.codificador} / "
          f"{mejor.bloque}  (MdMRE val {mejor.MdMRE_val})")

    emb, pos = embs[mejor.codificador]
    cfg = BLOQUES[mejor.bloque]

    print("\n--- rejilla de hiperparametros sobre validacion ---")
    grid = []
    for a in [0.1, 1.0, 3.0, 10.0, 30.0]:
        m, p, _ = por_proyecto(tr, va, emb, pos, cfg, lambda a=a: Ridge(alpha=a))
        grid.append(dict(alg="ridge", cfg=f"alpha={a}", MdMRE=round(m, 4),
                         PRED25=round(p, 4)))
        print(f"  ridge alpha={a:<5} MdMRE {m:.4f}  PRED25 {p:.4f}")
    for it, dep, lr in itertools.product([200, 400], [4, 6], [0.03, 0.06]):
        m, p, _ = por_proyecto(
            tr, va, emb, pos, cfg,
            lambda it=it, dep=dep, lr=lr: HistGradientBoostingRegressor(
                max_iter=it, max_depth=dep, learning_rate=lr, l2_regularization=1.0,
                random_state=0))
        grid.append(dict(alg="hgb", cfg=f"it={it},d={dep},lr={lr}",
                         MdMRE=round(m, 4), PRED25=round(p, 4)))
        print(f"  hgb it={it} d={dep} lr={lr}  MdMRE {m:.4f}  PRED25 {p:.4f}")

    g = pd.DataFrame(grid).sort_values("MdMRE")
    print(f"\nmejor hiperparametro: {g.iloc[0].alg} {g.iloc[0].cfg}  "
          f"(MdMRE val {g.iloc[0].MdMRE})")

    mejor_ridge = g[g.alg == "ridge"].iloc[0]
    mejor_hgb = g[g.alg == "hgb"].iloc[0]
    alpha = float(mejor_ridge.cfg.split("=")[1])
    it, dep, lr = [float(x.split("=")[1]) for x in mejor_hgb.cfg.split(",")]

    m, p, _ = por_proyecto(tr, va, emb, pos, cfg, lambda: StackingRegressor(
        [("ridge", Ridge(alpha=alpha)),
         ("hgb", HistGradientBoostingRegressor(
             max_iter=int(it), max_depth=int(dep), learning_rate=lr,
             l2_regularization=1.0, random_state=0)),
         ("knn", KNeighborsRegressor(n_neighbors=10, metric="cosine",
                                     weights="distance"))],
        final_estimator=Ridge(alpha=1.0), cv=3, n_jobs=1))
    print(f"  apilamiento             MdMRE {m:.4f}  PRED25 {p:.4f}")
    g = pd.concat([g, pd.DataFrame([dict(alg="stacking", cfg="ridge+hgb+knn",
                                         MdMRE=round(m, 4), PRED25=round(p, 4))])])

    elegido = g.sort_values("MdMRE").iloc[0]
    seleccion = dict(codificador=mejor.codificador, bloque=mejor.bloque,
                     algoritmo=elegido.alg, config=elegido.cfg,
                     MdMRE_val=float(elegido.MdMRE),
                     ridge_alpha=alpha,
                     hgb=dict(max_iter=int(it), max_depth=int(dep),
                              learning_rate=lr))
    with open("results/seleccion_val.json", "w", encoding="utf8") as fh:
        json.dump(seleccion, fh, indent=2, ensure_ascii=False)
    print(f"\nSELECCION CONGELADA -> results/seleccion_val.json")
    print(json.dumps(seleccion, indent=2, ensure_ascii=False))
    r.to_csv("results/seleccion_bloques.csv", index=False)
    g.to_csv("results/seleccion_grid.csv", index=False)


if __name__ == "__main__":
    main()
