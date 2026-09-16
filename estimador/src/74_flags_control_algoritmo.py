"""Same-algorithm control for the flags experiment.

The flags winner is a LinearSVR, while the vector references are
HistGradientBoosting. A difference between them could therefore come from the
information (flags vs vector) or from the loss (LinearSVR's epsilon-insensitive
loss estimates something closer to the conditional median in log space, which
suits log-normal hours). This control fits the SAME LinearSVR on the vector, so
the only thing that differs from the flags winner is the input.

It also breaks the bias down by source on the fixed test block: flags carry no
explicit organisation signal, and a pooled training set dominated by the public
Jira block (median 0.83 h) would pull TAWOS and SiP predictions downwards if the
flags cannot tell the organisations apart -- something the raw vector can, from
writing style alone.
"""
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVR

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 240)
spec = importlib.util.spec_from_file_location("m73", "src/73_modelos_flags.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

RES = Path("results/flags")
GANADORES = ["F1 · grupos k-means | SVR lineal",
             "F1+F2+F3 · todo lo derivado del vector | SVR lineal"]


def main():
    d = pd.read_parquet(M.FLAGS)
    y = d.effort_hours.to_numpy()
    m = {b: (d.bloque == b).to_numpy() for b in ("train",) + M.BLOQUES}
    v4 = pd.read_parquet(M.V4).set_index("task_id").loc[d.task_id]
    emb = np.load(f"{M.EMB}.npy")
    ids = np.load(f"{M.EMB}.ids.npy", allow_pickle=True)
    pos = {t: i for i, t in enumerate(ids)}
    X = emb[[pos[t] for t in d.task_id]]
    oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=25).fit(
        v4.loc[m["train"], ["task_kind", "source"]].astype(str))
    sc = StandardScaler().fit(np.log1p(v4.loc[m["train"], M.NUM].to_numpy(float)))
    completo = np.hstack([X, oh.transform(v4[["task_kind", "source"]].astype(str)),
                          sc.transform(np.log1p(v4[M.NUM].to_numpy(float)))])

    pr = pd.read_parquet(RES / "predicciones.parquet")
    preds = {}
    for col in GANADORES + [" | ".join(M.REF_V), " | ".join(M.REF_C)]:
        preds[col] = {b: pr.loc[pr.bloque == b, col].to_numpy() for b in M.BLOQUES}
    for b in M.BLOQUES:                                   # mismo orden de filas
        assert (pr.loc[pr.bloque == b, "task_id"].to_numpy() == d.task_id[m[b]].to_numpy()).all()

    for nombre, Xm in (("referencia | vector 768 · SVR lineal", X),
                       ("referencia | modelo actual · SVR lineal", completo)):
        svr = LinearSVR(C=0.5, max_iter=5000, random_state=0).fit(Xm[m["train"]], np.log(y[m["train"]]))
        preds[nombre] = {b: np.exp(svr.predict(Xm[m[b]])) for b in M.BLOQUES}
        print(f"ajustado {nombre}", flush=True)

    filas = []
    for nombre, pb in preds.items():
        for b in M.BLOQUES:
            filas.append(dict(modelo=nombre, bloque=b, **M.metricas(y[m[b]], pb[b])))
    t = pd.DataFrame(filas)
    t.to_csv(RES / "control_mismo_algoritmo.csv", index=False)
    cols = ["modelo", "MdMRE", "MdAElog", "sesgo", "compresion", "spearman", "SA"]
    for b in M.BLOQUES:
        print(f"\n{M.SEP}\n{b}\n{M.SEP}")
        print(t[t.bloque == b][cols].sort_values("MdAElog").round(4).to_string(index=False))

    print(f"\n{M.SEP}\nMISMO ALGORITMO: flags (SVR) frente a vector (SVR)\n{M.SEP}")
    veredictos = {}
    for g in GANADORES:
        for b in ("prueba_d1", "externo"):
            v = M.veredicto(y[m[b]], preds[g][b], preds["referencia | vector 768 · SVR lineal"][b])
            veredictos[f"{g} · {b}"] = v
            print(f"  {g:55s} {b:10s} {v}")

    print(f"\n{M.SEP}\nSESGO (mediana pred/real) POR FUENTE en la prueba fija\n{M.SEP}")
    fuente = v4.source.to_numpy()[m["prueba_d1"]]
    yy = y[m["prueba_d1"]]
    sesgos = []
    for nombre in (GANADORES[0], "referencia | vector 768 · SVR lineal", " | ".join(M.REF_V),
                   " | ".join(M.REF_C)):
        fila = {"modelo": nombre}
        for s in ("TAWOS", "SiP"):
            k = fuente == s
            fila[f"sesgo_{s}"] = float(np.median(preds[nombre]["prueba_d1"][k] / yy[k]))
        sesgos.append(fila)
    s = pd.DataFrame(sesgos)
    print(s.round(3).to_string(index=False))
    s.to_csv(RES / "sesgo_por_fuente.csv", index=False)
    json.dump(veredictos, open(RES / "control_veredictos.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\nescrito {RES}/control_mismo_algoritmo.csv, sesgo_por_fuente.csv, control_veredictos.json")


if __name__ == "__main__":
    main()
