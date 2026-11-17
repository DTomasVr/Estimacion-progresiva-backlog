"""What the candidate actually answers: worked examples in and out.

Three readings, in the order a reviewer would want them:

  A  the real deployment input -- tasks written by the generation module -- with
     the number and the band the artefact would show
  B  real held-out tasks where the truth is known, sampled across effort
     buckets, so the hits and the misses are both visible
  C  the error by task size, which is where the model's compression shows

The model is fed the ENGLISH title without normalising the vectors, as the
artefact metadata says.
"""
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 240)
pd.set_option("display.max_colwidth", 88)
DATA = "data/processed/pbe_effort_v4.parquet"
EMB_TIT = "data/processed/embeddings/v4_mpnet_titulos"
ART = "models/candidato_titulos.joblib"
GEN = Path("F:/Software Task generation Model/salida_gen.json")
SEP = "=" * 132


def main():
    import torch
    from sentence_transformers import SentenceTransformer

    a = joblib.load(ART)
    reg, oh = a["regresor"], a["onehot"]
    k80 = a["bandas"]["validación sin Apache"]["factor80"]
    print(f"artefacto: {a['variante']} · corpus {a['corpus']} · {a['n_train']:,} filas · "
          f"banda 80 % ×{k80:.2f}")

    def predecir(X, kinds):
        if oh is not None:
            X = np.hstack([X, oh.transform(pd.DataFrame({"task_kind": kinds}).astype(str))])
        return np.exp(reg.predict(X))

    # ------------------------------------------------ A · entrada real del generador
    st = SentenceTransformer(a["codificador"],
                             device="cuda" if torch.cuda.is_available() else "cpu")
    gen = json.load(open(GEN, encoding="utf-8"))
    filas = []
    for dom, d in gen.items():
        for h in d["historias"]:
            es, en = h.get("tareas_es") or [], h.get("tareas") or []
            for i, t in enumerate(es):
                if i < len(en):
                    filas.append(dict(dominio=dom, es=t, en=en[i]))
    g = pd.DataFrame(filas)
    Xg = st.encode(g.en.tolist(), normalize_embeddings=False, show_progress_bar=False)
    g["horas"] = predecir(Xg, ["task"] * len(g))
    g["bajo"], g["alto"] = g.horas / k80, g.horas * k80

    print(f"\n{SEP}\nA · LO QUE RESPONDE A LAS TAREAS DEL GENERADOR (n={len(g)})\n{SEP}")
    for dom in g.dominio.unique():
        v = g[g.dominio == dom].sort_values("horas")
        print(f"\n  — {dom} —")
        for _, r in pd.concat([v.head(3), v.tail(3)]).iterrows():
            print(f"    {r.horas:5.2f} h  [{r.bajo:4.1f}–{r.alto:5.1f}]  {r.es[:78]}")
    print(f"\n  resumen: mediana {g.horas.median():.2f} h · "
          f"p10–p90 {g.horas.quantile(.1):.2f}–{g.horas.quantile(.9):.2f} h · "
          f"min {g.horas.min():.2f} · max {g.horas.max():.2f}")
    print(f"  suma del backlog completo: {g.horas.sum():.0f} h "
          f"({g.horas.sum() / 8:.0f} días-persona) para {len(g)} tareas")

    # ------------------------------------------------ B · tareas reales con verdad
    df = pd.read_parquet(DATA)
    sw = df[df.is_software_task == 1].copy()
    sw["apache"] = sw.fuente_jira.eq("Apache")
    emb = np.load(f"{EMB_TIT}.npy")
    ids = np.load(f"{EMB_TIT}.ids.npy", allow_pickle=True)
    pos = {t: i for i, t in enumerate(ids)}
    sw["fe"] = [pos[t] for t in sw.task_id]

    bloques = {"prueba SiP": sw[(sw.split_t3 == "test") & (sw.source == "SiP")],
               "externo no Apache": sw[(sw.split_t3 == "externo") & (~sw.apache)]}
    tramos = [(0, 1, "<1 h"), (1, 4, "1–4 h"), (4, 16, "4–16 h"), (16, 1e9, ">16 h")]
    rng = np.random.default_rng(11)
    for nombre, d in bloques.items():
        d = d.copy()
        d["pred"] = predecir(emb[d.fe.to_numpy()], d.task_kind.astype(str).tolist())
        d["razon"] = d.pred / d.effort_hours
        print(f"\n{SEP}\nB · TAREAS REALES DE «{nombre}» (n={len(d):,})\n{SEP}")
        for lo, hi, etq in tramos:
            v = d[(d.effort_hours >= lo) & (d.effort_hours < hi)]
            if not len(v):
                continue
            m = v.iloc[rng.choice(len(v), min(2, len(v)), replace=False)]
            for _, r in m.iterrows():
                marca = "ok " if .75 <= r.razon <= 1.33 else "   "
                print(f"  {marca}real {r.effort_hours:7.2f} h -> predice {r.pred:5.2f} h "
                      f"(×{r.razon:5.2f})  {str(r.title)[:66]}")
        print(f"\n  {'tramo':8s} {'n':>6s} {'real med':>9s} {'pred med':>9s} "
              f"{'sesgo':>7s} {'PRED25':>7s}")
        for lo, hi, etq in tramos:
            v = d[(d.effort_hours >= lo) & (d.effort_hours < hi)]
            if not len(v):
                continue
            p25 = float(np.mean(np.abs(v.pred - v.effort_hours) / v.effort_hours <= .25))
            print(f"  {etq:8s} {len(v):6d} {v.effort_hours.median():9.2f} "
                  f"{v.pred.median():9.2f} {v.razon.median():7.2f} {p25:7.3f}")

    print(f"\n{SEP}\nC · LO QUE SE VE EN LOS TRAMOS\n{SEP}")
    print("  El modelo comprime: exagera lo pequeño y se queda corto en lo grande.")
    print("  Es la razón de que la banda sea ancha y de que PRED(25) no suba.")


if __name__ == "__main__":
    main()
