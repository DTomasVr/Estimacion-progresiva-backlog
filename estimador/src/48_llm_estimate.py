"""Blind LLM estimation, calibration, and combination with the text model.

An LLM (Claude Opus 5, the assistant running this project) was shown the text
of 100 held-out test tasks WITHOUT their recorded hours, and produced an hour
estimate for each. The estimates are in data/processed/estimacion_llm.csv and
were written before any label was read -- the same design as the controlled
experiments of Pavlic et al. (2024) and Villarrubia et al. (2026).

The sample excludes every task whose hours had already appeared in the working
session, so the estimator had not seen the answers.

This script scores those estimates, learns a single multiplicative correction
on half of them, and measures the calibrated and combined predictors on the
other half.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402

pd.set_option("display.width", 250)
DATA = "data/processed/pbe_effort_v3.parquet"
NUEVAS = ["n_links", "n_links_estructurales", "n_componentes", "en_sprint",
          "tiene_asignado", "Title_Changed_After_Estimation",
          "Story_Point_Changed_After_Estimation"]
CATS = ["task_kind", "source", "desarrollador"]


def modelo_texto(sw, destino):
    emb = np.load("data/processed/embeddings/"
                  "paraphrase-multilingual-mpnet-base-v2.npy")
    ids = pd.read_parquet("data/processed/pbe_effort_clean.parquet",
                          columns=["task_id"]).task_id.to_numpy()
    pos = {t: i for i, t in enumerate(ids)}
    tr = sw[sw.split_t3 == "train"]
    rows = lambda d: emb[[pos[t] for t in d.task_id]]
    oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                       min_frequency=5)
    sc = StandardScaler()
    Xtr = np.hstack([rows(tr), oh.fit_transform(tr[CATS].astype(str)),
                     sc.fit_transform(np.log1p(tr[NUEVAS].to_numpy(float)))])
    m = Ridge(alpha=1.0).fit(Xtr, np.log(tr.effort_hours.to_numpy()))
    X = np.hstack([rows(destino), oh.transform(destino[CATS].astype(str)),
                   sc.transform(np.log1p(destino[NUEVAS].to_numpy(float)))])
    return np.exp(m.predict(X))


def main():
    llm = pd.read_csv("data/processed/estimacion_llm.csv")
    df = pd.read_parquet(DATA)
    sw = df[df.is_software_task == 1]
    m = sw.merge(llm, on="task_id", how="inner").reset_index(drop=True)
    y, p = m.effort_hours.to_numpy(), m.horas_llm.to_numpy()
    pm = modelo_texto(sw, m)
    print(f"tareas estimadas a ciegas: {len(m)}   "
          f"todas del bloque de prueba\n")

    print("--- sobre las 100 ---")
    filas = []
    for lbl, pred in [("linea base", np.full(len(m), sw[sw.split_t3 == "train"]
                                             .effort_hours.median())),
                      ("modelo de texto", pm),
                      ("estimacion LLM", p)]:
        mm = E.point_metrics(y, pred)
        filas.append(dict(metodo=lbl, MdMRE=round(mm["MdMRE"], 4),
                          PRED25=round(mm["PRED25"], 4),
                          spearman=round(float(pd.Series(y).corr(
                              pd.Series(pred), method="spearman")), 3)))
        E.record("llm", lbl.replace(" ", "_"), "muestra100", mm)
    print(pd.DataFrame(filas).to_string(index=False))

    print("\n--- sesgo del LLM por tamano real ---")
    b = pd.cut(y, [0, 1, 4, 16, np.inf], labels=["<1h", "1-4h", "4-16h", ">16h"])
    g = pd.DataFrame({"real": y, "llm": p, "banda": b}).groupby(
        "banda", observed=True).agg(n=("real", "size"), real=("real", "median"),
                                    llm=("llm", "median"))
    g["sesgo"] = (g.llm / g.real).round(2)
    print(g.round(2).to_string())

    print("\n--- calibracion: 50 para aprender, 50 para medir ---")
    rng = np.random.default_rng(7)
    idx = rng.permutation(len(m))
    cal, tst = idx[:50], idx[50:]
    off = float(np.median(np.log(y[cal]) - np.log(p[cal])))
    print(f"  correccion aprendida: {np.exp(off):.3f}x")

    p_cal = np.exp(np.log(p[tst]) + off)
    comb = np.exp(0.5 * (np.log(p_cal) + np.log(pm[tst])))
    filas = []
    for lbl, pred in [("LLM crudo", p[tst]), ("LLM calibrado", p_cal),
                      ("modelo de texto", pm[tst]),
                      ("LLM calibrado + modelo", comb)]:
        mm = E.point_metrics(y[tst], pred)
        r = np.abs(np.log(y[cal]) - np.log(p[cal])
                   - (off if "calibrado" in lbl else 0.0))
        q = float(np.quantile(r, 0.80))
        filas.append(dict(metodo=lbl, MdMRE=round(mm["MdMRE"], 4),
                          PRED25=round(mm["PRED25"], 4),
                          PRED50=round(mm["PRED50"], 4),
                          MdAE_h=round(mm["MdAE_h"], 2),
                          banda80=round(float(np.exp(2 * q)), 2),
                          spearman=round(float(pd.Series(y[tst]).corr(
                              pd.Series(pred), method="spearman")), 3)))
        E.record("llm", lbl.replace(" ", "_"), "muestra50", mm)
    print(pd.DataFrame(filas).to_string(index=False))
    print("\n  referencia medida antes: la estimacion de un desarrollador")
    print("  humano sobre SiP da banda del 80% de 4,00x, muy por debajo de")
    print("  los 16x del LLM calibrado y del modelo de texto.")


if __name__ == "__main__":
    main()
