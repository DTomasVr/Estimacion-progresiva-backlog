"""E61 - En que proyectos falla el total, y si el motivo es el modelo o el dato.

El MdMRE del total es 0,355, pero eso es una mediana: hay proyectos que fallan por
factores de cuatro. Antes de dar la cifra por buena conviene saber si esos
proyectos tienen algo en comun, porque hay dos historias muy distintas:

  si fallan los proyectos con textos raros o poco parecidos al corpus, es un
  problema del MODELO y se puede trabajar

  si fallan los proyectos cuyas horas registradas no son creibles -- decenas de
  tareas a 0,05 h, o medias de menos de una hora por tarea en un equipo de
  software -- es un problema del DATO, y la cifra que se reporta esta castigada
  por proyectos que ningun modelo podria acertar porque el numero de referencia
  no significa lo que dice

E47 ya probo a PREDECIR el error con rasgos agregados y no se pudo (R2 -0,03).
Esto es distinto: no se trata de predecirlo sino de describirlo, que es lo que
hace falta para saber que decir en la defensa.
"""
import json
import os
import sys
import warnings
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "4"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/donde_falla")
S = "=" * 118
MIN_TAREAS = 8


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet("data/processed/agix_effort_v4.parquet")
    sw = df[df.is_software_task == 1].copy()
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    y = sw.effort_hours.to_numpy()
    est = np.clip(np.load("results/modelo_total/pred_total.npy"), 1e-6, None)
    E5 = np.load(CACHE / "e5-large.npy")
    Vn = E5 / np.maximum(np.linalg.norm(E5, axis=1, keepdims=True), 1e-9)
    d = pd.DataFrame({"p": sw.project.to_numpy(), "real": y, "est": est,
                      "fuente": sw.source.to_numpy(),
                      "nw": sw.title.fillna("").str.split().str.len().to_numpy(),
                      "desc": sw.description.fillna("").str.len().gt(0).to_numpy()})
    tam = d.groupby("p").size()
    d = d[d.p.isin(tam[tam >= MIN_TAREAS].index)].reset_index(drop=True)
    g = d.groupby("p")
    t = pd.DataFrame({
        "n": g.size(), "real": g.real.sum(), "est": g.est.sum(),
        "h_por_tarea": g.real.median(),
        "frac_micro": g.real.apply(lambda q: float((q <= .25).mean())),
        "frac_bajo1h": g.real.apply(lambda q: float((q < 1).mean())),
        "nw_media": g.nw.mean(), "frac_desc": g.desc.mean(),
        "fuente": g.fuente.agg(lambda q: q.mode().iloc[0]),
    })
    t["razon"] = t.est / t.real
    t["err_log"] = np.abs(np.log(t.razon))
    t["mre"] = np.abs(t.est - t.real) / t.real
    print(f"{len(t)} proyectos · MdMRE {t.mre.median():.4f}")

    print(f"\n{S}\nCORRELACION DE CADA RASGO CON EL ERROR DEL TOTAL (Spearman)\n{S}")
    co = {}
    for c in ("n", "h_por_tarea", "frac_micro", "frac_bajo1h", "nw_media", "frac_desc"):
        co[c] = float(t[c].corr(t.err_log, method="spearman"))
    for k, v in sorted(co.items(), key=lambda a: -abs(a[1])):
        print(f"  {k:14s} {v:+.3f}")
    print("  (positivo = ese rasgo va con MAS error)")

    print(f"\n{S}\nERROR SEGUN LO CREIBLE QUE SEA EL REGISTRO DE HORAS\n{S}")
    t["tramo"] = pd.cut(t.h_por_tarea, [0, .5, 1, 2, 4, 1e9],
                        labels=["<0,5 h", "0,5-1 h", "1-2 h", "2-4 h", ">4 h"])
    r = t.groupby("tramo", observed=True).apply(lambda q: pd.Series(dict(
        proyectos=len(q), h_por_tarea=q.h_por_tarea.median(),
        MdMRE=q.mre.median(), PRED25=float((q.mre <= .25).mean()),
        factor2=float((np.maximum(q.razon, 1 / q.razon) <= 2).mean()),
        sesgo_x=q.razon.median())))
    print(r.round(3).to_string())

    print(f"\n{S}\nY SI SE QUITAN LOS PROYECTOS CON REGISTRO POCO CREIBLE\n{S}")
    for u, etq in ((0, "todos"), (.5, "quitando los de menos de 0,5 h por tarea"),
                   (1.0, "quitando los de menos de 1 h por tarea")):
        k = t.h_por_tarea >= u
        print(f"  {etq:42s} n={int(k.sum()):3d} · MdMRE {t.mre[k].median():.4f} · "
              f"PRED25 {float((t.mre[k] <= .25).mean()):.4f} · "
              f"factor2 {float((np.maximum(t.razon, 1 / t.razon)[k] <= 2).mean()):.4f}")
    print("\n  esto NO es una mejora del modelo: es medir sobre un subconjunto. Solo")
    print("  sirve para saber cuanto de la cifra reportada se lo lleva el ruido del")
    print("  registro, que es lo que conviene tener claro antes de que lo pregunten")

    print(f"\n{S}\nLOS OCHO PROYECTOS QUE MAS FALLAN\n{S}")
    w = t.reindex(t.err_log.sort_values(ascending=False).index).head(8)
    print(w[["n", "real", "est", "h_por_tarea", "frac_micro", "razon",
             "fuente"]].round(3).to_string())
    t.to_csv(OUT / "proyectos.csv")
    json.dump(co, open(OUT / "correlaciones.json", "w"), indent=1)
    print(f"\nescrito {OUT}/proyectos.csv")


if __name__ == "__main__":
    main()


def con_anclas():
    """Si el problema es la escala del proyecto, una tarea cerrada deberia
    arreglarlo. Se repite la tabla por tramo con 1 y 3 tareas cerradas."""
    df = pd.read_parquet("data/processed/agix_effort_v4.parquet")
    sw = df[df.is_software_task == 1].copy()
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    y = sw.effort_hours.to_numpy()
    est = np.clip(np.load("results/modelo_total/pred_total.npy"), 1e-6, None)
    d = pd.DataFrame({"p": sw.project.to_numpy(), "real": y, "est": est})
    tam = d.groupby("p").size()
    d = d[d.p.isin(tam[tam >= MIN_TAREAS].index)].reset_index(drop=True)
    nom = list(pd.unique(d.p))
    idx = {p: np.flatnonzero(d.p.to_numpy() == p) for p in nom}
    re_, es_ = d.real.to_numpy(), d.est.to_numpy()
    le = np.log(es_)
    g = d.groupby("p")
    real = g.real.sum().reindex(nom).to_numpy()
    hpt = g.real.median().reindex(nom).to_numpy()
    rng = np.random.default_rng(0)
    tabla = {}
    for k in (0, 1, 3):
        ac = np.zeros(len(nom))
        S_ = 1 if k == 0 else 40
        for _ in range(S_):
            tt = []
            for p in nom:
                i = idx[p]
                if k == 0 or len(i) <= k:
                    tt.append(float(es_[i].sum()))
                    continue
                s = rng.choice(i, k, replace=False)
                off = np.log(re_[s].sum() / es_[s].sum()) * k / (k + 12)
                tt.append(float(np.exp(le[i] + off).sum()))
            ac += np.array(tt)
        tabla[k] = ac / S_
    t = pd.DataFrame({"real": real, "hpt": hpt}, index=nom)
    t["tramo"] = pd.cut(t.hpt, [0, .5, 1, 2, 4, 1e9],
                        labels=["<0,5 h", "0,5-1 h", "1-2 h", "2-4 h", ">4 h"])
    print(f"\n{S}\nLA MISMA TABLA, PERO CON TAREAS YA CERRADAS\n{S}")
    filas = []
    for tr, gg in t.groupby("tramo", observed=True):
        k_ = t.index.isin(gg.index)
        fila = dict(tramo=str(tr), proyectos=len(gg))
        for k in (0, 1, 3):
            r = np.abs(tabla[k][k_] - gg.real.to_numpy()) / gg.real.to_numpy()
            fila[f"MdMRE_{k}"] = float(np.median(r))
            fila[f"sesgo_{k}"] = float(np.median(tabla[k][k_] / gg.real.to_numpy()))
        filas.append(fila)
    e = pd.DataFrame(filas)
    print(e.round(3).to_string(index=False))
    print("  MdMRE_k y sesgo_k son con k tareas cerradas; k=0 es en frio")
    e.to_csv(OUT / "por_tramo_con_anclas.csv", index=False)


if __name__ == "__main__":
    con_anclas()
