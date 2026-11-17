"""E-265 - La curva del total por AVANCE, con el orden real de cierre.

E-264 midio la curva por fraccion cerrada sorteando que tareas se cierran. Aqui
se repite con el orden en que de verdad se cerraron (columna `orden`) y con
cortes por tiempo (2, 4 y 8 semanas desde el primer cierre), que es lo que ve un
equipo: «tras el primer sprint».

Solo entran las tareas con fecha y los proyectos que conservan 8 o mas. Se
supone que el backlog entero se conoce desde el principio, que es como trabaja
PBE (el backlog se genera al arrancar); en estos proyectos reales parte de las
tareas se crearon despues, asi que la curva es algo optimista en ese punto.
Anclaje: el desplegado (cociente de sumas, lambda 12).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("results/avance_orden_real")
MIN_TAREAS, SORTEOS, LAMBDA = 8, 60, 12
FRACS = (.05, .10, .20, .30, .40, .50, .70)
SEMANAS = (2, 4, 8, 12)
ETIQ = ["8-15", "16-40", "41-120", "121-400", "mas de 400"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet("data/processed/pbe_effort_v4.parquet")
    sw = df[df.is_software_task == 1].copy()
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    est = np.clip(np.load("results/modelo_total/pred_total.npy"), 1e-6, None)
    d = pd.DataFrame({"p": sw.project.to_numpy(), "real": sw.effort_hours.to_numpy(),
                      "est": est, "orden": sw.orden.to_numpy()})
    todas = len(d)
    d = d[d.orden.notna()]
    tam = d.groupby("p").size()
    d = d[d.p.isin(tam[tam >= MIN_TAREAS].index)].sort_values(["p", "orden"]).reset_index(drop=True)
    nom = list(pd.unique(d.p))
    P = len(nom)
    G = [g for _, g in d.groupby("p", sort=False)]
    re_ = [g.real.to_numpy() for g in G]
    es_ = [g.est.to_numpy() for g in G]
    dias = [(g.orden - g.orden.iloc[0]).dt.total_seconds().to_numpy() / 86400 for g in G]
    n = np.array([len(x) for x in re_])
    real = np.array([x.sum() for x in re_])
    frio = np.array([x.sum() for x in es_])
    tramo = np.asarray(pd.cut(n, [0, 15, 40, 120, 400, 10 ** 6], labels=ETIQ).astype(str))
    print(f"{P} proyectos · {len(d):,} tareas con fecha de {todas:,} · {time.time() - t0:.0f} s", flush=True)

    def total(j, sel):
        k = len(sel)
        if k == 0:
            return frio[j]
        m = np.ones(n[j], bool)
        m[sel] = False
        off = np.log(re_[j][sel].sum() / es_[j][sel].sum()) * k / (k + LAMBDA)
        return re_[j][sel].sum() + np.exp(off) * es_[j][m].sum()

    def fila(nombre, tot, k_med):
        ok = np.abs(tot - real) / real <= .25
        f = dict(corte=nombre, tareas_cerradas_mediana=float(k_med), PRED25=float(ok.mean()),
                 MdMRE=float(np.median(np.abs(tot - real) / real)),
                 factor2=float(np.mean(np.maximum(tot / real, real / tot) <= 2)))
        for e in ETIQ:
            f[e] = float(ok[tramo == e].mean())
        return f

    filas = [fila("en frio", frio, 0)]
    rng = np.random.default_rng(265)
    for f in FRACS:
        ks = np.minimum(np.maximum(1, np.round(f * n).astype(int)), n - 1)
        filas.append(fila(f"orden real · {int(f * 100)} % cerrado",
                          np.array([total(j, np.arange(ks[j])) for j in range(P)]), np.median(ks)))
        az = [fila("", np.array([total(j, rng.choice(n[j], ks[j], replace=False)) for j in range(P)]),
                   np.median(ks)) for _ in range(SORTEOS)]
        m = pd.DataFrame(az).drop(columns="corte").mean().to_dict()
        filas.append(dict(corte=f"al azar · {int(f * 100)} % cerrado", **m))
        print(f"  {int(f * 100)} % listo · {time.time() - t0:.0f} s", flush=True)
    for s in SEMANAS:
        ks = np.array([min(int((dias[j] <= 7 * s).sum()), n[j] - 1) for j in range(P)])
        f = fila(f"orden real · {s} semanas", np.array([total(j, np.arange(ks[j])) for j in range(P)]),
                 np.median(ks))
        f["fraccion_cerrada_mediana"] = float(np.median(ks / n))
        filas.append(f)
    t = pd.DataFrame(filas)
    t.insert(1, "proyectos", P)
    t.to_csv(OUT / "resumen.csv", index=False)
    pd.Series(tramo).value_counts().reindex(ETIQ).to_csv(OUT / "proyectos_por_tramo.csv")
    print(t.round(3).to_string(index=False))
    print("proyectos por tramo:", pd.Series(tramo).value_counts().reindex(ETIQ).to_dict())
    print(f"escrito {OUT} · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
