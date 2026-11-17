"""E-278 - Entradas del usuario CON el orden real de cierre.

E-265 midio el orden real sin entradas del usuario (38 proyectos: 50,0 % con el
20 % cerrado frente a 71,3 % al azar). E-268 midio las estimaciones del usuario
con cierre al azar (41 proyectos). Aqui se juntan: proyectos con fecha de cierre
y con estimacion registrada en al menos el 80 % de sus tareas.

Configuracion desplegada: sin estimaciones, lambda_c = 12; con estimaciones,
lambda_u = 12 y lambda'_c = 30. El usuario estima todas las tareas que tienen
estimacion registrada. Salida: results/orden_real_con_estimaciones/resumen.csv
"""
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("results/orden_real_con_estimaciones")
MIN_TAREAS, COBERTURA, SORTEOS = 8, .80, 60
FRACS = (0, .10, .20, .30, .50)
LAM_U, LAM_C_SOLO, LAM_C_EST = 12, 12, 30


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet("data/processed/pbe_effort_v4.parquet")
    sw = df[df.is_software_task == 1].copy()
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    d = pd.DataFrame({"p": sw.project.to_numpy(), "real": sw.effort_hours.to_numpy(),
                      "est": np.clip(np.load("results/modelo_total/pred_total.npy"), 1e-6, None),
                      "usu": sw.HoursEstimate.where(sw.HoursEstimate > 0).to_numpy(),
                      "orden": sw.orden.to_numpy()})
    d = d[d.orden.notna()]
    tam = d.groupby("p").size()
    d = d[d.p.isin(tam[tam >= MIN_TAREAS].index)].sort_values(["p", "orden"]).reset_index(drop=True)
    G = [g for _, g in d.groupby("p", sort=False)]
    G = [g for g in G if g.usu.notna().mean() >= COBERTURA]
    P = len(G)
    re_ = [g.real.to_numpy() for g in G]
    es_ = [g.est.to_numpy() for g in G]
    us_ = [g.usu.to_numpy() for g in G]
    con = [np.flatnonzero(~np.isnan(x)) for x in us_]
    n = np.array([len(x) for x in re_])
    real = np.array([x.sum() for x in re_])
    print(f"{P} proyectos con orden real y estimaciones · tareas: mediana {int(np.median(n))}", flush=True)

    def total(j, estima, cer):
        base = es_[j].copy()
        lc = LAM_C_SOLO
        if estima and len(con[j]):
            ei, m = con[j], len(con[j])
            base = base * np.exp(np.log(us_[j][ei].sum() / es_[j][ei].sum()) * m / (m + LAM_U))
            base[ei] = us_[j][ei]
            lc = LAM_C_EST
        k = len(cer)
        if not k:
            return base.sum()
        ab = np.ones(n[j], bool)
        ab[cer] = False
        fc = np.exp(np.log(re_[j][cer].sum() / base[cer].sum()) * k / (k + lc))
        return re_[j][cer].sum() + fc * base[ab].sum()

    def pred(t):
        return float(np.mean(np.abs(t - real) / real <= .25))

    rng = np.random.default_rng(278)
    filas = []
    for estima in (False, True):
        for f in FRACS:
            ks = np.minimum(np.round(f * n).astype(int), n - 1)
            if f:
                ks = np.maximum(ks, 1)
            orden = pred(np.array([total(j, estima, np.arange(ks[j])) for j in range(P)]))
            azar = np.mean([pred(np.array([total(j, estima, rng.choice(n[j], ks[j], replace=False))
                                           for j in range(P)])) for _ in range(SORTEOS)])
            filas.append(dict(usuario_estima="todo" if estima else "nada", fraccion_cerrada=f,
                              proyectos=P, PRED25_orden_real=orden, PRED25_azar=float(azar)))
    t = pd.DataFrame(filas)
    t.to_csv(OUT / "resumen.csv", index=False)
    print(t.round(3).to_string(index=False))
    print(f"un proyecto = {100 / P:.1f} puntos · escrito {OUT} · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
