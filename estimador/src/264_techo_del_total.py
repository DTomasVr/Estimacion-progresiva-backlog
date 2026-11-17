"""E-264 - ¿Hasta donde puede llegar el +-25 % del total, y por que camino?

El total esta en 36,8 % en frio y 53,3 % con 20 tareas cerradas, y las vias de
mejorar la prediccion por tarea estan cerradas. Antes de lanzar mas experimentos
se mide donde esta el techo y que lo fija:

  1 TECHO POR RUIDO. Aunque se supiera la escala exacta del proyecto, el total es
    una suma de horas muy asimetricas: con pocas tareas, la suma no se concentra.
    Se remuestrea cada proyecto con su escala REAL y se mide cuantas veces el
    total cae dentro de +-25 %. Es el techo de cualquier anclaje.

  2 CURVA POR AVANCE. La curva publicada fija k (1..20 tareas), pero 20 tareas
    son el 5 % de un proyecto de 400 y el 100 % de uno de 20. Se mide por
    fraccion del backlog cerrada.

  3 EL ESTIMADOR DEL ANCLA. Cociente de sumas contraido (lo desplegado) frente a
    otras lambdas, a la media de log-cocientes y a mezclar con horas por tarea.

Mismo protocolo que E67: predicciones fuera de pliegue de la cabeza del total,
proyectos con 8 tareas o mas, se promedian las METRICAS entre sorteos.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("results/techo_del_total")
MIN_TAREAS, SORTEOS, LAMBDA = 8, 60, 12
FRACS = (.05, .10, .20, .30, .40, .50, .70)
KS = (5, 10, 20)
S = "=" * 110


def carga():
    df = pd.read_parquet("data/processed/pbe_effort_v4.parquet")
    sw = df[df.is_software_task == 1].copy()
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    est = np.clip(np.load("results/modelo_total/pred_total.npy"), 1e-6, None)
    d = pd.DataFrame({"p": sw.project.to_numpy(), "real": sw.effort_hours.to_numpy(), "est": est})
    tam = d.groupby("p").size()
    return d[d.p.isin(tam[tam >= MIN_TAREAS].index)].reset_index(drop=True)


def dentro(t, real):
    return np.abs(t - real) / real <= .25


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    d = carga()
    nom = list(pd.unique(d.p))
    idx = {p: np.flatnonzero(d.p.to_numpy() == p) for p in nom}
    re_, es_ = d.real.to_numpy(), d.est.to_numpy()
    n_t = np.array([len(idx[p]) for p in nom])
    real_tot = np.array([re_[idx[p]].sum() for p in nom])
    frio = np.array([es_[idx[p]].sum() for p in nom])
    P = len(nom)
    tramo = pd.cut(n_t, [0, 15, 40, 120, 400, 10 ** 6],
                   labels=["8-15", "16-40", "41-120", "121-400", "mas de 400"])
    print(f"{P} proyectos · {len(d):,} tareas · en frio {dentro(frio, real_tot).mean():.3f}", flush=True)

    # 1 · techo por ruido: escala real conocida, se remuestrea el proyecto
    rng = np.random.default_rng(264)
    techo_mod, techo_cte = np.zeros(P), np.zeros(P)
    for j, p in enumerate(nom):
        i = idx[p]
        c = re_[i].sum() / es_[i].sum()
        m = re_[i].mean()
        b = rng.choice(i, (2000, len(i)), replace=True)
        tot = re_[b].sum(1)
        techo_mod[j] = dentro(c * es_[b].sum(1), tot).mean()
        techo_cte[j] = dentro(np.full(2000, m * len(i)), tot).mean()
    t1 = (pd.DataFrame({"tramo": tramo, "frio": dentro(frio, real_tot),
                        "techo_escala_exacta": techo_mod, "techo_solo_media": techo_cte})
          .groupby("tramo", observed=True).agg(proyectos=("frio", "size"), frio=("frio", "mean"),
                                               techo_escala_exacta=("techo_escala_exacta", "mean"),
                                               techo_solo_media=("techo_solo_media", "mean")).reset_index())
    t1.loc[len(t1)] = ["todos", P, dentro(frio, real_tot).mean(), techo_mod.mean(), techo_cte.mean()]
    t1.to_csv(OUT / "techo_ruido.csv", index=False)
    print(f"\n{S}\n1 · TECHO POR RUIDO (escala del proyecto conocida exactamente)\n{S}")
    print(t1.round(3).to_string(index=False), flush=True)

    # anclas: devuelven el factor que multiplica a lo abierto
    sesgo_log = float(np.mean([np.mean(np.log(re_[idx[p]] / es_[idx[p]]))
                               - np.log(re_[idx[p]].sum() / es_[idx[p]].sum()) for p in nom]))

    def ancla(sel, modo):
        k = len(sel)
        if modo.startswith("sumas"):
            lam = float(modo.split("=")[1])
            return np.exp(np.log(re_[sel].sum() / es_[sel].sum()) * k / (k + lam))
        if modo == "logs":
            return np.exp((np.mean(np.log(re_[sel] / es_[sel])) - sesgo_log) * k / (k + LAMBDA))
        if modo == "recortada":
            q = re_[sel] / es_[sel]
            tope = np.quantile(q, .9)
            w = np.minimum(q, tope) * es_[sel]
            return np.exp(np.log(w.sum() / es_[sel].sum() * 1.0) * k / (k + LAMBDA))
        raise ValueError(modo)

    MODOS = ["sumas=12", "sumas=4", "sumas=30", "logs", "recortada"]

    def corre(cuantas, etiqueta):
        filas = []
        rng = np.random.default_rng(9000)
        acc = {m: [] for m in MODOS}
        acc_ab = []
        por_tramo = []
        for _ in range(SORTEOS):
            tot = {m: np.zeros(P) for m in MODOS}
            ab_ok = np.full(P, np.nan)
            for j, p in enumerate(nom):
                i = idx[p]
                k = cuantas(len(i))
                if k <= 0 or k >= len(i):
                    for m in MODOS:
                        tot[m][j] = es_[i].sum() if k <= 0 else re_[i].sum()
                    continue
                sel = rng.choice(i, k, replace=False)
                resto = np.setdiff1d(i, sel)
                for m in MODOS:
                    tot[m][j] = re_[sel].sum() + ancla(sel, m) * es_[resto].sum()
                ab = ancla(sel, "sumas=12") * es_[resto].sum()
                ab_ok[j] = abs(ab - re_[resto].sum()) / re_[resto].sum() <= .25
            for m in MODOS:
                acc[m].append(dentro(tot[m], real_tot).mean())
            acc_ab.append(np.nanmean(ab_ok))
            por_tramo.append(pd.Series(dentro(tot["sumas=12"], real_tot)).groupby(tramo, observed=True).mean())
        fila = dict(corte=etiqueta, **{f"PRED25 {m}": float(np.mean(acc[m])) for m in MODOS},
                    PRED25_solo_lo_abierto=float(np.mean(acc_ab)))
        fila.update({f"tramo {c}": float(v) for c, v in pd.concat(por_tramo, axis=1).mean(1).items()})
        return fila

    filas = [corre(lambda n, k=k: k, f"k={k}") for k in KS]
    print(f"  cortes por k listos · {time.time() - t0:.0f} s", flush=True)
    filas += [corre(lambda n, f=f: int(round(f * n)), f"{int(f * 100)} % cerrado") for f in FRACS]
    t2 = pd.DataFrame(filas)
    t2.to_csv(OUT / "curva_avance.csv", index=False)
    print(f"\n{S}\n2 y 3 · CURVA POR AVANCE Y ESTIMADOR DEL ANCLA\n{S}")
    print(t2.round(3).to_string(index=False))
    json.dump({"proyectos": P, "sorteos": SORTEOS, "sesgo_log": sesgo_log,
               "segundos": round(time.time() - t0)}, open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT} · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
