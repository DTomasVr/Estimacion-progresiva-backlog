"""E67 - ¿Las tareas cerradas ARREGLAN la compresion de la escala, o solo la alivian?

Se sabe que anclar con tareas cerradas mejora el total (MdMRE 0,355 -> 0,292 con
cinco). Lo que NO se ha comprobado es lo que de verdad importa para decidir el
camino:

  1 ¿Arregla la COMPRESION? La escala predicha tiene dispersion 0,271 frente a
    0,791 real y pendiente 1,374. Si con tres o cinco tareas cerradas la
    dispersion sube a ~0,79 y la pendiente baja a ~1, el problema del grafico
    horizontal esta resuelto y no hace falta nada mas. Si se queda a medias, hay
    que seguir por el otro lado.

  2 ¿Es SIGNIFICATIVO? La mejora de E66 (modelo de escala aparte) no lo era:
    Wilcoxon p = 0,44 sobre 114 proyectos. Antes de construir nada sobre las
    tareas cerradas hay que contrastarlo igual, y ademas por sorteo, porque
    cuales son las tareas que se cierran primero es aleatorio.

  3 ¿CUANTAS hacen falta? Si con una ya se arregla, el producto no tiene que
    esperar a nada. Si hacen falta veinte, en un proyecto de treinta tareas llega
    tarde.

Protocolo, el mismo de E53: cociente de sumas, contraccion lambda 12, las tareas
cerradas se SABEN, y se promedian las METRICAS entre sorteos, no las
predicciones.
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "10"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import binomtest, wilcoxon  # noqa: E402

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
OUT = Path("results/anclas_escala")
S = "=" * 120
MIN_TAREAS, SORTEOS, LAMBDA = 8, 60, 12
KS = (1, 2, 3, 5, 10, 20)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
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
    g = d.groupby("p")
    real_tot = g.real.sum().reindex(nom).to_numpy()
    n_t = g.size().reindex(nom).to_numpy().astype(float)
    esc_r = np.log(g.real.mean().reindex(nom).to_numpy())     # escala real
    frio = g.est.sum().reindex(nom).to_numpy()
    P = len(nom)
    print(f"{P} proyectos · {len(d):,} tareas · {SORTEOS} sorteos · "
          f"{time.time() - t0:.0f} s", flush=True)

    def escala_y_total(k, rng):
        """Un sorteo: devuelve (total estimado, escala estimada) por proyecto."""
        tot = np.zeros(P)
        for j, p in enumerate(nom):
            i = idx[p]
            if len(i) <= k:
                tot[j] = es_[i].sum()
                continue
            sel = rng.choice(i, k, replace=False)
            resto = np.setdiff1d(i, sel)
            off = np.log(re_[sel].sum() / es_[sel].sum()) * k / (k + LAMBDA)
            tot[j] = re_[sel].sum() + (es_[resto] * np.exp(off)).sum()
        return tot, np.log(tot / n_t)

    def mide_escala(e):
        return dict(sd=float(e.std()),
                    err_log=float(np.median(np.abs(e - esc_r))),
                    spearman=float(pd.Series(e).corr(pd.Series(esc_r), method="spearman")),
                    pendiente=float(np.polyfit(e, esc_r, 1)[0]))

    def mide_total(t):
        r = np.abs(t - real_tot) / real_tot
        q = np.log(t / real_tot)
        m = float(np.median(q))
        return dict(MdMRE=float(np.median(r)), PRED25=float(np.mean(r <= .25)),
                    factor2=float(np.mean(np.maximum(t / real_tot, real_tot / t) <= 2)),
                    MAE_h=float(np.abs(t - real_tot).mean()),
                    SSPB_pc=float(np.sign(m) * (np.exp(abs(m)) - 1) * 100))

    esc0 = np.log(frio / n_t)
    err0 = np.abs(np.log(frio / real_tot))
    hit0 = np.abs(frio - real_tot) / real_tot <= .25
    filas = [dict(k=0, **mide_escala(esc0), **mide_total(frio),
                  p_wilcoxon=np.nan, p_mcnemar=np.nan, sorteos_signif=np.nan)]
    for k in KS:
        rng = np.random.default_rng(7000 + k)
        me, mt_, pw, pm = [], [], [], []
        for _ in range(SORTEOS):
            tot, e = escala_y_total(k, rng)
            me.append(mide_escala(e))
            mt_.append(mide_total(tot))
            err = np.abs(np.log(tot / real_tot))
            pw.append(float(wilcoxon(err, err0).pvalue))
            h = np.abs(tot - real_tot) / real_tot <= .25
            a, b = int((h & ~hit0).sum()), int((~h & hit0).sum())
            pm.append(binomtest(a, a + b, .5).pvalue if a + b else 1.)
        filas.append(dict(k=k,
                          **{c: float(np.mean([m[c] for m in me])) for c in me[0]},
                          **{c: float(np.mean([m[c] for m in mt_])) for c in mt_[0]},
                          p_wilcoxon=float(np.median(pw)),
                          p_mcnemar=float(np.median(pm)),
                          sorteos_signif=float(np.mean(np.array(pw) < .05))))
        print(f"  k={k} listo · {time.time() - t0:.0f} s", flush=True)
    t = pd.DataFrame(filas)
    print(f"\n{S}\n1 · ¿ARREGLA LA COMPRESION DE LA ESCALA?\n{S}")
    print(t[["k", "sd", "err_log", "spearman", "pendiente"]].round(4).to_string(index=False))
    print(f"  objetivo: sd -> {esc_r.std():.3f} (la real) y pendiente -> 1,0")
    print(f"  k=0 es en frio: sd {t.sd.iloc[0]:.3f}, pendiente {t.pendiente.iloc[0]:.3f}")

    print(f"\n{S}\n2 · ¿SE NOTA EN EL TOTAL, Y ES SIGNIFICATIVO?\n{S}")
    print(t[["k", "MdMRE", "PRED25", "factor2", "MAE_h", "SSPB_pc",
             "p_wilcoxon", "p_mcnemar", "sorteos_signif"]].round(4).to_string(index=False))
    print("  p_wilcoxon y p_mcnemar son la MEDIANA entre sorteos, contra el brazo en frio")
    print("  sorteos_signif es la fraccion de sorteos en que el Wilcoxon sale < 0,05")
    v = t[t.k > 0]
    prim = v[v.sorteos_signif >= .8]
    if len(prim):
        kk = int(prim.k.iloc[0])
        print(f"\n  a partir de {kk} tarea{'s' if kk > 1 else ''} cerrada"
              f"{'s' if kk > 1 else ''} la mejora es significativa en al menos el "
              f"80 % de los sorteos")
    else:
        print("\n  en ningun k la mejora sale significativa en el 80 % de los sorteos")
    t.to_csv(OUT / "resumen.csv", index=False)
    json.dump({"sorteos": SORTEOS, "lambda": LAMBDA, "sd_real": float(esc_r.std())},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/resumen.csv · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
