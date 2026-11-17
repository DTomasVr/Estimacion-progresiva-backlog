"""E70 - ¿Y si las tareas que se cierran primero NO son una muestra al azar?

Toda la curva de anclaje supone que las k tareas ya cerradas son un sorteo
aleatorio de las del proyecto. En un equipo real eso es dudoso: se suele empezar
por lo facil, y si las primeras en cerrarse son sistematicamente las mas cortas,
el desplazamiento estimado sale bajo y el total se queda corto. Seria un sesgo
que la evaluacion no ve y el producto si sufriria.

No hace falta suponer nada, porque el dataset guarda la fecha de cierre: 82 de
los 114 proyectos la tienen en al menos el 80 % de sus tareas. Asi que se puede
usar el ORDEN REAL en que se cerraron.

  a  al azar                        lo que supone la curva publicada
  b  las mas cortas de verdad       el peor caso: el equipo despacha lo facil
  c  las que el modelo cree cortas  mas realista que b, porque el equipo elige
                                    por tamano PERCIBIDO, no por el real
  d  el orden real de cierre        la verdad, sin suponer nada
  e  las mas largas primero         el otro extremo, para acotar

Todo sobre los mismos 82 proyectos, para que la comparacion sea limpia. Lo que
mas importa aqui no es el MdMRE sino el SESGO: si b o d lo disparan, la curva
publicada es optimista y hay que decirlo.
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "8"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 215)
OUT = Path("results/orden_cierre")
S = "=" * 118
MIN_TAREAS, SORTEOS, LAMBDA, COB = 8, 60, 12, .8
KS = (1, 2, 3, 5, 10)


def mt(real, pred):
    real = np.asarray(real, float)
    pred = np.clip(np.asarray(pred, float), 1e-6, None)
    r = np.abs(pred - real) / real
    q = np.log(pred / real)
    m = float(np.median(q))
    lr = np.log(real)
    return dict(MAE_h=float(np.abs(pred - real).mean()),
                MdMRE=float(np.median(r)),
                R2_log=float(1 - ((lr - np.log(pred)) ** 2).mean() / lr.var()),
                PRED25=float(np.mean(r <= .25)), PRED50=float(np.mean(r <= .50)),
                factor2=float(np.mean(np.maximum(pred / real, real / pred) <= 2)),
                SSPB_pc=float(np.sign(m) * (np.exp(abs(m)) - 1) * 100))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet("data/processed/pbe_effort_v4.parquet")
    sw = df[df.is_software_task == 1].copy()
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    est = np.clip(np.load("results/modelo_total/pred_total.npy"), 1e-6, None)
    d = pd.DataFrame({"p": sw.project.to_numpy(), "real": sw.effort_hours.to_numpy(),
                      "est": est, "cierre": sw.completed.to_numpy()})
    tam = d.groupby("p").size()
    d = d[d.p.isin(tam[tam >= MIN_TAREAS].index)]
    cob = d.groupby("p").cierre.apply(lambda q: q.notna().mean())
    val = cob[cob >= COB].index
    d = d[d.p.isin(val)].reset_index(drop=True)
    nom = list(pd.unique(d.p))
    idx = {p: np.flatnonzero(d.p.to_numpy() == p) for p in nom}
    re_, es_ = d.real.to_numpy(), d.est.to_numpy()
    le = np.log(es_)
    g = d.groupby("p")
    real_tot = g.real.sum().reindex(nom).to_numpy()
    n_t = g.size().reindex(nom).to_numpy().astype(float)
    frio = g.est.sum().reindex(nom).to_numpy()
    print(f"{len(nom)} proyectos con fecha de cierre en al menos el {COB:.0%} de sus "
          f"tareas · {len(d):,} tareas · {time.time() - t0:.0f} s", flush=True)

    # orden de cierre real, con las que no la tienen al final
    ordr = {}
    for p in nom:
        i = idx[p]
        c = d.cierre.to_numpy()[i]
        clave = np.where(pd.isna(c), np.datetime64("2999-01-01"), c)
        ordr[p] = i[np.argsort(clave, kind="stable")]
    cortas = {p: idx[p][np.argsort(re_[idx[p]], kind="stable")] for p in nom}
    cortas_m = {p: idx[p][np.argsort(es_[idx[p]], kind="stable")] for p in nom}
    largas = {p: idx[p][np.argsort(-re_[idx[p]], kind="stable")] for p in nom}

    def total(k, elegir, rng=None):
        t = np.zeros(len(nom))
        for j, p in enumerate(nom):
            i = idx[p]
            if len(i) <= k:
                t[j] = es_[i].sum()
                continue
            sel = elegir(p, k, rng)
            resto = np.setdiff1d(i, sel)
            off = np.log(re_[sel].sum() / es_[sel].sum()) * k / (k + LAMBDA)
            t[j] = re_[sel].sum() + (es_[resto] * np.exp(off)).sum()
        return t

    reglas = {
        "a · al azar": lambda p, k, r: r.choice(idx[p], k, replace=False),
        "b · las mas cortas de verdad": lambda p, k, r: cortas[p][:k],
        "c · las que el modelo cree cortas": lambda p, k, r: cortas_m[p][:k],
        "d · el orden real de cierre": lambda p, k, r: ordr[p][:k],
        "e · las mas largas primero": lambda p, k, r: largas[p][:k],
    }
    filas = [dict(k=0, regla="en frio", **mt(real_tot, frio))]
    for k in KS:
        for etq, f in reglas.items():
            if etq.startswith("a"):
                rng = np.random.default_rng(9000 + k)
                ms = [mt(real_tot, total(k, f, rng)) for _ in range(SORTEOS)]
                filas.append(dict(k=k, regla=etq,
                                  **{c: float(np.mean([m[c] for m in ms])) for c in ms[0]}))
            else:
                filas.append(dict(k=k, regla=etq, **mt(real_tot, total(k, f))))
        print(f"  k={k} listo · {time.time() - t0:.0f} s", flush=True)
    t = pd.DataFrame(filas)
    print(f"\n{S}\nSEGUN QUE TAREAS SE CIERREN PRIMERO ({len(nom)} proyectos)\n{S}")
    for c in ("SSPB_pc", "MdMRE", "PRED25"):
        print(f"\n{c}:")
        print(t[t.k > 0].pivot(index="regla", columns="k", values=c).round(4).to_string())
    print(f"\n  en frio: SSPB {t.SSPB_pc.iloc[0]:+.2f} % · MdMRE {t.MdMRE.iloc[0]:.4f} · "
          f"PRED25 {t.PRED25.iloc[0]:.4f}")

    print(f"\n{S}\nLECTURA\n{S}")
    az = t[(t.k == 3) & (t.regla.str.startswith("a"))].iloc[0]
    rl = t[(t.k == 3) & (t.regla.str.startswith("d"))].iloc[0]
    co = t[(t.k == 3) & (t.regla.str.startswith("b"))].iloc[0]
    print(f"  con 3 tareas cerradas:")
    print(f"    al azar          SSPB {az.SSPB_pc:+6.2f} % · MdMRE {az.MdMRE:.4f} · "
          f"PRED25 {az.PRED25:.4f}")
    print(f"    orden REAL       SSPB {rl.SSPB_pc:+6.2f} % · MdMRE {rl.MdMRE:.4f} · "
          f"PRED25 {rl.PRED25:.4f}")
    print(f"    las mas cortas   SSPB {co.SSPB_pc:+6.2f} % · MdMRE {co.MdMRE:.4f} · "
          f"PRED25 {co.PRED25:.4f}")
    dif = abs(rl.SSPB_pc - az.SSPB_pc)
    print(f"\n  la diferencia de sesgo entre el orden real y el azar es de "
          f"{dif:.1f} puntos")
    print("  " + ("la curva publicada NO es optimista por este motivo"
                  if dif < 10 else
                  "OJO: el orden real mete un sesgo que la curva publicada no ve"))
    t.to_csv(OUT / "resumen.csv", index=False)
    json.dump({"proyectos": len(nom), "cobertura_min": COB, "sorteos": SORTEOS},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/resumen.csv · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
