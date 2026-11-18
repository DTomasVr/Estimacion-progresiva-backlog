"""E-277 - El anclaje frente a una extrapolación simple, solo para saberlo.

La guía de la asesora (cambio 13) pedía declarar que el anclaje no se comparó con una
extrapolación simple. Desde la V4 del paper el resultado entra en PI1, con el IC por
proyectos de la diferencia anclaje - extrapolación (guía V4, cambio 1). Tres reglas sobre los mismos proyectos, sorteos y criterio
que E-275 (results/curva_anclaje_dos_criterios):

  anclaje lambda=12    lo desplegado: cociente de sumas contraído con k/(k+12)
  sin contracción      el mismo cociente sin contraer (lambda = 0)
  extrapolación        horas cerradas + media de las cerradas x número de abiertas;
                       no usa el modelo para nada

Criterios: «todos» (114 proyectos; el que cerró todo tiene el total conocido) y
«mayores» (solo los proyectos con más de k tareas).
"""
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("results/extrapolacion_simple")
SORTEOS = 60
KS = (1, 3, 5, 10, 20)

s = importlib.util.spec_from_file_location("e264", "src/264_techo_del_total.py")
e264 = importlib.util.module_from_spec(s)
s.loader.exec_module(e264)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    d = e264.carga()
    nom = list(pd.unique(d.p))
    idx = {p: np.flatnonzero(d.p.to_numpy() == p) for p in nom}
    re_, es_ = d.real.to_numpy(), d.est.to_numpy()
    n = np.array([len(idx[p]) for p in nom])
    real = np.array([re_[idx[p]].sum() for p in nom])
    frio = np.array([es_[idx[p]].sum() for p in nom])
    reglas = ("anclaje lambda=12", "sin contraccion (lambda=0)", "extrapolacion simple")

    def mide(t, m):
        r = np.abs(t[m] - real[m]) / real[m]
        return np.array([np.mean(r <= .25), np.median(r)])

    filas = [dict(k=0, regla=r, criterio=c, proyectos=int(m.sum()), PRED25=v[0], MdMRE=v[1])
             for r in reglas for c, m in (("todos", n > 0), ("mayores", n > 0))
             for v in [mide(frio, m) if r != "extrapolacion simple" else np.array([np.nan, np.nan])]]
    ic = []
    for k in KS:
        rng = np.random.default_rng(7000 + k)
        acc = {(r, c): [] for r in reglas for c in ("todos", "mayores")}
        acierto = {r: np.zeros(len(nom)) for r in reglas}   # fraccion de sorteos dentro de +-25 %
        for _ in range(SORTEOS):
            tot = {r: np.zeros(len(nom)) for r in reglas}
            for j, p in enumerate(nom):
                i = idx[p]
                if k >= len(i):
                    for r in reglas:
                        tot[r][j] = real[j]
                    continue
                sel = rng.choice(i, k, replace=False)
                resto = np.setdiff1d(i, sel)
                c, ab = re_[sel].sum(), es_[resto].sum()
                lc = np.log(c / es_[sel].sum())
                tot["anclaje lambda=12"][j] = c + np.exp(lc * k / (k + 12)) * ab
                tot["sin contraccion (lambda=0)"][j] = c + np.exp(lc) * ab
                tot["extrapolacion simple"][j] = c + re_[sel].mean() * len(resto)
            for r in reglas:
                acierto[r] += (np.abs(tot[r] - real) / real <= .25) / SORTEOS
                for cnom, m in (("todos", np.ones(len(nom), bool)), ("mayores", n > k)):
                    acc[(r, cnom)].append(mide(tot[r], m))
        for (r, cnom), v in acc.items():
            v = np.mean(v, 0)
            filas.append(dict(k=k, regla=r, criterio=cnom, proyectos=int((n > k).sum() if cnom == "mayores" else len(nom)),
                              PRED25=v[0], MdMRE=v[1]))
        # IC95 por proyectos de la diferencia de PRED(25), solo en los mayores
        m = n > k
        dif = (acierto["anclaje lambda=12"] - acierto["extrapolacion simple"])[m]
        b = np.random.default_rng(77).choice(dif, (10000, len(dif))).mean(1)
        lo, hi = np.quantile(b, [.025, .975])
        ic.append(dict(k=k, proyectos=int(m.sum()), dif_PRED25=float(dif.mean()), ic95_lo=float(lo),
                       ic95_hi=float(hi), proyectos_mejor_anclaje=int((dif > 0).sum()),
                       proyectos_mejor_extrapolacion=int((dif < 0).sum())))
        print(f"  k={k} listo · {time.time() - t0:.0f} s", flush=True)
    t = pd.DataFrame(filas)
    t.to_csv(OUT / "resumen.csv", index=False)
    pd.DataFrame(ic).to_csv(OUT / "diferencia_ic.csv", index=False)
    print("\nanclaje - extrapolacion, PRED25 en los mayores, IC95 por proyectos:")
    print(pd.DataFrame(ic).round(3).to_string(index=False))
    for cnom in ("todos", "mayores"):
        x = t[(t.criterio == cnom) & (t.k > 0)]
        print(f"\nPRED25 del total · {cnom}")
        print(x.pivot(index="k", columns="regla", values="PRED25").round(3).to_string())
        print(f"MdMRE del total · {cnom}")
        print(x.pivot(index="k", columns="regla", values="MdMRE").round(3).to_string())
    print(f"\nescrito {OUT} · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
