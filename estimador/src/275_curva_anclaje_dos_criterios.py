"""E-275 - La curva del anclaje, contando bien los proyectos pequenos.

La curva publicada (E67, results/anclas_escala) deja con la estimacion EN FRIO a los
proyectos que tienen k tareas o menos: un proyecto de 15 tareas «con 20 cerradas»
se evalua como si no se hubiera cerrado ninguna. Es pesimista sin motivo, porque si
todas sus tareas estan cerradas el total ya se conoce. Aqui se mide con dos
criterios que si responden a una pregunta real:

  todos      los 114 proyectos; el que ya cerro todas sus tareas tiene el total
             conocido (acierto seguro, y se dice)
  mayores    solo los proyectos con MAS de k tareas, donde las k cerradas son de
             verdad una parte del trabajo. Sin ayuda mecanica.

y ademas la regla «k tareas o el 20 % del backlog, lo que sea mayor», que es como
avanza un proyecto de verdad. Mismo anclaje (cociente de sumas, lambda 12), mismas
predicciones fuera de pliegue y mismas semillas que E67.
"""
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("results/curva_anclaje_dos_criterios")
SORTEOS, LAMBDA = 60, 12
KS = (1, 2, 3, 5, 10, 20)

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

    def sorteo(ks, rng):
        tot = np.zeros(len(nom))
        for j, p in enumerate(nom):
            i = idx[p]
            if ks[j] >= len(i):
                tot[j] = real[j]
                continue
            sel = rng.choice(i, ks[j], replace=False)
            resto = np.setdiff1d(i, sel)
            off = np.log(re_[sel].sum() / es_[sel].sum()) * ks[j] / (ks[j] + LAMBDA)
            tot[j] = re_[sel].sum() + np.exp(off) * es_[resto].sum()
        return tot

    def mide(t, m):
        r = np.abs(t[m] - real[m]) / real[m]
        return np.array([np.mean(r <= .25), np.mean(np.maximum(t[m] / real[m], real[m] / t[m]) <= 2), np.median(r)])

    todos = np.ones(len(nom), bool)
    filas = [dict(k=0, criterio="todos", proyectos=len(nom), **dict(zip(("PRED25", "factor2", "MdMRE"), mide(frio, todos))))]
    for k in KS:
        rng = np.random.default_rng(7000 + k)
        ks = np.full(len(nom), k)
        ms = [sorteo(ks, rng) for _ in range(SORTEOS)]
        g = n > k
        for criterio, m in (("todos", todos), ("mayores", g)):
            v = np.mean([mide(t, m) for t in ms], 0)
            filas.append(dict(k=k, criterio=criterio, proyectos=int(m.sum()), PRED25=v[0], factor2=v[1], MdMRE=v[2],
                              PRED25_en_frio=float(mide(frio, m)[0])))
        rng = np.random.default_rng(7500 + k)
        ks = np.maximum(k, np.round(.2 * n).astype(int))
        v = np.mean([mide(sorteo(ks, rng), todos) for _ in range(SORTEOS)], 0)
        filas.append(dict(k=k, criterio="k o el 20 % del backlog", proyectos=len(nom), PRED25=v[0], factor2=v[1], MdMRE=v[2]))
        print(f"  k={k} listo · {time.time() - t0:.0f} s", flush=True)
    t = pd.DataFrame(filas)
    t.to_csv(OUT / "resumen.csv", index=False)
    print(t.round(3).to_string(index=False))
    print(f"escrito {OUT} · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
