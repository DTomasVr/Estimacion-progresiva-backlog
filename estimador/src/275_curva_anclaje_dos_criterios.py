"""E-275 - The anchoring curve, counting small projects correctly.

The earlier curve (E67, results/anclas_escala) kept the COLD-START estimate for
projects with k tasks or fewer: a 15-task project "with 20 closed" was scored as if
nothing had been closed. That is needlessly pessimistic, because once every task is
closed the total is known. Here two criteria that answer a real question are used:

  all        the 114 projects; a project that has closed all its tasks has a known
             total (a guaranteed hit, and it is stated as such)
  larger     only projects with MORE than k tasks, where the k closed tasks really
             are a part of the work. No mechanical help.

plus the rule "k tasks or 20 % of the backlog, whichever is larger", which is how a
real project advances. Same anchoring (ratio of sums, lambda 12), same out-of-fold
predictions and same seeds as E67.

Output: results/curva_anclaje_dos_criterios/resumen.csv
"""
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("results/curva_anclaje_dos_criterios")
DRAWS, LAMBDA = 60, 12
KS = (1, 2, 3, 5, 10, 20)

s = importlib.util.spec_from_file_location("e264", "src/264_techo_del_total.py")
e264 = importlib.util.module_from_spec(s)
s.loader.exec_module(e264)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    d = e264.carga()
    names = list(pd.unique(d.p))
    idx = {p: np.flatnonzero(d.p.to_numpy() == p) for p in names}
    re_, es_ = d.real.to_numpy(), d.est.to_numpy()
    n = np.array([len(idx[p]) for p in names])
    real = np.array([re_[idx[p]].sum() for p in names])
    cold = np.array([es_[idx[p]].sum() for p in names])

    def draw(ks, rng):
        tot = np.zeros(len(names))
        for j, p in enumerate(names):
            i = idx[p]
            if ks[j] >= len(i):
                tot[j] = real[j]
                continue
            sel = rng.choice(i, ks[j], replace=False)
            rest = np.setdiff1d(i, sel)
            off = np.log(re_[sel].sum() / es_[sel].sum()) * ks[j] / (ks[j] + LAMBDA)
            tot[j] = re_[sel].sum() + np.exp(off) * es_[rest].sum()
        return tot

    def measure(t, m):
        r = np.abs(t[m] - real[m]) / real[m]
        return np.array([np.mean(r <= .25), np.mean(np.maximum(t[m] / real[m], real[m] / t[m]) <= 2), np.median(r)])

    every = np.ones(len(names), bool)
    rows = [dict(k=0, criterion="all", projects=len(names),
                 **dict(zip(("PRED25", "factor2", "MdMRE"), measure(cold, every))))]
    for k in KS:
        rng = np.random.default_rng(7000 + k)
        ks = np.full(len(names), k)
        ms = [draw(ks, rng) for _ in range(DRAWS)]
        g = n > k
        for criterion, m in (("all", every), ("larger", g)):
            v = np.mean([measure(t, m) for t in ms], 0)
            rows.append(dict(k=k, criterion=criterion, projects=int(m.sum()), PRED25=v[0], factor2=v[1], MdMRE=v[2],
                             PRED25_cold=float(measure(cold, m)[0])))
        rng = np.random.default_rng(7500 + k)
        ks = np.maximum(k, np.round(.2 * n).astype(int))
        v = np.mean([measure(draw(ks, rng), every) for _ in range(DRAWS)], 0)
        rows.append(dict(k=k, criterion="k or 20 % of the backlog", projects=len(names), PRED25=v[0], factor2=v[1],
                         MdMRE=v[2]))
        print(f"  k={k} done · {time.time() - t0:.0f} s", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "resumen.csv", index=False)
    print(t.round(3).to_string(index=False))
    print(f"written {OUT} · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
