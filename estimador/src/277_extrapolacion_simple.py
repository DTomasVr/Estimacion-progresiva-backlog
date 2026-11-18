"""E-277 - Anchoring against a simple extrapolation.

The advisor's guide (change 13) first asked to state that anchoring had not been compared
with a simple extrapolation. Since version 4 of the paper the result is reported in RQ1,
with a project-level CI of the anchoring - extrapolation difference (guide V4, change 1).
Three rules on the same projects, draws and criteria as E-275
(results/curva_anclaje_dos_criterios):

  anchoring lambda=12     the deployed rule: ratio of sums shrunk with k/(k+12)
  no shrinkage            the same ratio without shrinkage (lambda = 0)
  simple extrapolation    closed hours + mean of the closed tasks x number of open
                          tasks; it does not use the model at all

Criteria: "all" (114 projects; one that has closed everything has a known total) and
"larger" (only projects with more than k tasks).

Outputs: results/extrapolacion_simple/resumen.csv and diferencia_ic.csv
"""
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("results/extrapolacion_simple")
DRAWS = 60
KS = (1, 3, 5, 10, 20)

s = importlib.util.spec_from_file_location("e264", "src/264_techo_del_total.py")
e264 = importlib.util.module_from_spec(s)
s.loader.exec_module(e264)

ANCHOR, NO_SHRINK, EXTRAP = "anchoring lambda=12", "no shrinkage (lambda=0)", "simple extrapolation"


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
    rules = (ANCHOR, NO_SHRINK, EXTRAP)

    def measure(t, m):
        r = np.abs(t[m] - real[m]) / real[m]
        return np.array([np.mean(r <= .25), np.median(r)])

    rows = [dict(k=0, rule=r, criterion=c, projects=int(m.sum()), PRED25=v[0], MdMRE=v[1])
            for r in rules for c, m in (("all", n > 0), ("larger", n > 0))
            for v in [measure(cold, m) if r != EXTRAP else np.array([np.nan, np.nan])]]
    ci = []
    for k in KS:
        rng = np.random.default_rng(7000 + k)
        acc = {(r, c): [] for r in rules for c in ("all", "larger")}
        hit = {r: np.zeros(len(names)) for r in rules}   # share of draws within +-25 %
        for _ in range(DRAWS):
            tot = {r: np.zeros(len(names)) for r in rules}
            for j, p in enumerate(names):
                i = idx[p]
                if k >= len(i):
                    for r in rules:
                        tot[r][j] = real[j]
                    continue
                sel = rng.choice(i, k, replace=False)
                rest = np.setdiff1d(i, sel)
                c, op = re_[sel].sum(), es_[rest].sum()
                lc = np.log(c / es_[sel].sum())
                tot[ANCHOR][j] = c + np.exp(lc * k / (k + 12)) * op
                tot[NO_SHRINK][j] = c + np.exp(lc) * op
                tot[EXTRAP][j] = c + re_[sel].mean() * len(rest)
            for r in rules:
                hit[r] += (np.abs(tot[r] - real) / real <= .25) / DRAWS
                for cname, m in (("all", np.ones(len(names), bool)), ("larger", n > k)):
                    acc[(r, cname)].append(measure(tot[r], m))
        for (r, cname), v in acc.items():
            v = np.mean(v, 0)
            rows.append(dict(k=k, rule=r, criterion=cname,
                             projects=int((n > k).sum() if cname == "larger" else len(names)), PRED25=v[0], MdMRE=v[1]))
        # project-level CI95 of the PRED(25) difference, larger projects only
        m = n > k
        diff = (hit[ANCHOR] - hit[EXTRAP])[m]
        b = np.random.default_rng(77).choice(diff, (10000, len(diff))).mean(1)
        lo, hi = np.quantile(b, [.025, .975])
        ci.append(dict(k=k, projects=int(m.sum()), diff_PRED25=float(diff.mean()), ci95_lo=float(lo),
                       ci95_hi=float(hi), projects_anchoring_better=int((diff > 0).sum()),
                       projects_extrapolation_better=int((diff < 0).sum())))
        print(f"  k={k} done · {time.time() - t0:.0f} s", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "resumen.csv", index=False)
    pd.DataFrame(ci).to_csv(OUT / "diferencia_ic.csv", index=False)
    print("\nanchoring - extrapolation, PRED25 on the larger projects, project-level CI95:")
    print(pd.DataFrame(ci).round(3).to_string(index=False))
    for cname in ("all", "larger"):
        x = t[(t.criterion == cname) & (t.k > 0)]
        print(f"\nPRED25 of the total · {cname}")
        print(x.pivot(index="k", columns="rule", values="PRED25").round(3).to_string())
        print(f"MdMRE of the total · {cname}")
        print(x.pivot(index="k", columns="rule", values="MdMRE").round(3).to_string())
    print(f"\nwritten {OUT} · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
