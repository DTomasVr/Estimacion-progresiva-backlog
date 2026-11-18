"""E-268 - The full rule: each task is CLOSED, ESTIMATED by the user or ALONE.

E-267 measured separately giving real hours and giving estimates. In the product they
coexist: some tasks are already closed (real hours), others have the user's estimate
and the rest only have the model. The rule to be implemented:

  base of a task   = the user's estimate if there is one; otherwise model x f_u
  f_u              = (user sum / model sum, on the estimated tasks) ^ (m / (m + lambda_u))
  f_c              = (real sum / base sum, on the closed tasks)     ^ (k / (k + lambda_c))
  total            = real sum of the closed tasks + f_c x base sum of the open ones

With m = 0 and k = 0 it is the cold start; with a single input only that one weighs.
It is measured on the projects that carry a real user estimate (80 % of their tasks or
more), for every combination of how much the user estimates and how many tasks are
closed.

Output: results/tres_estados/resumen.csv
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("results/tres_estados")
DRAWS = 60
ESTIMATED = (0, .25, .50, 1.0)
CLOSED = (0, 3, 5, 10, 20)
LAM_U = (12, 30)
LAM_C = (12, 30, 96)

s = importlib.util.spec_from_file_location("e267", "src/267_entrada_del_usuario.py")
e267 = importlib.util.module_from_spec(s)
s.loader.exec_module(e267)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    d = e267.carga()
    G = [g for _, g in d.groupby("p", sort=False)]
    G = [g for g in G if g.usu.notna().mean() >= e267.COBERTURA]
    P = len(G)
    re_ = [g.real.to_numpy() for g in G]
    es_ = [g.est.to_numpy() for g in G]
    us_ = [g.usu.to_numpy() for g in G]
    has = [np.flatnonzero(~np.isnan(x)) for x in us_]
    n = np.array([len(x) for x in re_])
    real = np.array([x.sum() for x in re_])
    clean = np.array([g.copiada.mean() < .25 for g in G])
    print(f"{P} projects with a user estimate · {int(clean.sum())} with few copied rows · "
          f"tasks per project: median {int(np.median(n))}", flush=True)

    def total(j, est_idx, closed_idx, lu, lc):
        base = es_[j].copy()
        m = len(est_idx)
        if m:
            fu = np.exp(np.log(us_[j][est_idx].sum() / es_[j][est_idx].sum()) * m / (m + lu))
            base = base * fu
            base[est_idx] = us_[j][est_idx]
        k = len(closed_idx)
        if not k:
            return base.sum()
        op = np.ones(n[j], bool)
        op[closed_idx] = False
        fc = np.exp(np.log(re_[j][closed_idx].sum() / base[closed_idx].sum()) * k / (k + lc))
        return re_[j][closed_idx].sum() + fc * base[op].sum()

    rows = []
    for fe in ESTIMATED:
        for k in CLOSED:
            rng = np.random.default_rng(26800 + int(fe * 100) * 100 + k)
            acc = {}
            for _ in range(DRAWS):
                sel = []
                for j in range(P):
                    m = min(int(round(fe * n[j])), len(has[j]))
                    ei = rng.choice(has[j], m, replace=False) if m else np.array([], int)
                    kk = min(k, n[j] - 1)
                    ci = rng.choice(n[j], kk, replace=False) if kk else np.array([], int)
                    sel.append((ei, ci))
                for lu in LAM_U:
                    for lc in LAM_C:
                        if (fe == 0 and lu != LAM_U[0]) or (k == 0 and lc != LAM_C[0]):
                            continue
                        t = np.array([total(j, *sel[j], lu, lc) for j in range(P)])
                        r = np.abs(t - real) / real
                        acc.setdefault((lu, lc), []).append(
                            (np.mean(r <= .25), np.median(r), np.mean(r[clean] <= .25),
                             np.mean(np.maximum(t / real, real / t) <= 2)))
            for (lu, lc), v in acc.items():
                v = np.mean(v, 0)
                rows.append(dict(user_estimates=f"{int(fe * 100)} %", closed=k, lambda_u=lu, lambda_c=lc,
                                 PRED25=float(v[0]), MdMRE=float(v[1]), PRED25_clean=float(v[2]),
                                 factor2=float(v[3])))
        print(f"  user estimates {int(fe * 100)} % done · {time.time() - t0:.0f} s", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "resumen.csv", index=False)
    best = t.sort_values("PRED25", ascending=False).groupby(["user_estimates", "closed"], sort=False).head(1)
    order = [f"{int(f * 100)} %" for f in ESTIMATED]
    for col, title in (("PRED25", f"PRED25 of the total, the {P} projects"),
                       ("PRED25_clean", f"PRED25 of the total, the {int(clean.sum())} with few copied rows"),
                       ("MdMRE", "MdMRE of the total")):
        print(f"\n{title} (best lambda per cell; rows: how much the user estimates · columns: closed tasks)")
        print(best.pivot(index="user_estimates", columns="closed", values=col).reindex(order).round(3).to_string())
    fixed = t[(t.lambda_u == 12) & (t.lambda_c == 30) | (t.user_estimates == "0 %") & (t.lambda_c == 30)
              | (t.closed == 0) & (t.lambda_u == 12)]
    fixed = fixed.drop_duplicates(["user_estimates", "closed"])
    print("\nPRED25 with ONE configuration (lambda_u = 12, lambda_c = 30), which is what would be deployed:")
    print(fixed.pivot(index="user_estimates", columns="closed", values="PRED25").reindex(order).round(3).to_string())
    print("\nlambdas that win in each cell:")
    print(best.assign(l=best.lambda_u.astype(str) + "/" + best.lambda_c.astype(str))
          .pivot(index="user_estimates", columns="closed", values="l").reindex(order).to_string())
    json.dump({"projects": P, "clean": int(clean.sum()), "draws": DRAWS,
               "seconds": round(time.time() - t0)}, open(OUT / "meta.json", "w"), indent=1)
    print(f"\nwritten {OUT} · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
