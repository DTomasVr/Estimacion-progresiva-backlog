"""E-278 - User inputs WITH the real closing order.

E-265 measured the real order without user inputs (38 projects: 50,0 % with 20 % of the
tasks closed against 71,3 % at random). E-268 measured user estimates with random
closing (41 projects). Here both are combined: projects with closing dates and with a
recorded estimate for at least 80 % of their tasks.

Deployed configuration: without estimates, lambda_c = 12; with estimates, lambda_u = 12
and lambda'_c = 30. The user estimates every task that has a recorded estimate.
Output: results/orden_real_con_estimaciones/resumen.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("results/orden_real_con_estimaciones")
MIN_TASKS, COVERAGE, DRAWS = 8, .80, 60
FRACS = (0, .10, .20, .30, .50)
LAM_U, LAM_C_ALONE, LAM_C_EST = 12, 12, 30


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
    size = d.groupby("p").size()
    d = d[d.p.isin(size[size >= MIN_TASKS].index)].sort_values(["p", "orden"]).reset_index(drop=True)
    G = [g for _, g in d.groupby("p", sort=False)]
    G = [g for g in G if g.usu.notna().mean() >= COVERAGE]
    P = len(G)
    re_ = [g.real.to_numpy() for g in G]
    es_ = [g.est.to_numpy() for g in G]
    us_ = [g.usu.to_numpy() for g in G]
    has = [np.flatnonzero(~np.isnan(x)) for x in us_]
    n = np.array([len(x) for x in re_])
    real = np.array([x.sum() for x in re_])
    print(f"{P} projects with real order and estimates · tasks: median {int(np.median(n))}", flush=True)

    def total(j, estimates, closed):
        base = es_[j].copy()
        lc = LAM_C_ALONE
        if estimates and len(has[j]):
            ei, m = has[j], len(has[j])
            base = base * np.exp(np.log(us_[j][ei].sum() / es_[j][ei].sum()) * m / (m + LAM_U))
            base[ei] = us_[j][ei]
            lc = LAM_C_EST
        k = len(closed)
        if not k:
            return base.sum()
        op = np.ones(n[j], bool)
        op[closed] = False
        fc = np.exp(np.log(re_[j][closed].sum() / base[closed].sum()) * k / (k + lc))
        return re_[j][closed].sum() + fc * base[op].sum()

    def pred(t):
        return float(np.mean(np.abs(t - real) / real <= .25))

    rng = np.random.default_rng(278)
    rows = []
    for estimates in (False, True):
        for f in FRACS:
            ks = np.minimum(np.round(f * n).astype(int), n - 1)
            if f:
                ks = np.maximum(ks, 1)
            real_order = pred(np.array([total(j, estimates, np.arange(ks[j])) for j in range(P)]))
            rnd = np.mean([pred(np.array([total(j, estimates, rng.choice(n[j], ks[j], replace=False))
                                          for j in range(P)])) for _ in range(DRAWS)])
            rows.append(dict(user_estimates="all" if estimates else "none", closed_fraction=f,
                             projects=P, PRED25_real_order=real_order, PRED25_random=float(rnd)))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "resumen.csv", index=False)
    print(t.round(3).to_string(index=False))
    print(f"one project = {100 / P:.1f} points · written {OUT} · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
