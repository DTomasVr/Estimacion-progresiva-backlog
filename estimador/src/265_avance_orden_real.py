"""E-265 - The project-total curve by PROGRESS, with the real closing order.

E-264 measured the curve by closed fraction, drawing at random which tasks get closed.
Here it is repeated with the order in which they really closed (column `orden`) and
with cuts by time (2, 4, 8 and 12 weeks from the first closure), which is what a team
sees: "after the first sprint".

Only tasks with a date enter, and only projects that keep 8 or more. The whole backlog
is assumed known from the start, which is how the model works (the backlog is
generated at kick-off); in these real projects part of the tasks were created later,
so the curve is somewhat optimistic on that point. Anchoring: the deployed one (ratio
of sums, lambda 12).

Output: results/avance_orden_real/resumen.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("results/avance_orden_real")
MIN_TASKS, DRAWS, LAMBDA = 8, 60, 12
FRACS = (.05, .10, .20, .30, .40, .50, .70)
WEEKS = (2, 4, 8, 12)
LABELS = ["8-15", "16-40", "41-120", "121-400", "over 400"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet("data/processed/pbe_effort_v4.parquet")
    sw = df[df.is_software_task == 1].copy()
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    est = np.clip(np.load("results/modelo_total/pred_total.npy"), 1e-6, None)
    d = pd.DataFrame({"p": sw.project.to_numpy(), "real": sw.effort_hours.to_numpy(),
                      "est": est, "orden": sw.orden.to_numpy()})
    total_tasks = len(d)
    d = d[d.orden.notna()]
    size = d.groupby("p").size()
    d = d[d.p.isin(size[size >= MIN_TASKS].index)].sort_values(["p", "orden"]).reset_index(drop=True)
    names = list(pd.unique(d.p))
    P = len(names)
    G = [g for _, g in d.groupby("p", sort=False)]
    re_ = [g.real.to_numpy() for g in G]
    es_ = [g.est.to_numpy() for g in G]
    days = [(g.orden - g.orden.iloc[0]).dt.total_seconds().to_numpy() / 86400 for g in G]
    n = np.array([len(x) for x in re_])
    real = np.array([x.sum() for x in re_])
    cold = np.array([x.sum() for x in es_])
    band = np.asarray(pd.cut(n, [0, 15, 40, 120, 400, 10 ** 6], labels=LABELS).astype(str))
    print(f"{P} projects · {len(d):,} dated tasks out of {total_tasks:,} · {time.time() - t0:.0f} s", flush=True)

    def total(j, sel):
        k = len(sel)
        if k == 0:
            return cold[j]
        m = np.ones(n[j], bool)
        m[sel] = False
        off = np.log(re_[j][sel].sum() / es_[j][sel].sum()) * k / (k + LAMBDA)
        return re_[j][sel].sum() + np.exp(off) * es_[j][m].sum()

    def row(name, tot, k_med):
        ok = np.abs(tot - real) / real <= .25
        f = dict(cut=name, closed_tasks_median=float(k_med), PRED25=float(ok.mean()),
                 MdMRE=float(np.median(np.abs(tot - real) / real)),
                 factor2=float(np.mean(np.maximum(tot / real, real / tot) <= 2)))
        for e in LABELS:
            f[e] = float(ok[band == e].mean())
        return f

    rows = [row("cold", cold, 0)]
    rng = np.random.default_rng(265)
    for f in FRACS:
        ks = np.minimum(np.maximum(1, np.round(f * n).astype(int)), n - 1)
        rows.append(row(f"real order · {int(f * 100)} % closed",
                        np.array([total(j, np.arange(ks[j])) for j in range(P)]), np.median(ks)))
        rnd = [row("", np.array([total(j, rng.choice(n[j], ks[j], replace=False)) for j in range(P)]),
                   np.median(ks)) for _ in range(DRAWS)]
        m = pd.DataFrame(rnd).drop(columns="cut").mean().to_dict()
        rows.append(dict(cut=f"random · {int(f * 100)} % closed", **m))
        print(f"  {int(f * 100)} % done · {time.time() - t0:.0f} s", flush=True)
    for w in WEEKS:
        ks = np.array([min(int((days[j] <= 7 * w).sum()), n[j] - 1) for j in range(P)])
        f = row(f"real order · {w} weeks", np.array([total(j, np.arange(ks[j])) for j in range(P)]),
                np.median(ks))
        f["closed_fraction_median"] = float(np.median(ks / n))
        rows.append(f)
    t = pd.DataFrame(rows)
    t.insert(1, "projects", P)
    t.to_csv(OUT / "resumen.csv", index=False)
    pd.Series(band).value_counts().reindex(LABELS).to_csv(OUT / "proyectos_por_tramo.csv")
    print(t.round(3).to_string(index=False))
    print("projects by size band:", pd.Series(band).value_counts().reindex(LABELS).to_dict())
    print(f"written {OUT} · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
