"""E-264 - How far can the +-25 % of the project total go, and by which path?

The total sat at 36,8 % at cold start and 53,3 % with 20 closed tasks, and the ways
of improving the per-task prediction were exhausted. Before launching more experiments
this measures where the ceiling is and what sets it:

  1 NOISE CEILING. Even if the exact scale of the project were known, the total is a
    sum of very skewed hours: with few tasks, the sum does not concentrate. Each
    project is resampled with its REAL scale and the share of totals within +-25 %
    is measured. It is the ceiling of any anchoring.

  2 CURVE BY PROGRESS. The published curve fixes k (1..20 tasks), but 20 tasks are
    5 % of a 400-task project and 100 % of a 20-task one. Here it is measured by
    the closed fraction of the backlog.

  3 THE ANCHOR ESTIMATOR. Shrunk ratio of sums (the deployed one) against other
    lambdas, against the mean of log-ratios and against a trimmed ratio.

Same protocol as E67: out-of-fold predictions of the total head, projects with 8 or
more tasks, METRICS averaged over draws.

Outputs: results/techo_del_total/techo_ruido.csv and curva_avance.csv.
carga() is also used by E-275 and E-277.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("results/techo_del_total")
MIN_TASKS, DRAWS, LAMBDA = 8, 60, 12
FRACS = (.05, .10, .20, .30, .40, .50, .70)
KS = (5, 10, 20)
S = "=" * 110


def carga():
    """Tasks of the projects with 8 or more tasks: project, real hours and out-of-fold estimate."""
    df = pd.read_parquet("data/processed/pbe_effort_v4.parquet")
    sw = df[df.is_software_task == 1].copy()
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    est = np.clip(np.load("results/modelo_total/pred_total.npy"), 1e-6, None)
    d = pd.DataFrame({"p": sw.project.to_numpy(), "real": sw.effort_hours.to_numpy(), "est": est})
    size = d.groupby("p").size()
    return d[d.p.isin(size[size >= MIN_TASKS].index)].reset_index(drop=True)


def within(t, real):
    return np.abs(t - real) / real <= .25


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    d = carga()
    names = list(pd.unique(d.p))
    idx = {p: np.flatnonzero(d.p.to_numpy() == p) for p in names}
    re_, es_ = d.real.to_numpy(), d.est.to_numpy()
    n_t = np.array([len(idx[p]) for p in names])
    real_tot = np.array([re_[idx[p]].sum() for p in names])
    cold = np.array([es_[idx[p]].sum() for p in names])
    P = len(names)
    band = pd.cut(n_t, [0, 15, 40, 120, 400, 10 ** 6],
                  labels=["8-15", "16-40", "41-120", "121-400", "over 400"])
    print(f"{P} projects · {len(d):,} tasks · cold start {within(cold, real_tot).mean():.3f}", flush=True)

    # 1 · noise ceiling: real scale known, the project is resampled
    rng = np.random.default_rng(264)
    ceil_model, ceil_const = np.zeros(P), np.zeros(P)
    for j, p in enumerate(names):
        i = idx[p]
        c = re_[i].sum() / es_[i].sum()
        m = re_[i].mean()
        b = rng.choice(i, (2000, len(i)), replace=True)
        tot = re_[b].sum(1)
        ceil_model[j] = within(c * es_[b].sum(1), tot).mean()
        ceil_const[j] = within(np.full(2000, m * len(i)), tot).mean()
    t1 = (pd.DataFrame({"size_band": band, "cold": within(cold, real_tot),
                        "ceiling_exact_scale": ceil_model, "ceiling_mean_only": ceil_const})
          .groupby("size_band", observed=True).agg(projects=("cold", "size"), cold=("cold", "mean"),
                                                   ceiling_exact_scale=("ceiling_exact_scale", "mean"),
                                                   ceiling_mean_only=("ceiling_mean_only", "mean")).reset_index())
    t1.loc[len(t1)] = ["all", P, within(cold, real_tot).mean(), ceil_model.mean(), ceil_const.mean()]
    t1.to_csv(OUT / "techo_ruido.csv", index=False)
    print(f"\n{S}\n1 · NOISE CEILING (project scale known exactly)\n{S}")
    print(t1.round(3).to_string(index=False), flush=True)

    # anchors: they return the factor that multiplies the open part
    log_bias = float(np.mean([np.mean(np.log(re_[idx[p]] / es_[idx[p]]))
                              - np.log(re_[idx[p]].sum() / es_[idx[p]].sum()) for p in names]))

    def anchor(sel, mode):
        k = len(sel)
        if mode.startswith("sums"):
            lam = float(mode.split("=")[1])
            return np.exp(np.log(re_[sel].sum() / es_[sel].sum()) * k / (k + lam))
        if mode == "logs":
            return np.exp((np.mean(np.log(re_[sel] / es_[sel])) - log_bias) * k / (k + LAMBDA))
        if mode == "trimmed":
            q = re_[sel] / es_[sel]
            cap = np.quantile(q, .9)
            w = np.minimum(q, cap) * es_[sel]
            return np.exp(np.log(w.sum() / es_[sel].sum() * 1.0) * k / (k + LAMBDA))
        raise ValueError(mode)

    MODES = ["sums=12", "sums=4", "sums=30", "logs", "trimmed"]

    def run(how_many, label):
        rng = np.random.default_rng(9000)
        acc = {m: [] for m in MODES}
        acc_open = []
        by_band = []
        for _ in range(DRAWS):
            tot = {m: np.zeros(P) for m in MODES}
            open_ok = np.full(P, np.nan)
            for j, p in enumerate(names):
                i = idx[p]
                k = how_many(len(i))
                if k <= 0 or k >= len(i):
                    for m in MODES:
                        tot[m][j] = es_[i].sum() if k <= 0 else re_[i].sum()
                    continue
                sel = rng.choice(i, k, replace=False)
                rest = np.setdiff1d(i, sel)
                for m in MODES:
                    tot[m][j] = re_[sel].sum() + anchor(sel, m) * es_[rest].sum()
                op = anchor(sel, "sums=12") * es_[rest].sum()
                open_ok[j] = abs(op - re_[rest].sum()) / re_[rest].sum() <= .25
            for m in MODES:
                acc[m].append(within(tot[m], real_tot).mean())
            acc_open.append(np.nanmean(open_ok))
            by_band.append(pd.Series(within(tot["sums=12"], real_tot)).groupby(band, observed=True).mean())
        row = dict(cut=label, **{f"PRED25 {m}": float(np.mean(acc[m])) for m in MODES},
                   PRED25_open_part_only=float(np.mean(acc_open)))
        row.update({f"band {c}": float(v) for c, v in pd.concat(by_band, axis=1).mean(1).items()})
        return row

    rows = [run(lambda n, k=k: k, f"k={k}") for k in KS]
    print(f"  cuts by k done · {time.time() - t0:.0f} s", flush=True)
    rows += [run(lambda n, f=f: int(round(f * n)), f"{int(f * 100)} % closed") for f in FRACS]
    t2 = pd.DataFrame(rows)
    t2.to_csv(OUT / "curva_avance.csv", index=False)
    print(f"\n{S}\n2 and 3 · CURVE BY PROGRESS AND ANCHOR ESTIMATOR\n{S}")
    print(t2.round(3).to_string(index=False))
    json.dump({"projects": P, "draws": DRAWS, "log_bias": log_bias,
               "seconds": round(time.time() - t0)}, open(OUT / "meta.json", "w"), indent=1)
    print(f"\nwritten {OUT} · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
