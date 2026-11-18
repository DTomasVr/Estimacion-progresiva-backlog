"""E55 - An interval for the project total, not just a number.

"The project is 640 hours" is a sentence nobody should say with an MdMRE of 0,36.
"Between 380 and 1.080 hours, with 80 % confidence" can be said, and it is what a
project manager needs to negotiate a deadline.

With 114 projects the interval is calibrated by conformal prediction, which gives
guaranteed coverage without assuming anything about the shape of the distribution,
only that projects are exchangeable. The cut is estimated LEAVING OUT the project
being evaluated.

Three things to look at, not only coverage:

  WIDTH        an interval that covers 80 % while being x100 wide is useless. It is
               given as a multiplicative factor, which is how it is read
  SPREAD       by project size. If it over-covers small projects and under-covers
               large ones, the overall coverage hides both
  HISTORY      how it narrows as the team closes tasks

With closed tasks the same as in E53 applies: the hours already closed are KNOWN and
not estimated, and the metric is averaged over draws instead of averaging the
predictions, which gave a falsely good coverage.

Outputs: results/banda_proyecto/resumen.csv and por_tamano.csv
"""
import json
import os
import sys
import warnings
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "8"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 215)
OUT = Path("results/banda_proyecto")
S = "=" * 116
MIN_TASKS, DRAWS, LAMBDA = 8, 30, 12
LEVELS = (.5, .8, .9)
LABELS = ["8-20", "21-60", "61-200", "over 200"]


def bands(pred, real, q, level, groups=None):
    """Conformal intervals, leaving out the project being evaluated."""
    a = (1 - level) / 2
    lo = np.zeros(len(real))
    hi = np.zeros(len(real))
    for i in range(len(real)):
        k = np.ones(len(real), bool)
        k[i] = False
        if groups is not None:
            kk = k & (groups == groups[i])
            if kk.sum() >= 12:
                k = kk
        lo[i] = pred[i] * np.exp(np.quantile(q[k], a))
        hi[i] = pred[i] * np.exp(np.quantile(q[k], 1 - a))
    return lo, hi


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet("data/processed/pbe_effort_v4.parquet")
    sw = df[df.is_software_task == 1].copy()
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    y = sw.effort_hours.to_numpy()
    est = np.clip(np.load("results/modelo_total/pred_total.npy"), 1e-6, None)
    d = pd.DataFrame({"p": sw.project.to_numpy(), "real": y, "est": est})
    size = d.groupby("p").size()
    d = d[d.p.isin(size[size >= MIN_TASKS].index)].reset_index(drop=True)
    names = list(pd.unique(d.p))
    idx = {p: np.flatnonzero(d.p.to_numpy() == p) for p in names}
    re_, es_ = d.real.to_numpy(), d.est.to_numpy()
    le = np.log(es_)
    g = d.groupby("p")
    real = g.real.sum().reindex(names).to_numpy()
    n_t = g.size().reindex(names).to_numpy()
    strata = np.digitize(n_t, [21, 61, 201])

    scen = {"cold": [g.est.sum().reindex(names).to_numpy()]}
    for k in (3, 10):
        rng = np.random.default_rng(2000 + k)
        M = []
        for _ in range(DRAWS):
            t = []
            for p in names:
                i = idx[p]
                if len(i) <= k:
                    t.append(float(es_[i].sum()))
                    continue
                s = rng.choice(i, k, replace=False)
                rest = np.setdiff1d(i, s)
                off = np.log(re_[s].sum() / es_[s].sum()) * k / (k + LAMBDA)
                t.append(float(re_[s].sum() + np.exp(le[rest] + off).sum()))
            M.append(np.array(t))
        scen[f"{k} closed tasks"] = M
    print(f"{len(names)} projects · {DRAWS} draws", flush=True)

    rows, cond = [], []
    for name, draws in scen.items():
        for level in LEVELS:
            acc = {"single": [], "by size band": []}
            wid = {"single": [], "by size band": []}
            byB, byS = [], []
            for pred in draws:
                q = np.log(real / pred)
                for tag, gr in (("single", None), ("by size band", strata)):
                    lo, hi = bands(pred, real, q, level, gr)
                    cov = (real >= lo) & (real <= hi)
                    acc[tag].append(float(cov.mean()))
                    wid[tag].append(float(np.median(hi / lo)))
                    if level == .8 and name == "cold":
                        row = [(float(cov[strata == s].mean()),
                                float(np.median((hi / lo)[strata == s])))
                               for s in range(4)]
                        (byB if tag == "by size band" else byS).append(row)
            for tag in acc:
                rows.append(dict(scenario=name, level=f"{level:.0%}", method=tag,
                                 coverage=float(np.mean(acc[tag])),
                                 sd_coverage=float(np.std(acc[tag])),
                                 width_x=float(np.mean(wid[tag]))))
            if level == .8 and name == "cold":
                for tag, dat in (("single interval", byS), ("by size band", byB)):
                    a = np.array(dat).mean(0)
                    for s in range(4):
                        cond.append(dict(size_band=LABELS[s], method=tag,
                                         projects=int((strata == s).sum()),
                                         coverage=float(a[s, 0]),
                                         width_x=float(a[s, 1])))
        print(f"  {name} done", flush=True)

    f = pd.DataFrame(rows)
    for name in scen:
        print(f"\n{S}\n{name}\n{S}")
        print(f[f.scenario == name].drop(columns=["scenario"]).round(3)
              .to_string(index=False))
    c = pd.DataFrame(cond).sort_values(["method", "size_band"])
    print(f"\n{S}\nCOVERAGE AT 80 % BY PROJECT SIZE (cold start)\n{S}")
    print(c.round(3).to_string(index=False))

    v = f[(f.scenario == "cold") & (f.level == "80%") & (f.method == "single")].iloc[0]
    v10 = f[(f.scenario == "10 closed tasks") & (f.level == "80%") &
            (f.method == "single")].iloc[0]
    pred0 = scen["cold"][0]
    q = np.log(real / pred0)
    ex = int(np.argsort(n_t)[len(n_t) // 2])
    lo, hi = bands(pred0, real, q, .8, None)
    print(f"\n{S}\nREADING\n{S}")
    print(f"  at 80 %, cold start: covers {v.coverage:.1%} with width x{v.width_x:.1f}")
    print(f"  at 80 %, with 10 closed tasks: {v10.coverage:.1%} with "
          f"x{v10.width_x:.1f}")
    print(f"\n  example with a median-size project ({n_t[ex]} tasks):")
    print(f"    point estimate {pred0[ex]:.0f} h · 80 % interval from "
          f"{lo[ex]:.0f} to {hi[ex]:.0f} h · real {real[ex]:.0f} h")
    f.to_csv(OUT / "resumen.csv", index=False)
    c.to_csv(OUT / "por_tamano.csv", index=False)
    json.dump({"levels": list(LEVELS), "draws": DRAWS},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nwritten {OUT}/resumen.csv")


if __name__ == "__main__":
    main()
