"""E114 - Narrowing the band of the total sooner: look at WHAT the closed tasks are like.

E107 left the band of the total calibrated by number of closed tasks and size band, and
it narrows well: x10,61 at cold start, x4,96 with 3, x3,75 with 5, x2,37 with 10 and
x2,05 with 20. But it only looks at HOW MANY tasks are closed, not at WHAT they are like.

There is unused signal there. After closing k tasks something is observed that the cold
start does not have: if those k agree with the prediction (low spread of
log(real/predicted)) the project offset is well determined and the open part can be
bounded more tightly; if they are scattered, it cannot.

It is NORMALISED conformal prediction, but at project level, where there is an
observable that anticipates the uncertainty. At TASK level it failed six times because
the text does not know which tasks are uncertain; here the data comes from the progress
of the project itself.

Features available at estimation time, all from the running project:

  k              closed tasks
  sd_closed      spread of log(real/predicted) over those k
  |off|          how much the anchoring corrects
  n_open         how many tasks are still to estimate
  frac_hours     share of the predicted hours already closed
  size band      project size

  A  the E107 band: by k and size band
  B  normalised: the residual is divided by sigma(features), fitted out of sample, and
     the conformal quantile is taken on the normalised residual
  C  B but also by size band
  D  the size-band quantile SHRUNK towards the global one (the deployed variants)

What matters is not the mean width but how early it narrows: at which k the width that
today needs 10 or 20 closed tasks is reached.

The NOMINAL level is tuned on the same data until the REAL coverage is 0,80, so widths
are compared at equal coverage (the paper states that this coverage is by construction).

Outputs: results/banda_que_cierra_antes/resumen.csv and cobertura_por_tramo.csv
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "10"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 215)
OUT = Path("results/banda_que_cierra_antes")
S = "=" * 112
MIN_TASKS, LAMBDA, DRAWS, LEVEL = 8, 12, 30, .8
KS = (1, 2, 3, 5, 10, 20)
CUTS = [20, 60, 200]
PAR = dict(max_iter=200, max_depth=4, learning_rate=.06, l2_regularization=1.,
           random_state=0)


def qconf(x, p, upper):
    s = np.sort(np.asarray(x, float))
    m = len(s)
    j = (int(np.ceil((m + 1) * p)) if upper else int(np.floor((m + 1) * p))) - 1
    return float(s[int(np.clip(j, 0, m - 1))])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet("data/processed/pbe_effort_v4.parquet")
    sw = df[df.is_software_task == 1]
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    pred = np.clip(np.load("results/modelo_total/pred_total.npy"), 1e-6, None)
    size = sw.groupby("project").size()
    m = sw.project.isin(size[size >= MIN_TASKS].index).to_numpy()
    sw, pred = sw[m].reset_index(drop=True), pred[m]
    y = sw.effort_hours.to_numpy(float)
    proj = sw.project.to_numpy()
    names = list(pd.unique(proj))
    idx = {p: np.flatnonzero(proj == p) for p in names}
    P = len(names)
    n_t = np.array([len(idx[p]) for p in names])
    sb = np.digitize(n_t, CUTS)
    print(f"{P} projects · {len(sw):,} tasks · {time.time() - t0:.0f} s",
          flush=True)

    # one record per (k, draw, project)
    rec = []
    for k in KS:
        for s in range(DRAWS):
            rng = np.random.default_rng(1200 + s)
            for j, p in enumerate(names):
                i = idx[p]
                kk = min(k, len(i) - 1)
                sel = rng.choice(i, kk, replace=False)
                rest = np.setdiff1d(i, sel)
                lr = np.log(y[sel] / pred[sel])
                off = float(np.log(y[sel].sum() / pred[sel].sum())
                            * kk / (kk + LAMBDA))
                op = float((pred[rest] * np.exp(off)).sum())
                rec.append((k, s, j, sb[j], kk, len(rest),
                            float(np.std(lr)) if kk > 1 else -1.0,
                            abs(off), float(y[sel].sum()),
                            float(np.log(max(op, 1e-6))),
                            float(pred[sel].sum() / pred[i].sum()),
                            float(np.log(y[rest].sum() / max(op, 1e-9)))))
    R = pd.DataFrame(rec, columns=["k", "s", "j", "size_band", "kk", "n_open", "sd_closed",
                                   "abs_off", "h_closed", "log_open", "frac_h", "rho"])
    print(f"  {len(R):,} records · {time.time() - t0:.0f} s", flush=True)

    FEATURES = ["kk", "n_open", "sd_closed", "abs_off", "log_open", "frac_h", "size_band"]
    X = R[FEATURES].to_numpy(np.float32)
    rho = R.rho.to_numpy()
    jj = R.j.to_numpy()

    # sigma(features) out of sample, leaving whole PROJECTS out
    group = jj % 4
    sig = np.zeros(len(R))
    for g in range(4):
        a, b = group != g, group == g
        mo = HistGradientBoostingRegressor(**PAR).fit(X[a], np.log(np.abs(rho[a]) + 1e-3))
        sig[b] = np.exp(mo.predict(X[b]))
    sig = np.clip(sig, 1e-3, None)
    print(f"  sigma fitted · corr(|rho|, sigma) = "
          f"{np.corrcoef(np.abs(rho), sig)[0, 1]:.4f} · "
          f"{time.time() - t0:.0f} s", flush=True)

    rows, by_size = [], []

    def build(tag, sk, nominal):
        """Band of a method at a given NOMINAL level. Returns (inside, width)."""
        aa = (1 - nominal) / 2
        Rk = R[sk]
        jk, sgk, rk, sbk = (Rk.j.to_numpy(), sig[sk], rho[sk], Rk.size_band.to_numpy())
        lo, hi = np.zeros(sk.sum()), np.zeros(sk.sum())
        for g in range(4):
            cal, ev = (jk % 4) != g, (jk % 4) == g
            if tag == "A":
                lo[ev], hi[ev] = qconf(rk[cal], aa, False), qconf(rk[cal], 1 - aa, True)
            elif tag == "A2":
                for t_ in np.unique(sbk):
                    c2, e2 = cal & (sbk == t_), ev & (sbk == t_)
                    if c2.sum() < 60 or not e2.any():
                        c2 = cal
                    lo[e2] = qconf(rk[c2], aa, False)
                    hi[e2] = qconf(rk[c2], 1 - aa, True)
            elif tag == "B":
                z = rk[cal] / sgk[cal]
                lo[ev] = qconf(z, aa, False) * sgk[ev]
                hi[ev] = qconf(z, 1 - aa, True) * sgk[ev]
            elif tag == "C":
                for t_ in np.unique(sbk):
                    c2, e2 = cal & (sbk == t_), ev & (sbk == t_)
                    if c2.sum() < 60 or not e2.any():
                        c2 = cal
                    z = rk[c2] / sgk[c2]
                    lo[e2] = qconf(z, aa, False) * sgk[e2]
                    hi[e2] = qconf(z, 1 - aa, True) * sgk[e2]
            else:
                # D: the size-band quantile SHRUNK towards the global one, with weight
                # n/(n+M). It keeps the conditional nature of E89 without paying the
                # width that calibrating each size band separately costs.
                M = float(tag[1:])
                lg, hg = qconf(rk[cal], aa, False), qconf(rk[cal], 1 - aa, True)
                for t_ in np.unique(sbk):
                    c2, e2 = cal & (sbk == t_), ev & (sbk == t_)
                    if not e2.any():
                        continue
                    n2 = int(c2.sum())
                    if n2 < 10:
                        lo[e2], hi[e2] = lg, hg
                        continue
                    w_ = n2 / (n2 + M)
                    lo[e2] = w_ * qconf(rk[c2], aa, False) + (1 - w_) * lg
                    hi[e2] = w_ * qconf(rk[c2], 1 - aa, True) + (1 - w_) * hg
        op, cl = np.exp(Rk.log_open.to_numpy()), Rk.h_closed.to_numpy()
        w = (cl + op * np.exp(hi)) / np.maximum(cl + op * np.exp(lo), 1e-9)
        return (rk >= lo) & (rk <= hi), w

    NAMES = {"A": "A · by k only", "A2": "A2 · by k and size band (deployed)",
             "B": "B · normalised", "C": "C · normalised + size band",
             "D200": "D · shrunk size band M=200", "D600": "D · shrunk size band M=600",
             "D1500": "D · shrunk size band M=1500"}
    for k in KS:
        sk = R.k.to_numpy() == k
        for tag in ("A", "A2", "B", "C", "D200", "D600", "D1500"):
            # the NOMINAL level is tuned until the REAL coverage is LEVEL,
            # so that widths are compared at equal coverage
            lo_n, hi_n = .60, .999
            for _ in range(12):
                md = (lo_n + hi_n) / 2
                d_, _ = build(tag, sk, md)
                if d_.mean() < LEVEL:
                    lo_n = md
                else:
                    hi_n = md
            inside, w = build(tag, sk, hi_n)
            sb_ = R[sk].size_band.to_numpy()
            for t_ in np.unique(sb_):
                by_size.append(dict(k=k, band=NAMES[tag], size_band=int(t_),
                                    projects=int(len(np.unique(R[sk].j.to_numpy()[sb_ == t_]))),
                                    coverage=float(inside[sb_ == t_].mean())))
            larger = R[sk].kk.to_numpy() == k     # projects with more tasks than the closed ones ("larger")
            rows.append(dict(k=k, band=NAMES[tag], nominal_level=float(hi_n),
                             coverage=float(inside.mean()),
                             median_width=float(np.median(w)),
                             width_p25=float(np.quantile(w, .25)),
                             width_p75=float(np.quantile(w, .75)),
                             median_width_larger=float(np.median(w[larger])),
                             coverage_larger=float(inside[larger].mean()),
                             projects_larger=int(len(np.unique(R[sk].j.to_numpy()[larger])))))
        print(f"  k={k} done · {time.time() - t0:.0f} s", flush=True)

    t = pd.DataFrame(rows)
    print(f"\n{S}\nMEDIAN WIDTH OF THE BAND OF THE TOTAL AT {LEVEL:.0%}\n{S}")
    print(t.pivot(index="k", columns="band",
                  values="median_width").round(2).to_string())
    print(f"\n{S}\nCOVERAGE\n{S}")
    print(t.pivot(index="k", columns="band", values="coverage").round(3).to_string())
    print(f"\n{S}\nTHE GOOD QUARTILE: PROJECTS WHOSE CLOSED TASKS AGREE\n{S}")
    print(t.pivot(index="k", columns="band",
                  values="width_p25").round(2).to_string())

    print(f"\n{S}\nREADING\n{S}")
    A = t[t.band.str.startswith("A2")].set_index("k")
    B = t[t.band.str.startswith("B")].set_index("k")
    C = t[t.band.str.startswith("C")].set_index("k")
    for k in KS:
        print(f"  k={k:2d}: deployed x{A.loc[k, 'median_width']:5.2f} · "
              f"B x{B.loc[k, 'median_width']:5.2f} · "
              f"C x{C.loc[k, 'median_width']:5.2f}  "
              f"(p25: {A.loc[k, 'width_p25']:5.2f} / "
              f"{B.loc[k, 'width_p25']:5.2f} / {C.loc[k, 'width_p25']:5.2f})")
    target = float(A.loc[10, "median_width"])
    for tag, D in (("B", B), ("C", C)):
        reach = [k for k in KS if D.loc[k, "median_width"] <= target]
        if reach:
            print(f"\n  {tag} reaches with k={min(reach)} the width that A needs k=10 for")
    t.to_csv(OUT / "resumen.csv", index=False)
    # coverage of each band by size band (guide V2: "coverage in each group")
    pd.DataFrame(by_size).to_csv(OUT / "cobertura_por_tramo.csv", index=False)
    json.dump({"projects": P, "draws": DRAWS, "level": LEVEL,
               "corr_sigma": float(np.corrcoef(np.abs(rho), sig)[0, 1])},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nwritten {OUT}/resumen.csv · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
