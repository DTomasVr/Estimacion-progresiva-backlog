"""E96 - Tuning the snap-to-grid step, and measuring it against the state of the art.

E95 left two things. The awkward one: predicting a constant gives more PRED(25) than
our model, because PRED(25) measures the kurtosis of estimate/real and the effort
distribution is a comb (ten values cover 46,8 %). And the useful one: SNAPPING the
prediction to the nearest frequent value raises the +-25 % from 15,06 % to 16,49 %
and the factor 2 from 0,390 to 0,431 without touching the MAE, the SA or the ranking
of tasks.

The aggressive rule (looking for the mode of a relative window) gave somewhat more
PRED(25) but sank the Spearman from 0,32 to 0,12 and the SA to negative: it would show
almost the same number for every task. Useless as a product.

So only the snap to the grid, which is free, is tuned here:

  how many values the grid must have, chosen OUT of fold
  frequency grid against geometric grid
  and whether it is better to snap in log or in linear scale

And it is measured where it can be compared: the block split_t3 == test, which is the
one used by the Deep-SE and GPT2SP reimplementations in 46_baselines_literatura.py.

COMPARABILITY WARNING: our predictions are out of fold with GroupKFold by WHOLE PROJECT,
so the model never saw the task or its project. The literature models are trained on
the train block, which DOES contain tasks of the same projects. The comparison works
against us, not the other way round; it should be said as it is.

Outputs: results/rejilla_valores/tamano_rejilla.csv and contra_literatura.csv
(the proposal and the 2,45 h constant of Table II in the paper)
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
OUT = Path("results/rejilla_valores")
S = "=" * 104
FOLDS, TOL = 5, .25
SIZES = (4, 6, 8, 10, 12, 16, 20, 30, 50)


def met(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, None)
    r = np.abs(p - y) / y
    mar = np.abs(np.median(y) - y).mean()
    return dict(PRED25=float(np.mean(r <= TOL)), MdMRE=float(np.median(r)),
                factor2=float(np.mean(np.maximum(p / y, y / p) <= 2)),
                MAE_h=float(np.abs(p - y).mean()),
                SA=float(1 - np.abs(p - y).mean() / mar),
                spearman=float(spearmanr(p, y).statistic))


def snap(p, grid):
    """To the nearest grid value IN LOG: what matters is the ratio."""
    return grid[np.argmin(np.abs(np.log(p[:, None] / grid[None])), 1)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet("data/processed/pbe_effort_v4.parquet")
    every = df[df.is_software_task == 1].reset_index(drop=True)
    sw = every[~every.fuente_jira.eq("Apache")].reset_index(drop=True)
    y = sw.effort_hours.to_numpy(float)
    proj = sw.project.to_numpy()
    p_cur = np.load("results/experto_cortas/predicciones.npz")["p_glob"]
    assert len(p_cur) == len(y)
    print(f"{len(y):,} tasks · {time.time() - t0:.0f} s", flush=True)

    # ---- 1. grid size, chosen out of fold ----
    rows = []
    for g in SIZES:
        for kind in ("frequency", "geometric"):
            p = np.empty(len(y))
            for a, b in GroupKFold(FOLDS).split(y[:, None], y, proj):
                if kind == "frequency":
                    v, c = np.unique(y[a], return_counts=True)
                    grid = np.sort(v[np.argsort(-c)[:g]])
                else:
                    grid = np.geomspace(np.quantile(y[a], .02),
                                        np.quantile(y[a], .98), g)
                p[b] = snap(p_cur[b], grid)
            rows.append(dict(size=g, grid=kind, **met(y, p)))
    t1 = pd.DataFrame(rows)
    print(f"\n{S}\nHOW MANY VALUES SHOULD THE GRID HAVE?\n{S}")
    print(t1.pivot(index="size", columns="grid",
                   values="PRED25").round(4).to_string())
    best = t1.loc[t1.PRED25.idxmax()]
    print(f"\n  best: {int(best['size'])} values, {best.grid} grid · ±25 % {best.PRED25:.2%} · factor2 "
          f"{best.factor2:.4f} · MAE {best.MAE_h:.3f} · SA {best.SA:.4f}")

    # the final grid, fitted on the whole corpus, to have it written down
    v, c = np.unique(y, return_counts=True)
    final_grid = np.sort(v[np.argsort(-c)[:int(best['size'])]])
    print(f"  the grid: {' · '.join(f'{x:g}' for x in final_grid)} h")

    # ---- 2. where it is compared with the literature ----
    te = (sw.split_t3 == "test").to_numpy()
    p_sn = np.empty(len(y))
    for a, b in GroupKFold(FOLDS).split(y[:, None], y, proj):
        vv, cc = np.unique(y[a], return_counts=True)
        p_sn[b] = snap(p_cur[b], np.sort(vv[np.argsort(-cc)[:int(best['size'])]]))
    const = np.empty(len(y))
    for a, b in GroupKFold(FOLDS).split(y[:, None], y, proj):
        cand = np.unique(y[a])
        const[b] = cand[int(np.argmax([np.mean(np.abs(x - y[a]) / y[a] <= TOL)
                                       for x in cand]))]
    print(f"\n{S}\nON THE BLOCK split_t3 == test ({int(te.sum()):,} tasks)\n{S}")
    f2 = [dict(model="ours, not snapped", **met(y[te], p_cur[te])),
          dict(model="ours, SNAPPED to the grid", **met(y[te], p_sn[te])),
          dict(model="best constant", **met(y[te], const[te]))]
    t2 = pd.DataFrame(f2)
    print(t2[["model", "PRED25", "MdMRE", "factor2", "MAE_h", "SA",
              "spearman"]].round(4).to_string(index=False))
    lit = pd.read_csv("results/baselines_literatura.csv")
    print("\n  the published models on the same block:")
    for _, x in lit.iterrows():
        print(f"    {x.model:42s} ±25 % {x.PRED25:.4f} · MdMRE {x.MdMRE:.4f}")

    print(f"\n{S}\nREADING\n{S}")
    a = float(t2[t2.model.str.startswith("ours, not")].PRED25.iloc[0])
    s_ = float(t2[t2.model.str.contains("SNAPPED")].PRED25.iloc[0])
    cc_ = float(t2[t2.model.str.startswith("best")].PRED25.iloc[0])
    top = float(lit.PRED25.max())
    name = lit.loc[lit.PRED25.idxmax(), "model"]
    print(f"  not snapped {a:.2%} · SNAPPED {s_:.2%} ({(s_ - a) * 100:+.1f} pts)")
    print(f"  the best published: {name} with {top:.2%}")
    print(f"  the best constant: {cc_:.2%}")
    if cc_ >= top:
        print("\n  NOTE: a constant beats EVERYTHING published on this metric.")
        print("  It is not a result about our model, it is about the metric:")
        print("  PRED(25) measures the kurtosis of estimate/real, not the ability to")
        print("  rank. It has to be reported together with SA and Spearman, where the")
        print("  constant scores 0 and we do not.")
    t1.to_csv(OUT / "tamano_rejilla.csv", index=False)
    t2.to_csv(OUT / "contra_literatura.csv", index=False)
    json.dump({"size": int(best['size']), "kind": best.grid,
               "grid": [float(x) for x in final_grid],
               "test_not_snapped": a, "test_snapped": s_,
               "test_constant": cc_, "best_published": top},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nwritten {OUT}/ · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
