"""E-267 - The user gives the hours of SOME tasks: how much does the total improve?

This is the approach to be implemented: the user types the hours of any tasks (one, five
or all) and the system uses them in two ways at once:

  - those tasks count by what the user said, not by what the model says;
  - the ratio between what the user said and what the model predicted for THOSE tasks
    gives a factor that rescales the others, shrunk by how many were given:
        factor = (user sum / model sum) ^ (m / (m + lambda))
    With m = 1 only that task is anchored and the factor barely moves; with more, it
    weighs more.

SIMULATION. Each project of the corpus is treated as new: its predictions are
out-of-fold by whole project (the model never saw it). The user input is simulated in
two ways:

  ideal        the real hours of those tasks: the best a user can give, and exactly
               what happens when the task has already been closed
  estimate     the estimate the user REALLY gave (HoursEstimate), in the projects that
               have it for 80 % of their tasks or more. It is not simulated:
               multiplying the real hours by a size-independent error gives an inflated
               total that does not look like a real user (tried: 17 % giving them all,
               against the 73 % measured)

It is measured on (1) the SAME project and (2) ANOTHER project of the same team: the
factor learnt on project A is applied cold to project B of the same organisation,
against the control of applying it to a project of another organisation.

METRICS are averaged over draws (which tasks the user gives is random).

Outputs: results/entrada_del_usuario/. carga() and COBERTURA are also used by E-268.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("results/entrada_del_usuario")
MIN_TASKS, DRAWS = 8, 60
MS = (1, 2, 3, 5, 10, 20)
FRACS = (.25, .50, 1.0)
LAMBDAS = (0, 4, 12, 30)
COBERTURA = .80          # share of tasks with a user estimate needed to use a project
LABELS = ["8-15", "16-40", "41-120", "121-400", "over 400"]


def org_of(p):
    if ":" in p:
        return p.split(":")[0]
    return "SiP" if p.startswith("SiP") else p


def carga():
    """Tasks of the projects with 8 or more tasks, with the user estimate when there is one."""
    df = pd.read_parquet("data/processed/pbe_effort_v4.parquet")
    sw = df[df.is_software_task == 1].copy()
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    d = pd.DataFrame({"p": sw.project.to_numpy(), "real": sw.effort_hours.to_numpy(),
                      "est": np.clip(np.load("results/modelo_total/pred_total.npy"), 1e-6, None),
                      "usu": sw.HoursEstimate.where(sw.HoursEstimate > 0).to_numpy(),
                      "copiada": sw.estimacion_igual_a_real.to_numpy()})
    size = d.groupby("p").size()
    return d[d.p.isin(size[size >= MIN_TASKS].index)].reset_index(drop=True)


def metrics(tot, real):
    r = np.abs(tot - real) / real
    return dict(PRED25=float(np.mean(r <= .25)), MdMRE=float(np.median(r)),
                factor2=float(np.mean(np.maximum(tot / real, real / tot) <= 2)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    d = carga()
    names = list(pd.unique(d.p))
    P = len(names)
    G = [g for _, g in d.groupby("p", sort=False)]
    assert [g.p.iloc[0] for g in G] == names
    re_ = [g.real.to_numpy() for g in G]
    es_ = [g.est.to_numpy() for g in G]
    n = np.array([len(x) for x in re_])
    real = np.array([x.sum() for x in re_])
    cold = np.array([x.sum() for x in es_])
    band = np.asarray(pd.cut(n, [0, 15, 40, 120, 400, 10 ** 6], labels=LABELS).astype(str))
    us_ = [g.usu.to_numpy() for g in G]
    has = [np.flatnonzero(~np.isnan(x)) for x in us_]
    sub = np.array([len(c) / len(x) >= COBERTURA for c, x in zip(has, us_)])
    clean = sub & np.array([g.copiada.mean() < .25 for g in G])
    print(f"{P} projects · {len(d):,} tasks · cold-start PRED25 {metrics(cold, real)['PRED25']:.3f}", flush=True)
    print(f"with a user estimate: {int(sub.sum())} projects (cold start "
          f"{metrics(cold[sub], real[sub])['PRED25']:.3f}) · with less than 25 % copied rows: "
          f"{int(clean.sum())} (cold start {metrics(cold[clean], real[clean])['PRED25']:.3f})", flush=True)

    def how_many(j, m):
        return n[j] if m == "all" else min(int(round(m * n[j])) if isinstance(m, float) else m, n[j])

    def one_draw(m, rng):
        """Totals per project for each (input, rule)."""
        tot = {}
        op = {}
        for j in range(P):
            k = max(1, how_many(j, m))
            sel = rng.choice(n[j], k, replace=False)
            rest = np.ones(n[j], bool)
            rest[sel] = False
            arms = [("ideal", re_[j][sel], sel, rest)]
            if sub[j]:
                s2 = rng.choice(has[j], min(k, len(has[j])), replace=False)
                r2 = np.ones(n[j], bool)
                r2[s2] = False
                arms += [("estimate", us_[j][s2], s2, r2), ("ideal, same projects", re_[j][s2], s2, r2)]
            for inp, hours, sel, rest in arms:
                su, sm, sr = hours.sum(), es_[j][sel].sum(), es_[j][rest].sum()
                tot.setdefault((inp, "replace only"), np.zeros(P))[j] = su + sr
                for lam in LAMBDAS:
                    f = np.exp(np.log(su / sm) * k / (k + lam)) if k + lam else 1.
                    tot.setdefault((inp, f"anchor lambda={lam}"), np.zeros(P))[j] = su + f * sr
                    if lam == 12 and rest.any():
                        op.setdefault(inp, np.full(P, np.nan))[j] = (
                            abs(f * sr - re_[j][rest].sum()) / re_[j][rest].sum() <= .25)
        return tot, op

    rows, by_band = [], []
    cuts = [(m, f"{m} task{'s' if m > 1 else ''}") for m in MS] + \
           [(f, "all" if f == 1.0 else f"{int(f * 100)} % of the backlog") for f in FRACS]
    for m, label in cuts:
        rng = np.random.default_rng(26700 + int(m * 100))
        acc, acc_op, acc_band = {}, {}, {}
        for _ in range(DRAWS):
            tot, op = one_draw(m, rng)
            for c, t in tot.items():
                msk = np.ones(P, bool) if c[0] == "ideal" else sub
                mm = metrics(t[msk], real[msk])
                if c[0] != "ideal":
                    mm["PRED25_clean"] = metrics(t[clean], real[clean])["PRED25"]
                acc.setdefault(c, []).append(mm)
                if c == ("ideal", "anchor lambda=12"):
                    ok = np.abs(t - real) / real <= .25
                    acc_band.setdefault(c[0], []).append([ok[band == e].mean() for e in LABELS])
            for e, v in op.items():
                acc_op.setdefault(e, []).append(np.nanmean(v if e == "ideal" else v[sub]))
        for (inp, rule), ms in acc.items():
            rows.append(dict(input=inp, user_gives=label, rule=rule,
                             **{k: float(np.mean([x[k] for x in ms])) for k in ms[0]},
                             PRED25_of_what_was_not_given=(float(np.mean(acc_op[inp]))
                                                           if rule == "anchor lambda=12" and inp in acc_op
                                                           else np.nan)))
        for inp, v in acc_band.items():
            by_band.append(dict(input=inp, user_gives=label, **dict(zip(LABELS, np.mean(v, 0)))))
        print(f"  same project · {label} done · {time.time() - t0:.0f} s", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "mismo_proyecto.csv", index=False)
    pd.DataFrame(by_band).to_csv(OUT / "mismo_proyecto_por_tamano.csv", index=False)

    order = [e for _, e in cuts]
    print(f"\n{'=' * 100}\nSAME PROJECT · PRED25 of the total (cold start {metrics(cold, real)['PRED25']:.3f})\n{'=' * 100}")
    for inp in ("ideal", "estimate", "ideal, same projects"):
        if inp != "ideal":
            print(f"\n(the {int(sub.sum())} projects with estimates; cold start "
                  f"{metrics(cold[sub], real[sub])['PRED25']:.3f})")
        piv = t[t.input == inp].pivot(index="user_gives", columns="rule", values="PRED25").reindex(order)
        print(f"\ninput {inp}:")
        print(piv.round(3).to_string())
    print("\nPRED25 of the tasks the user did NOT give only (anchor lambda=12):")
    print(t[t.rule == "anchor lambda=12"].pivot(index="user_gives", columns="input",
                                                values="PRED25_of_what_was_not_given").reindex(order).round(3)
          .to_string())
    print(f"\nOnly the {int(clean.sum())} projects with few copied rows · PRED25:")
    print(t[t.input == "estimate"].pivot(index="user_gives", columns="rule",
                                         values="PRED25_clean").reindex(order).round(3).to_string())
    print("\nBy project size (ideal input, anchor lambda=12):")
    print(pd.DataFrame(by_band).round(3).to_string(index=False))

    # 2 · another project of the same team
    org = np.array([org_of(p) for p in names])
    count = pd.Series(org).value_counts()
    with_sib = np.flatnonzero(pd.Series(org).map(count).to_numpy() >= 2)
    print(f"\n{len(with_sib)} projects with a sibling in {int((count >= 2).sum())} organisations: "
          f"{count[count >= 2].to_dict()}", flush=True)
    rows2 = []
    for m, label in [(5, "5 tasks"), (10, "10 tasks"), (20, "20 tasks"), (1.0, "all")]:
        rng = np.random.default_rng(26800 + int(m * 100))
        acc = {}
        for _ in range(DRAWS):
            ts, tc = np.zeros(len(with_sib)), np.zeros(len(with_sib))
            for a, j in enumerate(with_sib):
                sib = np.flatnonzero((org == org[j]) & (np.arange(P) != j))
                oth = np.flatnonzero(org != org[j])
                for target, cand in ((ts, sib), (tc, oth)):
                    A = rng.choice(cand)
                    k = max(1, how_many(A, m))
                    sel = rng.choice(n[A], k, replace=False)
                    f = np.exp(np.log(re_[A][sel].sum() / es_[A][sel].sum()) * k / (k + 12))
                    target[a] = f * cold[j]
            for name, tt in (("sibling's factor", ts), ("factor from another organisation (control)", tc)):
                acc.setdefault(name, []).append(metrics(tt, real[with_sib]))
        for name, ms in acc.items():
            rows2.append(dict(user_gives_on_A=label, arm=name,
                              **{k: float(np.mean([x[k] for x in ms])) for k in ms[0]}))
        print(f"  another project · {label} done · {time.time() - t0:.0f} s", flush=True)
    rows2.insert(0, dict(user_gives_on_A="-", arm="B cold start", **metrics(cold[with_sib], real[with_sib])))
    t2 = pd.DataFrame(rows2)
    t2.to_csv(OUT / "otro_proyecto_mismo_equipo.csv", index=False)
    print(f"\n{'=' * 100}\nANOTHER PROJECT OF THE SAME TEAM ({len(with_sib)} projects)\n{'=' * 100}")
    print(t2.round(3).to_string(index=False))
    json.dump({"projects": P, "draws": DRAWS, "with_estimate": int(sub.sum()), "clean": int(clean.sum()),
               "projects_with_sibling": int(len(with_sib)),
               "seconds": round(time.time() - t0)}, open(OUT / "meta.json", "w"), indent=1)
    print(f"\nwritten {OUT} · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
