"""How well does story point -> hours actually work, per project?

The global conversion is dead (35x spread across projects). This asks whether a
PER-PROJECT conversion is good enough to be worth using as training
augmentation, by measuring the error a project-calibrated converter makes on
that project's OWN real logged hours -- an optimistic upper bound, since it is
evaluated on the very rows used to fit it.

Reported per project:
  MdAE / MdMRE  of predicting effort_hours from Story_Point
  PRED(25)      share of predictions within 25% of the truth
  a, b, r2      of a within-project log-log fit
and, as the honest reference, the same metrics for a trivial baseline that
ignores story points and always predicts that project's median hours.
"""
import numpy as np
import pandas as pd

pd.set_option("display.width", 240)
MIN_N = 30


def loglog(sp, hrs):
    x, y = np.log(sp.to_numpy(float)), np.log(hrs.to_numpy(float))
    if len(np.unique(x)) < 2:
        return np.nan, np.nan, np.nan
    b, la = np.polyfit(x, y, 1)
    pred = la + b * x
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return float(np.exp(la)), float(b), (1 - ss_res / ss_tot) if ss_tot else np.nan


def metrics(true, pred):
    ae = np.abs(true - pred)
    mre = ae / true
    return dict(MdAE=np.median(ae), MdMRE=np.median(mre),
                PRED25=float((mre <= .25).mean()))


def main():
    cal = pd.read_parquet("data/processed/tawos_sp_hours.parquet")
    cal = cal[(cal.Story_Point > 0) & (cal.effort_hours > 0)]
    rows = []
    for proj, d in cal.groupby("project"):
        if len(d) < MIN_N:
            continue
        med_ratio = float((d.effort_hours / d.Story_Point).median())
        a, b, r2 = loglog(d.Story_Point, d.effort_hours)

        m_lin = metrics(d.effort_hours.to_numpy(), (med_ratio * d.Story_Point).to_numpy())
        m_pow = metrics(d.effort_hours.to_numpy(), (a * d.Story_Point ** b).to_numpy())
        m_base = metrics(d.effort_hours.to_numpy(),
                         np.full(len(d), d.effort_hours.median()))
        rows.append(dict(
            project=proj, n=len(d), h_per_sp=round(med_ratio, 2),
            a=round(a, 2), b=round(b, 2), r2=round(r2, 3),
            lin_MdMRE=round(m_lin["MdMRE"], 2), pow_MdMRE=round(m_pow["MdMRE"], 2),
            base_MdMRE=round(m_base["MdMRE"], 2),
            pow_PRED25=round(m_pow["PRED25"], 2),
            base_PRED25=round(m_base["PRED25"], 2),
        ))
    res = pd.DataFrame(rows).sort_values("n", ascending=False)
    print("--- per-project SP -> hours conversion quality (fit and scored on the "
          "same rows, so optimistic) ---")
    print(res.to_string(index=False))

    print("\nReading: pow_MdMRE is the story-point converter, base_MdMRE ignores "
          "story points\nand predicts the project median. If they are close, "
          "story points add nothing.")

    # pooled view
    print("\n--- pooled across these projects ---")
    sub = cal[cal.project.isin(res.project)]
    a, b, r2 = loglog(sub.Story_Point, sub.effort_hours)
    print(f"  pooled log-log fit: {a:.2f} * SP^{b:.2f}   r2 = {r2:.3f}")
    print(f"  spread of per-project h/SP: min {res.h_per_sp.min():.2f}, "
          f"max {res.h_per_sp.max():.2f}  ({res.h_per_sp.max()/res.h_per_sp.min():.0f}x)")

    # how much data would a per-project conversion actually unlock?
    gold = pd.read_parquet("data/processed/tawos_gold.parquet",
                           columns=["project", "Issue_Key"])
    allissues = pd.read_parquet("data/interim/Issue.parquet",
                                columns=["Project_ID", "Story_Point", "Timespent",
                                         "Title", "Description_Text"])
    proj = pd.read_parquet("data/interim/Project.parquet")[["ID", "Project_Key"]]
    allissues = allissues.merge(proj, left_on="Project_ID", right_on="ID", how="left")
    expandable = allissues[
        (allissues.Story_Point > 0)
        & (allissues.Timespent.isna() | (allissues.Timespent <= 0))
        & allissues.Project_Key.isin(res.project)
    ]
    print(f"\n--- expansion potential (SP>0, no logged time, calibratable project) ---")
    print(expandable.groupby("Project_Key").size().sort_values(ascending=False).to_string())
    print(f"  TOTAL expandable rows: {len(expandable):,}")


if __name__ == "__main__":
    main()
