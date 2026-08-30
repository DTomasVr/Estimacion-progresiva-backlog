"""Project-22: calibrate the story-point -> actual-hours relationship.

P22 is the only public dataset that pairs story points with clocked hours on
the same rows, so it is our reference for whether an SP->hours conversion is
defensible at all. We reproduce Derek Jones' published fit first, as a
correctness check on our own pipeline, then export the calibration.

Published (README.md): Actual = 1.01 * Estimate^1.09, variance explained 46%,
1 sd within a factor 1.9 of actual, 95% within a factor 3.4.
"""
import json

import numpy as np
import pandas as pd

RAW = "data/raw/jones/Project-22"
OUT = "data/processed"


def load():
    story = pd.read_csv(f"{RAW}/story-info.csv")
    review = pd.read_csv(f"{RAW}/review-info.csv")
    rev = review.groupby("Branch")["ReviewMinutes"].sum().rename("rev_mins")
    df = story.join(rev, on="Branch")
    df["rev_mins"] = df["rev_mins"].fillna(0.0)
    # act_hrs = developer hours + review hours   (analysis-P22.R:59)
    df["act_hrs"] = df["Total"] + df["rev_mins"] / 60.0
    return df


def fit_loglog(sp, hrs):
    """OLS of log(hours) on log(story points); returns a, b, r2 for h = a*sp^b."""
    x = np.log(sp.to_numpy(float))
    y = np.log(hrs.to_numpy(float))
    b, log_a = np.polyfit(x, y, 1)
    pred = log_a + b * x
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return float(np.exp(log_a)), float(b), 1 - ss_res / ss_tot


def main():
    df = load()
    # MCO = the implementation-related stories; drop SP=0 and missing values
    mco = df[(df.IsMCO == 1) & df.StoryPoints.notna() & (df.StoryPoints > 0)
             & df.act_hrs.notna() & (df.act_hrs > 0)].copy()
    print(f"P22 stories: {len(df)}  ->  MCO with SP and hours: {len(mco)}")

    a, b, r2 = fit_loglog(mco.StoryPoints, mco.act_hrs)
    print(f"\nfitted   Actual = {a:.2f} * Estimate^{b:.2f}   (R2 = {r2:.2f})")
    print(f"published        1.01 * Estimate^1.09        (R2 = 0.46)")

    # spread of actual/predicted, the factor-of statistic in the README
    ratio = mco.act_hrs / (a * mco.StoryPoints ** b)
    lr = np.log(ratio)
    print(f"\nfactor within 1 sd : {np.exp(lr.std()):.1f}  (published 1.9)")
    print(f"factor within 95%  : {np.exp(np.percentile(np.abs(lr), 95)):.1f}"
          f"  (published 3.4)")

    # per-story-point calibration table: the numbers we will actually reuse
    tbl = mco.groupby("StoryPoints")["act_hrs"].agg(
        n="count", median="median", mean="mean",
        p25=lambda s: s.quantile(.25), p75=lambda s: s.quantile(.75),
    ).round(2)
    tbl["h_per_point"] = (tbl["median"] / tbl.index).round(2)
    print(f"\nhours per story point (P22, MCO stories):\n{tbl.to_string()}")

    overall = float((mco.act_hrs / mco.StoryPoints).median())
    print(f"\noverall median hours/point: {overall:.2f}")

    tbl.to_csv(f"{OUT}/p22_sp_to_hours.csv")
    with open(f"{OUT}/p22_calibration.json", "w") as fh:
        json.dump({
            "source": "Derek-Jones/Software-estimation-datasets, Project-22",
            "n_stories": int(len(mco)),
            "power_law": {"a": round(a, 4), "b": round(b, 4), "r2": round(r2, 4)},
            "median_hours_per_point": round(overall, 4),
            "sd_factor": round(float(np.exp(lr.std())), 3),
        }, fh, indent=2)
    print(f"\nwrote {OUT}/p22_sp_to_hours.csv and {OUT}/p22_calibration.json")


if __name__ == "__main__":
    main()
