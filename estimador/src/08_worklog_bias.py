"""How do the 2.23% of TAWOS issues WITH a worklog differ from the 97.8% without?

Only issues someone bothered to log time against carry our target variable, so
the gold set is a self-selected sample. This quantifies the selection on every
dimension we can observe, so the thesis can state the bias precisely instead of
waving at it.
"""
import numpy as np
import pandas as pd

pd.set_option("display.width", 240)

COLS = ["ID", "Title", "Description_Text", "Type", "Priority", "Status",
        "Resolution", "Story_Point", "Timespent", "Resolution_Time_Minutes",
        "Creation_Date", "Resolution_Date", "Project_ID", "Sprint_ID",
        "Assignee_ID"]


def rate_table(df, mask, col, top=12):
    """Share of each category that has a worklog, vs the category's overall size."""
    t = (pd.crosstab(df[col].fillna("(nulo)"), mask)
           .rename(columns={False: "sin_log", True: "con_log"}))
    if True not in t.columns and "con_log" not in t.columns:
        return None
    t["total"] = t.sum(axis=1)
    t["tasa_%"] = (t["con_log"] / t["total"] * 100).round(2)
    return t.sort_values("total", ascending=False).head(top)


def main():
    df = pd.read_parquet("data/interim/Issue.parquet", columns=COLS)
    proj = pd.read_parquet("data/interim/Project.parquet")[["ID", "Project_Key"]]
    df = df.merge(proj, left_on="Project_ID", right_on="ID", how="left",
                  suffixes=("", "_p"))
    has = (df.Timespent.notna() & (df.Timespent > 0))
    df["logged"] = has
    base = has.mean() * 100
    print(f"issues: {len(df):,}   with worklog: {has.sum():,} ({base:.2f}%)\n")

    print("=" * 78)
    print("1. CONCENTRATION -- is the worklog habit a property of a few projects?")
    print("=" * 78)
    t = rate_table(df, has, "Project_Key", top=15)
    print(t.to_string())
    g = df[has].Project_Key.value_counts()
    print(f"\n  projects contributing any worklog : {(g > 0).sum()} of {df.Project_Key.nunique()}")
    print(f"  top-1 project share of the gold set: {g.iloc[0]/g.sum()*100:.1f}%  ({g.index[0]})")
    print(f"  top-3 project share               : {g.head(3).sum()/g.sum()*100:.1f}%")
    print(f"  top-5 project share               : {g.head(5).sum()/g.sum()*100:.1f}%")

    print("\n" + "=" * 78)
    print("2. TASK TYPE -- are logged issues a different kind of work?")
    print("=" * 78)
    print(rate_table(df, has, "Type", top=12).to_string())

    print("\n" + "=" * 78)
    print("3. PRIORITY")
    print("=" * 78)
    print(rate_table(df, has, "Priority", top=8).to_string())

    print("\n" + "=" * 78)
    print("4. RESOLUTION -- logged issues are presumably finished work")
    print("=" * 78)
    print(rate_table(df, has, "Resolution", top=8).to_string())

    print("\n" + "=" * 78)
    print("5. SIZE AND TEXT -- are logged issues bigger or better documented?")
    print("=" * 78)
    df["title_len"] = df.Title.fillna("").str.len()
    df["desc_len"] = df.Description_Text.fillna("").str.len()
    df["cal_h"] = (df.Resolution_Date - df.Creation_Date).dt.total_seconds() / 3600
    comp = df.groupby("logged")[["title_len", "desc_len", "cal_h",
                                 "Story_Point"]].median().T
    comp.columns = ["sin_log", "con_log"]
    comp["ratio"] = (comp["con_log"] / comp["sin_log"]).round(2)
    print(comp.round(2).to_string())
    print(f"\n  has a description : sin_log {(df[~has].desc_len>0).mean()*100:.1f}%"
          f"  | con_log {(df[has].desc_len>0).mean()*100:.1f}%")
    print(f"  has story points  : sin_log {(df[~has].Story_Point>0).mean()*100:.1f}%"
          f"  | con_log {(df[has].Story_Point>0).mean()*100:.1f}%")
    print(f"  has an assignee   : sin_log {df[~has].Assignee_ID.notna().mean()*100:.1f}%"
          f"  | con_log {df[has].Assignee_ID.notna().mean()*100:.1f}%")
    print(f"  is in a sprint    : sin_log {df[~has].Sprint_ID.notna().mean()*100:.1f}%"
          f"  | con_log {df[has].Sprint_ID.notna().mean()*100:.1f}%")

    print("\n" + "=" * 78)
    print("6. TIME -- was time tracking a phase the teams went through?")
    print("=" * 78)
    df["year"] = df.Creation_Date.dt.year
    y = (pd.crosstab(df.year, has).rename(columns={False: "sin_log", True: "con_log"}))
    y["total"] = y.sum(axis=1)
    y["tasa_%"] = (y["con_log"] / y["total"] * 100).round(2)
    print(y[y.total > 500].to_string())

    print("\n" + "=" * 78)
    print("7. WITHIN THE TOP PROJECT (MDL) -- selection even where logging happens")
    print("=" * 78)
    mdl = df[df.Project_Key == "MDL"]
    print(f"  MDL issues {len(mdl):,}, logged {mdl.logged.sum():,} "
          f"({mdl.logged.mean()*100:.2f}%)")
    print(rate_table(mdl, mdl.logged, "Type", top=8).to_string())


if __name__ == "__main__":
    main()
