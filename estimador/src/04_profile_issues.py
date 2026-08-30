"""Profile the TAWOS Issue table: how much usable text+hours data is really there.

The whole crossing plan hinges on the answers here, so this script asks the
questions explicitly rather than assuming:
  1. What unit is `Timespent` in? (Jira's REST API reports seconds.)
  2. How many issues have logged time at all, and with usable text?
  3. How many have BOTH story points and logged time -- the calibration subset?
  4. How is that spread across the 39 projects?
"""
import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
INTERIM = "data/interim"

COLS = ["ID", "Issue_Key", "Title", "Description_Text", "Type", "Priority",
        "Status", "Resolution", "Story_Point", "Timespent",
        "In_Progress_Minutes", "Total_Effort_Minutes",
        "Resolution_Time_Minutes", "Project_ID", "Sprint_ID", "Assignee_ID",
        "Story_Point_Changed_After_Estimation"]


def unit_check(df):
    """Timespent vs the derived *_Minutes columns tells us the unit."""
    both = df[df.Timespent.notna() & (df.Timespent > 0)
              & df.Total_Effort_Minutes.notna() & (df.Total_Effort_Minutes > 0)]
    print(f"\n--- unit check ({len(both):,} issues with Timespent and Total_Effort_Minutes) ---")
    if len(both):
        ratio = both.Timespent / both.Total_Effort_Minutes
        print(f"  Timespent / Total_Effort_Minutes: median {ratio.median():.2f}")
    print("  Timespent quantiles:")
    ts = df.Timespent.dropna()
    ts = ts[ts > 0]
    for q in (0.01, 0.25, 0.5, 0.75, 0.95, 0.99):
        v = ts.quantile(q)
        print(f"    p{int(q*100):02d} = {v:12,.0f}  "
              f"(= {v/3600:8.2f} h if seconds | {v/60:8.2f} h if minutes)")
    print(f"    min = {ts.min():,.0f}   max = {ts.max():,.0f}")


def main():
    df = pd.read_parquet(f"{INTERIM}/Issue.parquet", columns=COLS)
    print(f"issues: {len(df):,}")

    print("\n--- field availability ---")
    has_ts = df.Timespent.notna() & (df.Timespent > 0)
    has_sp = df.Story_Point.notna() & (df.Story_Point > 0)
    title = df.Title.fillna("").str.strip().str.strip('"').str.strip()
    desc = df.Description_Text.fillna("").str.strip().str.strip('"').str.strip()
    has_txt = (title.str.len() > 0) & (desc.str.len() > 0)
    for name, mask in [
        ("story points > 0", has_sp),
        ("Timespent > 0", has_ts),
        ("title + description", has_txt),
        ("SP and Timespent", has_sp & has_ts),
        ("Timespent + text", has_ts & has_txt),
        ("SP + Timespent + text", has_sp & has_ts & has_txt),
    ]:
        print(f"  {name:24s} {mask.sum():8,}  ({mask.mean()*100:5.2f}%)")

    unit_check(df)

    print("\n--- per project (issues with Timespent>0) ---")
    proj = pd.read_parquet(f"{INTERIM}/Project.parquet")[["ID", "Project_Key"]]
    g = (df.assign(has_ts=has_ts, has_sp=has_sp, has_txt=has_txt)
           .groupby("Project_ID")
           .agg(issues=("ID", "size"), with_sp=("has_sp", "sum"),
                with_ts=("has_ts", "sum"),
                with_all=("has_ts", lambda s: 0)))
    both = df.assign(k=(has_sp & has_ts & has_txt)).groupby("Project_ID")["k"].sum()
    g["with_all"] = both
    g = g.join(proj.set_index("ID")).sort_values("with_ts", ascending=False)
    print(g[g.with_ts > 0][["Project_Key", "issues", "with_sp", "with_ts", "with_all"]]
          .to_string())

    print("\n--- issue Type distribution (Timespent>0) ---")
    print(df[has_ts].Type.value_counts().head(15).to_string())


if __name__ == "__main__":
    main()
