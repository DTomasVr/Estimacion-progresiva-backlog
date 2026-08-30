"""Determine the unit of TAWOS `Timespent` empirically.

Jira's REST API reports `timespent` in SECONDS, but TAWOS is a derived dataset
and its paper describes the field as "work hours logged". Everything
downstream depends on getting this right, so we test it rather than assume.

Test: logged effort can never exceed the calendar time the issue was open.
`Resolution_Time_Minutes` is calendar minutes from creation to resolution.
If we read Timespent as MINUTES and it routinely exceeds calendar time, the
minutes reading is wrong. If it never does, seconds would make tasks
absurdly short (a median of two minutes of work per issue).
"""
import numpy as np
import pandas as pd

pd.set_option("display.width", 220)

df = pd.read_parquet(
    "data/interim/Issue.parquet",
    columns=["Issue_Key", "Type", "Timespent", "In_Progress_Minutes",
             "Total_Effort_Minutes", "Resolution_Time_Minutes",
             "Creation_Date", "Resolution_Date", "Project_ID"],
)
d = df[df.Timespent.notna() & (df.Timespent > 0)].copy()
print(f"issues with Timespent > 0: {len(d):,}\n")

print("--- raw distributions of the four time columns (issues with Timespent) ---")
print(d[["Timespent", "In_Progress_Minutes", "Total_Effort_Minutes",
         "Resolution_Time_Minutes"]]
      .describe(percentiles=[.25, .5, .75, .95]).round(1).to_string())

# calendar minutes the issue was actually open
d["calendar_min"] = (d.Resolution_Date - d.Creation_Date).dt.total_seconds() / 60
ok = d[d.calendar_min.notna() & (d.calendar_min > 0)].copy()
print(f"\nissues with a resolved calendar span: {len(ok):,}")

for unit, factor in [("seconds", 1 / 60), ("minutes", 1.0), ("hours", 60.0)]:
    eff_min = ok.Timespent * factor
    viol = (eff_min > ok.calendar_min).mean()
    med_h = float(np.median(eff_min / 60))
    print(f"  as {unit:7s}: median effort {med_h:8.2f} h | "
          f"exceeds calendar span in {viol*100:5.2f}% of issues")

print("\n--- Timespent value histogram (top 20 exact values) ---")
vc = d.Timespent.value_counts().head(20)
print(pd.DataFrame({
    "value": vc.index, "count": vc.values,
    "as_min_h": (vc.index / 60).round(2), "as_sec_h": (vc.index / 3600).round(3),
}).to_string(index=False))

print("\n--- are values multiples of 60? (a seconds field usually is) ---")
for m in (60, 30, 15):
    print(f"  divisible by {m:3d}: {(d.Timespent % m == 0).mean()*100:5.1f}%")

print("\n--- per-project median Timespent (top projects) ---")
proj = pd.read_parquet("data/interim/Project.parquet")[["ID", "Project_Key"]]
g = (d.groupby("Project_ID")["Timespent"]
       .agg(n="size", median="median", p90=lambda s: s.quantile(.9))
       .join(proj.set_index("ID")).sort_values("n", ascending=False).head(12))
g["median_as_h_if_min"] = (g["median"] / 60).round(2)
print(g.to_string())
