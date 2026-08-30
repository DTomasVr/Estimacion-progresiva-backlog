"""Build the TAWOS gold set: issues with real logged effort and usable text.

Decisions encoded here, all established empirically in earlier steps:
  * effort_hours = Timespent / 60      (05_timespent_units.py -- Timespent is minutes)
  * Title is 100% wrapped in "..."     and Description_Text 93.6% in \"\"\"...\"\"\"
    -- a load artifact of the TAWOS build, stripped here.
  * Total_Effort_Minutes / In_Progress_Minutes / Resolution_Time_Minutes are
    STATUS-derived calendar spans, not logged effort. Kept as features, never
    as the target.
"""
import numpy as np
import pandas as pd

pd.set_option("display.width", 220)

INTERIM, OUT = "data/interim", "data/processed"

COLS = ["ID", "Issue_Key", "Title", "Description_Text", "Description_Code",
        "Type", "Priority", "Status", "Resolution", "Story_Point", "Timespent",
        "In_Progress_Minutes", "Total_Effort_Minutes", "Resolution_Time_Minutes",
        "Creation_Date", "Resolution_Date", "Project_ID", "Sprint_ID",
        "Assignee_ID", "Reporter_ID", "Story_Point_Changed_After_Estimation",
        "Title_Changed_After_Estimation", "Description_Changed_After_Estimation"]


def clean_text(s):
    """Strip the CSV-quote wrappers the TAWOS build baked into the text."""
    s = s.fillna("").astype(str).str.strip()
    s = s.str.replace(r'^"""', "", regex=True).str.replace(r'"""$', "", regex=True)
    s = s.str.replace(r'^"', "", regex=True).str.replace(r'"$', "", regex=True)
    return s.str.strip()


def main():
    df = pd.read_parquet(f"{INTERIM}/Issue.parquet", columns=COLS)
    proj = pd.read_parquet(f"{INTERIM}/Project.parquet")[["ID", "Project_Key"]]
    proj = proj.rename(columns={"ID": "Project_ID", "Project_Key": "project"})

    df["title"] = clean_text(df.Title)
    df["description"] = clean_text(df.Description_Text)
    df["code_len"] = clean_text(df.Description_Code).str.len()

    g = df[df.Timespent.notna() & (df.Timespent > 0)].copy()
    print(f"issues with logged effort: {len(g):,}")

    g["effort_hours"] = g.Timespent / 60.0
    g = g.merge(proj, on="Project_ID", how="left")

    g["title_len"] = g.title.str.len()
    g["desc_len"] = g.description.str.len()
    g["title_words"] = g.title.str.split().str.len().fillna(0).astype(int)
    g["desc_words"] = g.description.str.split().str.len().fillna(0).astype(int)
    g["has_code"] = (g.code_len > 0).astype(int)
    g["has_desc"] = (g.desc_len > 0).astype(int)
    g["text"] = (g.title + ". " + g.description).str.strip()

    print(f"  with a non-empty title      : {(g.title_len>0).sum():,}")
    print(f"  with a non-empty description: {g.has_desc.sum():,}")
    print(f"  with story points too       : {(g.Story_Point>0).sum():,}")

    # ---- duplicate text: the main leakage risk for a text model ----
    print("\n--- duplicate text (leakage risk) ---")
    for col in ["title", "text"]:
        dup = g[g[col].str.len() > 0][col].duplicated(keep=False)
        print(f"  exact duplicate {col:6s}: {dup.sum():,} rows "
              f"({dup.mean()*100:.1f}%) in {g[g[col].str.len()>0][col].nunique():,} groups")

    # ---- effort distribution and outliers ----
    print("\n--- effort_hours distribution ---")
    q = g.effort_hours.describe(percentiles=[.01, .25, .5, .75, .95, .99])
    print(q.round(2).to_string())
    for thr in (0.25, 8, 40, 80, 160):
        print(f"  > {thr:5.2f} h : {(g.effort_hours>thr).sum():6,} "
              f"({(g.effort_hours>thr).mean()*100:5.2f}%)")
    print(f"  < 0.25 h (<15 min): {(g.effort_hours<0.25).sum():,}")

    # sanity: logged effort must not exceed the calendar span by much
    cal_h = (g.Resolution_Date - g.Creation_Date).dt.total_seconds() / 3600
    impossible = (cal_h.notna()) & (g.effort_hours > cal_h)
    print(f"\n  effort exceeds calendar span: {impossible.sum():,} "
          f"({impossible.mean()*100:.2f}%)  -- flagged, not dropped")
    g["effort_gt_calendar"] = impossible.astype(int)
    g["calendar_hours"] = cal_h

    keep = ["ID", "Issue_Key", "project", "Project_ID", "title", "description",
            "text", "Type", "Priority", "Status", "Resolution", "Story_Point",
            "effort_hours", "calendar_hours", "In_Progress_Minutes",
            "Total_Effort_Minutes", "Resolution_Time_Minutes", "Creation_Date",
            "Resolution_Date", "Sprint_ID", "Assignee_ID", "Reporter_ID",
            "title_len", "desc_len", "title_words", "desc_words", "has_code",
            "has_desc", "effort_gt_calendar",
            "Story_Point_Changed_After_Estimation",
            "Title_Changed_After_Estimation",
            "Description_Changed_After_Estimation"]
    out = g[keep].sort_values(["project", "ID"]).reset_index(drop=True)
    out.to_parquet(f"{OUT}/tawos_gold.parquet", compression="zstd", index=False)
    print(f"\nwrote {OUT}/tawos_gold.parquet  ({len(out):,} rows, {len(keep)} cols)")

    # the calibration subset: both story points and real hours
    cal = out[(out.Story_Point > 0) & out.effort_hours.notna()].copy()
    cal.to_parquet(f"{OUT}/tawos_sp_hours.parquet", compression="zstd", index=False)
    print(f"wrote {OUT}/tawos_sp_hours.parquet ({len(cal):,} rows) "
          f"across {cal.project.nunique()} projects")
    print("\n--- calibration subset per project ---")
    print(cal.groupby("project").agg(
        n=("effort_hours", "size"),
        med_h=("effort_hours", "median"),
        med_sp=("Story_Point", "median"),
        h_per_sp=("effort_hours", lambda s: np.nan),
    ).assign(h_per_sp=cal.groupby("project").apply(
        lambda d: (d.effort_hours / d.Story_Point).median(), include_groups=False))
     .sort_values("n", ascending=False).round(2).to_string())


if __name__ == "__main__":
    main()
