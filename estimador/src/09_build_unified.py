"""Harmonise TAWOS + SiP into one training table with leakage-safe splits.

Schema decisions:
  * `effort_hours` is always clocked time. TAWOS contributes Timespent/60,
    SiP contributes HoursActual. Nothing is converted from story points.
  * `task_kind` is a harmonised taxonomy over TAWOS `Type` and SiP
    `SubCategory`, for the categoriser stage. The raw label is kept alongside.
  * SiP contains non-software work (Marketing, Staff Recruitment, Board
    Meeting). Those rows are KEPT but flagged `is_software_task = 0`, and
    excluded from the default splits -- a reversible decision.

Splits:
  * `split`  -- 70/15/15 grouped on normalised text, so duplicated task text
    can never straddle train and test.
  * `split_project` -- whole projects held out, to measure how the model
    generalises to an organisation it has never seen. This is the setting that
    matters for PBE's cold start.
"""
import hashlib
import re

import numpy as np
import pandas as pd

pd.set_option("display.width", 240)
OUT = "data/processed"
SEED = 20260830

# TAWOS Type -> harmonised kind
TAWOS_KIND = {
    "Bug": "bug", "Defect": "bug",
    "Improvement": "improvement", "Enhancement Request": "improvement",
    "Suggestion": "improvement", "Enhancement": "improvement",
    "New Feature": "feature", "Story": "feature", "Feature Request": "feature",
    "Task": "task", "Sub-task": "task", "Technical task": "task",
    "Epic": "epic",
    "Documentation": "docs",
    "Test": "test",
    "Support Request": "support",
}
# SiP SubCategory -> harmonised kind
SIP_KIND = {
    "Bug": "bug",
    "Enhancement": "feature",
    "Support": "support", "In House Support": "support",
    "Client Support": "support",
    "Documentation": "docs", "General Documentation": "docs",
    "Business Specification": "spec", "Technical Specification": "spec",
    "Testing": "test",
    "Research": "research",
    "Release": "release",
    "Training": "training",
    "Conversion": "conversion",
    "Consultancy": "support",
    "Third Party": "other",
}
SIP_NON_SOFTWARE = {"Management"}


def norm(s):
    s = s.fillna("").astype(str).str.lower()
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s


def load_tawos():
    g = pd.read_parquet(f"{OUT}/tawos_gold.parquet")
    out = pd.DataFrame({
        "source": "TAWOS",
        "domain": "open_source_jira",
        "task_id": "TAWOS-" + g.Issue_Key.astype(str),
        "project": g.project,
        "title": g.title,
        "description": g.description,
        "raw_kind": g.Type.fillna("(none)"),
        "task_kind": g.Type.map(TAWOS_KIND).fillna("other"),
        "priority": g.Priority.fillna("(none)"),
        "effort_hours": g.effort_hours,
        "story_point": g.Story_Point,
        "created": g.Creation_Date,
        "has_desc": g.has_desc,
        "has_code": g.has_code,
        "effort_gt_calendar": g.effort_gt_calendar,
        "is_software_task": 1,
    })
    return out


def load_sip():
    t = pd.read_csv("data/raw/jones/SiP/Sip-task-info.csv", encoding="cp1252")
    # HoursActual is a task-level total, constant across the per-developer rows
    t = t.drop_duplicates("TaskNumber").copy()
    t = t[~t.StatusCode.isin(["TEMPLATE", "AUTHORISE", "ESTIMATED", "CANCELLED"])]
    t = t[t.HoursActual > 0]
    out = pd.DataFrame({
        "source": "SiP",
        "domain": "commercial_inhouse",
        "task_id": "SIP-" + t.TaskNumber.astype(str),
        "project": "SiP-" + t.ProjectCode.astype(str),
        "title": t.Summary.fillna("").str.strip(),
        "description": "",
        "raw_kind": t.Category.astype(str) + "/" + t.SubCategory.astype(str),
        "task_kind": t.SubCategory.map(SIP_KIND).fillna("other"),
        "priority": "P" + t.Priority.astype(str),
        "effort_hours": t.HoursActual,
        "story_point": np.nan,
        "created": pd.NaT,
        "has_desc": 0,
        "has_code": 0,
        "effort_gt_calendar": 0,
        "is_software_task": (~t.Category.isin(SIP_NON_SOFTWARE)).astype(int),
    })
    return out


def assign_split(df, rng, fracs=(0.70, 0.15, 0.15)):
    """70/15/15 by text-group, so identical text never straddles splits."""
    # to_numpy(object): rng.shuffle on an Arrow-backed pandas array can
    # duplicate elements, which would silently corrupt the split
    groups = np.asarray(df.text_hash.unique(), dtype=object)
    rng.shuffle(groups)
    assert len(set(groups)) == len(groups), "shuffle duplicated group ids"
    n = len(groups)
    n_tr, n_va = int(n * fracs[0]), int(n * (fracs[0] + fracs[1]))
    lut = {}
    for i, gid in enumerate(groups):
        lut[gid] = "train" if i < n_tr else ("val" if i < n_va else "test")
    return df.text_hash.map(lut)


def assign_project_split(df, rng, target=0.15):
    """Hold out whole projects until ~target of the software rows are covered."""
    sw = df[df.is_software_task == 1]
    sizes = sw.groupby(["source", "project"]).size().sort_values()
    # only consider projects small enough to not swallow the whole set
    cand = sizes[sizes <= len(sw) * 0.12].index.tolist()
    rng.shuffle(cand)
    held, total = [], 0
    for key in cand:
        if total >= len(sw) * target:
            break
        held.append(key)
        total += sizes[key]
    keys = set(held)
    is_held = df.set_index(["source", "project"]).index.isin(keys)
    return np.where(is_held, "test", "train"), held, total


def main():
    df = pd.concat([load_tawos(), load_sip()], ignore_index=True)
    print(f"combined rows: {len(df):,}")
    print(df.groupby(["source", "is_software_task"]).size().to_string())

    df["text"] = (df.title.fillna("") + ". " + df.description.fillna("")).str.strip()
    df["text"] = df.text.str.replace(r"^\. ", "", regex=True).str.strip(". ").str.strip()
    df["text_norm"] = norm(df.text)
    df["text_hash"] = [hashlib.md5(s.encode()).hexdigest()[:16] for s in df.text_norm]
    df["title_words"] = df.title.fillna("").str.split().str.len()
    df["log_effort"] = np.log(df.effort_hours)

    # ---- drop unusable rows ----
    before = len(df)
    df = df[(df.effort_hours > 0) & (df.text_norm.str.len() >= 3)].copy()
    print(f"\ndropped {before-len(df):,} rows with no effort or no usable text")

    print("\n--- duplicate text across the combined set ---")
    dup = df.text_hash.duplicated(keep=False)
    print(f"  rows sharing text with another row: {dup.sum():,} "
          f"({dup.mean()*100:.1f}%) in {df.text_hash.nunique():,} groups")
    cross = (df[dup].groupby("text_hash")["source"].nunique() > 1).sum()
    print(f"  text groups spanning BOTH sources : {cross}")

    rng = np.random.default_rng(SEED)
    sw = df.is_software_task == 1
    df["split"] = pd.NA
    df.loc[sw, "split"] = assign_split(df[sw], rng)

    pj, held, held_n = assign_project_split(df, rng)
    df["split_project"] = pd.NA
    df.loc[sw, "split_project"] = pj[sw.to_numpy()]

    print("\n--- split (grouped on text, software tasks only) ---")
    print(df[sw].split.value_counts().to_string())
    print("\n  leakage check: text groups appearing in more than one split")
    chk = df[sw].groupby("text_hash")["split"].nunique()
    print(f"    {(chk > 1).sum()} groups  (must be 0)")

    print(f"\n--- split_project: {len(held)} projects held out, "
          f"{held_n:,} rows ({held_n/sw.sum()*100:.1f}%) ---")
    print("   " + ", ".join(f"{s}:{p}" for s, p in held[:14])
          + (" ..." if len(held) > 14 else ""))

    print("\n--- harmonised task_kind ---")
    k = (df[sw].groupby(["task_kind", "source"]).size().unstack(fill_value=0))
    k["total"] = k.sum(axis=1)
    k["med_h"] = df[sw].groupby("task_kind").effort_hours.median().round(2)
    print(k.sort_values("total", ascending=False).to_string())

    print("\n--- effort_hours by source (software tasks) ---")
    print(df[sw].groupby("source").effort_hours.describe(
        percentiles=[.25, .5, .75, .95]).round(2).to_string())

    cols = ["task_id", "source", "domain", "project", "title", "description",
            "text", "text_hash", "raw_kind", "task_kind", "priority",
            "effort_hours", "log_effort", "story_point", "created",
            "title_words", "has_desc", "has_code", "effort_gt_calendar",
            "is_software_task", "split", "split_project"]
    df[cols].to_parquet(f"{OUT}/pbe_effort_dataset.parquet",
                        compression="zstd", index=False)
    df[cols].to_csv(f"{OUT}/pbe_effort_dataset.csv", index=False)
    print(f"\nwrote {OUT}/pbe_effort_dataset.parquet  ({len(df):,} rows)")
    print(f"wrote {OUT}/pbe_effort_dataset.csv")
    print(f"  software tasks with a split: {sw.sum():,}")


if __name__ == "__main__":
    main()
