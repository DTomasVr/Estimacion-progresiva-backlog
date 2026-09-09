"""Dataset v3: every variable the audit surfaced, plus the literature's split.

Adds to v2:
  * SiP    HoursEstimate, AssignedToID, RaisedByID, ProjectBreakdownCode
  * TAWOS  n_links / n_links_estructurales / n_componentes, the three
           *_Changed_After_Estimation flags, Assignee_ID, sprint membership
  * both   split_temporal -- a chronological within-project split, which is the
           setting Deep-SE / GPT2SP report and the one Agix reaches once a team
           has history

Leakage notes carried into the file rather than left implicit:
  * Issue_Link has no timestamp, so `n_links` may include links added during the
    work. `n_links_estructurales` (Depends/Blocks/Part) is the conservative
    subset; both are stored so the optimistic and defensible readings stay
    separable.
  * TaskPerformance and DeveloperPerformance are derived from the actual hours
    and are deliberately NOT imported.
"""
import sys

import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
IN = "data/processed/agix_effort_v2.parquet"
OUT = "data/processed/agix_effort_v3.parquet"
SEED = 20260830
TEST_FRAC = 0.25
ESTRUCTURAL = {"Depends", "Dependency", "Blocks", "Blockers", "Part",
               "Supercedes", "Incorporates"}


def sip_extras():
    t = pd.read_csv("data/raw/jones/SiP/Sip-task-info.csv", encoding="cp1252")
    t = t.drop_duplicates("TaskNumber")
    t["task_id"] = "SIP-" + t.TaskNumber.astype(str)
    return t.set_index("task_id")[["HoursEstimate", "AssignedToID",
                                   "RaisedByID", "ProjectBreakdownCode"]]


def tawos_extras():
    iss = pd.read_parquet(
        "data/interim/Issue.parquet",
        columns=["ID", "Issue_Key", "Assignee_ID", "Sprint_ID",
                 "Title_Changed_After_Estimation",
                 "Description_Changed_After_Estimation",
                 "Story_Point_Changed_After_Estimation"])
    iss["task_id"] = "TAWOS-" + iss.Issue_Key.astype(str)

    lk = pd.read_parquet("data/interim/Issue_Link.parquet")
    cp = pd.read_parquet("data/interim/Issue_Component.parquet")
    for d in (lk, cp):
        d["Issue_ID"] = pd.to_numeric(d.Issue_ID, errors="coerce").astype("Int64")
    lk, cp = lk[lk.Issue_ID.notna()], cp[cp.Issue_ID.notna()]

    feats = pd.DataFrame(index=iss.ID)
    feats["n_links"] = lk.groupby("Issue_ID").size()
    feats["n_links_estructurales"] = (
        lk[lk.Name.isin(ESTRUCTURAL)].groupby("Issue_ID").size())
    feats["n_componentes"] = cp.groupby("Issue_ID").size()
    feats = feats.fillna(0.0)

    out = iss.set_index("task_id").join(feats, on="ID")
    out["en_sprint"] = out.Sprint_ID.notna().astype(int)
    out["tiene_asignado"] = out.Assignee_ID.notna().astype(int)
    return out[["Assignee_ID", "en_sprint", "tiene_asignado", "n_links",
                "n_links_estructurales", "n_componentes",
                "Title_Changed_After_Estimation",
                "Description_Changed_After_Estimation",
                "Story_Point_Changed_After_Estimation"]]


def main():
    df = pd.read_parquet(IN)
    print(f"entrada v2: {len(df):,} filas "
          f"({(df.is_software_task==1).sum():,} de software)")

    s, t = sip_extras(), tawos_extras()
    df = df.join(s, on="task_id").join(t, on="task_id", rsuffix="_tw")

    # unified developer column across the two sources
    df["desarrollador"] = np.where(
        df.source == "SiP",
        "sip_" + df.AssignedToID.astype("string"),
        "tawos_" + df.Assignee_ID.astype("string"))
    df.loc[df.desarrollador.str.contains("<NA>", na=True), "desarrollador"] = pd.NA

    df["tiene_estimacion"] = (df.HoursEstimate.notna()
                              & (df.HoursEstimate > 0)).astype(int)
    for c in ["n_links", "n_links_estructurales", "n_componentes", "en_sprint",
              "tiene_asignado", "Title_Changed_After_Estimation",
              "Description_Changed_After_Estimation",
              "Story_Point_Changed_After_Estimation"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    print("\n--- cobertura de las variables nuevas ---")
    sw = df[df.is_software_task == 1]
    for c, etiqueta in [("tiene_estimacion", "estimacion humana"),
                        ("desarrollador", "desarrollador identificado"),
                        ("n_links", "con algun enlace"),
                        ("n_componentes", "con algun componente"),
                        ("en_sprint", "asignada a un sprint"),
                        ("ProjectBreakdownCode", "codigo de desglose")]:
        if c in ("desarrollador", "ProjectBreakdownCode"):
            cov = sw[c].notna().mean()
        elif c in ("tiene_estimacion", "en_sprint"):
            cov = (sw[c] == 1).mean()
        else:
            cov = (sw[c] > 0).mean()
        print(f"  {etiqueta:28s} {cov*100:5.1f}%   "
              f"({int(cov*len(sw)):,} de {len(sw):,})")

    # ---- chronological within-project split (the literature's setting) ----
    df["orden"] = df.created.fillna(pd.Timestamp("2000-01-01"))
    df = df.sort_values(["project", "orden", "task_id"]).reset_index(drop=True)
    df["split_temporal"] = pd.NA
    swm = df.is_software_task == 1
    for proj, idx in df[swm].groupby("project").groups.items():
        idx = list(idx)
        if len(idx) < 60:
            df.loc[idx, "split_temporal"] = "train"
            continue
        n_te = max(15, int(len(idx) * TEST_FRAC))
        df.loc[idx[:-n_te], "split_temporal"] = "train"
        df.loc[idx[-n_te:], "split_temporal"] = "test"

    st = df[swm].split_temporal.value_counts()
    print(f"\n--- particion temporal dentro de cada proyecto ---")
    print(f"  {st.to_dict()}   proyectos con prueba: "
          f"{df[swm & (df.split_temporal=='test')].project.nunique()}")

    # a chronological leak is possible only if the same text repeats over time
    fuga = (df[swm & (df.split_temporal == 'test')].text_hash
            .isin(df[swm & (df.split_temporal == 'train')].text_hash).sum())
    print(f"  textos de prueba que ya aparecen en entrenamiento: {int(fuga)}")

    print("\n--- relacion de las variables nuevas con el esfuerzo ---")
    for c in ["HoursEstimate", "n_links", "n_links_estructurales",
              "n_componentes", "en_sprint", "Title_Changed_After_Estimation",
              "Description_Changed_After_Estimation",
              "Story_Point_Changed_After_Estimation"]:
        d = sw[sw[c].notna()]
        if len(d) < 100:
            continue
        rho = d[c].corr(d.effort_hours, method="spearman")
        print(f"  {c:38s} Spearman {rho:+.4f}   n={len(d):,}")

    df.to_parquet(OUT, compression="zstd", index=False)
    df.to_csv(OUT.replace(".parquet", ".csv"), index=False)
    print(f"\nescrito {OUT}  ({len(df):,} filas, {df.shape[1]} columnas)")
    print(f"  tareas de software: {int(swm.sum()):,}")


if __name__ == "__main__":
    main()
