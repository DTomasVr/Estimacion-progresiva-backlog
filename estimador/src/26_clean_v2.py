"""Second cleaning pass: make every row mean "one developer's time on one task".

Agix's tracker measures a single developer working on a single task. Rows that
mean something else teach the model a quantity it will never be asked for.

  E1  drop TAWOS Epics -- containers by definition
  E2  drop container-language rows above 16 h (projects, plans, catch-alls)
  E3  SiP multi-developer tasks: use the largest contributor's hours instead of
      the sum across everyone, so the target stays per-developer
  E4  flag (do not drop) the quantisation of the target for the write-up

Splits are recomputed afterwards on the surviving rows.
"""
import sys

import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
IN = "data/processed/agix_effort_clean.parquet"
OUT = "data/processed/agix_effort_v2.parquet"
SEED = 20260830

CONTAINER = (r"\b(project|phase|programme|initiative|roadmap|plan|planning|"
             r"meeting|workshop|kick.?off|various|misc|miscellaneous|"
             r"catch.?all|assorted|month.?end|period close|ongoing|routine|"
             r"housekeeping)\b")


def main():
    df = pd.read_parquet(IN)
    n0 = len(df)
    print(f"entrada: {n0:,} filas ({(df.is_software_task==1).sum():,} de software)\n")

    # ---- E3: per-developer target for SiP ------------------------------
    t = pd.read_csv("data/raw/jones/SiP/Sip-task-info.csv", encoding="cp1252")
    g = t.groupby("TaskNumber")
    ndev = g.DeveloperID.nunique()
    dmax = g.DeveloperHoursActual.max()
    tn = df.task_id.str.replace("SIP-", "", regex=False)
    tn = pd.to_numeric(tn, errors="coerce")
    df["n_dev"] = tn.map(ndev).fillna(1).astype(int)
    df["effort_task_total"] = df.effort_hours
    per_dev = tn.map(dmax)
    adjust = (df.source == "SiP") & (df.n_dev > 1) & per_dev.notna() & (per_dev > 0)
    df.loc[adjust, "effort_hours"] = per_dev[adjust]
    print(f"E3  {int(adjust.sum()):,} filas SiP reexpresadas como horas del "
          f"desarrollador principal")
    print(f"    mediana {df.loc[adjust,'effort_task_total'].median():.2f} h -> "
          f"{df.loc[adjust,'effort_hours'].median():.2f} h")

    # ---- E1 / E2: rows that are not a single task ----------------------
    low = df.text.str.lower()
    is_epic = df.raw_kind.eq("Epic")
    cont = low.str.contains(CONTAINER, regex=True, na=False)
    drop = is_epic | (cont & (df.effort_hours > 16))
    print(f"\nE1  epics: {int(is_epic.sum())}")
    print(f"E2  contenedores > 16 h: {int((cont & (df.effort_hours>16)).sum())}")
    print(f"    total descartado: {int(drop.sum())}")
    df = df[~drop].copy()

    df["log_effort"] = np.log(df.effort_hours)

    # ---- E4: quantisation flag ----------------------------------------
    e = df.effort_hours
    df["objetivo_redondo"] = (np.abs(e - np.round(e)) < 1e-6).astype(int)
    print(f"\nE4  {int(df.objetivo_redondo.sum()):,} filas con esfuerzo en horas "
          f"enteras ({df.objetivo_redondo.mean()*100:.1f}%) -- marcadas, no alteradas")

    # ---- recompute splits on the survivors ------------------------------
    rng = np.random.default_rng(SEED)
    sw = df.is_software_task == 1
    keys = np.asarray(df.loc[sw, "group_key"].unique(), dtype=object)
    rng.shuffle(keys)
    assert len(set(keys)) == len(keys)
    n_tr, n_va = int(len(keys) * .70), int(len(keys) * .85)
    lut = {k: ("train" if i < n_tr else "val" if i < n_va else "test")
           for i, k in enumerate(keys)}
    df["split"] = pd.NA
    df.loc[sw, "split"] = df.loc[sw, "group_key"].map(lut)

    sizes = df[sw].groupby(["source", "project"]).size().sort_values()
    cand = [k for k in sizes.index if sizes[k] <= sw.sum() * 0.12]
    rng.shuffle(cand)
    held, tot = [], 0
    for k in cand:
        if tot >= sw.sum() * .15:
            break
        held.append(k)
        tot += sizes[k]
    idx = df.set_index(["source", "project"]).index
    df["split_project"] = pd.NA
    df.loc[sw, "split_project"] = np.where(
        idx.isin(set(held)), "test", "train")[sw.to_numpy()]

    swd = df[sw]
    print(f"\n--- resultado ---")
    print(f"  filas totales      : {len(df):,}   (antes {n0:,})")
    print(f"  tareas de software : {len(swd):,}")
    print(swd.groupby("source").effort_hours.describe(
        percentiles=[.5, .95]).round(2).to_string())
    print(f"\n  desviacion tipica en log: {np.std(np.log(swd.effort_hours)):.3f}")
    print(f"  tareas > 40 h: {int((swd.effort_hours>40).sum()):,} "
          f"({(swd.effort_hours>40).mean()*100:.2f}%)   maximo "
          f"{swd.effort_hours.max():.1f} h")
    print(f"  particion: {swd.split.value_counts().to_dict()}")
    chk = swd.groupby("group_key").split.nunique()
    print(f"  fuga: {int((chk>1).sum())} grupos en mas de un split (debe ser 0)")

    df.to_parquet(OUT, compression="zstd", index=False)
    df.to_csv(OUT.replace(".parquet", ".csv"), index=False)
    print(f"\nescrito {OUT}")


if __name__ == "__main__":
    main()
