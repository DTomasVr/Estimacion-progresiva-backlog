"""Do dependency and component counts carry size information the text lacks?

An issue with twelve blockers or spanning five components is probably large,
and its title will not say so. This tests link and component counts as features
on the TAWOS rows, which are the only ones with the relational structure.

Leakage caveat, stated up front: Issue_Link carries no timestamp, so we cannot
tell whether a link existed when the task was estimated or was added while the
work was done. Counts of `Blocks`/`Depends` are plausibly planning-time;
`Duplicate` and `Regression` are plausibly discovered later. The experiment
therefore reports a conservative subset (structural link types + components)
alongside the all-links version, so the optimistic and defensible readings are
visible separately.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402
import features as F  # noqa: E402

pd.set_option("display.width", 250)
E.DATA = "data/processed/pbe_effort_v2.parquet"

# link types that plausibly exist before the work starts
ESTRUCTURAL = {"Depends", "Dependency", "Blocks", "Blockers", "Part",
               "Supercedes", "Incorporates"}


def link_features():
    l = pd.read_parquet("data/interim/Issue_Link.parquet")
    c = pd.read_parquet("data/interim/Issue_Component.parquet")
    # the extractor only cast the columns in its NUMERIC map, so the foreign
    # keys of these tables came through as strings
    for d in (l, c):
        d["Issue_ID"] = pd.to_numeric(d.Issue_ID, errors="coerce").astype("Int64")
    l, c = l[l.Issue_ID.notna()], c[c.Issue_ID.notna()]
    tot = l.groupby("Issue_ID").size().rename("n_links")
    est = (l[l.Name.isin(ESTRUCTURAL)].groupby("Issue_ID").size()
           .rename("n_links_estructurales"))
    out = (l[l.Direction == "OUTBOUND"].groupby("Issue_ID").size()
           .rename("n_links_salientes"))
    ncomp = c.groupby("Issue_ID").size().rename("n_componentes")
    return pd.concat([tot, est, out, ncomp], axis=1).fillna(0.0)


def build(tr, te, emb, pos, extra_cols):
    fb = F.FeatureBuilder(use_kind=True, use_source=False, use_priority=False,
                          use_numeric=False, use_project=False)
    Xtr, Xte = fb.fit_transform(tr, [te], emb, pos)
    if extra_cols:
        a = np.log1p(tr[extra_cols].to_numpy(float))
        b = np.log1p(te[extra_cols].to_numpy(float))
        Xtr, Xte = np.hstack([Xtr, a]), np.hstack([Xte, b])
    return Xtr, Xte


def main():
    lf = link_features()
    df = pd.read_parquet(E.DATA)
    sw = df[(df.is_software_task == 1) & (df.source == "TAWOS")].copy()
    sw["Issue_ID"] = sw.task_id.str.replace("TAWOS-", "", regex=False)
    key = pd.read_parquet("data/interim/Issue.parquet",
                          columns=["ID", "Issue_Key"])
    key["Issue_ID"] = key.Issue_Key.astype(str)
    sw = sw.merge(key[["ID", "Issue_ID"]], on="Issue_ID", how="left")
    sw = sw.merge(lf, left_on="ID", right_index=True, how="left")
    for c in lf.columns:
        sw[c] = sw[c].fillna(0.0)
    print(f"tareas TAWOS: {len(sw):,}   con al menos un enlace: "
          f"{int((sw.n_links>0).sum()):,} ({(sw.n_links>0).mean()*100:.1f}%)   "
          f"con componente: {int((sw.n_componentes>0).sum()):,} "
          f"({(sw.n_componentes>0).mean()*100:.1f}%)")

    print("\n=== relacion cruda con el esfuerzo ===")
    for c in lf.columns:
        rho = sw[c].corr(sw.effort_hours, method="spearman")
        print(f"  {c:24s} Spearman {rho:+.4f}   mediana esfuerzo "
              f"si=0 {sw.loc[sw[c]==0,'effort_hours'].median():5.2f} h | "
              f">0 {sw.loc[sw[c]>0,'effort_hours'].median():5.2f} h")

    sets = {
        "solo_texto": [],
        "texto+componentes": ["n_componentes"],
        "texto+links_estructurales": ["n_links_estructurales", "n_componentes"],
        "texto+todos_los_links": list(lf.columns),
        "solo_links": None,
    }
    emb, pos = F.load_embeddings_and_pos()
    for split_col in ["split", "split_project"]:
        tr = sw[sw[split_col] == "train"]
        te = sw[sw[split_col] == "test"]
        if len(te) < 100:
            continue
        y_tr = np.log(tr.effort_hours.to_numpy())
        y_te = te.effort_hours.to_numpy()
        print(f"\n{'='*78}\nsplit={split_col}  train {len(tr):,}  test {len(te):,}"
              f"\n{'='*78}")
        rows = []
        for name, cols in sets.items():
            if cols is None:
                Xtr = np.log1p(tr[list(lf.columns)].to_numpy(float))
                Xte = np.log1p(te[list(lf.columns)].to_numpy(float))
            else:
                Xtr, Xte = build(tr, te, emb, pos, cols)
            for head, mk in [("ridge", lambda: Ridge(alpha=1.0)),
                             ("hgb", lambda: HistGradientBoostingRegressor(
                                 max_iter=400, max_depth=6, learning_rate=0.06,
                                 random_state=0))]:
                p = np.exp(mk().fit(Xtr, y_tr).predict(Xte))
                m = E.point_metrics(y_te, p)
                comp = float(np.std(np.log(p)) / np.std(np.log(y_te)))
                sp = float(pd.Series(y_te).corr(pd.Series(p), method="spearman"))
                rows.append(dict(bloques=name, head=head,
                                 MdMRE=round(m["MdMRE"], 4),
                                 PRED25=round(m["PRED25"], 4),
                                 RMSE_log=round(m["RMSE_log"], 4),
                                 compresion=round(comp, 3), spearman=round(sp, 3),
                                 grandes=f"{int(((y_te>16)&(p>16)).sum())}/"
                                         f"{int((y_te>16).sum())}"))
                E.record("enlaces", f"{head}_{name}", split_col,
                         {**m, "compresion": comp, "spearman": sp})
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
